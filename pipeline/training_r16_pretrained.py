"""Versioned R16 public-pretrained PEFT executor; never edits a frozen run.

Checkpoint selection uses the existing own-fold CV role. Shared local_fit is
read only after each job completes, for the later capped ensemble policy.
No Test or locked report panel is read during training/preflight.
"""
from __future__ import annotations
import gc, hashlib, json, math, os, random, signal, time
from pathlib import Path
import numpy as np
from pipeline.r16_multisupport_data import ROOT, OUT, REGIMES, role, read, write, sha, save, guard
from pipeline.training_r16_multisupport import (
    features, codes as old_codes, verify, cache_role, cache_read, atomic_pt, metric,
)

RUN = ROOT/'models/r16_pretrained_ft_r1'
OPS = ROOT/'ops/r16_pretrained_ft_r1'
CACHE = ROOT/'data/cache/r16_pretrained_ft_r1'
FAMILIES = ('dasheng_base', 'dinov2_small', 'timemoe')
TASKS = ('regression', 'classification')
CLASSIFIER_CHANNELS = (0,1,2,3,6,5,8,9,10,11,12,13,14,15,16)
SEED = 202610281901

def source_pins():
    pins = old_codes()
    for name in ('pipeline/training_r16_pretrained.py',
                 'pipeline/pretrained_timemoe_r16.py', 'pipeline/pretrained_audio2d_r16.py',
                 'pipeline/load_foundation_r5.py', 'pipeline/foundation_masks_r5.py',
                 'ops/run_r16_pretrained.py', 'ops/finalize_r16_pretrained.py'):
        pins[name] = sha(ROOT/name)
    return pins

def branch(family, task, device='cpu'):
    if family == 'timemoe':
        from pipeline.pretrained_timemoe_r16 import create_timemoe_branch
        return create_timemoe_branch(task, rank=8, dropout=.1, device=device, dtype='float32')
    from pipeline.pretrained_audio2d_r16 import create_audio2d_branch
    return create_audio2d_branch(family, task, rank=8, dropout=.1, device=device, dtype='float32')

def base_fingerprint(model):
    h = hashlib.sha256()
    for name, p in model.frozen_named_parameters():
        h.update(name.encode()); h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()

def cv_cache(regime):
    folder = CACHE/regime/'cv'
    if (folder/'manifest.json').exists():
        m=read(folder/'manifest.json')
        if m['data_manifest_sha256'] != sha(OUT/'manifest.json') or m['feature_code_sha256'] != old_codes():
            raise ValueError('CV cache identity changed')
        for name,h in m['artifact_sha256'].items():
            if sha(folder/name) != h: raise ValueError('CV cache checksum changed')
        return folder
    if folder.exists(): raise FileExistsError('Partial CV cache requires a new version')
    d=role(0,regime,'cv'); folder.mkdir(parents=True)
    xs=[];ws=[]
    for off in range(0,len(d['share']),256):
        x,w=features(d['db'][off:off+256]);xs.append(x);ws.append(w)
    arrays={'x':np.concatenate(xs),'w':np.concatenate(ws),
            **{k:d[k] for k in ('share','presence','cell_share','sample_id')}}
    for name,a in arrays.items(): np.save(folder/(name+'.npy'),a,allow_pickle=False)
    write(folder/'manifest.json',{'status':'completed','role':'own_fold0_cv',
        'n':len(d['share']),'data_manifest_sha256':sha(OUT/'manifest.json'),
        'feature_code_sha256':old_codes(),
        'artifact_sha256':{p.name:sha(p) for p in folder.glob('*.npy')}})
    return folder

def prepare():
    for r in REGIMES:
        cache_role(r,'train'); cv_cache(r)
    cache_role('D0','selection')
    write(OPS/'DATA_READY.json',{'status':'passed','data_manifest_sha256':sha(OUT/'manifest.json'),
         'TRAIN':'unchanged original paired D0/H0 Fold0, 61446 each',
         'early_stopping':'existing own-fold0 CV,1000 per regime',
         'ensemble_fit':'existing shared local_fit2400,400 unique backgrounds',
         'checkpoint_and_ensemble_fit_roles_disjoint':True,'Test_used':False})

def task_input(x,task):
    return x if task=='regression' else x[:,CLASSIFIER_CHANNELS]

def loss_and_values(logits,task,w,y,q,z):
    import torch
    if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite output')
    if task=='classification':
        if logits.shape != y.shape: raise ValueError('Classifier shape differs')
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,z)
        return loss,logits.sigmoid(),None
    if tuple(logits.shape) != tuple(w.shape): raise ValueError('Mask grid shape differs')
    qhat=logits.float().sigmoid()
    pred=(qhat*w.float()).sum((1,2))
    cell=((qhat-q.float()).square()*w.float()).sum((1,2))
    loss=(pred.sub(y).square()+.25*cell).mean()
    return loss,pred,cell.mean()

def predict(model,task,data,device,batch):
    import torch
    model.eval(); pp=[]
    with torch.no_grad():
        for off in range(0,len(data['share']),batch):
            x=torch.as_tensor(np.asarray(data['x'][off:off+batch]).copy(),device=device)
            w=torch.as_tensor(np.asarray(data['w'][off:off+batch]).copy(),device=device)
            logits=model(task_input(x,task))
            v=logits.sigmoid() if task=='classification' else (logits.float().sigmoid()*w).sum((1,2))
            if not torch.isfinite(v).all(): raise FloatingPointError('Nonfinite inference')
            pp.append(v.cpu().numpy())
    return np.concatenate(pp)

def sync(device):
    if device=='mps': __import__('torch').mps.synchronize()
    elif device=='cuda': __import__('torch').cuda.synchronize()

def preflight(family, batch=4, device='mps'):
    import torch
    torch.set_num_threads(2)
    if device=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable; no implicit CPU fallback')
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no implicit CPU fallback')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1': raise RuntimeError('Silent MPS fallback must be disabled')
    tr=cache_read(cache_role('D0','train')); results={}
    for task in TASKS:
        torch.manual_seed(SEED);model,provenance=branch(family,task,device)
        initial=base_fingerprint(model); times=[]
        model.set_stage(2);model.train()
        opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=1e-3)
        for step in range(3):
            sl=slice(step*batch,(step+1)*batch)
            x=torch.tensor(np.asarray(tr['x'][sl]),device=device)
            w=torch.tensor(np.asarray(tr['w'][sl]),device=device)
            y=torch.tensor(np.asarray(tr['share'][sl]),dtype=torch.float32,device=device)
            q=torch.tensor(np.asarray(tr['cell_share'][sl]),device=device)
            z=torch.tensor(np.asarray(tr['presence'][sl]),dtype=torch.float32,device=device)
            sync(device);start=time.perf_counter();opt.zero_grad(set_to_none=True)
            loss,_,_=loss_and_values(model(task_input(x,task)),task,w,y,q,z)
            loss.backward()
            for n,p in model.frozen_named_parameters():
                if p.grad is not None: raise AssertionError('Frozen base received gradient '+n)
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)):
                raise FloatingPointError('Nonfinite adaptation gradient')
            opt.step();sync(device);times.append(time.perf_counter()-start)
        unchanged=base_fingerprint(model)==initial
        if not unchanged: raise AssertionError('Permanent pretrained base changed')
        result={'passed':True,'family':family,'task':task,'device':device,'precision':'float32',
            'microbatch':batch,'warm_step_seconds':times,'steady_step_seconds':float(np.median(times[1:])),
            'rows_per_second':float(batch/np.median(times[1:])),
            'estimated_train_epoch_seconds':float(len(tr['share'])*np.median(times[1:])/batch),
            'frozen_base_unchanged':unchanged,'frozen_base_fingerprint':initial,
            'adaptation_parameters':sum(t.numel() for t in model.adapter_state().values()),
            'driver_bytes':int(torch.mps.driver_allocated_memory()) if device=='mps' else None,
            'cuda_peak_allocated_bytes':int(torch.cuda.max_memory_allocated()) if device=='cuda' else None,
            'provenance':provenance}
        results[task]=result;write(OPS/f'PREFLIGHT_{family}_{task}.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='provenance'}),flush=True)
        del opt,model,x,w,y,q,z,loss;gc.collect()
        if device=='mps':torch.mps.empty_cache()
    write(OPS/f'PREFLIGHT_{family}.json',{'status':'passed','models':results,
        'source_sha256':source_pins(),'data_manifest_sha256':sha(OUT/'manifest.json'),
        'disposable_steps_not_training_results':True})
    return results

def train_job(family,regime,task):
    import torch
    torch.set_num_threads(2)
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise RuntimeError('Silent MPS fallback forbidden')
    plan=read(OPS/'RUN_PLAN.json');verify(plan['source_sha256'])
    if plan['data_manifest_sha256']!=sha(OUT/'manifest.json'):raise ValueError('Data manifest changed')
    pf=read(OPS/f'PREFLIGHT_{family}.json')
    if pf['source_sha256']!=plan['source_sha256']:raise ValueError('Preflight source differs')
    settings=pf['models'][task]; device=settings['device'];batch=settings['microbatch']
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS required')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required')
    folder=RUN/family/regime/task
    if folder.exists():raise FileExistsError('Fresh job only; explicit resume implementation required for partial run')
    folder.mkdir(parents=True);tr=cache_read(cache_role(regime,'train'));va=cache_read(cv_cache(regime))
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    model,provenance=branch(family,task,device);before=base_fingerprint(model)
    warmup=plan['warmup_epochs'];max_stage2=plan['stage2_max_epochs'];patience=plan['patience']
    effective=plan['effective_batch'];best=math.inf if task=='regression' else -math.inf
    best_epoch=0;bad=0;steps=0;started=time.perf_counter();pause={'value':False}
    signal.signal(signal.SIGUSR1,lambda *_:pause.update(value=True))
    write(folder/'PLAN.json',{'family':family,'regime':regime,'task':task,'fold':0,
        'device':device,'microbatch':batch,'effective_batch':effective,'seed':SEED,
        'run_plan':plan,'provenance':provenance,'best_selection_role':'own_fold0_cv',
        'permanent_base_fingerprint':before,'train_n':len(tr['share']),'CV_n':len(va['share'])})
    stop='maximum_stage2_epochs'
    for stage,total in ((1,warmup),(2,max_stage2)):
        model.set_stage(stage)
        opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
            lr=1e-3 if stage==1 else 2e-4,weight_decay=1e-3)
        scheduler=None if stage==1 else torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=total,eta_min=1e-5)
        if stage==2:bad=0
        for local_epoch in range(1,total+1):
            epoch=local_epoch if stage==1 else warmup+local_epoch
            verify(plan['source_sha256']);model.train();start=time.perf_counter()
            order=np.random.default_rng([SEED,epoch]).permutation(len(tr['share']))
            loss_sum=0.;seen=0
            for begin in range(0,len(order),effective):
                group=order[begin:begin+effective];opt.zero_grad(set_to_none=True)
                for off in range(0,len(group),batch):
                    ids=group[off:off+batch]
                    x=torch.tensor(np.asarray(tr['x'][ids]),device=device)
                    w=torch.tensor(np.asarray(tr['w'][ids]),device=device)
                    y=torch.tensor(np.asarray(tr['share'][ids]),dtype=torch.float32,device=device)
                    q=torch.tensor(np.asarray(tr['cell_share'][ids]),device=device)
                    z=torch.tensor(np.asarray(tr['presence'][ids]),dtype=torch.float32,device=device)
                    loss,_,_=loss_and_values(model(task_input(x,task)),task,w,y,q,z)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                    (loss*(len(ids)/len(group))).backward()
                    loss_sum+=float(loss.detach().cpu())*len(ids);seen+=len(ids)
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
                opt.step();steps+=1
                if steps%20==0 or begin==0:
                    row={'phase':'training','family':family,'regime':regime,'task':task,'stage':stage,
                        'epoch':epoch,'stage_epoch':local_epoch,'stage_max':total,'rows_seen':seen,
                        'train_rows':len(order),'optimizer_steps':steps,'mean_loss':loss_sum/seen,
                        'epoch_elapsed_seconds':time.perf_counter()-start,'best_cv':None if best_epoch==0 else best,
                        'best_epoch':best_epoch,'patience_counter':bad}
                    write(OPS/'progress.json',row)
            p=predict(model,task,va,device,batch)
            cv=metric(va['share'],p,np.zeros_like(p)) if task=='regression' else metric(va['share'],np.zeros_like(p),p)
            irrelevant = ('classifier_auc','classifier_mean','classifier_ge099') if task=='regression' else ('rmse','bias','prediction_mean','prediction_rms','share_auc')
            for key in irrelevant:cv.pop(key,None)
            value=cv['rmse'] if task=='regression' else cv['classifier_auc']
            if value is None:raise ValueError('CV requires both classes')
            improved=value<best-1e-6 if task=='regression' else value>best+1e-6
            if improved:best=value;best_epoch=epoch;bad=0
            elif stage==2:bad+=1
            lr=opt.param_groups[0]['lr']
            if scheduler:scheduler.step()
            row={'family':family,'regime':regime,'task':task,'stage':stage,'epoch':epoch,
                'stage_epoch':local_epoch,'train_loss':loss_sum/seen,'cv':cv,'best_cv':best,
                'best_epoch':best_epoch,'patience_counter':bad,'learning_rate':lr,
                'epoch_seconds':time.perf_counter()-start,'optimizer_steps':steps,
                'driver_bytes':int(torch.mps.driver_allocated_memory()) if device=='mps' else None}
            checkpoint={'format':'r16_pretrained_adapter_checkpoint_v1','adapter_state':model.adapter_state(),
                'family':family,'regime':regime,'task':task,'stage':stage,'epoch':epoch,
                'best_cv':best,'best_epoch':best_epoch,'optimizer_state_dict':opt.state_dict(),
                'scheduler_state_dict':None if scheduler is None else scheduler.state_dict(),
                'torch_rng':torch.get_rng_state(),'mps_rng':torch.mps.get_rng_state() if device=='mps' else None,
                'cuda_rng':torch.cuda.get_rng_state() if device=='cuda' else None,
                'source_sha256':plan['source_sha256'],'data_manifest_sha256':plan['data_manifest_sha256'],
                'provenance':provenance,'permanent_base_fingerprint':before}
            atomic_pt(folder/'latest.pt',checkpoint)
            if improved:
                atomic_pt(folder/'best.pt',checkpoint)
                save(folder/'cv_best.npz',{'prediction':p,'target_share':va['share'],'sample_id':va['sample_id']})
            with (folder/'epochs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            write(folder/'progress.json',row);write(OPS/'progress.json',{'phase':'epoch_completed',**row})
            print(json.dumps(row,allow_nan=False),flush=True)
            if pause['value'] or (RUN/'PAUSE_AFTER_EPOCH').exists():
                write(OPS/'PAUSED.json',{'family':family,'regime':regime,'task':task,'epoch':epoch,
                    'latest_sha256':sha(folder/'latest.pt'),'explicit_resume_required':True})
                return 20
            if stage==2 and bad>=patience:stop='early_stopping';break
        if stage==2:break
    after=base_fingerprint(model)
    if before!=after:raise AssertionError('Permanent pretrained base changed during training')
    bestck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    model.load_adapter_state(bestck['adapter_state']);selection=cache_read(cache_role('D0','selection'))
    pred=predict(model,task,selection,device,batch)
    save(folder/'selection.npz',{'prediction':pred,'target_share':selection['share'],'sample_id':selection['sample_id']})
    write(folder/'summary.json',{'status':'completed','family':family,'regime':regime,'task':task,
        'epochs_run':epoch,'best_epoch':best_epoch,'best_cv':best,'stop_reason':stop,
        'seconds':time.perf_counter()-started,'permanent_base_unchanged':before==after,
        'data_manifest_sha256':plan['data_manifest_sha256'],'source_sha256':plan['source_sha256'],
        'artifact_sha256':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}})
    return 0
