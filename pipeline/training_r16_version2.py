"""Separate V2 ConvNeXt production runner; V1 artifacts/sources stay immutable.

Planning/checking does not train. Only explicit train calls create fresh Conv
jobs. No Test or locked report panel is read; finalization is a separate step.
"""
from __future__ import annotations
from pathlib import Path
import datetime, gc, hashlib, json, math, os, random, signal, time
import numpy as np

from pipeline.r16_multisupport_data import ROOT, OUT, REGIMES, read, write, sha, save
from pipeline.training_r16_multisupport import codes, verify, cache_read, atomic_pt, metric

RUN=ROOT/'models/r16_pretrained_version2_r1'
OPS=ROOT/'ops/r16_pretrained_version2_r1'
CONFIG=ROOT/'configs/r16_pretrained_v2_candidate.json'
PREFLIGHT=ROOT/'ops/r16_pretrained_v2_mps_r1/PREFLIGHT_BOTH_TASKS.json'
SPLIT=ROOT/'ops/r16_pretrained_masking_r1/LOCAL_FIT_SPLIT.json'
TRAIN_CACHE=ROOT/'data/cache/r16_multisupport_full_r1'
CV_CACHE=ROOT/'data/cache/r16_pretrained_ft_r1'
V1_RUN=ROOT/'models/r16_pretrained_cuda_bf16_r1/both'
FAMILY='convnextv2_base_fcmae'
TASKS=('regression','classification')
SEED=202610281901


def sources():
    result=codes()
    for name in ('pipeline/training_r16_version2.py','ops/train_r16_version2.py',
                 'pipeline/pretrained_convnext_v2_r16.py','ops/preflight_r16_v2_mps.py',
                 'pipeline/r16_feature_masking.py','pipeline/training_r16_pretrained_masked.py',
                 'pipeline/training_r16_production.py','configs/r16_pretrained_v2_candidate.json'):
        result[name]=sha(ROOT/name)
    return result


def check_preflight():
    """Check both task receipts, current source/config and public weight pins."""
    if not PREFLIGHT.is_file(): raise FileNotFoundError('V2 both-task preflight is missing')
    pf=read(PREFLIGHT)
    if pf.get('status')!='passed' or not pf.get('task_branches_independently_instantiated'):
        raise ValueError('Independent regression/classification preflight must pass')
    verify(pf['source_sha256'])
    rows={row['task']:row for row in pf['rows']}
    if set(rows)!=set(TASKS):raise ValueError('Both preflight task branches required')
    for task,row in rows.items():
        p=ROOT/row['result']
        if sha(p)!=row['result_sha256']:raise ValueError('Preflight receipt changed')
        receipt=read(p)
        if receipt.get('status')!='passed' or receipt.get('task')!=task:
            raise ValueError('Task preflight failed')
        if not receipt.get('frozen_base_unchanged') or not receipt.get('adapter_roundtrip_exact'):
            raise ValueError('Frozen base or adapter roundtrip gate failed')
        if receipt['provenance']['weight_sha256']!=pf['public_weight_sha256']:
            raise ValueError('Public weight identity differs')
        if receipt['device']!=pf['device']:raise ValueError('Preflight device differs')
    return pf


def _checked_cache(folder,manifest_sha):
    """Read an already-created cache. Never build/mutate V1 cache folders."""
    folder=Path(folder);mp=folder/'manifest.json'
    if not mp.is_file():raise FileNotFoundError('Existing immutable cache needed: '+str(folder))
    m=read(mp)
    if m.get('status')!='completed' or m['data_manifest_sha256']!=manifest_sha:
        raise ValueError('Cache data identity differs')
    cp=m.get('code_sha256',m.get('feature_code_sha256'))
    if cp!=codes():raise ValueError('Cache feature source identity differs')
    for name,h in m['artifact_sha256'].items():
        if sha(folder/name)!=h:raise ValueError('Cache checksum differs: '+name)
    arrays=cache_read(folder)
    n=len(arrays['share'])
    if arrays['x'].shape!=(n,17,15,25) or arrays['w'].shape!=(n,15,25):
        raise ValueError('Data tensor shape differs')
    return arrays,{'path':str(folder.relative_to(ROOT)),'manifest_sha256':sha(mp),
                   'artifact_sha256':m['artifact_sha256'],'n':n}


def data_roles():
    """Bind actual D/H TRAIN, source-separated checkpoint role and own-fold CV."""
    mh=sha(OUT/'manifest.json');m=read(OUT/'manifest.json');split=read(SPLIT)
    if split['local_fit_npz_sha256']!=m['selection']['sha256']:
        raise ValueError('Development partition data identity differs')
    if sha(ROOT/split['source_metadata'])!=split['source_metadata_sha256']:
        raise ValueError('Development source metadata changed')
    chosen=split['chosen'];a=np.asarray(chosen['checkpoint_selection']['selection_row_indices'],int)
    b=np.asarray(chosen['ensemble_fit']['selection_row_indices'],int)
    proof=chosen['crossrole_proof'][0]
    if proof['base_window_overlap'] or proof['background_group_overlap'] or proof['minimum_support_gap_hours']<24:
        raise ValueError('Development source separation/purge failed')
    selection,sid=_checked_cache(TRAIN_CACHE/'shared_selection/selection',mh)
    if set(a)&set(b) or sorted(np.r_[a,b])!=list(range(len(selection['share']))):
        raise ValueError('Development row partition differs')
    for key,idx in (('checkpoint_selection',a),('ensemble_fit',b)):
        if not np.array_equal(selection['sample_id'][idx].astype(str),np.asarray(chosen[key]['sample_ids']).astype(str)):
            raise ValueError('Development sample order differs')
    if len(a)!=1194 or len(b)!=1206:raise ValueError('Fixed 1194/1206 split differs')
    train={};cv={};ident={'selection':sid}
    for r in REGIMES:
        train[r],ident[r+'_train']=_checked_cache(TRAIN_CACHE/r/'train',mh)
        cv[r],ident[r+'_cv']=_checked_cache(CV_CACHE/r/'cv',mh)
    checkpoint={k:np.asarray(v[a]) for k,v in selection.items()}
    # The ensemble-fit half is never evaluated inside training.
    return train,checkpoint,cv,{'data_manifest_sha256':mh,'split_sha256':sha(SPLIT),
        'cache_identities':ident,'checkpoint_rows':len(a),'ensemble_fit_rows_excluded':len(b),
        'minimum_support_gap_hours':proof['minimum_support_gap_hours']}


def completed_reuse():
    """Check existing completed Dasheng/TimeMoE artifacts; no performance read."""
    rows=[]
    for family in ('dasheng_base','timemoe'):
        for regime in REGIMES:
            for task in TASKS:
                folder=V1_RUN/family/regime/task;p=folder/'summary.json'
                if not p.is_file():rows.append({'family':family,'regime':regime,'task':task,'status':'not_locally_available'});continue
                s=read(p)
                if s.get('status')!='completed':raise ValueError('Partial V1 model cannot be reused')
                for name,h in s['artifact_sha256'].items():
                    if sha(folder/name)!=h:raise ValueError('Completed V1 artifact changed: '+name)
                rows.append({'family':family,'regime':regime,'task':task,'status':'checksum_verified_completed',
                    'summary':str(p.relative_to(ROOT)),'summary_sha256':sha(p),
                    'artifact_sha256':s['artifact_sha256'],'retrain':False})
    return rows


def make_plan():
    """Explicit immutable plan only; no model creation or training."""
    cfg=read(CONFIG);pf=check_preflight()
    if cfg['new_family']!=FAMILY or cfg['folds']!=[0] or cfg['additional_kfold']:
        raise ValueError('Only new ConvNeXt Fold0 four-task scope allowed')
    tr,va,cv,identity=data_roles()
    from pipeline.r16_feature_masking import FeatureMaskingConfig
    plan={'format':'r16_version2_training_plan_v1','status':'fixed_before_training',
      'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'config_sha256':sha(CONFIG),'source_sha256':sources(),'data':identity,
      'preflight_sha256':sha(PREFLIGHT),'public_weight_sha256':pf['public_weight_sha256'],
      'runtime':{'python':pf['python'],'torch':pf['torch'],'device':pf['device'],'precision':'float32'},
      'family':FAMILY,'regimes':list(REGIMES),'tasks':list(TASKS),'folds':[0],
      'jobs':[{'family':FAMILY,'regime':r,'task':t,'batch':int(next(x for x in pf['rows'] if x['task']==t)['stages'][0]['batch'])} for r in REGIMES for t in TASKS],
      'warmup_epochs':3,'stage2_max_epochs':25,'patience':7,'optimizer':'Adam',
      'learning_rates':{'warmup':1e-3,'stage2':2e-4,'minimum':1e-5},
      'weight_decay':1e-3,'dropout':.1,'effective_batch':64,'gradient_clip':1.,
      'masking':FeatureMaskingConfig.for_variant('both').state_dict(),
      'regression_classifier_parameter_sharing':False,'score_share_weight':0.,
      'completed_models_for_later_reuse':completed_reuse(),
      'ensemble_policy':cfg['ensemble_policy'],'ensemble_fitting_performed':False,
      'Test_read':False,'report_read':False,'automatic_training':False,
      'partial_resume_supported':False,'new_convnext_trained':False,
      'remaining_after_training':['freeze and compare local gate predictions','reuse compatible completed models','separate policy fit and inference finalizer; no Test access here']}
    OPS.mkdir(parents=True,exist_ok=True)
    dest=OPS/'RUN_PLAN.json'
    if dest.exists():
        old=read(dest)
        for key in ('config_sha256','source_sha256','data','preflight_sha256','runtime','jobs'):
            if old[key]!=plan[key]:raise FileExistsError('Immutable V2 run plan differs; use new namespace')
        return old
    write(dest,plan)
    return plan


def _fingerprint(model):
    h=hashlib.sha256()
    for name,p in model.frozen_named_parameters():
        h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _device_gate(device):
    import torch
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise RuntimeError('Silent MPS fallback forbidden')
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')


def _task_metric(y,p,task):
    result=metric(y,p,np.zeros_like(p)) if task=='regression' else metric(y,np.zeros_like(p),p)
    remove=('classifier_auc','classifier_mean','classifier_ge099') if task=='regression' else ('rmse','bias','prediction_mean','prediction_rms','share_auc')
    return {k:v for k,v in result.items() if k not in remove}


def train_job(regime,task):
    """Explicit fresh Conv task only. Reject partial directories; no auto resume."""
    import torch
    from pipeline.pretrained_convnext_v2_r16 import create_r16_v2_branch
    from pipeline.r16_feature_masking import FeatureMaskingConfig,apply_r16_feature_masking,make_feature_masking_generator
    from pipeline.training_r16_pretrained_masked import task_input,loss_and_values,predict
    plan=read(OPS/'RUN_PLAN.json');verify(plan['source_sha256'])
    pf=check_preflight()
    if sha(PREFLIGHT)!=plan['preflight_sha256'] or sha(CONFIG)!=plan['config_sha256']:
        raise ValueError('Plan/preflight/config binding differs')
    if task not in TASKS or regime not in REGIMES:raise ValueError('Only Conv four-task scope supported')
    if str(Path(__import__('sys').executable).resolve())!=str(Path(plan['runtime']['python']).resolve()) or torch.__version__!=plan['runtime']['torch']:
        raise ValueError('Run with the exact preflight Python/Torch runtime')
    device=plan['runtime']['device'];_device_gate(device);torch.set_num_threads(2)
    train,va,cv,identity=data_roles()
    if identity!=plan['data']:raise ValueError('Data cache/split identity differs')
    batch=next(j['batch'] for j in plan['jobs'] if j['task']==task and j['regime']==regime)
    folder=RUN/FAMILY/regime/task
    if folder.exists():raise FileExistsError('Fresh job required; partial resume is not implemented')
    folder.mkdir(parents=True)
    tr=train[regime];torch.manual_seed(SEED);np.random.seed(SEED%(2**32));random.seed(SEED)
    model,provenance=create_r16_v2_branch(FAMILY,task,device=device,dtype='float32',dropout=.1)
    if provenance['weight_sha256']!=plan['public_weight_sha256']:raise ValueError('Model public weight differs')
    before=_fingerprint(model);config=FeatureMaskingConfig.for_variant('both')
    if config.state_dict()!=plan['masking']:raise ValueError('Feature masking differs')
    write(folder/'PLAN.json',{'family':FAMILY,'regime':regime,'task':task,'fold':0,'run_plan_sha256':sha(OPS/'RUN_PLAN.json'),
        'data':identity,'provenance':provenance,'frozen_fingerprint':before,'train_n':len(tr['share']),
        'checkpoint_n':len(va['share']),'microbatch':batch,'Test_read':False,'report_read':False})
    best=math.inf if task=='regression' else -math.inf;be=bad=steps=0;start=time.monotonic();pause={'value':False}
    signal.signal(signal.SIGUSR1,lambda *_:pause.update(value=True));stop='maximum_stage2_epochs'
    for stage,total in ((1,3),(2,25)):
        model.set_stage(stage);params=[p for p in model.parameters() if p.requires_grad]
        opt=torch.optim.Adam(params,lr=1e-3 if stage==1 else 2e-4,weight_decay=1e-3)
        scheduler=None if stage==1 else torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=25,eta_min=1e-5)
        if stage==2:bad=0
        for se in range(1,total+1):
            epoch=se if stage==1 else 3+se;verify(plan['source_sha256']);model.train();t0=time.monotonic()
            order=np.random.default_rng([SEED,epoch]).permutation(len(tr['share']))
            gen=make_feature_masking_generator(SEED,epoch=epoch,stream='train');loss_sum=0.;seen=0
            for begin in range(0,len(order),64):
                group=order[begin:begin+64];opt.zero_grad(set_to_none=True)
                for off in range(0,len(group),batch):
                    ix=group[off:off+batch]
                    x=torch.tensor(np.asarray(tr['x'][ix]),device=device)
                    w=torch.tensor(np.asarray(tr['w'][ix]),device=device)
                    y=torch.tensor(np.asarray(tr['share'][ix]),dtype=torch.float32,device=device)
                    q=torch.tensor(np.asarray(tr['cell_share'][ix]),device=device)
                    z=torch.tensor(np.asarray(tr['presence'][ix]),dtype=torch.float32,device=device)
                    xa,aug=apply_r16_feature_masking(x,task=task,training=True,generator=gen,config=config)
                    if not torch.equal(xa[:,5],x[:,5]):raise AssertionError('Observed mask changed')
                    loss,_,_=loss_and_values(model(task_input(xa,task)),task,w,y,q,z)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                    (loss*len(ix)/len(group)).backward();loss_sum+=float(loss.detach().cpu())*len(ix);seen+=len(ix)
                if any(p.grad is not None for _,p in model.frozen_named_parameters()):raise AssertionError('Frozen base gradient')
                if not torch.isfinite(torch.nn.utils.clip_grad_norm_(params,1.)):raise FloatingPointError('Nonfinite gradient')
                opt.step();steps+=1
                if begin==0 or steps%20==0:write(OPS/'progress.json',{'phase':'training','regime':regime,'task':task,'epoch':epoch,'stage':stage,'rows_seen':seen,'total_rows':len(order),'optimizer_steps':steps})
            pred=predict(model,task,va,device,batch);scores=_task_metric(va['share'],pred,task)
            value=scores['rmse' if task=='regression' else 'classifier_auc']
            if value is None:raise ValueError('Checkpoint class metric unavailable')
            improved=value<best-1e-6 if task=='regression' else value>best+1e-6
            if improved:best=value;be=epoch;bad=0
            elif stage==2:bad+=1
            lr=opt.param_groups[0]['lr']
            if scheduler:scheduler.step()
            ns=np.random.get_state()
            ck={'format':'r16_version2_adapter_checkpoint_v1','family':FAMILY,'regime':regime,'task':task,'stage':stage,'epoch':epoch,
                'adapter_state':model.adapter_state(),'optimizer_state_dict':opt.state_dict(),
                'scheduler_state_dict':None if scheduler is None else scheduler.state_dict(),
                'torch_rng':torch.get_rng_state(),'numpy_rng':(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),
                'python_rng':random.getstate(),'masking_rng':gen.get_state(),
                'device_rng':torch.mps.get_rng_state() if device=='mps' else (torch.cuda.get_rng_state() if device=='cuda' else None),
                'best_epoch':be,'best_value':best,'patience_counter':bad,'optimizer_steps':steps,
                'run_plan_sha256':sha(OPS/'RUN_PLAN.json'),'source_sha256':plan['source_sha256'],'data':identity,
                'masking':config.state_dict(),'provenance':provenance,'permanent_base_fingerprint':before}
            atomic_pt(folder/'latest.pt',ck)
            if improved:
                atomic_pt(folder/'best.pt',ck)
                save(folder/'checkpoint_validation_best.npz',{'prediction':pred,'target_share':va['share'],'sample_id':va['sample_id']})
            row={'regime':regime,'task':task,'stage':stage,'epoch':epoch,'train_loss':loss_sum/seen,
                 'checkpoint_selection':scores,'best_value':best,'best_epoch':be,'patience_counter':bad,
                 'learning_rate':lr,'epoch_seconds':time.monotonic()-t0,'optimizer_steps':steps,
                 'latest_sha256':sha(folder/'latest.pt'),'best_sha256':sha(folder/'best.pt')}
            with (folder/'epochs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            write(folder/'progress.json',row);write(OPS/'progress.json',{'phase':'epoch_completed',**row})
            print(json.dumps(row,allow_nan=False),flush=True)
            if pause['value'] or (RUN/'PAUSE_AFTER_EPOCH').exists():
                write(folder/'PAUSED.json',{'epoch':epoch,'latest_sha256':row['latest_sha256'],'resume_supported':False,'explicit_new_resume_implementation_required':True});return 20
            if stage==2 and bad>=7:stop='early_stopping';break
    if _fingerprint(model)!=before:raise AssertionError('Frozen public base changed')
    ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    if ck['run_plan_sha256']!=sha(OPS/'RUN_PLAN.json'):raise ValueError('Best checkpoint binding changed')
    model.load_adapter_state(ck['adapter_state'])
    own=predict(model,task,cv[regime],device,batch)
    save(folder/'cv_best.npz',{'prediction':own,'target_share':cv[regime]['share'],'sample_id':cv[regime]['sample_id']})
    write(folder/'summary.json',{'status':'completed','family':FAMILY,'regime':regime,'task':task,'fold':0,
        'epochs_run':epoch,'best_epoch':be,'best_value':best,'stop_reason':stop,'seconds':time.monotonic()-start,
        'ownfold_CV_after_checkpoint_frozen':_task_metric(cv[regime]['share'],own,task),
        'source_sha256':plan['source_sha256'],'data':identity,'run_plan_sha256':sha(OPS/'RUN_PLAN.json'),
        'frozen_base_unchanged':True,'Test_read':False,'report_read':False,'ensemble_fit_evaluated':False,
        'artifact_sha256':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}})
    del model,opt;gc.collect()
    if device=='mps':torch.mps.empty_cache()
    return 0
