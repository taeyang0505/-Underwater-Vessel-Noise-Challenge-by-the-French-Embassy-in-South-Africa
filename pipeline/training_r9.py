"""Fresh R9 mask-model training: exact100 epochs, no early stopping or resume.

Dataset items contain x[8,15,25], w, share, cell_share and presence. Query
forward receives x,w only. Dynamic TRAIN implements set_epoch(epoch); fixed
validation is never advanced. Dataset .identity is JSON-safe provenance.
Known labels retain their original energy basis; rounded w need not pool
known q to exactly y, so the explicit quantization bound is audited.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT=Path(__file__).resolve().parents[1]
CODE_NAMES=('pipeline/training_r9.py','pipeline/models_r9.py','pipeline/relative_grid_r6.py',
            'pipeline/relative_grid_r6_v2.py','pipeline/relative_models_r6_v2.py',
            'pipeline/hybrid_models_r5.py','pipeline/grid_models_r5.py','pipeline/models.py')
DEFAULTS={'epochs':100,'fixture_only':False,'optimizer':'adam','early_stopping_enabled':False,
          'learning_rate':1e-3,'weight_decay':3e-4,'gradient_clip':1.,'cell_loss_weight':.25,
          'presence_loss_weight':.05,'batch_size':128,'evaluation_batch_size':128,'num_workers':4,
          'threads':4,'seed':42,'bf16':True,'monitor':'window_rmse','model_settings':{},
          'site_balanced_loss':True,
          'quantization_target_tolerance':math.tanh(math.log(10.)*.05/20.)+1e-5}


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def _canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def _checkpoint(path,value):
    tmp=Path(str(path)+'.tmp');torch.save(value,tmp);tmp.replace(path)


def _settings(settings):
    if not isinstance(settings,dict) or set(settings)-set(DEFAULTS):raise ValueError('Unknown R9 training settings')
    s={**DEFAULTS,**settings};s['model_settings']=dict(s['model_settings'])
    if type(s['fixture_only']) is not bool:raise ValueError('Explicit fixture_only bool required')
    if type(s['site_balanced_loss']) is not bool:raise ValueError('Explicit site_balanced_loss bool required')
    if (type(s['epochs']) is not int or s['epochs']!=(2 if s['fixture_only'] else 100)
            or s['optimizer']!='adam' or s['early_stopping_enabled'] is not False):
        raise ValueError('Production requires100/noES/Adam; explicitly labelled fixtures require2epochs')
    for k,expected in [('learning_rate',1e-3),('weight_decay',3e-4),('gradient_clip',1.),
                       ('cell_loss_weight',.25),('presence_loss_weight',.05)]:
        if s[k]!=expected:raise ValueError('Fixed R9 training policy differs: '+k)
    if s['model_settings'].get('dropout',.1)!=.1:raise ValueError('R9 dropout must remain.1')
    for key in ['batch_size','evaluation_batch_size','threads']:
        if type(s[key]) is not int or s[key]<1:raise ValueError('Positive integer '+key+' required')
    if type(s['num_workers']) is not int or s['num_workers']<0:raise ValueError('Nonnegative num_workers required')
    if type(s['seed']) is not int or s['seed']<0:raise ValueError('Nonnegative seed required')
    if s['monitor'] not in ['window_rmse','site_macro_rmse']:raise ValueError('Explicit valid INNER scheduler monitor required')
    if s['quantization_target_tolerance']!=DEFAULTS['quantization_target_tolerance']:
        raise ValueError('Do not relax known-target rounded-weight tolerance')
    json.dumps(s,allow_nan=False)
    return s


def _identity(dataset,role,fixture):
    value=getattr(dataset,'identity',None)
    if callable(value):value=value()
    if value is None:
        if not fixture:raise ValueError(role+' Dataset must provide JSON-safe immutable identity')
        value={'fixture_only':True,'role':role,'rows':len(dataset)}
    if not isinstance(value,(dict,str)):raise ValueError('Dataset identity must be dict or fingerprint string')
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def _seed_worker(_worker):
    seed=torch.initial_seed()%2**32;np.random.seed(seed);random.seed(seed)
    torch.set_num_threads(1)


def _loader(dataset,s,device,*,train,epoch=0):
    generator=torch.Generator().manual_seed(s['seed']+epoch*1000003)
    args=dict(dataset=dataset,batch_size=s['batch_size'] if train else s['evaluation_batch_size'],
              shuffle=train,drop_last=False,num_workers=s['num_workers'],pin_memory=device.type=='cuda',
              generator=generator,worker_init_fn=_seed_worker,persistent_workers=False)
    if s['num_workers']:args['multiprocessing_context']='spawn'
    return DataLoader(**args)


def _amp(device,bf16):
    return torch.autocast(device_type='cuda',dtype=torch.bfloat16) if device.type=='cuda' and bf16 else nullcontext()


def _batch(batch,device,*,targets):
    if not isinstance(batch,dict) or not {'x','w'}.issubset(batch):raise ValueError('Expected explicit x,w batch dictionary')
    x=batch['x'].to(device=device,dtype=torch.float32,non_blocking=device.type=='cuda')
    w=batch['w'].to(device=device,dtype=torch.float32,non_blocking=device.type=='cuda')
    if not targets:return x,w,None
    if not {'share','cell_share','presence'}.issubset(batch):raise ValueError('Training/evaluation labels missing')
    labels=tuple(batch[k].to(device=device,dtype=torch.float32,non_blocking=device.type=='cuda')
                 for k in ['share','presence','cell_share'])
    return x,w,labels


def physical_multitask_loss(outputs,targets,w,*,quantization_target_tolerance=DEFAULTS['quantization_target_tolerance']):
    """Per-row exact MSE+.25*energy-weighted cell MSE+.05*derived BCE.

    Prediction pooling must match exactly within float32 tolerance. Original
    label pooling uses its bounded raw-versus-rounded tolerance, not 3e-6.
    The target values passed to this function are never redefined.
    """
    p,logit,q=outputs;y,z,truth=targets
    p,logit,q,y,z,truth,w=(v.float() for v in [p,logit,q,y,z,truth,w])
    if p.ndim!=1 or y.shape!=p.shape or z.shape!=p.shape or logit.shape!=p.shape or q.shape!=w.shape or truth.shape!=q.shape:
        raise ValueError('Output/target/weight alignment failed')
    if not all(torch.isfinite(v).all() for v in [p,logit,q,y,z,truth,w]):raise FloatingPointError('Nonfinite loss inputs')
    if ((y<0)|(y>1)).any() or ((truth<0)|(truth>1)).any() or ((q<0)|(q>1)).any() or (w<0).any():raise ValueError('Invalid share/q/weight range')
    if not torch.equal(z,(y>0).float()) or (truth[w==0]!=0).any() or (q[w==0]!=0).any():raise ValueError('Invalid controlled-addition presence/missing q')
    if not torch.allclose(w.sum((1,2)),torch.ones_like(p),atol=3e-6,rtol=0):raise ValueError('Weights must sum to one')
    if not torch.allclose(p,(q*w).sum((1,2)),atol=3e-6,rtol=0):raise ValueError('Prediction bypasses physical pooling')
    gap=((truth*w).sum((1,2))-y).abs()
    if (gap>quantization_target_tolerance).any():raise ValueError('Known q/y exceeds predeclared rounded-energy mismatch bound')
    share=(p-y).square();cell=((q-truth).square()*w).sum((1,2))
    presence=nn.functional.binary_cross_entropy_with_logits(logit,z,reduction='none')
    return share+.25*cell+.05*presence,{'share_mse':share,'weighted_cell_mse':cell,
        'derived_presence_bce':presence,'raw_label_vs_rounded_pool_gap':gap}


@torch.no_grad()
def predict_loader(model,loader,device='cuda',*,bf16=True,collect_targets=False):
    """Fixed model inference; returns share/score/cell_mask NumPy arrays.

    collect_targets=True additionally returns target_share/target_presence,
    site and sample_id for diagnostic metrics only. They are not model inputs.
    A query-only loader needs x,w only. This function never fits/updates data.
    """
    device=torch.device(device);enabled=bool(bf16 and device.type=='cuda' and torch.cuda.is_bf16_supported())
    model.eval();out={k:[] for k in ['share','score','cell_mask']};sites=[];ids=[];has_sites=True
    input_hash=hashlib.sha256()
    if collect_targets:out.update(target_share=[],target_presence=[])
    offset=0
    for batch in loader:
        x,w,labels=_batch(batch,device,targets=collect_targets)
        with _amp(device,enabled):p,z,q=model(x,w)
        if collect_targets:
            physical_multitask_loss((p,z,q),labels,w)
        for k,v in [('share',p),('score',z.float().sigmoid()),('cell_mask',q)]:
            a=v.float().cpu().numpy()
            if not np.isfinite(a).all():raise FloatingPointError('Nonfinite prediction: '+k)
            out[k].append(a)
        if collect_targets:
            for name in ['x','w','share','presence','cell_share']:
                a=batch[name].detach().cpu().contiguous().numpy()
                input_hash.update(json.dumps([name,str(a.dtype),list(a.shape)],separators=(',',':')).encode())
                input_hash.update(a.tobytes())
            out['target_share'].append(labels[0].cpu().numpy());out['target_presence'].append(labels[1].cpu().numpy())
            if 'site' not in batch:has_sites=False
            sites.extend([str(v) for v in batch.get('site',['']*len(x))])
            ids.extend([str(v) for v in batch.get('sample_id',range(offset,offset+len(x)))])
        offset+=len(x)
    if not offset:raise ValueError('Cannot predict empty loader')
    result={k:np.concatenate(v) for k,v in out.items()}
    if collect_targets:
        input_hash.update(json.dumps([sites,ids],separators=(',',':')).encode())
        result.update(site=np.asarray(sites),sample_id=np.asarray(ids),site_metadata_available=has_sites,
                      validation_input_sha256=input_hash.hexdigest())
    return result


def aggregate_metrics(pred):
    """Window risk and site-macro diagnostics on explicit known labels only."""
    from sklearn.metrics import roc_auc_score
    y=np.asarray(pred['target_share'],float);p=np.asarray(pred['share'],float);e=p-y
    result={'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(abs(e).mean()),'bias_pred_minus_truth':float(e.mean())}
    if pred['site_metadata_available']:
        site=pred['site'];slices={str(v):{'n':int((site==v).sum()),'rmse':float(np.sqrt(np.mean(e[site==v]**2)))} for v in sorted(set(site))}
        result.update(site_macro_rmse=float(np.mean([v['rmse'] for v in slices.values()])),site_slices=slices)
    else:result.update(site_macro_rmse=None,site_slices=None)
    z=pred['target_presence'];result['addition_presence_auc']=float(roc_auc_score(z,pred['score'])) if len(np.unique(z))==2 else None
    return result


def train_neural(kind,train_dataset,val_dataset,settings,outdir,device='cuda'):
    """Fresh train; production100epochs/noES, explicitly flagged fixtures2.

    Train item/worker identity is deterministic under epoch+seed. Workers are
    respawned each epoch so set_epoch cannot leave stale persistent copies.
    Checkpoints contain architecture/settings/input+code fingerprints. The
    last checkpoint includes optimizer/scheduler/RNG states; resume is refused.
    """
    from pipeline.models_r9 import make_r9_model
    s=_settings(settings);folder=Path(outdir)
    if folder.exists():raise FileExistsError('Fresh output required; failed/complete runs cannot be resumed')
    if not hasattr(train_dataset,'set_epoch'):raise ValueError('Dynamic TRAIN requires set_epoch(epoch)')
    if not len(train_dataset) or not len(val_dataset):raise ValueError('Nonempty TRAIN and validation required')
    device=torch.device(device)
    if device.type not in ['cpu','cuda'] or device.type=='cuda' and not torch.cuda.is_available():raise ValueError('Requested CPU/CUDA device is unavailable')
    train_identity=_identity(train_dataset,'train',s['fixture_only']);val_identity=_identity(val_dataset,'validation',s['fixture_only'])
    code={n:_sha(ROOT/n) for n in CODE_NAMES};input_fingerprint=_canonical({'train':train_identity,'validation':val_identity})
    folder.mkdir(parents=True);(folder/'checkpoints').mkdir();start=time.perf_counter();epoch=0;step=0
    try:
        torch.set_num_threads(s['threads']);torch.manual_seed(s['seed']);random.seed(s['seed']);np.random.seed(s['seed']%2**32)
        if device.type=='cuda':torch.cuda.manual_seed_all(s['seed'])
        amp=bool(s['bf16'] and device.type=='cuda' and torch.cuda.is_bf16_supported())
        model=make_r9_model(kind,**s['model_settings']).to(device)
        architecture=model.architecture();optimizer=torch.optim.Adam(model.parameters(),lr=s['learning_rate'],weight_decay=s['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=2,min_lr=1e-6)
        started={'kind':kind,'settings':s,'architecture':architecture,'input_fingerprint':input_fingerprint,
                 'train_identity':train_identity,'validation_identity':val_identity,'code_sha256':code,
                 'device':str(device),'precision':'CUDA BF16 operators / FP32 pooling+loss' if amp else 'FP32',
                 'fixture_only':s['fixture_only'],'early_stopping_enabled':False,'resume_allowed':False}
        _json(folder/'started.json',started)
        best_window=best_macro=float('inf');window_epoch=macro_epoch=None;history=[];validation_input_sha=None
        validation_loader=_loader(val_dataset,s,device,train=False)
        def checkpoint(epoch,metric,*,last=False):
            ck={'format':'r9_neural_training_checkpoint_v1','kind':kind,'epoch':epoch,
                'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'architecture':architecture,'settings':s,'input_fingerprint':input_fingerprint,
                'train_identity':train_identity,'validation_identity':val_identity,'code_sha256':code,
                'validation_metrics':metric,'optimizer_steps':step,'fixture_only':s['fixture_only']}
            ck['validation_input_sha256']=validation_input_sha
            if last:ck.update(optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),
                              torch_rng_state=torch.get_rng_state(),numpy_rng_state_repr=repr(np.random.get_state()),
                              cuda_rng_state_all=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
            return ck
        for epoch in range(1,s['epochs']+1):
            train_dataset.set_epoch(epoch)
            if _identity(train_dataset,'train',s['fixture_only'])!=train_identity or _identity(val_dataset,'validation',s['fixture_only'])!=val_identity:
                raise ValueError('Dataset source/fixed-validation identity changed')
            loader=_loader(train_dataset,s,device,train=True,epoch=epoch)
            model.train();seen=0;ss=loss_sum=unweighted_loss_sum=cell_sum=bce_sum=0.;max_gap=0.
            weight_sum=0.;weight_min=float('inf');weight_max=0.
            for batch in loader:
                x,w,targets=_batch(batch,device,targets=True);optimizer.zero_grad(set_to_none=True)
                with _amp(device,amp):outputs=model(x,w)
                # All objective arithmetic is explicitly outside autocast in FP32.
                losses,parts=physical_multitask_loss(outputs,targets,w,quantization_target_tolerance=s['quantization_target_tolerance'])
                weight=batch.get('loss_weight') if s['site_balanced_loss'] else None
                weight=torch.ones_like(losses) if weight is None else weight.to(device=device,dtype=torch.float32)
                if weight.shape!=losses.shape or not torch.isfinite(weight).all() or (weight<=0).any():
                    raise ValueError('TRAIN loss_weight must be a positive finite per-row vector')
                loss=(losses*weight).mean();loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),s['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();step+=1;count=len(x);seen+=count
                ss+=float(parts['share_mse'].detach().sum());cell_sum+=float(parts['weighted_cell_mse'].detach().sum())
                bce_sum+=float(parts['derived_presence_bce'].detach().sum());loss_sum+=float(loss.detach())*count
                unweighted_loss_sum+=float(losses.detach().sum());weight_sum+=float(weight.sum())
                weight_min=min(weight_min,float(weight.min()));weight_max=max(weight_max,float(weight.max()))
                max_gap=max(max_gap,float(parts['raw_label_vs_rounded_pool_gap'].detach().max()))
            if seen!=len(train_dataset):raise ValueError('TRAIN epoch did not visit every requested item')
            pred=predict_loader(model,validation_loader,device,bf16=amp,collect_targets=True);metric=aggregate_metrics(pred)
            if validation_input_sha is None:validation_input_sha=pred['validation_input_sha256']
            elif pred['validation_input_sha256']!=validation_input_sha:raise ValueError('Fixed validation values/order changed between epochs')
            if metric['n']!=len(val_dataset):raise ValueError('Validation row coverage mismatch')
            monitor=metric['rmse'] if s['monitor']=='window_rmse' else metric['site_macro_rmse']
            if monitor is None or not math.isfinite(monitor):raise FloatingPointError('Undefined/nonfinite validation monitor')
            lr=optimizer.param_groups[0]['lr'];scheduler.step(monitor)
            if metric['rmse']<best_window:
                best_window=metric['rmse'];window_epoch=epoch;_checkpoint(folder/'best_window.pt',checkpoint(epoch,metric))
                np.savez_compressed(folder/'best_window_validation_predictions.npz',**{k:v for k,v in pred.items() if isinstance(v,np.ndarray)})
            if metric['site_macro_rmse'] is not None and metric['site_macro_rmse']<best_macro:
                best_macro=metric['site_macro_rmse'];macro_epoch=epoch;_checkpoint(folder/'best_macro.pt',checkpoint(epoch,metric))
            if epoch%10==0 or epoch==s['epochs']:_checkpoint(folder/'checkpoints'/f'epoch_{epoch:03d}.pt',checkpoint(epoch,metric))
            # Save latest state each epoch; no training reads it for resuming.
            _checkpoint(folder/'last.pt',checkpoint(epoch,metric,last=True))
            row={'epoch':epoch,'train_online_rmse':math.sqrt(ss/seen),'train_online_loss':loss_sum/seen,
                 'train_balanced_multitask_loss':loss_sum/seen,'train_unweighted_multitask_loss':unweighted_loss_sum/seen,
                 'TRAIN_loss_weight_mean':weight_sum/seen,'TRAIN_loss_weight_min':weight_min,'TRAIN_loss_weight_max':weight_max,
                 'train_weighted_cell_mse':cell_sum/seen,'train_derived_presence_bce':bce_sum/seen,
                 'train_examples':seen,'validation':metric,'best_window_epoch':window_epoch,'best_macro_epoch':macro_epoch,
                 'learning_rate':lr,'next_learning_rate':optimizer.param_groups[0]['lr'],'optimizer_steps':step,
                 'max_raw_label_rounded_weight_pool_gap':max_gap,'elapsed_seconds':time.perf_counter()-start,
                 'TRAIN_set_epoch_called':epoch,'validation_identity_unchanged':True,
                 'validation_input_sha256':validation_input_sha}
            history.append(row)
            with (folder/'epochs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            _json(folder/'progress.json',row)
            print(json.dumps({'kind':kind,**row},allow_nan=False),flush=True)
        if _identity(val_dataset,'validation',s['fixture_only'])!=val_identity:raise ValueError('Fixed validation identity changed')
        if any(_sha(ROOT/n)!=h for n,h in code.items()):raise ValueError('Model/training code changed during fit')
        files=[folder/'best_window.pt',folder/'last.pt',folder/'best_window_validation_predictions.npz']
        if macro_epoch is not None:files.append(folder/'best_macro.pt')
        files+=sorted((folder/'checkpoints').glob('epoch_*.pt'))
        summary={**started,'status':'completed','epochs_run':len(history),'optimizer_steps':step,
                 'best_window_epoch':window_epoch,'best_window_rmse':best_window,'best_macro_epoch':macro_epoch,
                 'best_site_macro_rmse':None if macro_epoch is None else best_macro,'last_validation_metrics':history[-1]['validation'],
                 'stop_reason':'completed_fixed_epochs','elapsed_seconds':time.perf_counter()-start,
                 'training_metric_is_online_not_fixed_TRAIN_evaluation':True,'validation_used_for_checkpoint_selection':True,
                 'train_online_rmse_is_unweighted':True,'site_balance_changes_loss_not_sampling':s['site_balanced_loss'],
                 'site_balance_performance_improvement_claimed':False,
                 'validation_input_sha256':validation_input_sha,
                 'global_minimum_guaranteed':False,'artifact_sha256':{str(f.relative_to(folder)):_sha(f) for f in files}}
        _json(folder/'summary.json',summary);return summary
    except BaseException as exc:
        _json(folder/'FAILED.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc),
              'epoch':epoch,'optimizer_steps':step,'CUDA_OOM':isinstance(exc,torch.cuda.OutOfMemoryError),
              'resume_or_partial_reuse_allowed':False,'fixture_only':s['fixture_only'],'input_fingerprint':input_fingerprint,
              'created_utc':datetime.now(timezone.utc).isoformat()})
        raise
