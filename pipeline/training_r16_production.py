"""Fresh R16 CPU/MPS engine with relative8 feature mmap and decoupled heads.

No challenge Test or stress panels are read by training. Raw source provider
exposes TRAIN and fit-only selection. Features are computed in NumPy FP64
once, chunked into mmap; MPS receives FP32 tensors without unsupported-op
fallback or per-batch quantile/PCA work. Fixed100 epochs, Adam, no ES.
"""
from __future__ import annotations
import gc,hashlib,json,os,random,signal,sys,time
from pathlib import Path
import numpy as np
import torch
from pipeline.relative_grid_r6_v2 import relative_grid_v2
from pipeline.models_r16_production import make_r16_model,decoupled_loss,clip_decoupled_gradients

ROOT=Path(__file__).resolve().parents[1]
KINDS=('bg_cnn','fdy','itransformer','tfgrid_fdy_conformer')
SOURCE_NAMES=('pipeline/models.py','pipeline/models_r9.py','pipeline/grid_models_r5.py',
 'pipeline/hybrid_models_r5.py','pipeline/relative_grid_r6.py','pipeline/relative_grid_r6_v2.py',
 'pipeline/relative_models_r6_v2.py','pipeline/models_r16_production.py',
 'pipeline/training_r16_production.py','ops/train_r16_production_nn.py',
 'pipeline/r16_production_data.py','ops/r16_forward_data.py')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def atomic_json(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    q.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)
def read(p):return json.loads(Path(p).read_text())
def source_codes():return{n:sha(ROOT/n)for n in SOURCE_NAMES if (ROOT/n).exists()}
def verify_codes(expected):
    if source_codes()!=expected:raise ValueError('R16 neural source closure changed')
def guard(event,args):
    if event=='open'and args and isinstance(args[0],(str,bytes,os.PathLike)):
        n=os.fsdecode(args[0]).lower().replace('\\','/')
        if n.endswith('/test.csv')or '/eda_deep/'in n or '/r15_band_comparison_'in n:raise PermissionError('Training cannot read challenge query or query-derived EDA')
def cpu_state(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu()
    if isinstance(x,dict):return{k:cpu_state(v)for k,v in x.items()}
    if isinstance(x,list):return[cpu_state(v)for v in x]
    if isinstance(x,tuple):return tuple(cpu_state(v)for v in x)
    return x
def atomic_pt(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');torch.save(cpu_state(v),tmp);tmp.replace(p)
def atomic_npz(p,**arrays):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb')as f:np.savez_compressed(f,**arrays)
    tmp.replace(p)
def validate_config(path):
    c=read(path)
    tr=c['training']
    if (c['format']!='r16_neural_mac_v1'or c['input_channels']!=8 or c['input_policy']!='relative_grid_v2'
        or c['forbidden_features']!=['PCA','LF_anchor','q0','absolute_db','highshare']
        or tr['epochs']!=100 or tr['optimizer']!='Adam'or tr['early_stopping']is not False
        or tr['CPU_threads']!=2 or tr['dropout']!=.1 or list(c['models'])!=list(KINDS)
        or c['loss']!={'share_MSE':1.,'energy_weighted_cell_MSE':.25,'independent_classifier_BCE':1.,'contrast_BCE':0.}):raise ValueError('R16 neural fixed policy differs')
    if any(v['epochs']!=100 or v['device']not in('cpu','mps')for v in c['models'].values()):raise ValueError('100-epoch CPU/MPS model contract')
    return c
def auc(y,p):
    y=np.asarray(y)>0;p=np.asarray(p,float);n1=int(y.sum());n0=len(y)-n1
    if not n1 or not n0:return None
    ix=np.argsort(p,kind='stable');x=p[ix];rank=np.empty(len(x),float)
    lo=np.r_[0,np.flatnonzero(x[1:]!=x[:-1])+1];hi=np.r_[lo[1:],len(x)]
    for a,b in zip(lo,hi):rank[a:b]=(a+1+b)/2
    return float((rank[y[ix]].sum()-n1*(n1+1)/2)/(n1*n0))
def metrics(y,p,c):
    y=np.asarray(y,float);p=np.asarray(p,float);c=np.asarray(c,float);e=p-y
    return{'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(abs(e).mean()),'bias':float(e.mean()),
       'share_auc':auc(y,p),'classifier_auc':auc(y,c),'classifier_brier':float(np.mean((c-(y>0))**2)),
       'share_mean':float(p.mean()),'classifier_mean':float(c.mean()),'classifier_ge099':float(np.mean(c>=.99))}

def materialize_features(manifest,fold,role,folder):
    """Chunked FP64 NumPy reference features; no peak full x8 allocation."""
    from pipeline.r16_production_data import load_production_role,validate_production_manifest
    data_qa=validate_production_manifest(manifest,fold,verify_sources=True)
    folder=Path(folder)
    if folder.exists():
        meta=read(folder/'manifest.json')
        if meta['source_manifest_sha256']!=sha(manifest)or meta['source_code_sha256']!=source_codes():raise ValueError('Foreign or stale feature cache')
        for n,h in meta['sha256'].items():
            if sha(folder/n)!=h:raise ValueError('Feature cache changed')
        return meta
    d=load_production_role(manifest,fold,role);n=len(d['share']);folder.mkdir(parents=True,exist_ok=False)
    if not n or d['db'].shape!=(n,15,25)or not np.array_equal(d['share']>0,d['presence'].astype(bool)):raise ValueError('Raw train/selection schema differs')
    started=time.perf_counter()
    specs={'x':((n,8,15,25),np.float32),'w':((n,15,25),np.float32),'cell_share':((n,15,25),np.float32),
           'share':((n,),np.float64),'presence':((n,),np.float32)}
    mm={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=t,shape=s)for k,(s,t)in specs.items()}
    gap=0.
    for off in range(0,n,256):
        sl=slice(off,min(off+256,n));x,w=relative_grid_v2(d['db'][sl]);q=np.asarray(d['cell_share'][sl],np.float32);y=np.asarray(d['share'][sl],float)
        g=float(abs((q*w).sum((1,2))-y).max());gap=max(gap,g)
        if not np.isfinite(q).all()or(q<0).any()or(q>1).any()or not np.isfinite(y).all()or(y<0).any()or(y>1).any()or np.any(q[w==0]!=0)or g>.0058:raise ValueError('Raw supervision/quantized pool bounds differ')
        for k,a in [('x',x),('w',w),('cell_share',q),('share',y),('presence',d['presence'][sl])]:mm[k][sl]=a
    for a in mm.values():a.flush()
    del mm
    for name in ('sample_id','site','background_group','reference_component','base_window_index','operator'):
        if name in d:
            a=np.asarray(d[name]);a=a.astype(str)if a.dtype.kind in'OU'else a
            np.save(folder/(name+'.npy'),a,allow_pickle=False)
    meta={'status':'completed','format':'r16_relative8_CPU_mmap_v1','source_manifest_sha256':sha(manifest),
          'fold':fold,'role':role,'n':n,'source_code_sha256':source_codes(),'preprocessing_device':'cpu','feature_dtype_before_output':'numpyfloat64',
          'backbone_input_dtype':'float32','MPS_unsupported_fallback_used':False,'PCA_or_anchor':False,
          'data_source_validation':data_qa,'max_raw_q_vs_rounded_w_pool_gap':gap,'seconds':time.perf_counter()-started,
          'sha256':{p.name:sha(p)for p in folder.glob('*.npy')}}
    atomic_json(folder/'manifest.json',meta);return meta
def read_cache(folder):
    return{p.stem:np.load(p,mmap_mode='r',allow_pickle=False)for p in Path(folder).glob('*.npy')}

@torch.no_grad()
def predict_features(model,x,w,device,batch=128):
    model.eval();p=[];c=[];q=[]
    for off in range(0,len(x),batch):
        # mmap is read-only; tensor owns its bounded contiguous microbatch.
        xx=torch.from_numpy(np.array(x[off:off+batch],copy=True)).to(device)
        ww=torch.from_numpy(np.array(w[off:off+batch],copy=True)).to(device)
        pp,ll,qq=model(xx,ww);p.append(pp.cpu().numpy());c.append(ll.sigmoid().cpu().numpy());q.append(qq.cpu().numpy())
    return np.concatenate(p),np.concatenate(c),np.concatenate(q)

def rng_state(device):
    s=np.random.get_state()
    return{'torch':torch.get_rng_state(),'mps':torch.mps.get_rng_state()if device=='mps'else None,
       'numpy':{'algorithm':s[0],'state':s[1].tolist(),'position':s[2],'has_gauss':s[3],'cached_gaussian':s[4]},'python':random.getstate()}
def restore_rng(r,device):
    torch.set_rng_state(r['torch'])
    if device=='mps':torch.mps.set_rng_state(r['mps'])
    s=r['numpy'];np.random.set_state((s['algorithm'],np.asarray(s['state'],np.uint32),s['position'],s['has_gauss'],s['cached_gaussian']))
    random.setstate(r['python'])
def restore_checkpoint(path,*,expected_identity=None,device='cpu',allow_fixture=False):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['format']!='r16_production8_checkpoint_v1'or(ck['fixture_only']and not allow_fixture):raise ValueError('R16 production checkpoint required')
    verify_codes(ck['source_code_sha256'])
    if expected_identity is not None and ck['identity_sha256']!=expected_identity:raise ValueError('Checkpoint run identity differs')
    model=make_r16_model(ck['kind']);model.load_state_dict(ck['state_dict'],strict=True)
    if model.architecture()!=ck['architecture']:raise ValueError('R16 architecture reconstruction differs')
    return model.to(device).eval(),ck
def restore_completed_model(folder,device='cpu'):
    folder=Path(folder);s=read(folder/'summary.json');identity=read(folder.parent/'execution_identity.json')
    if s['status']!='completed'or s['epochs_run']!=100 or (folder.parent/'FAILED.json').exists():raise ValueError('Full100 completed unfailed model required')
    if s['identity_sha256']!=sha(folder.parent/'execution_identity.json'):raise ValueError('Completed run identity changed')
    manifest=ROOT/identity['data_manifest_path']
    if sha(manifest)!=identity['data_manifest_sha256']:raise ValueError('Completed source manifest changed')
    for n,h in s['artifact_sha256'].items():
        if sha(folder/n)!=h:raise ValueError('Completed R16 artifact differs')
    m,reg=restore_checkpoint(folder/'best_regression.pt',expected_identity=sha(folder.parent/'execution_identity.json'),device=device)
    _,clf=restore_checkpoint(folder/'best_classifier.pt',expected_identity=sha(folder.parent/'execution_identity.json'),device='cpu')
    m.classifier.load_state_dict({k[len('classifier.'):]:v for k,v in clf['state_dict'].items()if k.startswith('classifier.')},strict=True)
    return m.eval(),s
@torch.no_grad()
def predict_completed_model(folder,db,batch_size=128,device='cpu'):
    if isinstance(db,dict):raise TypeError('Query db only; no targets or site metadata')
    m,s=restore_completed_model(folder,device);a=np.asarray(db);out=[]
    for off in range(0,len(a),batch_size):
        x,w=relative_grid_v2(a[off:off+batch_size]);p,c,q=predict_features(m,x,w,device,batch_size)
        out.append((p,c,q,w))
    return{n:np.concatenate([r[i]for r in out])for i,n in enumerate(('share','score','cell_mask','weights'))}

def train(config_path,manifest,out,preflight,fold=0,resume=False,cache_root=None):
    from pipeline.r16_production_data import validate_production_manifest
    c=validate_config(config_path);pf=read(preflight);out=Path(out);manifest=Path(manifest);code=source_codes()
    source_validation=validate_production_manifest(manifest,fold,verify_sources=True)
    if (pf['status']!='passed'or pf.get('fold')!=fold or pf['config_sha256']!=sha(config_path)or pf['data_manifest_sha256']!=sha(manifest)or pf['source_code_sha256']!=code):raise ValueError('Capability preflight identity differs')
    if any(v['device']=='mps'for v in c['models'].values())and(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0'or not torch.backends.mps.is_available()):raise ValueError('Actual MPS fallback0 required')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if resume:
        if not(out/'PAUSED.json').exists()or(out/'FAILED.json').exists():raise ValueError('Only explicitly paused complete epoch may resume')
        identity=read(out/'execution_identity.json')
        if identity['config_sha256']!=sha(config_path)or identity['data_manifest_sha256']!=sha(manifest)or identity['source_code_sha256']!=code:raise ValueError('Paused run context differs')
    else:
        if out.exists():raise FileExistsError('Fresh run output required')
        out.mkdir(parents=True)
        identity={'format':'r16_production_neural_run_v1','config':c,'config_sha256':sha(config_path),'data_manifest_sha256':sha(manifest),
              'data_manifest_path':str(manifest.resolve().relative_to(ROOT)),
              'source_code_sha256':code,'fold':fold,'started_unix':time.time(),'PID':os.getpid(),
              'Python':sys.version,'torch':str(torch.__version__),'preflight_sha256':sha(preflight),'Test_or_stress_read':False,
              'full_epoch_budget':100,'training_data_static':'deterministic R16 fixed mixed TRAIN; all rows each epoch',
              'input_features':'precompute relative8 once CPU FP64 to bounded mmap; no per-batch CPUquantile',
              'data_source_validation':source_validation}
        atomic_json(out/'execution_identity.json',identity)
    identity_sha=sha(out/'execution_identity.json');cache=Path(cache_root)if cache_root else out/'feature_cache'
    cache=cache.resolve()
    if not cache.is_relative_to(ROOT):raise ValueError('Feature mmap cache must remain within the project')
    for role in ('train','selection'):materialize_features(manifest,fold,role,cache/role)
    tr=read_cache(cache/'train');va=read_cache(cache/'selection');selection_y=np.asarray(va['share'],float)
    pause={'requested':False}
    if hasattr(signal,'SIGUSR1'):signal.signal(signal.SIGUSR1,lambda *_:pause.update(requested=True))
    began=time.perf_counter();active=None;epoch=0
    try:
        for kind,spec in c['models'].items():
            active=kind;folder=out/kind
            if (folder/'summary.json').exists():
                restore_completed_model(folder,'cpu');continue
            folder.mkdir(exist_ok=True);device=spec['device'];batch=pf['models'][kind]['selected_batch_size']
            torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
            model=make_r16_model(kind).to(device)
            opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=1e-5)
            best=float('inf');best_clf=-1.;best_epoch=best_clf_epoch=steps=0;start_epoch=1
            if resume and(folder/'latest.pt').exists():
                model,ck=restore_checkpoint(folder/'latest.pt',expected_identity=identity_sha,device=device)
                opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003);opt.load_state_dict(ck['optimizer_state_dict'])
                for state in opt.state.values():
                    for k,v in state.items():
                        if isinstance(v,torch.Tensor)and k!='step':state[k]=v.to(device)
                scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=1e-5);scheduler.load_state_dict(ck['scheduler_state_dict'])
                restore_rng(ck['rng_state'],device);start_epoch=ck['epoch']+1;best=ck['best_rmse'];best_clf=ck['best_classifier_auc'];best_epoch=ck['best_regression_epoch'];best_clf_epoch=ck['best_classifier_epoch'];steps=ck['optimizer_steps']
            for epoch in range(start_epoch,101):
                verify_codes(code)
                if sha(config_path)!=identity['config_sha256']or sha(manifest)!=identity['data_manifest_sha256']:raise ValueError('Frozen config/data changed')
                order=np.random.default_rng(np.random.SeedSequence([c['seed'],fold,epoch])).permutation(len(tr['share']))
                model.train();sse=cell=bce=0.;seen=0;started=time.perf_counter()
                for off in range(0,len(order),batch):
                    ix=order[off:off+batch]
                    x=torch.from_numpy(np.array(tr['x'][ix],copy=True)).to(device);w=torch.from_numpy(np.array(tr['w'][ix],copy=True)).to(device)
                    labels=tuple(torch.from_numpy(np.array(tr[k][ix],dtype=np.float32,copy=True)).to(device)for k in('share','presence','cell_share'))
                    opt.zero_grad(set_to_none=True);pred=model(x,w);loss,parts=decoupled_loss(pred,labels,w);loss.backward();clip_decoupled_gradients(model);opt.step();steps+=1
                    n=len(ix);seen+=n;sse+=float(parts['share_mse'].detach().sum().cpu());cell+=float(parts['weighted_cell_mse'].detach().sum().cpu());bce+=float(parts['classification_bce'].detach().sum().cpu())
                    if off==0 or steps%100==0:atomic_json(out/'progress.json',{'status':'training','kind':kind,'epoch':epoch,'maximum_epochs':100,'rows_seen':seen,'rows_total':len(order),'steps':steps,'elapsed_seconds':time.perf_counter()-began})
                if seen!=len(order):raise ValueError('Incomplete TRAIN epoch')
                pp,cc,qq=predict_features(model,va['x'],va['w'],device,batch);mm=metrics(selection_y,pp,cc)
                if not np.isfinite(pp).all()or not np.isfinite(qq).all():raise FloatingPointError('Nonfinite validation output')
                improved=mm['rmse']<best;clf_improved=mm['classifier_auc']>best_clf
                if improved:best=mm['rmse'];best_epoch=epoch
                if clf_improved:best_clf=mm['classifier_auc'];best_clf_epoch=epoch
                lr=opt.param_groups[0]['lr'];scheduler.step()
                ck={'format':'r16_production8_checkpoint_v1','kind':kind,'epoch':epoch,'state_dict':model.state_dict(),'architecture':model.architecture(),
                  'identity_sha256':identity_sha,'config_sha256':identity['config_sha256'],'source_code_sha256':code,'data_manifest_sha256':identity['data_manifest_sha256'],
                  'fold':fold,'fixture_only':False,'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'rng_state':rng_state(device),
                  'optimizer_steps':steps,'best_rmse':best,'best_classifier_auc':best_clf,'best_regression_epoch':best_epoch,'best_classifier_epoch':best_clf_epoch,
                  'selection_metrics':mm,'feature_cache_manifests':{role:sha(cache/role/'manifest.json')for role in('train','selection')}}
                if improved:
                    atomic_pt(folder/'best_regression.pt',ck)
                    atomic_npz(folder/'best_regression_selection_predictions.npz',share=pp,cell_mask=qq,target_share=selection_y,sample_id=va['sample_id'])
                if clf_improved:
                    atomic_pt(folder/'best_classifier.pt',ck)
                    atomic_npz(folder/'best_classifier_selection_predictions.npz',classifier_prob=cc,target_presence=selection_y>0,sample_id=va['sample_id'])
                atomic_pt(folder/'latest.pt',ck)
                row={'kind':kind,'epoch':epoch,'maximum_epochs':100,'train_online_rmse':float(np.sqrt(sse/seen)),'train_weighted_cell_mse':cell/seen,'train_classifier_bce':bce/seen,
                  'selection':mm,'best_regression_rmse':best,'best_regression_epoch':best_epoch,'best_classifier_auc':best_clf,'best_classifier_epoch':best_clf_epoch,
                  'rows':seen,'optimizer_steps':steps,'learning_rate':lr,'next_learning_rate':opt.param_groups[0]['lr'],'epoch_seconds_including_validation_checkpoint':time.perf_counter()-started,
                  'MPS_driver_bytes':int(torch.mps.driver_allocated_memory())if device=='mps'else None,'device':device}
                with(folder/'epochs.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                atomic_json(folder/'progress.json',row);print(json.dumps(row),flush=True)
                if pause['requested']or(out/'PAUSE_AFTER_EPOCH').exists():
                    atomic_json(out/'PAUSED.json',{'status':'paused_at_complete_epoch','kind':kind,'epoch':epoch,'latest_sha256':sha(folder/'latest.pt'),'identity_sha256':identity_sha,'explicit_resume_required':True});return out/'PAUSED.json'
            atomic_json(folder/'summary.json',{'status':'completed','epochs_run':100,'early_stopping':False,'optimizer':'Adam','kind':kind,'architecture':model.architecture(),
                  'best_regression_epoch':best_epoch,'best_regression_rmse':best,'best_classifier_epoch':best_clf_epoch,'best_classifier_auc':best_clf,
                  'identity_sha256':identity_sha,'source_code_sha256':code,'data_manifest_sha256':identity['data_manifest_sha256'],
                  'artifact_sha256':{p.name:sha(p)for p in folder.iterdir()if p.is_file()}})
            del model,opt,scheduler;gc.collect()
            if device=='mps':torch.mps.empty_cache()
        final_source_validation=validate_production_manifest(manifest,fold,verify_sources=True)
        atomic_json(out/'ALL_COMPLETED.json',{'status':'completed','models':list(KINDS),'fold':fold,'identity_sha256':identity_sha,'elapsed_seconds':time.perf_counter()-began,
                                            'final_data_source_validation':final_source_validation})
        if (out/'PAUSED.json').exists():(out/'PAUSED.json').unlink()
        return out/'ALL_COMPLETED.json'
    except BaseException as e:
        atomic_json(out/'FAILED.json',{'status':'failed','kind':active,'epoch':epoch,'error':repr(e),'no_partial_resume':True});raise
