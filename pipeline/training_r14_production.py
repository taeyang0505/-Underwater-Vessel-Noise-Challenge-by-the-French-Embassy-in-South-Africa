"""Mac production14 engine, raw-epoch sharing and atomic bounded checkpoints.

CPU FP64 features -> chosen CPU/MPS FP32 backbone. Pure unweighted loss.
No Test/stress panels here; only source-pinned baseline selection chooses
checkpoints. First actual epoch is the stability gate, then fixed budget.
"""
from pathlib import Path
from datetime import datetime,timezone
import copy,gc,hashlib,json,os,random,subprocess,sys,time
import numpy as np
import torch
from pipeline.features_r14_production import ProductionFrontend
from pipeline.models_r14_production import make_production_model
from pipeline.training_r13 import physical_multitask_loss

ROOT=Path(__file__).resolve().parents[1]
SOURCE_PY='/Users/hongtaeyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
FAST=('itransformer','fdy','bg_cnn');TF='tfgrid_fdy_conformer'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic_json(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def cpu_state(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu()
    if isinstance(v,dict):return {k:cpu_state(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu_state(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu_state(x) for x in v)
    return v
def atomic_pt(p,v):
    tmp=Path(str(p)+'.tmp');torch.save(cpu_state(v),tmp);tmp.replace(p)
def source_codes():
    from pipeline.runner_r13 import CODE_NAMES
    extra=('pipeline/features_r14_torch.py','pipeline/features_r14_anchor.py','pipeline/features_r14_production.py',
        'pipeline/models_r14_production.py','pipeline/training_r14_production.py','ops/train_r14_production_nn.py')
    result={n:sha(ROOT/n) for n in dict.fromkeys((*CODE_NAMES,*extra))}
    overlay=ROOT/'ops/r14_production_runtime_py/manifest.json';m=json.loads(overlay.read_text())
    result[str(overlay.relative_to(ROOT))]=sha(overlay)
    for n,h in m['file_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Production runtime overlay changed:'+n)
        result[n]=h
    return result
def validate_config(path):
    p=Path(path).resolve();c=json.loads(p.read_text());approval=json.loads((ROOT/c['approval_path']).read_text())
    if (c['production_recipe_approved'] is not True or approval['production_recipe_approved'] is not True
        or c['input_channels']!=14 or c['neural_data_variant']!='baseline' or c['loss']!={'share_MSE':1.,'energy_weighted_cell_MSE':.25,'presence_or_contrast_BCE':0.,'target_weight_c':0.,'site_balanced_loss':False}
        or c['training']['early_stopping'] or c['training']['optimizer']!='Adam'
        or c['training']['CPU_threads']!=2 or c['raw_cache_maximum_epochs']!=2):raise ValueError('Foreign production policy')
    if sha(approval['user_instruction_path'])!=approval['user_instruction_sha256']:raise ValueError('User approval attachment differs')
    for k in FAST:
        if c['models'][k]['epochs']!=100:raise ValueError('Fast neural models require100 epochs')
    if c['models'][TF]['epochs']!=50:raise ValueError('Predeclared TF budget50 required')
    return c
def read_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def materialize(config_path,fold,epoch,cache):
    """Called only in the source Python subprocess; no model/backbone fit."""
    from pipeline.full_data_r13 import R13Sources,DynamicR13Dataset
    c=validate_config(config_path);cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    path=cache/f'epoch_{epoch:03d}.npz';meta=path.with_suffix('.json')
    if path.exists() or meta.exists():raise FileExistsError('Fresh complete cache pair required')
    policy={'data_variant':'baseline','include_shape':True,'production_recipe_approved':True,'approval_sha256':sha(ROOT/c['approval_path'])};policy['policy_sha256']=canonical(policy)
    source=R13Sources(ROOT/c['source_config_path'],bundle=ROOT/c['bundle_path'],runtime_policy=policy);ds=DynamicR13Dataset(source,fold,epoch=epoch)
    n=len(ds);db=np.empty((n,15,25),np.float32);q=np.empty_like(db);y=np.empty(n,np.float32);z=np.empty(n,np.float32);ids=[];base=np.empty(n,np.int64)
    started=time.perf_counter()
    for i in range(n):
        a=ds.raw_item(i);db[i]=a['db'];q[i]=a['cell_share'];y[i]=a['share'];z[i]=a['presence'];ids.append(a['sample_id']);base[i]=a['base_window_index']
    if len(set(base))!=n or not np.array_equal(y>0,z.astype(bool)) or not np.isfinite(q).all() or np.any(q[~np.isfinite(db)]!=0):raise ValueError('Raw epoch source/target invariants failed')
    tmp=Path(str(path)+'.tmp')
    with tmp.open('xb') as f:np.savez(f,db=db,cell_share=q,share=y,presence=z,sample_id=np.asarray(ids),base_window_index=base)
    tmp.replace(path);source.assert_unchanged()
    atomic_json(meta,{'format':'r14_production_raw_epoch_v1','config_sha256':sha(config_path),'fold':fold,'epoch':epoch,
        'rows':n,'source_identity':ds.identity,'schedule_audit':ds.epoch_audit(),'npz_sha256':sha(path),
        'source_code_sha256':source_codes(),'seconds':time.perf_counter()-started,'source_unchanged':True})
    return meta
def ensure_cache(config_path,fold,epoch,cache,code):
    path=cache/f'epoch_{epoch:03d}.npz';meta=path.with_suffix('.json')
    if not path.exists() and not meta.exists():
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT)+':/private/tmp/underwater-ml-deps:/private/tmp/underwater-eda-deps';env['DYLD_LIBRARY_PATH']='/opt/homebrew/opt/libomp/lib'
        log=cache/f'epoch_{epoch:03d}.log'
        with log.open('x') as f:r=subprocess.run([SOURCE_PY,'-m','ops.train_r14_production_nn','materialize','--config',str(config_path),'--fold',str(fold),'--epoch',str(epoch),'--cache',str(cache)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('Raw epoch producer failed; preserve partial and stop')
    if not path.is_file() or not meta.is_file():raise ValueError('Partial raw cache pair')
    m=json.loads(meta.read_text())
    if m['config_sha256']!=sha(config_path) or m['epoch']!=epoch or m['fold']!=fold or m['source_code_sha256']!=code or sha(path)!=m['npz_sha256']:raise ValueError('Raw cache binding differs')
    receipts=cache.parent/'epoch_receipts';receipts.mkdir(exist_ok=True)
    receipt=receipts/f"epoch_{epoch:03d}_{m['npz_sha256'][:16]}.json"
    if not receipt.exists():atomic_json(receipt,dict(m,cache_metadata_file_sha256=sha(meta)))
    elif json.loads(receipt.read_text())['npz_sha256']!=m['npz_sha256']:raise ValueError('Immutable epoch receipt collision')
    m['permanent_receipt_path']=str(receipt.relative_to(cache.parent));m['permanent_receipt_sha256']=sha(receipt)
    # Delete only our older complete cache files, never unrelated project data.
    for old in sorted(cache.glob('epoch_*.npz'),key=lambda p:(p.stat().st_mtime_ns,p.name))[:-2]:
        old.unlink();old.with_suffix('.json').unlink(missing_ok=True);old.with_suffix('.log').unlink(missing_ok=True)
    return read_npz(path),m
def feature_basis(config,fold):
    from pipeline.features_r12 import load_spectral_basis
    b=ROOT/config['bundle_path'];m=json.loads((b/'manifest.json').read_text());f=m['folds'][str(fold)]
    if m['data_hash']!=config['bundle_data_hash']:raise ValueError('Prepared bundle differs')
    basis=load_spectral_basis(b/f['basis_path'],expected_sha256=f['basis_sha256'],expected_training_binding=f['basis_training_binding'])
    p=b/f['variant_roles']['baseline']['selection']['path']
    if sha(p)!=f['variant_roles']['baseline']['selection']['sha256']:raise ValueError('Fixed selection differs')
    return basis,read_npz(p),m
def restore_checkpoint(path,*,expected_config_sha256=None,expected_fold=None,expected_data_hash=None,
                       expected_basis_file_sha256=None,expected_approval_sha256=None,device='cpu',allow_fixture=False):
    """Portable strict backbone+CPU frontend for frozen root inference."""
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['format']!='r14_production14_checkpoint_v1' or type(ck.get('fixture_only')) is not bool or (ck['fixture_only'] and not allow_fixture):raise ValueError('Production14 checkpoint required; fixtures require explicit allow_fixture')
    if not allow_fixture and any(v is None for v in (expected_config_sha256,expected_fold,expected_data_hash,expected_basis_file_sha256,expected_approval_sha256)):raise ValueError('Explicit production config/fold/data/PCA/approval context required')
    if expected_config_sha256 is not None and ck['config_sha256']!=expected_config_sha256:raise ValueError('Config binding differs')
    for expected,key in ((expected_fold,'fold'),(expected_data_hash,'data_hash'),(expected_basis_file_sha256,'basis_file_sha256'),(expected_approval_sha256,'approval_sha256')):
        if expected is not None and ck[key]!=expected:raise ValueError('Checkpoint context differs:'+key)
    if source_codes()!=ck['source_code_sha256']:raise ValueError('Checkpoint code closure differs')
    model=make_production_model(ck['kind']);model.load_state_dict(ck['state_dict'],strict=True)
    if model.architecture()!=ck['architecture']:raise ValueError('Architecture reconstruction differs')
    frontend=ProductionFrontend(ck['feature_basis']);frontend.verify_frozen_buffers()
    if frontend.metadata()!=ck['frontend_metadata']:raise ValueError('Feature state differs')
    return model.to(device).eval(),frontend,ck
def restore_completed_model(model_dir,device='cpu'):
    """Require complete budget+artifacts+fold-PCA, portable CPU best weights."""
    folder=Path(model_dir).resolve();summary=json.loads((folder/'summary.json').read_text());run=folder.parent
    identity=json.loads((run/'execution_identity.json').read_text());kind=summary['kind']
    if (run/'FAILED.json').exists() or summary.get('status')!='completed' or summary.get('fixture_only') is not False:raise ValueError('Completed unfailed production model required')
    if summary['epochs_run']!=identity['config']['models'][kind]['epochs'] or summary['source_code_sha256']!=source_codes():raise ValueError('Incomplete budget or foreign sources')
    if summary['config_sha256']!=identity['config_sha256'] or summary['data_hash']!=identity['data_hash']:raise ValueError('Run/summary binding differs')
    for n,h in summary['artifact_sha256'].items():
        p=folder/n
        if Path(n).name!=n or p.is_symlink() or not p.is_file() or sha(p)!=h:raise ValueError('Model artifact changed:'+n)
    bundle=ROOT/identity['config']['bundle_path'];m=json.loads((bundle/'manifest.json').read_text());f=m['folds'][str(identity['fold'])]
    model,frontend,ck=restore_checkpoint(folder/'best.pt',expected_config_sha256=identity['config_sha256'],expected_fold=identity['fold'],
        expected_data_hash=identity['data_hash'],expected_basis_file_sha256=f['basis_sha256'],expected_approval_sha256=identity['approval_sha256'],device=device)
    if ck['epoch']!=summary['best_epoch'] or ck['fold']!=identity['fold'] or ck['architecture']!=summary['architecture']:raise ValueError('Best checkpoint binding differs')
    from pipeline.features_r12 import load_spectral_basis
    b=load_spectral_basis(bundle/f['basis_path'],expected_sha256=f['basis_sha256'],expected_training_binding=f['basis_training_binding'])
    if b!=ck['feature_basis'] or m['data_hash']!=ck['data_hash']:raise ValueError('Foreign fold-TRAIN PCA or source data')
    return model,frontend,summary
@torch.no_grad()
def predict_completed_model(model_dir,db,batch_size=128,device='cpu'):
    if isinstance(db,dict) or type(batch_size)is not int or batch_size<1:raise ValueError('db-only numeric query and positive batch required')
    a=np.asarray(db)
    if a.ndim!=3 or a.shape[1:]!=(15,25) or not len(a) or a.dtype.kind!='f':raise ValueError('Floating db[N,15,25] required')
    model,frontend,summary=restore_completed_model(model_dir,device);ps=[];qs=[];ws=[]
    for i in range(0,len(a),batch_size):
        x,w=frontend(torch.from_numpy(a[i:i+batch_size]));p,s,q=model(x.to(device),w.to(device));ps.append(p.cpu().numpy());qs.append(q.cpu().numpy());ws.append(w.numpy())
    share=np.concatenate(ps);return {'share':share,'score':share.copy(),'cell_mask':np.concatenate(qs),'weights':np.concatenate(ws)}
def training(config_path,out,preflight,fold=0):
    """Fresh run; epoch1 gate is part of production training, not discarded."""
    config_path=Path(config_path).resolve();c=validate_config(config_path);out=Path(out).resolve();pf=json.loads(Path(preflight).read_text());code=source_codes()
    if out.exists():raise FileExistsError('No completed/failed/partial run reuse')
    if (pf.get('status')!='passed' or pf['config_sha256']!=sha(config_path) or pf['source_code_sha256']!=code or pf['fold']!=fold):raise ValueError('Actual14 capability preflight differs')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or not torch.backends.mps.is_available():raise ValueError('Actual MPS, fallback0 required')
    torch.set_num_threads(2);out.mkdir(parents=True);cache=out/'raw_cache';cache.mkdir();(out/'RUNNING.lock').write_text(str(os.getpid()))
    import shutil
    shutil.copyfile(config_path,out/'config.json');shutil.copyfile(ROOT/c['approval_path'],out/'approval.json');shutil.copyfile(preflight,out/'capability_preflight.json')
    basis,validation,bundle=feature_basis(c,fold);frontend=ProductionFrontend(basis);vx,vw=frontend(torch.from_numpy(validation['db']))
    identity={'format':'r14_production_neural_run_v1','config':c,'config_sha256':sha(config_path),'approval_sha256':sha(ROOT/c['approval_path']),
        'source_code_sha256':code,'data_hash':bundle['data_hash'],'fold':fold,'production_recipe_approved':True,
        'feature_basis':basis,'preflight_sha256':sha(preflight),'model_budgets':c['models'],'started_utc':datetime.now(timezone.utc).isoformat(),
        'PID':os.getpid(),'Python':sys.version,'torch':torch.__version__,'Test_or_stress_panels_read':False}
    atomic_json(out/'execution_identity.json',identity);torch.manual_seed(c['source_seed']);np.random.seed(c['source_seed']);random.seed(c['source_seed'])
    states={};step=0;epoch=0;kind=None;begin=time.perf_counter()
    try:
        for phase,kinds in [('fast3',FAST),('TF',[TF])]:
            states={}
            for kind in kinds:
                torch.manual_seed(c['source_seed']);device=c['models'][kind]['device'];model=make_production_model(kind).to(device)
                opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=c['models'][kind]['epochs'],eta_min=1e-5)
                folder=out/kind;folder.mkdir();states[kind]={'model':model,'optimizer':opt,'scheduler':sched,'best':float('inf'),'best_epoch':0,'steps':0,'folder':folder}
            for epoch in range(1,max(c['models'][k]['epochs'] for k in kinds)+1):
                if source_codes()!=code or sha(config_path)!=identity['config_sha256'] or sha(ROOT/c['approval_path'])!=identity['approval_sha256']:raise ValueError('Code/config/approval changed during run')
                raw,meta=ensure_cache(config_path,fold,epoch,cache,code)
                order=np.random.default_rng(np.random.SeedSequence([c['source_seed'],fold,epoch])).permutation(len(raw['share']))
                for kind in kinds:
                    if epoch>c['models'][kind]['epochs']:continue
                    st=states[kind];model,opt,sched=st['model'],st['optimizer'],st['scheduler'];device=c['models'][kind]['device'];bs=pf['models'][kind]['selected_batch_size']
                    model.train();seen=0;sse=cell=0.;started=time.perf_counter();maxgap=0.
                    for offset in range(0,len(order),bs):
                        ix=order[offset:offset+bs];xx,ww=frontend(torch.from_numpy(raw['db'][ix]));xx=xx.to(device);ww=ww.to(device)
                        labels=tuple(torch.from_numpy(raw[k][ix]).to(device) for k in ('share','presence','cell_share'))
                        opt.zero_grad(set_to_none=True);pred=model(xx,ww);loss,parts=physical_multitask_loss(pred,labels,ww)
                        loss.mean().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();st['steps']+=1;step+=1
                        count=len(ix);seen+=count;sse+=float(parts['share_mse'].detach().sum().cpu());cell+=float(parts['weighted_cell_mse'].detach().sum().cpu());maxgap=max(maxgap,float(parts['raw_label_vs_rounded_pool_gap'].max().cpu()))
                        if offset==0 or st['steps']%100==0:atomic_json(out/'progress.json',{'status':'training','phase':phase,'kind':kind,'epoch':epoch,'rows_seen':seen,'total_rows':len(order),'optimizer_steps':st['steps'],'elapsed_seconds':time.perf_counter()-begin})
                    if seen!=len(order):raise ValueError('Incomplete actual epoch')
                    model.eval();preds=[];masks=[]
                    with torch.no_grad():
                        for start in range(0,len(vx),bs):
                            p,s,q=model(vx[start:start+bs].to(device),vw[start:start+bs].to(device));preds.append(p.cpu().numpy());masks.append(q.cpu().numpy())
                    vp=np.concatenate(preds);qm=np.concatenate(masks)
                    if not np.isfinite(vp).all() or not np.isfinite(qm).all():raise FloatingPointError('Nonfinite fixed selection')
                    metric={'rmse':float(np.sqrt(np.mean((vp.astype(float)-validation['share'])**2))),'n':len(vp)}
                    lr=opt.param_groups[0]['lr'];sched.step();seconds=time.perf_counter()-started
                    rng_np=np.random.get_state()
                    ck={'format':'r14_production14_checkpoint_v1','kind':kind,'epoch':epoch,'state_dict':model.state_dict(),'architecture':model.architecture(),
                        'feature_basis':basis,'basis_file_sha256':bundle['folds'][str(fold)]['basis_sha256'],
                        'approval_sha256':identity['approval_sha256'],'user_instruction_sha256':json.loads((out/'approval.json').read_text())['user_instruction_sha256'],
                        'frontend_metadata':frontend.metadata(),'config_sha256':identity['config_sha256'],'source_code_sha256':code,
                        'data_hash':bundle['data_hash'],'fold':fold,'device_used':device,'batch_size':bs,'optimizer_steps':st['steps'],'selection_metrics':metric,
                        'fixture_only':False,'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':sched.state_dict(),
                        'torch_rng_state':torch.get_rng_state(),'MPS_rng_state':torch.mps.get_rng_state(),
                        'numpy_rng_state':{'algorithm':rng_np[0],'state_uint32':rng_np[1].tolist(),'position':rng_np[2],'has_gauss':rng_np[3],'cached_gaussian':rng_np[4]},
                        'python_random_state':random.getstate(),'raw_epoch_receipt_path':meta['permanent_receipt_path'],'raw_epoch_receipt_sha256':meta['permanent_receipt_sha256']}
                    if metric['rmse']<st['best']:
                        st['best']=metric['rmse'];st['best_epoch']=epoch;atomic_pt(st['folder']/'best.pt',ck)
                        temp=st['folder']/'best_selection_predictions.npz.tmp'
                        with temp.open('wb') as f:np.savez_compressed(f,share=vp,cell_mask=qm,sample_id=validation['sample_id'],target_share=validation['share'])
                        temp.replace(st['folder']/'best_selection_predictions.npz')
                    atomic_pt(st['folder']/'latest.pt',ck)
                    row={'kind':kind,'epoch':epoch,'budget':c['models'][kind]['epochs'],'train_online_rmse':(sse/seen)**.5,'train_cell_mse':cell/seen,
                        'selection':metric,'best_selection_rmse':st['best'],'best_epoch':st['best_epoch'],'actual_epoch_seconds_including_validation_checkpoint':time.perf_counter()-started,
                        'training_and_validation_seconds_before_checkpoint':seconds,'raw_cache_materialization_seconds':meta['seconds'],
                        'raw_epoch_receipt_path':meta['permanent_receipt_path'],'raw_epoch_receipt_sha256':meta['permanent_receipt_sha256'],
                        'rows':seen,'optimizer_steps':st['steps'],'learning_rate':lr,'next_learning_rate':opt.param_groups[0]['lr'],'max_rounded_pool_gap':maxgap,
                        'site_or_target_row_weighting':False,'cache_shared_in_phase':list(kinds),'source_epoch':epoch,'memory_driver_bytes_sampled':torch.mps.driver_allocated_memory()}
                    with (st['folder']/'epochs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                    atomic_json(st['folder']/'progress.json',row);print(json.dumps(row),flush=True)
                    if epoch==1:
                        atomic_json(st['folder']/'EPOCH1_STABILITY_GATE.json',{'status':'passed','complete_rows':seen,'finite_optimizer_steps':st['steps'],
                            'source_epoch_count_exact':seen==len(order),'continues_same_fresh_training_to_fixed_budget':True,'epoch1_seconds':time.perf_counter()-started})
                atomic_json(out/'run_state.json',{'status':'running','phase':phase,'complete_epoch':epoch,'elapsed_seconds':time.perf_counter()-begin})
            for kind in kinds:
                st=states[kind];folder=st['folder'];artifacts={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
                atomic_json(folder/'summary.json',{'status':'completed','kind':kind,'epochs_run':c['models'][kind]['epochs'],'early_stopping':False,'optimizer':'Adam',
                    'parameters':sum(p.numel() for p in st['model'].parameters()),'best_epoch':st['best_epoch'],'best_selection_rmse':st['best'],
                    'architecture':st['model'].architecture(),'config_sha256':identity['config_sha256'],'source_code_sha256':code,'data_hash':bundle['data_hash'],
                    'fold':fold,'basis_sha256':basis['basis_sha256'],'fixture_only':False,'artifact_sha256':artifacts})
            states.clear();gc.collect();torch.mps.empty_cache()
            if phase=='fast3':
                atomic_json(out/'FAST3_COMPLETED.json',{'status':'completed','kinds':list(kinds),'summary_sha256':{k:sha(out/k/'summary.json') for k in kinds}})
                atomic_json(out/'FAST_MODELS_FROZEN.json',{'format':'r14_fast3_neural_registry_v1','status':'completed','fold':fold,'config_sha256':identity['config_sha256'],
                    'data_hash':bundle['data_hash'],'source_code_sha256':code,'models':{k:{'summary_sha256':sha(out/k/'summary.json'),'best_checkpoint_sha256':sha(out/k/'best.pt')} for k in kinds}})
        atomic_json(out/'ALL_NEURAL_FROZEN.json',{'status':'completed','fold':fold,'models':{k:sha(out/k/'summary.json') for k in (*FAST,TF)},'config_sha256':identity['config_sha256'],'source_code_sha256':code})
        atomic_json(out/'ALL_FOLD0_MODELS_FROZEN.json',{'format':'r14_full4_neural_registry_v1','status':'completed','fold':fold,'config_sha256':identity['config_sha256'],
            'data_hash':bundle['data_hash'],'source_code_sha256':code,'models':{k:{'summary_sha256':sha(out/k/'summary.json'),'best_checkpoint_sha256':sha(out/k/'best.pt')} for k in (*FAST,TF)}})
        atomic_json(out/'run_state.json',{'status':'completed','elapsed_seconds':time.perf_counter()-begin,'all_neural_frozen':True})
    except BaseException as e:
        atomic_json(out/'FAILED.json',{'status':'failed','kind':kind,'epoch':epoch,'optimizer_steps':step,'error_type':type(e).__name__,'error':str(e),'partial_reuse_allowed':False})
        atomic_json(out/'run_state.json',{'status':'failed','error':str(e)});raise
    finally:(out/'RUNNING.lock').unlink(missing_ok=True)
