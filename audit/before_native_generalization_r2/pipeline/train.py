"""Explicit training on verified prebuilt V2 arrays, without implicit sampling.

One package defines one held-out split. Group CV is orchestrated separately with
--cv-mode. Model seeds are never called CV folds. Existing runs are immutable.
"""
from pathlib import Path
import argparse,hashlib,json,platform,subprocess,sys,time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import mean_squared_error,roc_auc_score
from pipeline.mixtures import fixed_features,cnn_channels
from pipeline.models import make_model
ROOT=Path(__file__).resolve().parents[1]
MODEL_KINDS=('lgbm','lgbm_physics','lgbm_retrieval','tabm','cnn','fdy','itransformer')

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def rmse(y,p):return float(np.sqrt(mean_squared_error(y,p)))

def load_prebuilt_data(folder,cv_mode=False):
    """Check source gates, exact file hashes and saved energy-target identity."""
    folder=Path(folder).resolve();gate_path=folder/'readiness_manifest.json'
    gate=json.loads(gate_path.read_text())
    if gate.get('ready_to_train') is not True:raise ValueError('Prebuilt ready_to_train gate is not true')
    if not str(gate.get('version','')).startswith('quality_v2'):raise ValueError('An explicitly prepared quality_v2 package is required')
    for flag in ['source_checks_passed','reference_group_and_duplicate_isolation_passed','background_week_and_purge_isolation_passed','sealed_holdout_locked']:
        if gate.get(flag) is not True:raise ValueError('Required source gate failed: '+flag)
    for flag in ['challenge_test_accessed','existing_sealed_NOAA_accessed','sealed_holdout_materialized','frozen_v1_data_modified']:
        if gate.get(flag) is not False:raise ValueError('Unsafe or unspecified provenance flag: '+flag)
    files=gate.get('file_sha256')
    if not isinstance(files,dict) or not files:raise ValueError('Missing file_sha256 map')
    for name,digest in files.items():
        if Path(name).name!=name or name in ['', '.', '..']:raise ValueError('Unsafe package file name')
        # In nested CV, outer/alias arrays belong to the orchestrator only.
        # Their declared hashes identify the package, but this trainer never
        # opens their values or bytes. Verify just the two training inputs.
        if cv_mode and name not in ['train.npz','validation.npz','train_recipes.json','validation_recipes.json']:continue
        path=folder/name
        if path.is_symlink() or not path.is_file():raise ValueError('Missing or linked package file: '+name)
        if sha256(path)!=digest:raise ValueError('Package SHA-256 mismatch: '+name)
    package_hash=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    if package_hash!=gate.get('data_package_hash'):raise ValueError('data_package_hash mismatch')
    version_dir=Path(gate['source_version_dir'])
    if not version_dir.is_absolute():version_dir=ROOT/version_dir
    if sha256(version_dir/'manifest.json')!=gate['source_version_manifest_sha256']:
        raise ValueError('Source-version manifest changed; refresh readiness after verification')
    if cv_mode:
        if not gate.get('cv_protocol'):raise ValueError('--cv-mode requires an explicit cv_protocol in the readiness gate')
        splits=['train','validation']
    else:
        splits=['train','validation','calibration']
        if (folder/'domain_holdout.npz').exists():splits.append('domain_holdout')
    if (folder/'sealed_holdout.npz').exists():raise ValueError('Sealed holdout must remain outside development packages')
    data={};checks={}
    for split in splits:
        name=split+'.npz'
        if name not in files:raise ValueError('Required split is not fingerprinted: '+name)
        with np.load(folder/name,allow_pickle=False) as z:
            if not {'db','share','presence','cell_share'}.issubset(z.files):raise ValueError('Missing training arrays in '+name)
            d={k:z[k] for k in ['db','share','presence','cell_share']}
        n=len(d['share']);db=d['db'];q=d['cell_share'];y=d['share'];presence=d['presence']
        if not n or db.shape!=(n,15,25) or q.shape!=db.shape or y.shape!=(n,) or presence.shape!=(n,):raise ValueError('Invalid dimensions: '+split)
        if any(v.dtype!=np.float32 for v in d.values()):raise ValueError('Expected float32 arrays: '+split)
        partial_bands=(gate.get('feature_version')==3 or gate.get('preprocessing_version')==3 or 'physics' in str(gate.get('version','')))
        observed_minutes=np.isfinite(db).any(2) if partial_bands else np.isfinite(db).all(2)
        if np.isinf(db).any() or not (observed_minutes.sum(1)>=3).all():raise ValueError('Invalid observation coverage: '+split)
        if not all(np.isfinite(v).all() for v in [q,y,presence]):raise ValueError('Nonfinite targets: '+split)
        if not ((y>=0)&(y<=1)).all() or not ((q>=0)&(q<=1)).all():raise ValueError('Targets outside [0,1]: '+split)
        if not np.array_equal(presence,(y>0).astype('float32')):raise ValueError('Presence/share mismatch: '+split)
        error=0.
        for start in range(0,n,512):
            x=db[start:start+512].astype('float64');observed=np.isfinite(x)
            energy=np.where(observed,10**((x-np.nanmax(x,(1,2),keepdims=True))/10),0)
            recovered=(q[start:start+512]*energy).sum((1,2))/energy.sum((1,2))
            error=max(error,float(np.max(np.abs(recovered-y[start:start+512]))))
        if error>3e-6:raise ValueError('Saved cell/window energy targets disagree: '+split)
        part=gate.get('partitions',{}).get(split,{})
        if part.get('examples')!=n:raise ValueError('Readiness split count mismatch: '+split)
        if part.get('positive_draw_became_zero_count')!=0 or part.get('frozen_generation_replayed') is not True:raise ValueError('Generation target/replay gate failed: '+split)
        checks[split]={'examples':n,'max_cell_target_error':error,'npz_sha256':files[name]};data[split]=d
    return data,gate,{'readiness_manifest_sha256':sha256(gate_path),'data_package_hash':package_hash,'splits':checks,
                      'unopened_nontraining_entries':sorted(set(files)-{'train.npz','validation.npz','train_recipes.json','validation_recipes.json'}) if cv_mode else []}

@torch.inference_mode()
def predict_deep(model,kind,data,device,batch=256,inputs=None,feature_version=None):
    model.eval();shares=[];scores=[]
    version=getattr(model,'feature_version',1) if feature_version is None else feature_version
    if hasattr(model,'feature_version') and version!=model.feature_version:raise ValueError('Requested features differ from checkpoint architecture')
    if inputs is None:inputs=fixed_features(data['db'],version=version) if kind=='tabm' else cnn_channels(data['db'],version=version)
    for start in range(0,len(data['share']),batch):
        sl=slice(start,start+batch)
        if kind=='tabm':
            z=model(torch.from_numpy(inputs[sl]).to(device)).sigmoid().mean(1);share,score=z[:,0],z[:,1]
        else:
            c,w=inputs;share,logit,_=model(torch.from_numpy(c[sl]).to(device),torch.from_numpy(w[sl]).to(device));score=logit.sigmoid()
        shares.append(share.cpu().numpy());scores.append(score.cpu().numpy())
    return np.concatenate(shares),np.concatenate(scores)

def train_deep(kind,train,val,device,args,folder):
    torch.manual_seed(args.seed);torch.set_num_threads(args.threads)
    architecture={'architecture_version':3 if args.feature_version==3 else 2,'dropout':args.dropout,'model_width':args.width if kind=='tabm' else getattr(args,kind+'_width')}
    if args.feature_version==3:architecture.update({'n_features':1258} if kind=='tabm' else {'input_channels':6})
    model=make_model(kind,width=args.width,k=args.k,**architecture).to(device)
    model.feature_version=args.feature_version
    if args.optimizer=='adam':opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    elif args.optimizer=='nadam':opt=torch.optim.NAdam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    else:opt=torch.optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,momentum=args.momentum,nesterov=True)
    if args.scheduler=='plateau':scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=args.lr_patience,min_lr=args.min_lr)
    elif args.scheduler=='cosine':scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=args.min_lr)
    else:scheduler=None
    regularizer=None
    if args.contrastive_weight:
        if kind=='tabm':raise ValueError('Contrastive hidden representations are supported for cnn/fdy/itransformer only')
        from pipeline.contrastive import contrastive_regularizer
        regularizer=contrastive_regularizer
    tx=fixed_features(train['db'],version=args.feature_version) if kind=='tabm' else cnn_channels(train['db'],version=args.feature_version)
    vx=fixed_features(val['db'],version=args.feature_version) if kind=='tabm' else cnn_channels(val['db'],version=args.feature_version)
    y=torch.from_numpy(train['share']).to(device);z=torch.from_numpy(train['presence']).to(device)
    cell=torch.from_numpy(train['cell_share']).to(device) if kind!='tabm' else None
    best=float('inf');patience_best=float('inf');state=None;history=[];stale=0;best_epoch=0;stop_reason='max_epochs_reached';rng=np.random.default_rng(args.seed)
    for epoch in range(1,args.epochs+1):
        model.train();loss_sum=0.;contrastive_sum=0.;seen=0;t0=time.perf_counter();lr=opt.param_groups[0]['lr']
        for batch_index,ids in enumerate(np.array_split(rng.permutation(len(y)),max(1,int(np.ceil(len(y)/args.batch))))):
            opt.zero_grad(set_to_none=True)
            if kind=='tabm':
                logits=model(torch.from_numpy(tx[ids]).to(device))
                loss=F.mse_loss(logits[:,:,0].sigmoid(),y[ids,None].expand(-1,logits.shape[1]))
                loss=loss+.03*F.binary_cross_entropy_with_logits(logits[:,:,1],z[ids,None].expand(-1,logits.shape[1]))
            else:
                c,w=tx;tw=torch.from_numpy(w[ids]).to(device)
                p,logit,mask=model(torch.from_numpy(c[ids]).to(device),tw)
                loss=F.mse_loss(p,y[ids])+.03*F.binary_cross_entropy_with_logits(logit,z[ids])+.03*((mask-cell[ids]).square()*tw).sum((1,2)).mean()
            if regularizer is not None and len(ids)>1:
                extra=regularizer(model,kind,train['db'][ids],device,temperature=args.contrastive_temperature,seed=args.seed+epoch*1000003+batch_index)
                if extra.ndim!=0 or not torch.isfinite(extra):raise ValueError('Invalid contrastive scalar loss')
                loss=loss+args.contrastive_weight*extra;contrastive_sum+=float(extra.detach().cpu())*len(ids)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();loss_sum+=float(loss.detach().cpu())*len(ids);seen+=len(ids)
        train_pred,_=predict_deep(model,kind,train,device,batch=args.batch,inputs=tx)
        pred,_=predict_deep(model,kind,val,device,batch=args.batch,inputs=vx);value=rmse(val['share'],pred)
        if not np.isfinite(value):raise ValueError('Nonfinite validation RMSE')
        if value<best:
            best=value;best_epoch=epoch;state={name:v.detach().cpu().clone() for name,v in model.state_dict().items()}
        if value<patience_best-args.min_delta:patience_best=value;stale=0
        else:stale+=1
        if scheduler is not None:
            if args.scheduler=='plateau':scheduler.step(value)
            else:scheduler.step()
        history.append({'epoch':epoch,'epochs_run':epoch,'loss':loss_sum/seen,'contrastive_loss':contrastive_sum/seen,'train_rmse':rmse(train['share'],train_pred),'validation_rmse':value,'best_epoch':best_epoch,'best_validation_rmse':best,'epochs_without_improvement':stale,'learning_rate':lr,'next_learning_rate':opt.param_groups[0]['lr'],'seconds':time.perf_counter()-t0})
        print(kind,history[-1],flush=True)
        if stale>=args.patience:stop_reason='early_stopping_patience_reached';break
    if state is None:raise RuntimeError('No finite best checkpoint')
    model.load_state_dict(state);model.eval();restored,_=predict_deep(model,kind,val,device,batch=args.batch,inputs=vx);restored_rmse=rmse(val['share'],restored)
    if not np.isclose(restored_rmse,best,rtol=1e-5,atol=1e-7):raise RuntimeError('Restored best checkpoint did not reproduce its validation RMSE')
    summary={'max_epochs':args.epochs,'epochs_run':len(history),'best_epoch':best_epoch,'best_validation_rmse':best,'restored_validation_rmse':restored_rmse,'last_validation_rmse':history[-1]['validation_rmse'],'stop_reason':stop_reason,'early_stopping_patience':args.patience,'min_delta':args.min_delta,'optimizer':opt.__class__.__name__,'learning_rate':args.lr,'weight_decay':args.weight_decay,'momentum':args.momentum if args.optimizer=='sgd_nesterov' else None,'nesterov':args.optimizer=='sgd_nesterov','scheduler':args.scheduler,'dropout':args.dropout,'architecture':architecture,'contrastive_weight':args.contrastive_weight,'contrastive_temperature':args.contrastive_temperature,'contrastive_source_groups_available':False,'contrastive_limit':'Optional identity-projection ablation; source-level false negatives cannot all be excluded without batch source IDs.','selection_metric':'validation_rmse','best_checkpoint_restored':True,'convergence_claim':'Stopping by patience or epoch budget does not prove a global optimum or sufficient training.'}
    for h in history:h['stop_reason']=stop_reason if h['epoch']==len(history) else 'continued'
    summary.update({'feature_version':args.feature_version,'input_shape':[1258 if args.feature_version==3 else 1254] if kind=='tabm' else [6 if args.feature_version==3 else 4,15,25],
                    'loss_components':{'share_MSE':1.,'presence_BCE':.03,'energy_weighted_cell_MSE':0. if kind=='tabm' else .03},'spectrum_share_is_energy_weighted_sigmoid_cell_mask':kind!='tabm'})
    torch.save({'state_dict':state,'kind':kind,'width':args.width,'k':args.k,'seed':args.seed,'n_features':(1258 if args.feature_version==3 else 1254) if kind=='tabm' else None,
                'feature_version':args.feature_version,'preprocessing_version':args.feature_version,'input_shape':summary['input_shape'],
                'parameters':sum(p.numel() for p in model.parameters()),'architecture':architecture,'training_summary':summary},folder/f'{kind}.pt')
    (folder/f'{kind}_history.json').write_text(json.dumps(history,indent=2));(folder/f'{kind}_training_summary.json').write_text(json.dumps(summary,indent=2));model.training_summary_=summary
    return model

def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True,help='Verified prebuilt V2 package')
    p.add_argument('--tag',default='quality_v2');p.add_argument('--output-dir','--run-dir',dest='output_dir',type=Path,help='Fresh run directory; never overwritten')
    p.add_argument('--validate-only',action='store_true');p.add_argument('--cv-mode',action='store_true',help='Explicit group-CV fold package; train/validation only')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=7)
    p.add_argument('--optimizer',choices=['adam','nadam','sgd_nesterov'],default='adam')
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--weight-decay',type=float,default=.0003);p.add_argument('--momentum',type=float,default=.9);p.add_argument('--min-delta',type=float,default=0.)
    p.add_argument('--scheduler',choices=['plateau','cosine','none'],default='plateau');p.add_argument('--lr-patience',type=int,default=2);p.add_argument('--min-lr',type=float,default=1e-6)
    p.add_argument('--dropout',type=float,default=.1);p.add_argument('--contrastive-weight',type=float,default=0.);p.add_argument('--contrastive-temperature',type=float,default=.2)
    p.add_argument('--feature-version',type=int,choices=[1,3],default=1,help='Version 3: 6 spectral channels and 4 extra high-frequency features')
    p.add_argument('--models',nargs='+',choices=MODEL_KINDS,default=['lgbm','tabm','cnn']);p.add_argument('--device',default='cpu')
    p.add_argument('--batch',type=int,default=128);p.add_argument('--width',type=int,default=256,help='TabM hidden width only')
    p.add_argument('--cnn-width',type=int,default=32);p.add_argument('--fdy-width',type=int,default=32);p.add_argument('--itransformer-width',type=int,default=96)
    p.add_argument('--k',type=int,default=16);p.add_argument('--seed',type=int,default=42);p.add_argument('--trees',type=int,default=700);p.add_argument('--threads',type=int,default=6)
    return p

def validate_arguments(args):
    if not all(np.isfinite(getattr(args,k)) for k in ['lr','weight_decay','momentum','min_delta','min_lr','dropout','contrastive_weight','contrastive_temperature']):raise ValueError('Training hyperparameters must be finite')
    if not 1<=args.epochs<=100:raise ValueError('epochs must be between 1 and the user maximum of 100')
    if args.patience<1 or args.lr_patience<0 or min(args.batch,args.threads,args.k,args.trees)<1:raise ValueError('Invalid training budget')
    if not 0<=args.dropout<1 or args.contrastive_weight<0 or args.contrastive_temperature<=0:raise ValueError('Invalid dropout/contrastive setting')
    if args.lr<=0 or args.weight_decay<0 or args.min_delta<0 or not 0<=args.min_lr<=args.lr:raise ValueError('Invalid optimizer/scheduler setting')
    if not 0<args.momentum<1:raise ValueError('Nesterov momentum must be in (0,1)')
    if any(getattr(args,k)<=0 for k in ['width','cnn_width','fdy_width','itransformer_width']):raise ValueError('Widths must be positive')
    if any(getattr(args,k)%4 for k in ['cnn_width','fdy_width','itransformer_width']):raise ValueError('Conv/attention widths must be multiples of four')
    if len(set(args.models))!=len(args.models):raise ValueError('Duplicate model kinds')
    if args.contrastive_weight and 'tabm' in args.models:raise ValueError('TabM hidden contrastive representation is unsupported')
    if args.contrastive_weight and not set(args.models)&{'cnn','fdy','itransformer'}:raise ValueError('Contrastive loss requires at least one supported neural model')
    if Path(args.tag).name!=args.tag or args.tag in ['','.','..']:raise ValueError('tag must be a simple label')

def main(args):
    validate_arguments(args);data,gate,verification=load_prebuilt_data(args.data_dir,cv_mode=args.cv_mode)
    if 'physics' in str(gate.get('version','')) and args.feature_version!=3:raise ValueError('Physics V3 package requires --feature-version 3')
    run=f'{args.tag}_{gate["data_package_hash"][:8]}_seed{args.seed}';folder=args.output_dir.resolve() if args.output_dir else ROOT/'models'/run
    if folder.exists():raise FileExistsError('Run already exists; choose a new tag/output directory: '+str(folder))
    if args.validate_only:
        print(json.dumps({'status':'validated; no training executed and no run created','run_directory':str(folder),'verification':verification,'max_epochs':args.epochs,'patience':args.patience,'optimizer':args.optimizer,'scheduler':args.scheduler,'dropout':args.dropout,'cv_mode':args.cv_mode,'validation_protocol':gate.get('cv_protocol') if args.cv_mode else 'one frozen package holdout; seeds are not K-fold CV'},indent=2));return
    from pipeline.metrics import evaluate_regression
    torch.set_num_threads(args.threads);device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable')
    folder.mkdir(parents=True,exist_ok=False);t0=time.perf_counter()
    info={'run':folder.name,'arguments':vars(args),'data_folder':str(args.data_dir.resolve()),'data_package':verification,'external_manifest':{'source_hash':gate['source_hash'],'origin':'V2 readiness manifest'},'prebuilt_readiness_manifest':gate,'code_sha256_at_start':{p.name:sha256(p) for p in (ROOT/'pipeline').glob('*.py')},'validation_protocol':gate['cv_protocol'] if args.cv_mode else {'kind':'single_frozen_prebuilt_holdout','is_k_fold':False,'seeds_are_folds':False},'platform':platform.platform(),'test_accessed':False,'sealed_holdout_accessed':False,'status':'training_in_progress','training_summaries':{},'results':{}}
    info['feature_version']=args.feature_version
    def save_manifest():
        info['elapsed_seconds']=time.perf_counter()-t0;(folder/'run_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2,default=str))
    save_manifest()
    try:
        predictions={s:{} for s in data if s!='train'};baseline=float(data['train']['share'].mean())
        for split in predictions:info['results']['constant_'+split]={'rmse':rmse(data[split]['share'],np.full(len(data[split]['share']),baseline))}
        for kind in ['lgbm','lgbm_physics','lgbm_retrieval']:
            if kind not in args.models:continue
            subprocess.run([sys.executable,'-m','pipeline.train_lgbm',str(args.data_dir.resolve()),str(folder),'--trees',str(args.trees),'--threads',str(args.threads),'--seed',str(args.seed),'--feature-version',str(args.feature_version),'--data-package-hash',gate['data_package_hash'],'--variant',kind.removeprefix('lgbm_') if kind!='lgbm' else 'base']+(['--cv-mode'] if args.cv_mode else []),check=True,cwd=ROOT)
            info['training_summaries'][kind]=json.loads((folder/(kind+'_training_summary.json')).read_text())
            for split in predictions:
                with np.load(folder/f'{kind}_{split}.npz') as p:predictions[split][kind]=(p['share'],p['score'])
        for kind in ['tabm','cnn','fdy','itransformer']:
            if kind not in args.models:continue
            model=train_deep(kind,data['train'],data['validation'],device,args,folder);info['training_summaries'][kind]=model.training_summary_
            for split in predictions:predictions[split][kind]=predict_deep(model,kind,data[split],device,batch=args.batch)
            del model;save_manifest()
        for split,members in predictions.items():
            out={'share_true':data[split]['share'],'presence_true':data[split]['presence']}
            for kind,(p,s) in members.items():
                metrics=evaluate_regression(data[split]['share'],p,presence=data[split]['presence'],scores=s,db=data[split]['db']);metrics['rmse']=rmse(data[split]['share'],p);metrics['score_1_minus_rmse']=1-metrics['rmse']
                metrics['addition_auc']=float(roc_auc_score(data[split]['presence'],s)) if len(np.unique(data[split]['presence']))>1 else None
                info['results'][kind+'_'+split]=metrics;out[kind+'_share']=p;out[kind+'_score']=s
            np.savez_compressed(folder/f'{split}_predictions.npz',**out)
        if sha256(args.data_dir/'readiness_manifest.json')!=verification['readiness_manifest_sha256']:raise RuntimeError('Readiness changed during training')
        for split,c in verification['splits'].items():
            if sha256(args.data_dir/(split+'.npz'))!=c['npz_sha256']:raise RuntimeError('Input arrays changed during training')
        info['status']='completed frozen-prebuilt experiment; not an official leaderboard result';save_manifest()
    except Exception as exc:
        info['status']='failed; partial artifacts must not be used';info['error']={'type':type(exc).__name__,'message':str(exc)};save_manifest();raise
    print(json.dumps({'run':str(folder),'status':info['status'],'elapsed_seconds':info['elapsed_seconds']},indent=2),flush=True)

if __name__=='__main__':main(build_parser().parse_args())
