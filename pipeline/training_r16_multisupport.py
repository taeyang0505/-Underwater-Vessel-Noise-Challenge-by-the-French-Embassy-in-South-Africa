"""Full paired-regime learners. Fixed100/Adam, separate regression/classifier.

Only TRAIN and the predeclared local_fit are read during optimization.
Features are stateless FP64 CPU calculations cached once as FP32 mmap.
"""
from pathlib import Path
import gc,json,os,random,signal,time
import numpy as np
from pipeline.r16_multisupport_data import ROOT,OUT,REGIMES,KINDS,PARENT,read,write,sha,save,role,guard

MODELS=ROOT/'models/r16_multisupport_full_r1'
CACHE=ROOT/'data/cache/r16_multisupport_full_r1'
SEED=2026102817
CODE_NAMES=('pipeline/r16_multisupport_data.py','pipeline/training_r16_multisupport.py',
 'pipeline/models_r16_multisupport.py','pipeline/models_r16_f1_preflight.py','pipeline/models_r9.py',
 'pipeline/models_r16_production.py','pipeline/models.py','pipeline/grid_models_r5.py','pipeline/hybrid_models_r5.py',
 'pipeline/trees_r16_production.py','pipeline/physics_features_r6_v2.py','pipeline/relative_grid_r6.py',
 'pipeline/relative_grid_r6_v2.py','ops/r16_temporal_features.py')
def codes():return{n:sha(ROOT/n)for n in CODE_NAMES}
def verify(pins):
 for n,h in pins.items():
  if sha(ROOT/n)!=h:raise ValueError('Frozen source changed '+n)
def features(db):
 from pipeline.relative_grid_r6_v2 import relative_grid_v2
 from ops.r16_temporal_features import temporal_grid
 x,w=relative_grid_v2(db);f,_=temporal_grid(db)
 return np.concatenate([x,f],1),w
def tree_features(db,kind):
 from pipeline.trees_r16_production import transform
 from ops.r16_temporal_features import temporal_statistics
 x=transform(db,kind);extra,_=temporal_statistics(db);return np.concatenate([x,extra],1).astype(np.float32)
def metric(y,p,c,weights=None):
 y=np.asarray(y,float);p=np.asarray(p,float);c=np.asarray(c,float)
 w=np.ones(len(y))/len(y)if weights is None else np.asarray(weights)/np.sum(weights)
 z=y>0
 return {'n':len(y),'rmse':float(np.sqrt(w@((p-y)**2))),'bias':float(w@(p-y)),
  'prediction_mean':float(w@p),'prediction_rms':float(np.sqrt(w@(p*p))),
  'share_auc':weighted_auc(z,p,w),
  'classifier_auc':weighted_auc(z,c,w),
  'classifier_mean':float(w@c),'classifier_ge099':float(w@(c>=.99))}
def weighted_auc(z,p,w):
 z=np.asarray(z,bool);p=np.asarray(p,float);w=np.asarray(w,float)
 if not z.any()or z.all():return None
 order=np.argsort(p,kind='stable');pp=p[order];zz=z[order];ww=w[order]
 lo=np.r_[0,np.flatnonzero(pp[1:]!=pp[:-1])+1]
 positive=np.add.reduceat(ww*zz,lo);negative=np.add.reduceat(ww*~zz,lo)
 return float(np.sum(positive*(np.cumsum(negative)-.5*negative))/(positive.sum()*negative.sum()))
def train_trees():
 import lightgbm as lgb
 import sys
 sys.addaudithook(guard);pins=codes();manifest=OUT/'manifest.json';datahash=sha(manifest)
 selection=role(0,'D0','selection');folders={};start=time.perf_counter()
 for f in range(5):
  for regime in REGIMES:
   folder=MODELS/'trees'/regime/f'fold_{f:02d}'
   if folder.exists():raise FileExistsError('Fresh Tree run only; no partial reuse')
   folder.mkdir(parents=True);train=role(f,regime,'train');pred={};history={};iterations={}
   write(folder/'PLAN.json',{'status':'fixed_before_fit','fold':f,'regime':regime,'maximum_trees':1500,'patience':40,'CPU_threads':2,'train_n':len(train['share']),'selection_n':len(selection['share']),'code_sha256':pins,'data_manifest_sha256':datahash,'Test_used':False})
   for kind in('regression','classification'):
    width=5160 if kind=='regression'else 2033;tmp=folder/'temporary_train_features.npy';x=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(train['share']),width))
    for off in range(0,len(x),256):x[off:off+256]=tree_features(train['db'][off:off+256],kind)
    x.flush();v=tree_features(selection['db'],kind)
    args=dict(n_estimators=1500,learning_rate=.035,num_leaves=31,min_child_samples=50,max_bin=127,n_jobs=2,
      random_state=SEED+f,verbosity=-1,colsample_bytree=.85,subsample=.85,subsample_freq=1,deterministic=True,force_col_wise=True)
    clf=kind=='classification';model=lgb.LGBMClassifier(objective='binary',metric='binary_logloss',**args)if clf else lgb.LGBMRegressor(objective='regression_l2',metric='None',**args)
    key='presence'if clf else'share';hist={}
    def callback(env):
     if env.iteration==0 or(env.iteration+1)%25==0:
      progress={'stage':'trees','fold':f,'regime':regime,'kind':kind,'round':env.iteration+1,'elapsed_seconds':time.perf_counter()-start}
      write(MODELS/'progress.json',progress);print(json.dumps(progress),flush=True)
    callback.order=30
    model.fit(x,train[key],eval_set=[(v,selection[key])],eval_metric='binary_logloss'if clf else lambda y,p:('clipped_rmse',float(np.sqrt(np.mean((np.clip(p,0,1)-y)**2))),False),
      callbacks=[lgb.early_stopping(40,verbose=False),lgb.record_evaluation(hist),callback])
    booster=model.booster_;booster.save_model(str(folder/(kind+'.txt')));p=booster.predict(v,num_threads=2);pred[kind]=p if clf else np.clip(p,0,1)
    history[kind]=hist;iterations[kind]=int(model.best_iteration_);del model,booster,x,v;gc.collect();tmp.unlink()
   save(folder/'selection.npz',{'share':pred['regression'],'classifier_prob':pred['classification'],'target_share':selection['share'],'sample_id':selection['sample_id']})
   write(folder/'history.json',history);verify(pins)
   summary={'status':'completed','fold':f,'regime':regime,'iterations':iterations,'train_n':len(train['share']),
    'feature_counts':{'regression':5160,'classification':2033},'selection_metrics':metric(selection['share'],pred['regression'],pred['classification']),
    'data_manifest_sha256':datahash,'code_sha256':pins,'artifact_sha256':{p.name:sha(p)for p in folder.iterdir()if p.is_file()},'Test_used':False}
   write(folder/'summary.json',summary);folders[f'{regime}_{f}']=str(folder.relative_to(ROOT));del train;gc.collect()
 write(MODELS/'ALL_TREES_COMPLETED.json',{'status':'completed','models':folders,'code_sha256':pins,'data_manifest_sha256':datahash,'elapsed_seconds':time.perf_counter()-start})
def cache_role(regime,name):
 folder=CACHE/(regime if name=='train'else 'shared_selection')/name
 if folder.exists():
  m=read(folder/'manifest.json')
  if m['code_sha256']!=codes()or m['data_manifest_sha256']!=sha(OUT/'manifest.json'):raise ValueError('Cache binding changed')
  for n,h in m['artifact_sha256'].items():
   if sha(folder/n)!=h:raise ValueError('Cache artifact changed')
  return folder
 d=role(0,regime,name);folder.mkdir(parents=True);n=len(d['share']);x=np.lib.format.open_memmap(folder/'x.npy',mode='w+',dtype=np.float32,shape=(n,17,15,25));w=np.lib.format.open_memmap(folder/'w.npy',mode='w+',dtype=np.float32,shape=(n,15,25))
 for off in range(0,n,256):x[off:off+256],w[off:off+256]=features(d['db'][off:off+256])
 if np.max(abs((d['cell_share']*w).sum((1,2))-d['share']))>.0058:raise ValueError('Raw target versus rounded weight bound differs')
 x.flush();w.flush();del x,w
 for k in('share','presence','cell_share','sample_id'):np.save(folder/(k+'.npy'),d[k],allow_pickle=False)
 write(folder/'manifest.json',{'status':'completed','n':n,'code_sha256':codes(),'data_manifest_sha256':sha(OUT/'manifest.json'),'artifact_sha256':{p.name:sha(p)for p in folder.glob('*.npy')}})
 return folder
def cache_read(folder):return{p.stem:np.load(p,mmap_mode='r',allow_pickle=False)for p in Path(folder).glob('*.npy')}
def prepare_cache():
 import sys
 sys.addaudithook(guard)
 for regime in REGIMES:
  for name in('train','selection'):cache_role(regime,name)
def atomic_pt(p,value):
 from pipeline.training_r16_production import cpu_state
 import torch
 p=Path(p);t=p.with_name(p.name+'.tmp');torch.save(cpu_state(value),t);t.replace(p)
def predict_model(model,x,w,device,batch):
 import torch
 model.eval();pp=[];cc=[]
 with torch.no_grad():
  for off in range(0,len(x),batch):
   xx=torch.tensor(np.asarray(x[off:off+batch]),device=device);ww=torch.tensor(np.asarray(w[off:off+batch]),device=device)
   p,c,q=model(xx,ww);pp.append(p.cpu().numpy());cc.append(c.sigmoid().cpu().numpy())
 return np.concatenate(pp),np.concatenate(cc)
def preflight():
 import torch
 from pipeline.models_r16_multisupport import MultiSupportModel,decoupled_loss
 torch.set_num_threads(2)
 if not torch.backends.mps.is_available():raise RuntimeError('MPS required; no silent fallback')
 tr=cache_read(cache_role('H0','train'));result={}
 for kind in KINDS:
  device='cpu'if kind=='itransformer'else'mps';batch=128 if kind in('fdy','itransformer')else 64
  torch.manual_seed(SEED);m=MultiSupportModel(kind).to(device);opt=torch.optim.Adam(m.parameters(),lr=.001);times=[]
  for j in range(3):
   opt.zero_grad(set_to_none=True);sl=slice(j*batch,(j+1)*batch);x=torch.tensor(np.asarray(tr['x'][sl]),device=device);w=torch.tensor(np.asarray(tr['w'][sl]),device=device)
   target=tuple(torch.tensor(np.asarray(tr[k][sl]),dtype=torch.float32,device=device)for k in('share','presence','cell_share'))
   if device=='mps':torch.mps.synchronize()
   start=time.perf_counter();loss,_=decoupled_loss(m(x,w),target,w);loss.backward();opt.step()
   if device=='mps':torch.mps.synchronize()
   times.append(time.perf_counter()-start)
  result[kind]={'device':device,'batch':batch,'step_seconds_after_warmup':float(np.median(times[1:])),
   'driver_bytes':int(torch.mps.driver_allocated_memory())if device=='mps'else None,'architecture':m.architecture()}
  print(json.dumps({'stage':'actual_preflight','kind':kind,**result[kind]}),flush=True);del m,opt;gc.collect()
  if device=='mps':torch.mps.empty_cache()
 write(MODELS/'ACTUAL_PREFLIGHT.json',{'status':'passed','models':result,'disposable_steps_only':True,'code_sha256':codes(),'data_manifest_sha256':sha(OUT/'manifest.json')})
def train_neural():
 import torch,sys
 from pipeline.models_r16_multisupport import MultiSupportModel,decoupled_loss,clip_decoupled_gradients
 from pipeline.training_r16_production import rng_state
 sys.addaudithook(guard);torch.set_num_threads(2);pf=read(MODELS/'ACTUAL_PREFLIGHT.json');pins=codes();datahash=sha(OUT/'manifest.json')
 if pf['code_sha256']!=pins or pf['data_manifest_sha256']!=datahash:raise ValueError('Preflight identity changed')
 va=cache_read(cache_role('D0','selection'));pause={'requested':False};signal.signal(signal.SIGUSR1,lambda *_:pause.update(requested=True));began=time.perf_counter()
 for kind in KINDS:
  for regime in REGIMES:
   folder=MODELS/'neural'/regime/kind
   if folder.exists():raise FileExistsError('Fresh NN only; no automatic partial reuse')
   folder.mkdir(parents=True);tr=cache_read(cache_role(regime,'train'));device=pf['models'][kind]['device'];batch=pf['models'][kind]['batch']
   torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);model=MultiSupportModel(kind).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
   scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=1e-5);best=float('inf');bc=-1.;be=bce=steps=0
   write(folder/'PLAN.json',{'epochs':100,'Adam':True,'early_stopping':False,'regime':regime,'kind':kind,'batch':batch,'device':device,'code_sha256':pins,'data_manifest_sha256':datahash,'architecture':model.architecture()})
   for epoch in range(1,101):
    verify(pins);order=np.random.default_rng([SEED,epoch]).permutation(len(tr['share']));sse=cell=clf=0.;start=time.perf_counter();seen=0;model.train()
    for off in range(0,len(order),batch):
     ix=order[off:off+batch];x=torch.tensor(np.asarray(tr['x'][ix]),device=device);w=torch.tensor(np.asarray(tr['w'][ix]),device=device)
     target=tuple(torch.tensor(np.asarray(tr[k][ix]),dtype=torch.float32,device=device)for k in('share','presence','cell_share'))
     opt.zero_grad(set_to_none=True);pred=model(x,w);loss,parts=decoupled_loss(pred,target,w);loss.backward();clip_decoupled_gradients(model);opt.step();steps+=1;seen+=len(ix)
     sse+=float(parts['share_mse'].detach().sum().cpu());cell+=float(parts['weighted_cell_mse'].detach().sum().cpu());clf+=float(parts['classification_bce'].detach().sum().cpu())
     if off==0 or steps%100==0:write(MODELS/'progress.json',{'stage':'neural','regime':regime,'kind':kind,'epoch':epoch,'rows_seen':seen,'total_rows':len(order),'optimizer_steps':steps,'elapsed_seconds':time.perf_counter()-began})
    p,c=predict_model(model,va['x'],va['w'],device,batch);metrics=metric(va['share'],p,c);ri=metrics['rmse']<best;ci=metrics['classifier_auc']>bc
    if ri:best=metrics['rmse'];be=epoch
    if ci:bc=metrics['classifier_auc'];bce=epoch
    lr=opt.param_groups[0]['lr'];scheduler.step()
    checkpoint={'format':'r16_multisupport_full_checkpoint_v1','state_dict':model.state_dict(),'kind':kind,'regime':regime,'epoch':epoch,
     'architecture':model.architecture(),'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'rng':rng_state(device),
     'best_regression_epoch':be,'best_regression_rmse':best,'best_classifier_epoch':bce,'best_classifier_auc':bc,'optimizer_steps':steps,
     'code_sha256':pins,'data_manifest_sha256':datahash}
    if ri:atomic_pt(folder/'best_regression.pt',checkpoint);save(folder/'selection_regression.npz',{'share':p,'sample_id':va['sample_id'],'target_share':va['share']})
    if ci:atomic_pt(folder/'best_classifier.pt',checkpoint);save(folder/'selection_classification.npz',{'classifier_prob':c,'sample_id':va['sample_id'],'target_share':va['share']})
    atomic_pt(folder/'latest.pt',checkpoint)
    row={'regime':regime,'kind':kind,'epoch':epoch,'train_rmse':float(np.sqrt(sse/seen)),'train_cell_mse':cell/seen,'train_classifier_BCE':clf/seen,
     'selection':metrics,'best_RMSE':best,'best_regression_epoch':be,'best_classifier_AUC':bc,'best_classifier_epoch':bce,'learning_rate':lr,
     'epoch_seconds':time.perf_counter()-start,'driver_bytes':int(torch.mps.driver_allocated_memory())if device=='mps'else None}
    with(folder/'epochs.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    write(folder/'progress.json',row);print(json.dumps(row),flush=True)
    if pause['requested']or(MODELS/'PAUSE_AFTER_EPOCH').exists():write(MODELS/'PAUSED.json',{'kind':kind,'regime':regime,'epoch':epoch,'latest_sha256':sha(folder/'latest.pt'),'explicit_resume_required':True});return False
   verify(pins);write(folder/'summary.json',{'status':'completed','epochs':100,'early_stopping':False,'regime':regime,'kind':kind,'best_regression_epoch':be,'best_rmse':best,
    'best_classifier_epoch':bce,'best_classifier_auc':bc,'code_sha256':pins,'data_manifest_sha256':datahash,'artifact_sha256':{p.name:sha(p)for p in folder.iterdir()if p.is_file()}})
   del tr,model,opt,scheduler;gc.collect()
   if device=='mps':torch.mps.empty_cache()
 write(MODELS/'ALL_NEURAL_COMPLETED.json',{'status':'completed','regimes':list(REGIMES),'kinds':list(KINDS),'epochs_each':100,'code_sha256':pins,'data_manifest_sha256':datahash,'elapsed_seconds':time.perf_counter()-began});return True
