"""New small 2x2 support/temporal ablation learners, separate from frozen R16.

Feature0 reproduces the old 8-channel small mask CNN / compact466 and
independent temporal700 Tree. Feature1 appends stateless ordered features.
NN regression/classification parameters and checkpoint selection are separate.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,os,sys,tempfile,time,multiprocessing,resource
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from torch import nn
import lightgbm as lgb
from pipeline.relative_grid_r6_v2 import relative_grid_v2
from pipeline.relative_models_r6_v2 import TinyMaskCNNv2
from pipeline.features_r12 import compact_relative_features
from ops.r16_clean_models import temporal_features,metrics,SmallDecoupledCNN
from ops.r16_temporal_features import temporal_grid,temporal_statistics,GRID_NAMES,fixture as feature_fixture

ROOT=Path(__file__).resolve().parents[1];SEED=2026102817
FORBIDDEN_PANEL_PATHS=set()
OPERATORS=('control_eq6','mfhf_shelf6','mfhf_shelf12','mfhf_shelf12_lfminus6','alpha_preEQ_inserted','independent_subband_snr')
SOURCE_NAMES=('ops/r16_support_temporal_models.py','ops/r16_temporal_features.py','ops/r16_clean_models.py',
 'pipeline/features_r12.py','pipeline/relative_grid_r6_v2.py','pipeline/relative_grid_r6.py','pipeline/relative_models_r6_v2.py')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def codes():return{n:sha(ROOT/n)for n in SOURCE_NAMES}
def verify(pins):
 for n,h in pins.items():
  if sha(ROOT/n)!=h:raise ValueError('Frozen pilot source changed: '+n)
def write(path,d):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def save_npz(path,**arrays):
 path=Path(path);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb')as f:np.savez_compressed(f,**arrays)
 tmp.replace(path)
def save_pt(path,d):
 path=Path(path);tmp=path.with_name(path.name+'.tmp');torch.save(d,tmp);tmp.replace(path)
def guard(event,args):
 if event=='open'and args and isinstance(args[0],(str,bytes,os.PathLike)):
  name=os.fsdecode(args[0]).lower().replace('\\','/')
  if Path(name).name in('test.csv','sample_submission.csv')or'/eda_deep/'in name or'/r15_band_comparison_'in name:
   raise PermissionError('No official query or query-derived EDA in small ablation')
  if os.path.realpath(os.fsdecode(args[0]))in FORBIDDEN_PANEL_PATHS:
   raise PermissionError('Locked gate/report/domain/long panel is forbidden during training')

def configure_training_access(manifest):
 """Read metadata only, then deny every declared non-TRAIN/non-fit panel."""
 mp=Path(manifest).resolve();m=json.loads(mp.read_text());blocked=set()
 for role,entries in m.get('panels',{}).items():
  if role=='local_fit':continue
  for e in entries.values():
   blocked.add(os.path.realpath((ROOT if e.get('scope')=='project'else mp.parent)/e['path']))
 for e in m.get('long_validation',{}).values():
  blocked.add(os.path.realpath((ROOT if e.get('scope')=='project'else mp.parent)/e['path']))
 FORBIDDEN_PANEL_PATHS.update(blocked)
 return len(blocked)

def inputs(db,feature_mode):
 if feature_mode not in(0,1):raise ValueError('Feature mode must be 0 or 1')
 x,w=relative_grid_v2(np.asarray(db))
 if feature_mode:
  extra,_=temporal_grid(db);x=np.concatenate([x,extra],1)
 if not np.isfinite(x).all()or not np.isfinite(w).all():raise ValueError('Invalid neural features')
 return x,w

def tree_features(db,feature_mode):
 if feature_mode not in(0,1):raise ValueError('Feature mode must be 0 or 1')
 reg=compact_relative_features(db,include_shape=False);clf=temporal_features(db)
 names=[]
 if feature_mode:
  extra,names=temporal_statistics(db);reg=np.concatenate([reg,extra],1);clf=np.concatenate([clf,extra],1)
 if not np.isfinite(reg).all()or not np.isfinite(clf).all():raise ValueError('Invalid Tree features')
 return reg.astype(np.float32),clf.astype(np.float32),names

class TemporalDecoupledCNN(nn.Module):
 def __init__(self,feature_mode):
  super().__init__()
  if feature_mode not in(0,1):raise ValueError('Feature mode must be 0 or 1')
  self.feature_mode=int(feature_mode);extra=len(GRID_NAMES)if feature_mode else 0
  self.reg=TinyMaskCNNv2(input_channels=8+extra,width=24,blocks=2,dropout=.1,global_context=True)
  self.clf=nn.Sequential(nn.Conv2d(6+extra,16,3,padding=1),nn.GELU(),nn.Dropout2d(.1),nn.Conv2d(16,16,3,padding=1),nn.GELU())
  self.clf_head=nn.Sequential(nn.Linear(32,16),nn.GELU(),nn.Dropout(.1),nn.Linear(16,1))
 def forward(self,x,w):
  h=self.reg.body(x);pooled=torch.cat([h[:,:,:,a:b].mean(3)for a,b in[(10,20),(20,23),(15,23)]],1)
  context=torch.nn.functional.gelu(self.reg.context_projection(pooled))*torch.sigmoid(self.reg.context_gate(pooled))
  q=self.reg.mask_head(h+context.unsqueeze(3))[:,0].sigmoid();obs=w>0;q=torch.where(obs,q,torch.zeros_like(q));share=(q*w).sum((1,2))
  channels=[0,1,2,3,6,5]+list(range(8,x.shape[1]));h=self.clf(x[:,channels]);om=obs[:,None].to(h.dtype);den=om.sum((2,3)).clamp_min(1)
  avg=(h*om).sum((2,3))/den;mx=h.masked_fill(~obs[:,None],-torch.inf).amax((2,3));logit=self.clf_head(torch.cat([avg,mx],1))[:,0]
  return share,q,logit
 def parameter_counts(self):
  reg=sum(p.numel()for p in self.reg.parameters());cl=sum(p.numel()for n,p in self.named_parameters()if not n.startswith('reg.'))
  return{'regression':reg,'classification':cl,'total':reg+cl}

def make_model(feature_mode,seed=SEED):
 """Both F arms start with identical common weights; extra inputs start at0."""
 torch.manual_seed(seed);base=TemporalDecoupledCNN(0)
 if not feature_mode:return base
 torch.manual_seed(seed);model=TemporalDecoupledCNN(1);old=base.state_dict();state=model.state_dict()
 for name,new in state.items():
  prior=old[name]
  if new.shape==prior.shape:new.copy_(prior)
  elif new.ndim==4 and new.shape[0]==prior.shape[0]and new.shape[2:]==prior.shape[2:]and new.shape[1]>prior.shape[1]:
   new.zero_();new[:,:prior.shape[1]].copy_(prior)
  else:raise ValueError('Unexpected feature-arm architecture mismatch: '+name)
 model.load_state_dict(state);return model

@torch.no_grad()
def nn_predict(model,x,w,batch=256):
 model.eval();p=[];c=[]
 for off in range(0,len(x),batch):
  pp,_,cc=model(torch.from_numpy(x[off:off+batch]).contiguous(memory_format=torch.channels_last),torch.from_numpy(w[off:off+batch]));p.append(pp.numpy());c.append(cc.sigmoid().numpy())
 return np.concatenate(p),np.concatenate(c)

def _labels(d):
 y=np.asarray(d['share'],np.float32);q=np.asarray(d['cell_share'],np.float32);z=np.asarray(d['presence'],np.int64)
 if y.ndim!=1 or q.shape!=(len(y),15,25)or not np.isfinite(y).all()or not np.isfinite(q).all()or((y<0)|(y>1)).any()or((q<0)|(q>1)).any()or not np.array_equal(z,y>0):raise ValueError('Physical labels invalid')
 return y,q,z

def train_arm(train,fit,output,arm,feature_mode,source_binding,*,epochs=100,fixture_only=False):
 """Parent supplies source-verified train/fit dictionaries, never locked panels."""
 if not fixture_only and(epochs!=100 or len(train['share'])>6000):raise ValueError('Predeclared bounded pilot requires100epoch and <=6000 TRAIN')
 if source_binding.get('Test_used')is not False or source_binding.get('selection_role')!='local_fit'or source_binding.get('locked_roles_used_for_fit')is not False:raise ValueError('Explicit train/local_fit-only binding required')
 if 'base_window_index'in train and'base_window_index'in fit and set(train['base_window_index'])&set(fit['base_window_index']):raise ValueError('Train/fit background IDs overlap')
 if 'reference_component'in train and'reference_component'in fit:
  refs=lambda d:{str(v)for v in d['reference_component']if str(v)not in('-1','None','')}
  if refs(train)&refs(fit):raise ValueError('Train/fit reference components overlap')
 sys.addaudithook(guard);torch.set_num_threads(2);torch.manual_seed(SEED);np.random.seed(SEED)
 folder=Path(output)/arm
 if folder.exists():raise FileExistsError('Fresh arm only; no partial reuse')
 folder.mkdir(parents=True);began=time.perf_counter();pins=codes()
 tx,tw=inputs(train['db'],feature_mode);vx,vw=inputs(fit['db'],feature_mode);y,q,z=_labels(train);vy,_,vz=_labels(fit)
 if np.max(abs((q*tw).sum((1,2))-y))>.0058:raise ValueError('Rounded physical labels exceed known bound')
 plan={'status':'frozen_before_fit','arm':arm,'feature_mode':int(feature_mode),'epochs':epochs,'NN_early_stopping':False,'optimizer':'Adam',
  'dropout':.1,'seed':SEED,'source_binding':source_binding,'source_code_sha256':pins,'train_rows':len(y),'fit_rows':len(vy),
  'new_channels':list(GRID_NAMES)if feature_mode else[],'regression_loss':'Share MSE + .25 energyweighted cell MSE',
  'classifier':'separate parameters and best checkpoint; temporal only; no sqrtw/profile/share','fixture_only':fixture_only,'Test_used':False}
 write(folder/'PLAN.json',plan)
 try:
  r,c,names=tree_features(train['db'],feature_mode);vr,vc,_=tree_features(fit['db'],feature_mode)
  common=dict(n_estimators=25 if fixture_only else 600,learning_rate=.04,num_leaves=23,max_depth=-1,min_child_samples=8 if fixture_only else 30,
   reg_lambda=2.,feature_fraction=.85,n_jobs=2,verbosity=-1,random_state=SEED,deterministic=True,force_col_wise=True)
  reg=lgb.LGBMRegressor(objective='regression',**common);classifier=lgb.LGBMClassifier(objective='binary',**common)
  reg.fit(r,y,eval_set=[(vr,vy)],eval_metric='rmse',callbacks=[lgb.early_stopping(5 if fixture_only else 40,verbose=False)])
  classifier.fit(c,z,eval_set=[(vc,vz)],eval_metric='binary_logloss',callbacks=[lgb.early_stopping(5 if fixture_only else 40,verbose=False)])
  reg.booster_.save_model(str(folder/'tree_regressor.txt'));classifier.booster_.save_model(str(folder/'tree_classifier.txt'))
  tp=np.clip(reg.predict(vr),0,1);tc=classifier.predict_proba(vc)[:,1]
  save_npz(folder/'fit_tree_predictions.npz',share=tp,classifier_prob=tc,target_share=vy)
  write(folder/'TREE_COMPLETED.json',{'status':'completed','feature_mode':feature_mode,'regression_features':r.shape[1],'classification_features':c.shape[1],
   'additional_feature_names':names,'best_iteration_regression':reg.best_iteration_,'best_iteration_classifier':classifier.best_iteration_,
   'fit_metrics':metrics(vy,tp,tc),'fixture_only':fixture_only})
  del r,c,vr,vc,reg,classifier;gc.collect()
  model=make_model(feature_mode).to(memory_format=torch.channels_last);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
  scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-5)
  xt=torch.from_numpy(tx);wt=torch.from_numpy(tw);yt=torch.from_numpy(y);qt=torch.from_numpy(q);zt=torch.from_numpy(z.astype(np.float32))
  best=float('inf');best_c=-1.;best_epoch=best_c_epoch=0
  for epoch in range(1,epochs+1):
   verify(pins);model.train();ixs=np.random.default_rng(np.random.SeedSequence([SEED,epoch])).permutation(len(y));sse=0.;seen=0
   for off in range(0,len(ixs),128):
    ix=ixs[off:off+128];opt.zero_grad(set_to_none=True);pp,qq,cc=model(xt[ix].contiguous(memory_format=torch.channels_last),wt[ix]);mse=(pp-yt[ix]).square()
    regloss=mse+.25*((qq-qt[ix]).square()*wt[ix]).sum((1,2));closs=nn.functional.binary_cross_entropy_with_logits(cc,zt[ix],reduction='none');loss=regloss.mean()+closs.mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pilot loss')
    loss.backward();nn.utils.clip_grad_norm_(model.reg.parameters(),1.,error_if_nonfinite=True);nn.utils.clip_grad_norm_(list(model.clf.parameters())+list(model.clf_head.parameters()),1.,error_if_nonfinite=True);opt.step()
    sse+=float(mse.detach().sum());seen+=len(ix)
   pp,cc=nn_predict(model,vx,vw);mm=metrics(vy,pp,cc)
   ck={'state_dict':model.state_dict(),'epoch':epoch,'arm':arm,'feature_mode':feature_mode,'source_code_sha256':pins,'fixture_only':fixture_only}
   if mm['rmse']<best:best=mm['rmse'];best_epoch=epoch;save_pt(folder/'best_regression.pt',ck)
   if mm['auc_classifier']>best_c:best_c=mm['auc_classifier'];best_c_epoch=epoch;save_pt(folder/'best_classifier.pt',ck)
   row={'epoch':epoch,'train_online_rmse':float(np.sqrt(sse/seen)),'fit_rmse':mm['rmse'],'fit_classifier_auc':mm['auc_classifier'],
    'best_regression_rmse':best,'best_regression_epoch':best_epoch,'best_classifier_auc':best_c,'best_classifier_epoch':best_c_epoch,'elapsed_seconds':time.perf_counter()-began}
   with(folder/'epochs.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
   write(folder/'progress.json',row);scheduler.step()
   if epoch==1 or epoch%10==0:print(json.dumps({'event':'epoch','arm':arm,**row}),flush=True)
  model.load_state_dict(torch.load(folder/'best_regression.pt',map_location='cpu',weights_only=True)['state_dict']);pp,_=nn_predict(model,vx,vw)
  model.load_state_dict(torch.load(folder/'best_classifier.pt',map_location='cpu',weights_only=True)['state_dict']);_,cc=nn_predict(model,vx,vw)
  save_npz(folder/'fit_nn_predictions.npz',share=pp,classifier_prob=cc,target_share=vy)
  verify(pins)
  result={'status':'completed','arm':arm,'feature_mode':feature_mode,'epochs':epochs,'NN_early_stopping':False,'optimizer':'Adam','parameter_counts':model.parameter_counts(),
   'best_regression_epoch':best_epoch,'best_classifier_epoch':best_c_epoch,'fit_nn_metrics':metrics(vy,pp,cc),'fit_tree_metrics':metrics(vy,tp,tc),
   'fit_fixed60tree40nn_metrics':metrics(vy,.6*tp+.4*pp,.5*tc+.5*cc),'elapsed_seconds':time.perf_counter()-began,
   'source_code_sha256':pins,'source_binding':source_binding,'fixture_only':fixture_only,'Test_used':False,
   'artifacts_sha256':{p.name:sha(p)for p in folder.iterdir()if p.is_file()}}
  write(folder/'COMPLETED.json',result);return result
 except BaseException as e:write(folder/'FAILED.json',{'error':repr(e),'no_partial_reuse':True});raise

def predict_arm(folder,panel,*,allow_fixture=False):
 folder=Path(folder);s=json.loads((folder/'COMPLETED.json').read_text())
 if s['status']!='completed'or(folder/'FAILED.json').exists()or(s.get('fixture_only')and not allow_fixture):raise ValueError('Complete production pilot arm required')
 verify(s['source_code_sha256'])
 for n,h in s['artifacts_sha256'].items():
  if sha(folder/n)!=h:raise ValueError('Saved pilot artifact changed')
 db=panel['db']if isinstance(panel,dict)else panel
 feature_mode=s['feature_mode'];x,w=inputs(db,feature_mode);model=make_model(feature_mode).to(memory_format=torch.channels_last)
 model.load_state_dict(torch.load(folder/'best_regression.pt',map_location='cpu',weights_only=True)['state_dict']);p,_=nn_predict(model,x,w)
 model.load_state_dict(torch.load(folder/'best_classifier.pt',map_location='cpu',weights_only=True)['state_dict']);_,c=nn_predict(model,x,w)
 reg=lgb.Booster(model_file=str(folder/'tree_regressor.txt'));cl=lgb.Booster(model_file=str(folder/'tree_classifier.txt'));r,v,_=tree_features(db,feature_mode)
 tp=np.clip(reg.predict(r,num_threads=2),0,1);tc=cl.predict(v,num_threads=2)
 return{'nn_share':p,'nn_classifier':c,'tree_share':tp,'tree_classifier':tc,'balanced_share':.6*tp+.4*p,'balanced_classifier':.5*tc+.5*c}

def concatenate(panels):
 keys=('db','share','cell_share','presence','site','sample_id','base_window_index','reference_component')
 return{k:np.concatenate([d[k]for d in panels])for k in keys if all(k in d for d in panels)}

def _train_worker(manifest,output,data_arm,feature_mode,source,data_sha):
 """Spawn-safe isolated CPU worker; opens only declared TRAIN/local_fit."""
 from ops.r16_support_temporal_data import load_training,load_panel
 sys.addaudithook(guard);manifest=Path(manifest).resolve();configure_training_access(manifest);verify(source)
 if sha(manifest)!=data_sha:raise ValueError('Pilot data manifest changed before worker')
 torch.set_num_threads(2)
 fit=concatenate([load_panel(manifest,'local_fit',op)for op in OPERATORS]);train=load_training(manifest,arm=data_arm)
 if len(train['share'])!=4000 or len(fit['share'])!=2400:raise ValueError('Predeclared 4000 TRAIN / 2400 fit rows required')
 arm=data_arm+'F'+str(feature_mode)
 result=train_arm(train,fit,output,arm,feature_mode,{'Test_used':False,'selection_role':'local_fit','locked_roles_used_for_fit':False,
  'manifest_sha256':data_sha,'data_arm':data_arm})
 verify(source)
 if sha(manifest)!=data_sha:raise ValueError('Pilot data manifest changed during worker')
 return arm,result

def train_all(manifest,output,*,workers=1):
 if workers not in(1,2,4):raise ValueError('Bounded CPU worker count must be 1,2 or4')
 sys.addaudithook(guard);manifest=Path(manifest).resolve();output=Path(output).resolve()
 if not output.is_relative_to(ROOT)or not manifest.is_relative_to(ROOT):raise ValueError('Project-contained production paths required')
 if output.exists():raise FileExistsError('Fresh factorial pilot output required')
 output.mkdir(parents=True);source=codes();source['ops/r16_support_temporal_data.py']=sha(ROOT/'ops/r16_support_temporal_data.py');data_sha=sha(manifest)
 blocked_count=configure_training_access(manifest)
 plan={'status':'frozen_before_fit','arms':['D0F0','D0F1','D1F0','D1F1'],'training_rows_per_arm':4000,'epochs':100,'NN_early_stopping':False,'optimizer':'Adam',
  'Tree_maximum_trees':600,'Tree_early_stopping_patience':40,'manifest':str(manifest.relative_to(ROOT)),'manifest_sha256':data_sha,
  'code_sha256':source,'selection_role':'local_fit','locked_roles_used_for_fit':False,'fixed_ensemble':{'regression_tree':.6,'regression_NN':.4,'classification_tree':.5,'classification_NN':.5},
  'common_initial_weights_equal_and_extra_input_weights_zero':True,'Test_used':False,'seed':SEED,
  'CPU_processes':workers,'CPU_threads_per_process':2,'process_start_method':'spawn'if workers>1 else'in_process',
  'audit_blocked_locked_panel_paths':blocked_count,'failure_policy':'Drain all started workers to terminal state; no partial reuse'}
 write(output/'PLAN.json',plan)
 try:
  jobs=[(d,f)for d in('D0','D1')for f in(0,1)];results={};failures={}
  write(output/'progress.json',{'status':'training','workers':workers,'completed_arms':[]})
  if workers==1:
   for d,f in jobs:
    arm,result=_train_worker(manifest,output,d,f,source,data_sha);results[arm]=result
    write(output/'progress.json',{'status':'training','workers':workers,'completed_arms':list(results)})
  else:
   # Each process independently seeds the same shared initial parameters and
   # writes its unique arm directory. Parent alone aggregates terminal states.
   with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))as pool:
    futures={pool.submit(_train_worker,str(manifest),str(output),d,f,source,data_sha):d+'F'+str(f)for d,f in jobs}
    for future in as_completed(futures):
     name=futures[future]
     try:
      arm,result=future.result();results[arm]=result
     except BaseException as e:failures[name]=repr(e)
     write(output/'progress.json',{'status':'draining_failed_workers'if failures else'training','workers':workers,
      'completed_arms':sorted(results),'failed_arms':failures,'terminal_arms':len(results)+len(failures)})
   if failures:raise RuntimeError('All CPU workers reached terminal states; failed arms: '+json.dumps(failures))
  verify(source)
  if sha(manifest)!=data_sha:raise ValueError('Pilot data changed')
  done={'status':'all_four_factorial_arms_completed','results':{a:results[a]for a in plan['arms']},'plan_sha256':sha(output/'PLAN.json'),'code_sha256':source,'manifest_sha256':data_sha,
   'Test_used':False,'locked_report_or_gate_used_for_fit':False}
  write(output/'ALL_COMPLETED.json',done);write(output/'progress.json',done);return done
 except BaseException as e:write(output/'FAILED.json',{'error':repr(e),'no_partial_reuse':True});raise

def fixture():
 began=time.perf_counter();torch.set_num_threads(2);rng=np.random.default_rng(SEED);d=np.round(rng.normal(80,5,(180,15,25)),1);d[:25,0]=np.nan
 # Physical synthetic fixture only; discarded learners and no real benchmark.
 db=np.power(10.,d/10.);w=np.where(np.isfinite(db),db,0);w/=w.sum((1,2))[:,None,None]
 q=np.full(d.shape,.2,np.float32);q[::5]=0;q[~np.isfinite(d)]=0;y=(q*w).sum((1,2));p={'db':d,'share':y,'cell_share':q,'presence':y>0,'base_window_index':np.arange(len(d))}
 train={k:v[:140]for k,v in p.items()};fit={k:v[140:]for k,v in p.items()}
 x,ww=inputs(d[:4],1);model=make_model(1);pp,qq,cl=model(torch.from_numpy(x),torch.from_numpy(ww));cl.sum().backward()
 if any(a.grad is not None for a in model.reg.parameters()):raise AssertionError('Classification changed regression parameters')
 pool=float(abs(pp.detach().numpy()-(qq.detach().numpy()*ww).sum((1,2))).max())
 if pool>3e-6:raise AssertionError('Physical pooling changed')
 # Baseline architecture parity includes exact init and forward in eval mode.
 torch.manual_seed(SEED);legacy=SmallDecoupledCNN().eval();torch.manual_seed(SEED);baseline=TemporalDecoupledCNN(0).eval();bx,bw=inputs(d[:4],0)
 with torch.no_grad():
  old=legacy(torch.from_numpy(bx),torch.from_numpy(bw));new=baseline(torch.from_numpy(bx),torch.from_numpy(bw))
 parity=max(float(abs(a-b).max())for a,b in zip(old,new))
 if parity!=0:raise AssertionError('Feature0 CNN differs from frozen small baseline architecture')
 with torch.no_grad():
  zero_extra=make_model(1).eval();initial=zero_extra(torch.from_numpy(x),torch.from_numpy(ww))
 initgap=max(float(abs(a-b).max())for a,b in zip(new,initial))
 if initgap>2e-6:raise AssertionError('F0/F1 common initial predictions differ')
 with tempfile.TemporaryDirectory(dir='/private/tmp',prefix='r16-support-temporal-model-')as temp:
  result=train_arm(train,fit,temp,'fixtureF1',1,{'Test_used':False,'selection_role':'local_fit','locked_roles_used_for_fit':False},epochs=2,fixture_only=True)
  pr=predict_arm(Path(temp)/'fixtureF1',fit['db'],allow_fixture=True)
  if not all(np.isfinite(a).all()for a in pr.values()):raise AssertionError('Fixture reload produced nonfinite predictions')
  blocked=Path(temp)/'locked_panel.npz';blocked.write_bytes(b'fixture');FORBIDDEN_PANEL_PATHS.add(os.path.realpath(blocked));rejected=False
  try:
   try:blocked.read_bytes()
   except PermissionError:rejected=True
   if not rejected:raise AssertionError('Locked input audit hook did not reject')
  finally:FORBIDDEN_PANEL_PATHS.remove(os.path.realpath(blocked))
 return{'status':'passed','temporal_features':feature_fixture(),'feature0_legacy_CNN_forward_max_abs':parity,'F0_F1_initial_forward_max_abs':initgap,'classifier_gradient_to_regression':False,
  'physical_pool_max_abs':pool,'feature0_parameter_counts':baseline.parameter_counts(),'feature1_parameter_counts':model.parameter_counts(),
  'tiny_learner_reload_finite':True,'tiny_rows_train':140,'tiny_epochs':2,'fixture_only_learners_discarded':True,'elapsed_seconds':time.perf_counter()-began,
  'locked_input_open_rejected':rejected,'tiny_fit_metrics':result['fit_fixed60tree40nn_metrics'],
  'Test_used':False,'real_benchmark_training':False,'peak_process_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  'source_code_sha256':codes()}

def parallel_fixture(workers):
 """Discarded synthetic learners test spawn/reload/isolated output paths."""
 began=time.perf_counter()
 with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))as pool:
  results=[f.result()for f in[pool.submit(fixture)for _ in range(workers)]]
 if not all(r['status']=='passed'for r in results):raise AssertionError('Spawn fixture failed')
 if len({json.dumps(r['tiny_fit_metrics'],sort_keys=True)for r in results})!=1:raise AssertionError('Concurrent seeds differ')
 return{'status':'passed','workers':workers,'start_method':'spawn','threads_each':2,'fixture_only':True,'Test_used':False,
  'same_seed_concurrent_metrics_exact':True,'locked_inputs_open_rejected':all(r['locked_input_open_rejected']for r in results),
  'real_benchmark_training':False,'worker_peak_RSS_bytes':[r['peak_process_RSS_bytes']for r in results],
  'worker_elapsed_seconds':[r['elapsed_seconds']for r in results],'elapsed_seconds':time.perf_counter()-began,
  'source_code_sha256':codes()}

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['fixture','train']);parser.add_argument('--manifest');parser.add_argument('--output',default='models/r16_support_temporal_r1');parser.add_argument('--workers',type=int,choices=[1,2,4],default=1);args=parser.parse_args()
 if args.command=='fixture':result=fixture()if args.workers==1 else parallel_fixture(args.workers)
 else:
  if not args.manifest:parser.error('--manifest required')
  result=train_all(args.manifest,args.output,workers=args.workers)
 print(json.dumps(result,ensure_ascii=False))
