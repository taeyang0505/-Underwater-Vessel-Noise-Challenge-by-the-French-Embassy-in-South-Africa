"""Isolated 8-channel physical-background-prior residual CNN pilot.

Only frozen TRAIN fits the background prior. Every query map receives db/prior
only. No outer/Test/calibration/sealed arrays or frozen-project edits.
"""
from pathlib import Path
import argparse,builtins,io,json,math,os,platform,sys,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_native_generalization_pilot import digest,array_digest,route,rmse

class BackgroundPriorResidualCNN(nn.Module):
 def __init__(self,width=32,dropout=.1,prior_clip=.01,prior_strength=.5):
  super().__init__()
  from pipeline.models import EnergyMaskCNN
  self.network=EnergyMaskCNN(width=width,dropout=dropout,input_channels=8)
  nn.init.zeros_(self.network.mask_head.weight);nn.init.zeros_(self.network.mask_head.bias)
  self.prior_clip=prior_clip;self.prior_strength=prior_strength
 def forward(self,x,weights,return_residual=False):
  h=self.network.body(x);residual=self.network.mask_head(h)[:,0]
  q0=x[:,6].clamp(self.prior_clip,1-self.prior_clip)
  q=torch.sigmoid(self.prior_strength*torch.logit(q0)+residual)
  share=(q*weights).sum((1,2));presence=self.network.presence_head(h.mean((2,3)))[:,0]
  return (share,presence,q,residual) if return_residual else (share,presence,q)

@torch.inference_mode()
def predict(model,inputs,weights,device,batch):
 model.eval();p=[];s=[]
 for start in range(0,len(inputs),batch):
  ids=slice(start,start+batch);share,logit,_=model(torch.from_numpy(inputs[ids]).to(device),torch.from_numpy(weights[ids]).to(device));p.append(share.cpu().numpy());s.append(logit.sigmoid().cpu().numpy())
 return np.concatenate(p),np.concatenate(s)

def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--project',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=7);p.add_argument('--seed',type=int,default=42)
 p.add_argument('--batch',type=int,default=128);p.add_argument('--map-batch',type=int,default=512);p.add_argument('--threads',type=int,default=4);p.add_argument('--clusters',type=int,default=32)
 p.add_argument('--learning-rate',type=float,default=.001);p.add_argument('--weight-decay',type=float,default=.0003);p.add_argument('--dropout',type=float,default=.1);p.add_argument('--residual-l2',type=float,default=.001)
 p.add_argument('--prior-clip',type=float,default=.01);p.add_argument('--prior-strength',type=float,default=.5)
 p.add_argument('--smoke-rows',type=int,default=0);return p

def main(a):
 project=a.project.resolve();data_dir=a.data_dir.resolve();output=a.output.resolve();start=time.perf_counter()
 if output.exists():raise FileExistsError('Fresh isolated output required')
 if any(output.is_relative_to(project/p) for p in ['data','models','pipeline','configs']):raise ValueError('Output must stay outside frozen paths')
 if not 1<=a.epochs<=100 or min(a.patience,a.batch,a.map_batch,a.threads,a.clusters)<1 or a.residual_l2<0:raise ValueError('Invalid pilot budget')
 if not 0<a.prior_clip<.5 or not 0<a.prior_strength<=1:raise ValueError('Invalid softened prior')
 sys.path.insert(0,str(project));sys.path.insert(0,str(Path(__file__).resolve().parent))
 from pipeline.models import EnergyMaskCNN
 from pipeline.mixtures import cnn_channels
 from pipeline.train import load_prebuilt_data
 from background_prior_features import fit_background_prior,background_prior_maps
 real_open,real_io=builtins.open,io.open;opened=set();forbidden={'outer_validation.npz','outer_validation_recipes.json','test.csv','calibration.npz','sealed_holdout.npz'}
 def guard(opener):
  def wrapped(file,*args,**kwargs):
   if isinstance(file,(str,bytes,os.PathLike)):
    path=Path(os.fsdecode(file)).resolve()
    if path.name.lower() in forbidden:raise PermissionError('Forbidden heldout access')
    if path.is_relative_to(project/'data'):
     if not path.is_relative_to(data_dir) or path.name.startswith(('outer_','calibration')) or 'sealed' in path.name.lower():raise PermissionError('Nondevelopment data access')
     opened.add(str(path.relative_to(project)))
   return opener(file,*args,**kwargs)
  return wrapped
 builtins.open=guard(real_open);io.open=guard(real_io)
 output.mkdir(parents=True,exist_ok=False);m={'status':'initializing physical-prior hybrid','arguments':vars(a),'outer_Test_calibration_sealed_accessed':False,'frozen_inputs_modified':False,'full_source_pilot':a.smoke_rows==0}
 def save():
  m['elapsed_seconds']=time.perf_counter()-start;m['data_files_opened']=sorted(opened);(output/'manifest.json').write_text(json.dumps(m,indent=2,default=str))
 save()
 try:
  data,gate,verified=load_prebuilt_data(data_dir,cv_mode=True);train=data['train'];inner=data['validation']
  recipes=[i for b in json.loads((data_dir/'validation_recipes.json').read_text())['batches'] for i in b['items']];routes=np.array([route(i) for i in recipes])
  if len(routes)!=len(inner['share']):raise ValueError('Inner order mismatch')
  if a.smoke_rows:
   train={k:v[:a.smoke_rows].copy() for k,v in train.items()};inner={k:v[:max(8,a.smoke_rows//2)].copy() for k,v in inner.items()};routes=routes[:len(inner['share'])]
  fingerprints={s:{k:array_digest(v) for k,v in d.items()} for s,d in [('train',train),('inner',inner)]}
  core={str(p.relative_to(project)):digest(p) for p in (project/'pipeline').glob('*.py')};prior_module=Path(sys.modules[fit_background_prior.__module__].__file__).resolve()
  m.update(train_rows=len(train['share']),inner_rows=len(inner['share']),data_package_hash=gate['data_package_hash'],readiness_sha256=verified['readiness_manifest_sha256'],core_sha256=core,trainer_sha256=digest(__file__),prior_helper_sha256=digest(prior_module),prior_fit_split='train only; B=(1-cell_share)*observed mixture energy',query_inputs='db and fitted aggregate prior only; no query q, source IDs or route tags',augmentation='none',loss='share MSE + .03 presence BCE + .03 energy-weighted cell MSE + residual_l2*energy-weighted residual-logit squared',residual_l2=a.residual_l2,optimizer='Adam',scheduler='ReduceLROnPlateau(.5,patience2)',baseline_frozen_CNN_inner_RMSE=.32115460396931594,baseline_frozen_CNN_native_only_RMSE=.48047649145035676)
  prior_start=time.perf_counter();prior=fit_background_prior(train['db'],train['cell_share'],seed=a.seed,clusters=a.clusters)
  (output/'background_prior.json').write_text(json.dumps(prior,indent=2));m['prior_fit_seconds']=time.perf_counter()-prior_start;m['prior_sha256']=digest(output/'background_prior.json')
  def inputs_for(db):
   chunks=[];weight_chunks=[];prior_shares=[];audits=[]
   for offset in range(0,len(db),a.map_batch):
    part=db[offset:offset+a.map_batch];q0,bfraction,audit=background_prior_maps(part,prior);channels,weights=cnn_channels(part,version=3)
    if q0.shape!=part.shape or bfraction.shape!=part.shape or not np.isfinite(q0).all() or not np.isfinite(bfraction).all():raise ValueError('Invalid prior maps')
    if ((q0<0)|(q0>1)).any() or ((bfraction<0)|(bfraction>10+1e-6)).any():raise ValueError('Invalid bounded prior ratios')
    valid=np.isfinite(part)
    if (q0[~valid]!=0).any() or (bfraction[~valid]!=0).any():raise ValueError('Prior map filled unobserved cells')
    bg_channel=np.log1p(np.clip(bfraction,0,10))/np.log(11.)
    chunks.append(np.concatenate([channels,q0[:,None],bg_channel[:,None]],1).astype('float32'));weight_chunks.append(weights);prior_shares.append((q0*weights).sum((1,2)));audits.append(audit)
   return np.concatenate(chunks),np.concatenate(weight_chunks),np.concatenate(prior_shares),audits
  tx,tw,train_prior,train_audits=inputs_for(train['db']);vx,vw,inner_prior,inner_audits=inputs_for(inner['db'])
  m.update(raw_prior_train_rmse=rmse(train['share'],train_prior),raw_prior_inner_rmse=rmse(inner['share'],inner_prior),raw_prior_inner_routes={str(k):{'n':int((routes==k).sum()),'rmse':rmse(inner['share'][routes==k],inner_prior[routes==k]),'mean_prediction':float(inner_prior[routes==k].mean()),'mean_true':float(inner['share'][routes==k].mean())} for k in np.unique(routes)},train_prior_audits=train_audits,inner_prior_audits=inner_audits)
  torch.set_num_threads(a.threads);torch.manual_seed(a.seed);device=torch.device(a.device)
  if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
  architecture={'class':'BackgroundPriorResidualCNN','version':1,'input_channels':8,'width':32,'dropout':a.dropout,'prior_clip':a.prior_clip,'prior_strength':a.prior_strength,'channel7':'q0=clip(1-Bhat/E,0,1)','channel8':'log1p(clip(Bhat/E,0,10))/log(11)','residual_initialization':'zero logits'}
  model=BackgroundPriorResidualCNN(width=32,dropout=a.dropout,prior_clip=a.prior_clip,prior_strength=a.prior_strength).to(device)
  optimizer=torch.optim.Adam(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=2,min_lr=1e-6)
  y=torch.from_numpy(train['share']).to(device);z=torch.from_numpy(train['presence']).to(device);q=torch.from_numpy(train['cell_share']).to(device)
  def assess(p):return {str(k):{'n':int((routes==k).sum()),'rmse':rmse(inner['share'][routes==k],p[routes==k]),'mean_prediction':float(p[routes==k].mean()),'mean_true':float(inner['share'][routes==k].mean())} for k in np.unique(routes)}
  initial,initial_score=predict(model,vx,vw,device,a.batch);best=rmse(inner['share'],initial);best_epoch=0;stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};history=[];stop='max_epochs_reached'
  m.update(status='running isolated background-prior hybrid',architecture=architecture,initial_inner_rmse=best,initial_inner_routes=assess(initial),torch_version=str(torch.__version__),CUDA_runtime=torch.version.cuda,platform=platform.platform(),selection_metric='fixed inner RMSE; epoch0 physical prior is an eligible best state')
  def checkpoint(pred,score):
   torch.save({'state_dict':state,'kind':'background_prior_residual_cnn','seed':a.seed,'architecture':architecture,'prior_sha256':m['prior_sha256'],'feature_version':'ops_background_prior_8channel_v1'},output/'best.pt')
   np.savez_compressed(output/'best_inner_predictions.npz',share_true=inner['share'],presence_true=inner['presence'],hybrid_share=pred,hybrid_score=score,route=routes)
  checkpoint(initial,initial_score);save();rng=np.random.default_rng(a.seed)
  for epoch in range(1,a.epochs+1):
   t0=time.perf_counter();loss_sum=0.;seen=0;lr=optimizer.param_groups[0]['lr'];model.train()
   for ids in np.array_split(rng.permutation(len(y)),math.ceil(len(y)/a.batch)):
    optimizer.zero_grad(set_to_none=True);weights=torch.from_numpy(tw[ids]).to(device)
    p,logit,cell,residual=model(torch.from_numpy(tx[ids]).to(device),weights,return_residual=True)
    loss=F.mse_loss(p,y[ids])+.03*F.binary_cross_entropy_with_logits(logit,z[ids])+.03*((cell-q[ids]).square()*weights).sum((1,2)).mean()+a.residual_l2*(residual.square()*weights).sum((1,2)).mean()
    if not torch.isfinite(loss):raise ValueError('Nonfinite hybrid loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_sum+=float(loss.detach().cpu())*len(ids);seen+=len(ids)
   pt,_=predict(model,tx,tw,device,a.batch);pv,sv=predict(model,vx,vw,device,a.batch);value=rmse(inner['share'],pv);scheduler.step(value)
   if value<best:best=value;best_epoch=epoch;stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};checkpoint(pv,sv)
   else:stale+=1
   entry={'epoch':epoch,'loss':loss_sum/seen,'train_rmse':rmse(train['share'],pt),'inner_rmse':value,'inner_bias':float(np.mean(pv-inner['share'])),'best_inner_rmse':best,'best_epoch':best_epoch,'epochs_without_improvement':stale,'learning_rate':lr,'route_metrics':assess(pv),'seconds':time.perf_counter()-t0}
   history.append(entry);(output/'history.json').write_text(json.dumps(history,indent=2));m.update(epochs_run=epoch,best_epoch=best_epoch,best_inner_rmse=best,last_epoch=entry);save();print(json.dumps({k:v for k,v in entry.items() if k!='route_metrics'}),flush=True)
   if stale>=a.patience:stop='early_stopping_patience_reached';break
  model.load_state_dict(state);final,score=predict(model,vx,vw,device,a.batch)
  prior_reloaded=json.loads((output/'background_prior.json').read_text());q_reload,b_reload,_=background_prior_maps(inner['db'][:min(128,len(inner['share']))],prior_reloaded)
  check_channels,check_weights=cnn_channels(inner['db'][:len(q_reload)],version=3);check_bg=np.log1p(np.clip(b_reload,0,10))/np.log(11.);check_x=np.concatenate([check_channels,q_reload[:,None],check_bg[:,None]],1).astype('float32')
  loaded=torch.load(output/'best.pt',map_location='cpu',weights_only=True);reloaded=BackgroundPriorResidualCNN(width=32,dropout=a.dropout,prior_clip=a.prior_clip,prior_strength=a.prior_strength).to(device);reloaded.load_state_dict(loaded['state_dict']);replay,_=predict(reloaded,check_x,check_weights,device,a.batch)
  reload_error=float(np.max(np.abs(replay-final[:len(replay)])))
  if not np.allclose(replay,final[:len(replay)],atol=2e-5,rtol=1e-5):raise RuntimeError('Hybrid prior/checkpoint reload failed')
  if not np.isclose(rmse(inner['share'],final),best,rtol=1e-5,atol=1e-7):raise RuntimeError('Best state restore failed')
  for name,d in [('train',train),('inner',inner)]:
   if any(array_digest(v)!=fingerprints[name][k] for k,v in d.items()):raise RuntimeError('Input RAM arrays changed')
  if any(digest(project/name)!=value for name,value in core.items()) or digest(prior_module)!=m['prior_helper_sha256'] or digest(__file__)!=m['trainer_sha256']:raise RuntimeError('Frozen code changed')
  if digest(data_dir/'readiness_manifest.json')!=verified['readiness_manifest_sha256']:raise RuntimeError('Frozen readiness changed')
  for split,check in verified['splits'].items():
   if digest(data_dir/(split+'.npz'))!=check['npz_sha256']:raise RuntimeError('Frozen input NPZ changed')
  m.update(status='completed isolated background-prior hybrid; inner development only',best_epoch=best_epoch,best_inner_rmse=best,best_inner_routes=assess(final),epochs_run=len(history),stop_reason=stop,checkpoint_prior_reload_max_error=reload_error,selection_caveat='Single predefined development pilot; inner selection is not unbiased outer/official performance.');save()
  print(json.dumps({'status':m['status'],'best_epoch':best_epoch,'best_inner_rmse':best,'native_only_rmse':m['best_inner_routes'].get('native_only',{}).get('rmse'),'reload_max_error':reload_error,'output':str(output)}),flush=True)
 except BaseException as exc:m.update(status='failed isolated hybrid; preserve logs',error={'type':type(exc).__name__,'message':str(exc)});save();raise
 finally:builtins.open=real_open;io.open=real_io
if __name__=='__main__':main(parser().parse_args())
