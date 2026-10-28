"""Two explicitly configured, isolated CNN development pilots.

Frozen train/inner data only. No outer/Test/calibration/sealed access, no edits
of the original project or failed baseline. Generated augmentation is ephemeral.
"""
from pathlib import Path
import argparse,builtins,hashlib,io,json,math,os,platform,sys,time
import numpy as np
import torch
from torch.nn import functional as F

def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--mode',choices=['common_response','reference_smooth'],required=True);p.add_argument('--probability',type=float,default=.75);p.add_argument('--strength-db',type=float,required=True)
 p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=7);p.add_argument('--seed',type=int,default=42)
 p.add_argument('--batch',type=int,default=128);p.add_argument('--augmentation-batch',type=int,default=512);p.add_argument('--threads',type=int,default=4)
 p.add_argument('--learning-rate',type=float,default=.001);p.add_argument('--weight-decay',type=float,default=.0003);p.add_argument('--dropout',type=float,default=.1)
 p.add_argument('--smoke-rows',type=int,default=0,help='Explicit incomplete smoke only; zero uses all train/inner rows')
 return p

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return h.hexdigest()

def array_digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def route(item):
 refs=item['references'];native=[r['kind']=='native_contiguous_15min_crop' for r in refs]
 return 'zero' if not refs else 'native_only' if all(native) else 'mixed_native_short' if any(native) else 'single_short' if len(refs)==1 else 'multi_short'

def rmse(y,p):return float(np.sqrt(np.mean((np.asarray(y,dtype='float64')-np.asarray(p,dtype='float64'))**2)))

def main(a):
 project=a.project.resolve();data_dir=a.data_dir.resolve();output=a.output.resolve()
 if output.exists():raise FileExistsError('Fresh isolated pilot output required: '+str(output))
 if any(output.is_relative_to(project/p) for p in ['data','models','pipeline','configs']):raise ValueError('Pilot outputs must stay outside frozen data/models/code/config paths')
 if not 1<=a.epochs<=100 or a.patience<1 or min(a.batch,a.augmentation_batch,a.threads)<1:raise ValueError('Invalid training budget')
 if not 0<=a.probability<=1 or not 0<a.strength_db<=12:raise ValueError('Invalid declared augmentation settings')
 if a.mode=='common_response' and a.strength_db!=4:raise ValueError('Explicit common-response pilot is fixed at 4 dB')
 if a.mode=='reference_smooth' and a.strength_db!=12:raise ValueError('Explicit reference-spectrum pilot is fixed at 12 dB')
 sys.path.insert(0,str(project));sys.path.insert(0,str(Path(__file__).resolve().parent))
 from pipeline.train import load_prebuilt_data,predict_deep
 from pipeline.models import make_model
 from pipeline.mixtures import cnn_channels
 from energy_domain_augmentation import augment_components
 # Data consumers are audit-guarded. Module imports above do not open arrays.
 forbidden={'outer_validation.npz','outer_validation_recipes.json','test.csv','calibration.npz','sealed_holdout.npz'}
 real_builtin_open=builtins.open;real_io_open=io.open;opened=set()
 def guard(opener):
  def wrapped(file,*args,**kwargs):
   if isinstance(file,(str,bytes,os.PathLike)):
    path=Path(os.fsdecode(file)).resolve()
    if path.name.lower() in forbidden:raise PermissionError('Forbidden pilot data access: '+str(path))
    if path.is_relative_to(project/'data'):
     if not path.is_relative_to(data_dir):raise PermissionError('Pilot tried another data partition: '+str(path))
     if path.name.startswith('outer_') or path.name.startswith('calibration') or 'sealed' in path.name.lower():raise PermissionError('Forbidden heldout data access')
     opened.add(str(path.relative_to(project)))
   return opener(file,*args,**kwargs)
  return wrapped
 builtins.open=guard(real_builtin_open);io.open=guard(real_io_open)
 output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();manifest={'status':'initializing','arguments':vars(a),'full_source_pilot':a.smoke_rows==0,'outer_Test_calibration_sealed_accessed':False,'frozen_inputs_modified':False}
 def save_manifest():
  manifest['elapsed_seconds']=time.perf_counter()-start;manifest['data_files_opened']=sorted(opened)
  (output/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
 save_manifest()
 try:
  data,gate,verified=load_prebuilt_data(data_dir,cv_mode=True)
  if 'physics' not in gate['version']:raise ValueError('Verified physics R1 package required')
  train=data['train'];inner=data['validation'];items=[i for b in json.loads((data_dir/'validation_recipes.json').read_text())['batches'] for i in b['items']]
  if len(items)!=len(inner['share']):raise ValueError('Inner recipes do not align')
  routes=np.array([route(i) for i in items])
  if a.smoke_rows:
   train={k:v[:a.smoke_rows].copy() for k,v in train.items()};inner={k:v[:max(8,a.smoke_rows//2)].copy() for k,v in inner.items()};routes=routes[:len(inner['share'])]
  n=len(train['share']);fingerprint={k:array_digest(v) for k,v in train.items()};held_fingerprint={k:array_digest(v) for k,v in inner.items()}
  core={str(p.relative_to(project)):digest(p) for p in (project/'pipeline').glob('*.py')}
  helper=Path(sys.modules[augment_components.__module__].__file__).resolve()
  manifest.update(data_package_hash=gate['data_package_hash'],readiness_sha256=verified['readiness_manifest_sha256'],core_sha256=core,augmentation_helper_sha256=digest(helper),trainer_sha256=digest(__file__),train_rows=n,inner_rows=len(inner['share']),route_counts={str(k):int((routes==k).sum()) for k in np.unique(routes)},selection_metric='fixed inner RMSE',heldout_recipe_changed=False,augmentation_scope='same probability across all rows; zero added component remains zero for reference-only augmentation; route never enters model input',baseline_frozen_CNN_inner_RMSE=.32115460396931594,baseline_frozen_CNN_native_only_RMSE=.48047649145035676)
  torch.set_num_threads(a.threads);torch.manual_seed(a.seed);device=torch.device(a.device)
  if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
  model=make_model('cnn',architecture_version=3,dropout=a.dropout,model_width=32,input_channels=6).to(device)
  optimizer=torch.optim.Adam(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay)
  scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=2,min_lr=1e-6)
  original_inputs=cnn_channels(train['db'],version=3);inner_inputs=cnn_channels(inner['db'],version=3)
  best=float('inf');best_epoch=0;stale=0;state=None;history=[];stop_reason='max_epochs_reached'
  order_rng=np.random.default_rng(a.seed)
  manifest.update(status='running isolated CNN pilot',torch_version=str(torch.__version__),CUDA_runtime=torch.version.cuda,platform=platform.platform(),optimizer='Adam',scheduler='ReduceLROnPlateau(.5,patience2)',loss='share MSE + .03 presence BCE + .03 energy-weighted cell MSE',dropout=a.dropout,architecture={'architecture_version':3,'model_width':32,'input_channels':6,'dropout':a.dropout});save_manifest()
  for epoch in range(1,a.epochs+1):
   epoch_start=time.perf_counter();aug_rng=np.random.default_rng(a.seed+epoch*1000003)
   parts={k:[] for k in ['db','share','presence','cell_share']};audits=[]
   for offset in range(0,n,a.augmentation_batch):
    generated,audit=augment_components(train['db'][offset:offset+a.augmentation_batch],train['cell_share'][offset:offset+a.augmentation_batch],aug_rng,mode=a.mode,probability=a.probability,strength_db=a.strength_db)
    for k in parts:parts[k].append(np.asarray(generated[k],dtype='float32'))
    audits.append(audit)
   augmented={k:np.concatenate(v) for k,v in parts.items()}
   if any(array_digest(train[k])!=v for k,v in fingerprint.items()):raise RuntimeError('Augmentation modified original train arrays')
   if augmented['db'].shape!=train['db'].shape or not np.array_equal(np.isfinite(augmented['db']),np.isfinite(train['db'])):raise ValueError('Augmentation changed observed support')
   if not np.array_equal(augmented['presence'],(augmented['share']>0).astype('float32')):raise ValueError('Augmentation label mismatch')
   if ((train['presence']>0)&(augmented['presence']==0)).any():raise ValueError('A positive training example silently became zero')
   inputs,weights=cnn_channels(augmented['db'],version=3)
   recovered=(augmented['cell_share']*weights).sum((1,2))
   conservation=float(np.max(np.abs(recovered-augmented['share'])))
   if conservation>3e-6:raise ValueError('Augmented cell-to-share conservation failed')
   target=torch.from_numpy(augmented['share']).to(device);presence=torch.from_numpy(augmented['presence']).to(device);cell=torch.from_numpy(augmented['cell_share']).to(device)
   model.train();loss_sum=0.;seen=0;epoch_lr=optimizer.param_groups[0]['lr']
   for ids in np.array_split(order_rng.permutation(n),math.ceil(n/a.batch)):
    optimizer.zero_grad(set_to_none=True);tw=torch.from_numpy(weights[ids]).to(device)
    prediction,logit,mask=model(torch.from_numpy(inputs[ids]).to(device),tw)
    loss=F.mse_loss(prediction,target[ids])+.03*F.binary_cross_entropy_with_logits(logit,presence[ids])+.03*((mask-cell[ids]).square()*tw).sum((1,2)).mean()
    if not torch.isfinite(loss):raise ValueError('Nonfinite pilot loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_sum+=float(loss.detach().cpu())*len(ids);seen+=len(ids)
   aug_pred,_=predict_deep(model,'cnn',augmented,device,batch=a.batch,inputs=(inputs,weights))
   original_pred,_=predict_deep(model,'cnn',train,device,batch=a.batch,inputs=original_inputs)
   val_pred,val_score=predict_deep(model,'cnn',inner,device,batch=a.batch,inputs=inner_inputs);value=rmse(inner['share'],val_pred)
   scheduler.step(value)
   if value<best:
    best=value;best_epoch=epoch;stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save({'state_dict':state,'kind':'cnn','seed':a.seed,'width':256,'k':16,'n_features':None,'input_shape':[6,15,25],
                'feature_version':3,'preprocessing_version':3,'architecture':manifest['architecture']},output/'best.pt')
    np.savez_compressed(output/'best_inner_predictions.npz',share_true=inner['share'],presence_true=inner['presence'],cnn_share=val_pred,cnn_score=val_score,route=routes)
   else:stale+=1
   route_metrics={str(k):{'n':int((routes==k).sum()),'rmse':rmse(inner['share'][routes==k],val_pred[routes==k]),'mean_true':float(inner['share'][routes==k].mean()),'mean_prediction':float(val_pred[routes==k].mean())} for k in np.unique(routes)}
   entry={'epoch':epoch,'loss':loss_sum/seen,'augmented_train_rmse':rmse(augmented['share'],aug_pred),'original_train_rmse':rmse(train['share'],original_pred),'inner_rmse':value,'inner_bias':float(np.mean(val_pred-inner['share'])),'best_inner_rmse':best,'best_epoch':best_epoch,'epochs_without_improvement':stale,'learning_rate':epoch_lr,'augmentation_conservation_max_error':conservation,'augmented_target_mean':float(augmented['share'].mean()),'augmented_target_zero_fraction':float((augmented['share']==0).mean()),'route_metrics':route_metrics,'augmentation_batch_audits':audits,'seconds':time.perf_counter()-epoch_start}
   history.append(entry);(output/'history.json').write_text(json.dumps(history,indent=2,default=str));print(json.dumps({k:v for k,v in entry.items() if k not in ['augmentation_batch_audits','route_metrics']}) ,flush=True)
   manifest.update(epochs_run=epoch,best_epoch=best_epoch,best_inner_rmse=best,last_epoch=entry);save_manifest()
   del augmented,inputs,weights,target,presence,cell,aug_pred,original_pred
   if stale>=a.patience:stop_reason='early_stopping_patience_reached';break
  if state is None:raise RuntimeError('No best checkpoint')
  model.load_state_dict(state);final_pred,final_score=predict_deep(model,'cnn',inner,device,batch=a.batch,inputs=inner_inputs)
  if not np.isclose(rmse(inner['share'],final_pred),best,rtol=1e-5,atol=1e-7):raise RuntimeError('Best-state restoration failed')
  if any(array_digest(v)!=held_fingerprint[k] for k,v in inner.items()):raise RuntimeError('Inner arrays changed')
  if any(digest(project/name)!=value for name,value in core.items()):raise RuntimeError('Frozen pipeline code changed during pilot')
  if digest(helper)!=manifest['augmentation_helper_sha256'] or digest(__file__)!=manifest['trainer_sha256']:raise RuntimeError('Pilot helper code changed during execution')
  if digest(data_dir/'readiness_manifest.json')!=verified['readiness_manifest_sha256']:raise RuntimeError('Frozen data readiness changed')
  for split,check in verified['splits'].items():
   if digest(data_dir/(split+'.npz'))!=check['npz_sha256']:raise RuntimeError('Frozen input NPZ changed')
  manifest.update(status='completed isolated CNN pilot; inner-development evidence only',stop_reason=stop_reason,epochs_run=len(history),best_epoch=best_epoch,best_inner_rmse=best,best_inner_metrics_by_route={str(k):{'n':int((routes==k).sum()),'rmse':rmse(inner['share'][routes==k],final_pred[routes==k]),'mean_true':float(inner['share'][routes==k].mean()),'mean_prediction':float(final_pred[routes==k].mean())} for k in np.unique(routes)},best_checkpoint_restored=True,selection_caveat='Two explicit pilots use the same inner development set; this is not an unbiased outer or official score.');save_manifest()
  print(json.dumps({'status':manifest['status'],'mode':a.mode,'best_epoch':best_epoch,'epochs_run':len(history),'best_inner_rmse':best,'best_native_only_rmse':manifest['best_inner_metrics_by_route'].get('native_only',{}).get('rmse'),'output':str(output)}),flush=True)
 except BaseException as exc:
  manifest.update(status='failed isolated pilot; preserve logs',error={'type':type(exc).__name__,'message':str(exc)});save_manifest();raise
 finally:builtins.open=real_builtin_open;io.open=real_io_open
if __name__=='__main__':main(parser().parse_args())
