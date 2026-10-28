"""Validated epoch-boundary continuation; frozen original trainer is untouched.

Same data, architectures, loss, 100 epochs, Adam, scheduling, batch/order and
RNG. Only process lifecycle and restore/skip-complete handling are added.
"""
import sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.training_r16_multisupport import *
RESUME=ROOT/'ops/r16_multisupport_resume_20261028_r1'

def validate_finished(folder,kind,regime):
 s=read(folder/'summary.json')
 if s['status']!='completed' or s['epochs']!=100 or s['kind']!=kind or s['regime']!=regime or s['code_sha256']!=codes() or s['data_manifest_sha256']!=sha(OUT/'manifest.json'):raise ValueError('Completed run identity differs')
 for n,h in s['artifact_sha256'].items():
  if sha(folder/n)!=h:raise ValueError('Completed run artifact differs')

def validate_partial(folder,kind,regime,device,batch,architecture):
 import torch
 plan=read(folder/'PLAN.json');ck=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
 if ck['format']!='r16_multisupport_full_checkpoint_v1' or ck['code_sha256']!=codes() or ck['data_manifest_sha256']!=sha(OUT/'manifest.json'):raise ValueError('Partial identity differs')
 if any(ck[k]!=v for k,v in [('kind',kind),('regime',regime),('architecture',architecture)]):raise ValueError('Partial architecture differs')
 if plan['device']!=device or plan['batch']!=batch or plan['epochs']!=100 or plan['early_stopping'] is not False or plan['code_sha256']!=codes():raise ValueError('Training plan differs')
 log=[json.loads(v) for v in (folder/'epochs.jsonl').read_text().splitlines()]
 if [v['epoch'] for v in log]!=list(range(1,ck['epoch']+1)):raise ValueError('Checkpoint and epoch log differ; no silent repair')
 row=log[-1]
 if row['best_regression_epoch']!=ck['best_regression_epoch'] or row['best_classifier_epoch']!=ck['best_classifier_epoch'] or row['best_RMSE']!=ck['best_regression_rmse']:raise ValueError('Best metrics differ')
 if not 1<=ck['epoch']<100 or ck['scheduler_state_dict']['last_epoch']!=ck['epoch']:raise ValueError('Not a complete epoch checkpoint')
 n=read(CACHE/regime/'train/manifest.json')['n']
 if ck['optimizer_steps']!=ck['epoch']*math.ceil(n/batch):raise ValueError('Optimizer steps differ')
 for name,key in [('best_regression.pt','best_regression_epoch'),('best_classifier.pt','best_classifier_epoch')]:
  best=torch.load(folder/name,map_location='cpu',weights_only=True)
  if best['epoch']!=ck[key] or best['code_sha256']!=codes() or best['data_manifest_sha256']!=ck['data_manifest_sha256']:raise ValueError('Best artifact differs')
 return ck

def continue_neural_verified():
 import torch,sys
 from pipeline.models_r16_multisupport import MultiSupportModel,decoupled_loss,clip_decoupled_gradients
 from pipeline.training_r16_production import rng_state,restore_rng
 sys.addaudithook(guard);torch.set_num_threads(2);pf=read(MODELS/'ACTUAL_PREFLIGHT.json');pins=codes();datahash=sha(OUT/'manifest.json')
 if pf['code_sha256']!=pins or pf['data_manifest_sha256']!=datahash:raise ValueError('Preflight identity changed')
 va=cache_read(cache_role('D0','selection'));pause={'requested':False};signal.signal(signal.SIGUSR1,lambda *_:pause.update(requested=True));began=time.perf_counter()
 for kind in KINDS:
  for regime in REGIMES:
   folder=MODELS/'neural'/regime/kind
   existing=folder.exists()
   if existing and (folder/'summary.json').exists():
    validate_finished(folder,kind,regime);continue
   if not existing:folder.mkdir(parents=True)
   tr=cache_read(cache_role(regime,'train'));device=pf['models'][kind]['device'];batch=pf['models'][kind]['batch']
   torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);model=MultiSupportModel(kind).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
   scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=1e-5);best=float('inf');bc=-1.;be=bce=steps=0
   first_epoch=1
   if existing:
    ck=validate_partial(folder,kind,regime,device,batch,model.architecture())
    model.load_state_dict(ck['state_dict'],strict=True);opt.load_state_dict(ck['optimizer_state_dict']);scheduler.load_state_dict(ck['scheduler_state_dict'])
    best=ck['best_regression_rmse'];bc=ck['best_classifier_auc'];be=ck['best_regression_epoch'];bce=ck['best_classifier_epoch'];steps=ck['optimizer_steps'];first_epoch=ck['epoch']+1
    restore_rng(ck['rng'],device);del ck
   else:
    write(folder/'PLAN.json',{'epochs':100,'Adam':True,'early_stopping':False,'regime':regime,'kind':kind,'batch':batch,'device':device,'code_sha256':pins,'data_manifest_sha256':datahash,'architecture':model.architecture()})
   for epoch in range(first_epoch,101):
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
