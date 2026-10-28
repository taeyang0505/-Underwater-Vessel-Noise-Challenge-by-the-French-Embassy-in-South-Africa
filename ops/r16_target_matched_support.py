"""One predeclared target-matched natural-support pilot; immutable previous runs.

No query data, no hyperparameter sweep. Train/fit only until both arms finish.
"""
from pathlib import Path
import argparse, csv, hashlib, json, multiprocessing, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'reports/r16_support_temporal_r1'
OUT=ROOT/'reports/r16_target_matched_support_r1'
MODELS=ROOT/'models/r16_target_matched_support_r1'
MANIFEST=BASE/'data/manifest.json'
ARMS=('S1_TM_H0_F1',)
OPS=('control_eq6','mfhf_shelf6','mfhf_shelf12','mfhf_shelf12_lfminus6','alpha_preEQ_inserted','independent_subband_snr')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 with t.open('wb')as f:np.savez_compressed(f,**d)
 t.replace(p)
def guard(event,args):
 if event=='open'and args and isinstance(args[0],(str,bytes,os.PathLike)):
  s=os.fsdecode(args[0]).lower().replace('\\','/')
  if Path(s).name in('test.csv','sample_submission.csv')or any(x in s for x in('/eda_deep/','/submissions/','/r15_band_comparison_')):raise PermissionError('No query or query-derived access in clean support check')
def verify(pins):
 for n,h in pins.items():
  if sha(ROOT/n)!=h:raise ValueError('Pinned file changed: '+n)

def build():
 sys.addaudithook(guard)
 if (OUT/'DATA_READY.json').exists():raise FileExistsError('Fresh target-matched data only')
 if not (OUT/'PROTOCOL.md').exists():raise ValueError('Preregister target-matched protocol first')
 d0p=BASE/'data/train/D0.npz';h0p=ROOT/'reports/r16_clean_support_r1/data/S1_H0_F1.npz'
 pins={str(p.relative_to(ROOT)):sha(p)for p in(d0p,h0p,MANIFEST,Path(__file__),OUT/'PROTOCOL.md')}
 d0=load(d0p);h0=load(h0p);data={k:v.copy()for k,v in h0.items()};positive=d0['share']>0
 assert np.array_equal(positive,h0['share']>0)
 B=h0['background_observed_power'];A=h0['added_observed_power'];O=h0['observed_mask'];y=d0['share']
 factor=np.ones(len(y),float)
 factor[positive]=y[positive]/(1-y[positive])*B.sum((1,2))[positive]/A.sum((1,2))[positive]
 added=h0['added_full_power']*factor[:,None,None];oa=np.where(O,added,0.);E=B+oa
 # Recover each row's original common dB offset without changing its B.
 residual=np.where(O,h0['unrounded_db']-10*np.log10(np.where(O,h0['mixture_observed_power'],1.)),np.nan)
 offset=np.nanmedian(residual,axis=(1,2));assert np.nanmax(abs(residual-offset[:,None,None]))<1e-10
 raw=np.where(O,10*np.log10(np.where(O,E,1.))+offset[:,None,None],np.nan)
 q=np.divide(oa,E,out=np.zeros_like(E),where=E>0);physical=oa.sum((1,2))/E.sum((1,2))
 data.update(db=np.round(raw,1).astype(np.float32),unrounded_db=raw,cell_share=q,share=physical,
  presence=(physical>0).astype(np.int64),added_full_power=added,added_observed_power=oa,mixture_observed_power=E)
 data['target_matched_scale']=factor
 assert np.max(abs(physical-y))<1e-12
 assert np.array_equal((q>0).any(2),(h0['cell_share']>0).any(2))
 for k,v in h0.items():np.testing.assert_array_equal(data[k][~positive],v[~positive],err_msg=k)
 support=(q>0).any(2).sum(1);path=OUT/'data/S1_TM_H0_F1.npz';save(path,data)
 pins[str(path.relative_to(ROOT))]=sha(path)
 stats={'rows':len(y),'positive':int(positive.sum()),'zero':int((~positive).sum()),'HF_replaced':0,
   'target_mean':float(physical.mean()),'target_rms':float(np.sqrt((physical**2).mean())),
   'target_difference_from_D0_max':float(abs(physical-y).max()),'active_positive_median':float(np.median(support[positive])),
   'active_ge12':int((support>=12).sum()),'scale_quantiles':np.quantile(factor[positive],[0,.1,.5,.9,1]).tolist(),
   'all_negative_arrays_unchanged':True,'all_positive_support_unchanged':True,'physical_labels_recomputed':True}
 write(OUT/'DATA_READY.json',{'status':'completed','source_sha256':pins,'statistics':stats,'Test_used':False,
   'method':'per-pair exact D0 y, alpha = y/(1-y)*sum(OB)/sum(OA); natural support unchanged'})
 print(json.dumps({'event':'target_matched_data_ready',**stats}),flush=True)

def worker(arm,pins):
 from ops.r16_support_temporal_models import configure_training_access,train_arm,concatenate
 from ops.r16_support_temporal_data import load_panel
 sys.addaudithook(guard);configure_training_access(MANIFEST);verify(pins)
 data=load(OUT/'data'/f'{arm}.npz');fit=concatenate([load_panel(MANIFEST,'local_fit',op)for op in OPS])
 result=train_arm(data,fit,MODELS,arm,1,{'Test_used':False,'selection_role':'local_fit','locked_roles_used_for_fit':False,'data_arm':arm,'clean_data_sha256':sha(OUT/'data'/f'{arm}.npz'),'parent_manifest_sha256':sha(MANIFEST),'new_wrapper_sha256':sha(__file__)})
 verify(pins);return arm,result

def train():
 from ops.r16_support_temporal_models import codes,configure_training_access
 sys.addaudithook(guard);configure_training_access(MANIFEST)
 ready=read(OUT/'DATA_READY.json');pins={**ready['source_sha256'],**codes()};verify(pins)
 if MODELS.exists():raise FileExistsError('Fresh models only; no partial reuse')
 MODELS.mkdir(parents=True)
 plan={'status':'fixed_before_training','arms':ARMS,'source_sha256':pins,'seed':2026102817,'epochs':100,'Adam':True,'early_stopping_NN':False,'CPU_processes':2,'CPU_threads_each':2,'fixed_regression':{'Tree':.6,'NN':.4},'fixed_score':{'Tree_classifier':.5,'NN_classifier':.5,'share':0},'extra_score_candidate':'D0F1 classifier unchanged; no fitting weights on locked panels','Test_used':False}
 write(MODELS/'PLAN.json',plan);began=time.perf_counter();results={};errors={}
 with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'))as pool:
  fs={pool.submit(worker,a,pins):a for a in ARMS}
  for f in as_completed(fs):
   try:a,v=f.result();results[a]=v
   except BaseException as e:errors[fs[f]]=repr(e)
   write(MODELS/'progress.json',{'completed':list(results),'errors':errors,'elapsed_seconds':time.perf_counter()-began})
 verify(pins)
 if errors:write(MODELS/'FAILED.json',errors);raise RuntimeError(str(errors))
 write(MODELS/'ALL_COMPLETED.json',{'status':'completed','results':results,'source_sha256':pins,'elapsed_seconds':time.perf_counter()-began,'Test_used':False})
 print(json.dumps({'event':'all_completed','elapsed_seconds':time.perf_counter()-began}),flush=True)

def evaluate():
 import torch
 from ops.r16_support_temporal_models import predict_arm
 from ops.r16_support_temporal_data import load_panel,load_long_panel
 from ops.evaluate_r16_support_temporal import measure
 sys.addaudithook(guard);torch.set_num_threads(2);torch.set_num_interop_threads(1)
 all_done=read(MODELS/'ALL_COMPLETED.json');assert all_done['status']=='completed';pins=dict(all_done['source_sha256']);verify(pins)
 pins[str((MODELS/'ALL_COMPLETED.json').relative_to(ROOT))]=sha(MODELS/'ALL_COMPLETED.json')
 for arm in ARMS:
  folder=MODELS/arm;c=read(folder/'COMPLETED.json');assert c['epochs']==100
  pins[str((folder/'COMPLETED.json').relative_to(ROOT))]=sha(folder/'COMPLETED.json')
  for n,h in c['artifacts_sha256'].items():pins[str((folder/n).relative_to(ROOT))]=h
 old_result=read(BASE/'evaluation/RESULTS.json')
 for n,h in old_result['predictions_sha256'].items():pins[str((BASE/'evaluation'/n).relative_to(ROOT))]=h
 for p in [BASE/'evaluation/RESULTS.json',OUT/'PROTOCOL.md',OUT/'DATA_READY.json']:pins[str(p.relative_to(ROOT))]=sha(p)
 verify(pins)
 policy={'regression':{'Tree':.6,'NN':.4},'score':{'Tree_classifier':.5,'NN_classifier':.5,'share':0},'fixed_decoupled_comparison':'same new regressor plus unchanged saved D0F1 classifier','Test_used':False,'selection_scope':'previously exposed local evaluation, not new blind performance','promotion_rule':'PROTOCOL.md fixed before new training'}
 write(OUT/'POLICY.json',policy);pins[str((OUT/'POLICY.json').relative_to(ROOT))]=sha(OUT/'POLICY.json')
 freeze={'status':'completed','all_models_frozen':True,'ensemble_policy_frozen':True,'data_manifest_sha256':sha(MANIFEST),'artifact_sha256':pins,'scope':'one target-matched support arm and prior saved D0F1 score; exposed evaluation','Test_used':False}
 write(OUT/'GLOBAL_FREEZE.json',freeze);freeze_path=OUT/'GLOBAL_FREEZE.json'
 result_dir=OUT/'evaluation'
 if result_dir.exists():raise FileExistsError('Fresh evaluation only')
 result_dir.mkdir();(result_dir/'predictions').mkdir();rows=[];preds={};pred_pins={};began=time.perf_counter()
 def evaluate_one(name,d,role,op='',weights=None):
  verify(pins);old=load(BASE/'evaluation/predictions'/f'{name}.npz');ids=d['sample_id'].astype(str)
  assert np.array_equal(ids,old['sample_id'])and np.array_equal(d['share'],old['target_share'])
  base_index=list(old['model_order']).index('D0F1');pr={a:predict_arm(MODELS/a,d)for a in ARMS}
  arrays={'sample_id':ids,'target_share':d['share'],'model_order':np.asarray(ARMS),'sample_weight':np.ones(len(ids))if weights is None else weights,'D0F1_classifier':old['balanced_classifier'][:,base_index]}
  for k in ('site','base_window_index','reference_component'):
   if k in d:arrays[k]=d[k]
  for key in pr[ARMS[0]]:arrays[key]=np.column_stack([pr[a][key]for a in ARMS])
  p=result_dir/'predictions'/f'{name}.npz';save(p,arrays);pred_pins[str(p.relative_to(result_dir))]=sha(p);preds[name]=arrays
  for arm in ARMS:
   v=pr[arm]
   for method,pk,ck in (('NN','nn_share','nn_classifier'),('Tree','tree_share','tree_classifier'),('fixed60T40N','balanced_share','balanced_classifier'),('reg_plus_D0F1score','balanced_share',None)):
    score=v[ck]if ck else arrays['D0F1_classifier'];mm=measure(d['share'],v[pk],score,weights);mm.update(panel=name,role=role,operator=op,arm=arm,model=method)
    if weights is not None:mm['unweighted_rmse']=measure(d['share'],v[pk],score)['rmse']
    if 'site'in d:
     ss=d['site'].astype(str);mm['site_slices']={s:measure(d['share'][ss==s],v[pk][ss==s],score[ss==s])for s in np.unique(ss)}
    rows.append(mm)
  print(json.dumps({'event':'panel','panel':name,'elapsed_seconds':time.perf_counter()-began}),flush=True)
 for role in ('local_gate','local_report','domain_holdout'):
  panels=[]
  for op in OPS:
   d=load_panel(MANIFEST,role,op,freeze_receipt=freeze_path);panels.append(d);evaluate_one(role+'__'+op,d,role,op)
  y=np.concatenate([d['share']for d in panels])
  for arm_index,arm in enumerate(ARMS):
   for method,pk,ck in (('NN','nn_share','nn_classifier'),('Tree','tree_share','tree_classifier'),('fixed60T40N','balanced_share','balanced_classifier'),('reg_plus_D0F1score','balanced_share',None)):
    pp=np.concatenate([preds[role+'__'+op][pk][:,arm_index]for op in OPS]);cc=np.concatenate([preds[role+'__'+op][ck][:,arm_index]if ck else preds[role+'__'+op]['D0F1_classifier']for op in OPS])
    rows.append({'panel':role+'__POOLED','role':role,'operator':'POOLED','arm':arm,'model':method,**measure(y,pp,cc)})
 long=[load_long_panel(MANIFEST,'native12_positive',operator=op,freeze_receipt=freeze_path)for op in OPS]
 pos={k:np.concatenate([d[k]for d in long])for k in long[0]if all(k in d for d in long)}
 for name in ('hf_high_negative','st_francis_dense_negative'):
  neg=load_long_panel(MANIFEST,name,freeze_receipt=freeze_path);pair={k:np.concatenate([pos[k],neg[k]])for k in set(pos)&set(neg)}
  weights=np.r_[np.full(len(pos['share']),.5/len(pos['share'])),np.full(len(neg['share']),.5/len(neg['share']))]
  evaluate_one('native12_VS_'+name,pair,'independent_long',weights=weights)
 # Preregistered fixed comparisons only. No learned policy weights.
 ix={(r['panel'],r['arm'],r['model']):r for r in rows};oldix={(r['panel'],r['arm'],r['model']):r for r in old_result['rows']};gates=[]
 for arm in ARMS:
  for method in ('fixed60T40N','reg_plus_D0F1score'):
   checks=[]
   for role in ('local_gate','local_report','domain_holdout'):
    p=role+'__POOLED';b=oldix[p,'D0F1','fixed60T40N'];a=ix[p,arm,method]
    checks.append({'name':role,'rmse':a['rmse'],'base_rmse':b['rmse'],'auc':a['auc_classifier'],'base_auc':b['auc_classifier'],'passed':bool(a['rmse']<=b['rmse']+max(.005,.05*b['rmse'])and a['auc_classifier']>=b['auc_classifier']-.02)})
   for name,limit in [('hf_high_negative',.11),('st_francis_dense_negative',.12)]:
    a=ix['native12_VS_'+name,arm,method]
    checks.append({'name':'native12_vs_'+name,'rmse':a['rmse'],'rmse_maximum':limit,'auc':a['auc_classifier'],'AUC_minimum':.75,'passed':bool(a['rmse']<=limit and a['auc_classifier']>=.75)})
   # Do not require score below an invented absolute 0.15: compare to D0F1.
   for name in ('hf_high_negative','st_francis_dense_negative'):
    p='native12_VS_'+name;a=ix[p,arm,method];b=oldix[p,'D0F1','fixed60T40N']
    checks.append({'name':name+'_negative_preservation','classifier_mean':a['negative']['classifier_mean'],'baseline_classifier_mean':b['negative']['classifier_mean'],'share_mean':a['negative']['prediction_mean'],'baseline_share_mean':b['negative']['prediction_mean'],'passed':bool(a['negative']['classifier_mean']<=b['negative']['classifier_mean']+.02 and a['negative']['prediction_mean']<=b['negative']['prediction_mean']+.01)})
   gates.append({'arm':arm,'method':method,'passed':all(x['passed']for x in checks),'checks':checks})
 verify(pins)
 out={'status':'completed','rows':rows,'candidate_gates':gates,'any_gate_passed':any(x['passed']for x in gates),'elapsed_seconds':time.perf_counter()-began,'predictions_sha256':pred_pins,'Test_used':False,'no_repeated_search':True,'selection_is_development_not_blind':True,'baseline_data_and_models_unchanged':True}
 write(result_dir/'RESULTS.json',out)
 cols=['panel','arm','model','n','rmse','bias','prediction_mean','prediction_rms','auc_share','auc_classifier','classifier_mean']
 with(result_dir/'metrics.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
 print(json.dumps({'event':'evaluation_completed','any_gate_passed':out['any_gate_passed'],'gates':gates}),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','train','evaluate']);args=parser.parse_args()
 {'build':build,'train':train,'evaluate':evaluate}[args.action]()
