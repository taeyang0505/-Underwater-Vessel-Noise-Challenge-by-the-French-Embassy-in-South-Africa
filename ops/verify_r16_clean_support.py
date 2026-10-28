"""Separate numerical checks for clean-support arrays and saved predictions."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r16_clean_support_r1'
BASE=ROOT/'reports/r16_support_temporal_r1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def data():
 d0=load(BASE/'data/train/D0.npz');d1=load(BASE/'data/train/D1.npz');zero=d0['share']==0
 ready=json.loads((OUT/'DATA_READY.json').read_text());checks={}
 for n,h in ready['source_sha256'].items():assert sha(ROOT/n)==h,n
 for arm,nchanged in [('S1_H0_F1',0),('S1_H15_F1',120)]:
  p=OUT/'data'/f'{arm}.npz';a=load(p)
  for k in a:
   np.testing.assert_array_equal(a[k][~zero],d1[k][~zero],err_msg=k)
  changed=zero&(a['base_window_index']!=d0['base_window_index']);assert changed.sum()==nchanged
  for k in a:np.testing.assert_array_equal(a[k][zero&~changed],d0[k][zero&~changed],err_msg=k)
  for k in ('site','recorder','sensor_group','effective_mask_bits','observed_mask'):np.testing.assert_array_equal(a[k],d0[k])
  for k in a:
   if k=='sample_id':continue
   np.testing.assert_array_equal(a[k][changed],d1[k][changed],err_msg=k)
  E=a['mixture_observed_power'];A=a['added_observed_power'];B=a['background_observed_power']
  np.testing.assert_allclose(E,B+A,rtol=1e-12,atol=1e-12)
  y=A.sum((1,2))/E.sum((1,2));assert abs(y-a['share']).max()<1e-12
  checks[arm]={'positive_arrays_exact_D1':True,'untouched_negative_arrays_exact_D0':True,'changed_negative_arrays_exact_D1':True,'replacement_count':int(changed.sum()),'raw_y_error':float(abs(y-a['share']).max()),'all_4000_masks_recorders_sites_exact':True}
 write(OUT/'DATA_RECOMPUTED_QA.json',{'status':'passed','checks':checks,'Test_used':False})
 print(json.dumps({'event':'independent_data_checks_passed','checks':checks}))

def results():
 from sklearn.metrics import roc_auc_score,mean_squared_error
 d=json.loads((OUT/'evaluation/RESULTS.json').read_text());freeze=json.loads((OUT/'GLOBAL_FREEZE.json').read_text())
 for n,h in freeze['artifact_sha256'].items():assert sha(ROOT/n)==h,n
 pred={}
 for n,h in d['predictions_sha256'].items():
  p=OUT/'evaluation'/n;assert sha(p)==h;pred[p.stem]=load(p)
 def arrays(panel,arm,method):
  if panel.endswith('__POOLED'):
   role=panel.removesuffix('__POOLED');names=[x for x in pred if x.startswith(role+'__')]
   # Concatenation order must match canonical six operators, not filename order.
   names=[role+'__'+op for op in ('control_eq6','mfhf_shelf6','mfhf_shelf12','mfhf_shelf12_lfminus6','alpha_preEQ_inserted','independent_subband_snr')]
  else:names=[panel]
  ys=[];ps=[];cs=[];ws=[]
  for name in names:
   a=pred[name];j=list(a['model_order']).index(arm)
   pk,ck={'NN':('nn_share','nn_classifier'),'Tree':('tree_share','tree_classifier'),'fixed60T40N':('balanced_share','balanced_classifier'),'reg_plus_D0F1score':('balanced_share',None)}[method]
   ys.append(a['target_share']);ps.append(a[pk][:,j]);cs.append(a[ck][:,j]if ck else a['D0F1_classifier']);ws.append(a['sample_weight'])
   np.testing.assert_allclose(a['balanced_share'][:,j],.6*a['tree_share'][:,j]+.4*a['nn_share'][:,j],rtol=1e-7,atol=1e-7)
   np.testing.assert_allclose(a['balanced_classifier'][:,j],.5*a['tree_classifier'][:,j]+.5*a['nn_classifier'][:,j],rtol=1e-7,atol=1e-7)
  return tuple(np.concatenate(v)for v in (ys,ps,cs,ws))
 worst=0
 for r in d['rows']:
  y,p,c,w=arrays(r['panel'],r['arm'],r['model']);w=w/w.sum()
  vals={'rmse':float(np.sqrt(mean_squared_error(y,p,sample_weight=w))),'bias':float(np.average(p-y,weights=w)),'prediction_mean':float(np.average(p,weights=w)),'classifier_mean':float(np.average(c,weights=w))}
  for k,v in vals.items():worst=max(worst,abs(v-r[k]))
  if (y>0).any()and(y==0).any():worst=max(worst,abs(float(roc_auc_score(y>0,c,sample_weight=w))-r['auc_classifier']))
 assert worst<1e-10,worst
 write(OUT/'RESULTS_RECOMPUTED_QA.json',{'status':'passed','recomputed_rows':len(d['rows']),'maximum_metric_absolute_error':worst,'frozen_artifacts':len(freeze['artifact_sha256']),'prediction_files':len(pred),'new_training_or_inference':False,'Test_used':False,'gate_unchanged':True})
 print(json.dumps({'event':'result_QA_passed','rows':len(d['rows']),'maximum_error':worst}))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('scope',choices=['data','results']);s=a.parse_args();{'data':data,'results':results}[s.scope]()
