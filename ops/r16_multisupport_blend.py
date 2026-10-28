"""Blend already-saved D0F1/S1 predictions. No training, inference or Test access."""
from pathlib import Path
import json, hashlib, sys, os, time
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/r16_support_temporal_r1'
CLEAN=ROOT/'reports/r16_clean_support_r1'
OUT=ROOT/'reports/r16_multisupport_r1'
ARMS=('S1_H0_F1','S1_H15_F1')
PRIMARY=('local_gate','local_report','domain_holdout')
OPS=('control_eq6','mfhf_shelf6','mfhf_shelf12','mfhf_shelf12_lfminus6','alpha_preEQ_inserted','independent_subband_snr')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def guard(event,args):
 if event=='open'and args and isinstance(args[0],(str,bytes,os.PathLike)):
  s=os.fsdecode(args[0]).lower()
  if Path(s).name=='test.csv'or'/eda_deep/'in s or'/submissions/'in s:raise PermissionError('Saved-prediction audit only')
def metric(y,p,c,w):
 w=w/w.sum();z=y>0
 out={'n':len(y),'rmse':float(np.sqrt(np.sum(w*(p-y)**2))),'bias':float(np.sum(w*(p-y))),
      'prediction_mean':float(np.sum(w*p)),'classifier_mean':float(np.sum(w*c)),
      'auc_classifier':float(roc_auc_score(z,c,sample_weight=w))}
 for mask,name in ((~z,'negative'),(z,'positive')):
  v=w[mask];v=v/v.sum()
  out[name]={'share_mean':float(np.sum(v*p[mask])),'classifier_mean':float(np.sum(v*c[mask]))}
 return out
def fit_arrays(folder):
 n=load(folder/'fit_nn_predictions.npz');t=load(folder/'fit_tree_predictions.npz')
 assert np.array_equal(n['target_share'],t['target_share'])
 return n['target_share'],.6*t['share']+.4*n['share'],.5*t['classifier_prob']+.5*n['classifier_prob']

def main():
 sys.addaudithook(guard);began=time.perf_counter()
 if OUT.exists():raise FileExistsError('New immutable blend report only')
 OUT.mkdir()
 protocol={'format':'r16_multisupport_blend_v1','new_training':False,'new_inference':False,'Test_used':False,
  'regression_weight_fit':'closed-form bounded MSE on local_fit only',
  'classifier_weight_fit_grid':list(np.linspace(0,1,11)),
  'fixed_comparisons':[{'ws':.4,'wc':.5},{'ws':.5,'wc':.5},{'ws':.35,'wc':.5}],
  'preference':'fit candidates H0 then H15; fixed candidates descriptive; locked metrics never fit weights',
  'gate':'Inherited clean-support PROTOCOL; no tolerance changes',
  'causality':'Target marginal shift is measured, not a proven sole cause; convexity does not guarantee all gates'}
 write(OUT/'PROTOCOL.json',protocol)
 pins={str((CLEAN/'GLOBAL_FREEZE.json').relative_to(ROOT)):sha(CLEAN/'GLOBAL_FREEZE.json'),
       str((BASE/'GLOBAL_FREEZE.json').relative_to(ROOT)):sha(BASE/'GLOBAL_FREEZE.json'),'ops/r16_multisupport_blend.py':sha(__file__)}
 for folder in (CLEAN,BASE):
  f=json.loads((folder/'GLOBAL_FREEZE.json').read_text())
  for name,h in f['artifact_sha256'].items():assert sha(ROOT/name)==h,name
 dy,dp,dc=fit_arrays(ROOT/'models/r16_support_temporal_r1/D0F1');candidates=[]
 for arm in ARMS:
  folder=ROOT/'models/r16_clean_support_r1'/arm;y,p,c=fit_arrays(folder);assert np.array_equal(dy,y)
  for f in folder.glob('fit_*predictions.npz'):pins[str(f.relative_to(ROOT))]=sha(f)
  delta=p-dp;ws=float(np.clip(np.dot(delta,y-dp)/np.dot(delta,delta),0,1))
  values=[(float(roc_auc_score(y>0,(1-v)*dc+v*c)),float(v))for v in np.linspace(0,1,11)]
  wc=sorted(values,key=lambda x:(-x[0],x[1]))[0][1]
  policies=[('fit_only',ws,wc),('fixed_60D0_40S1',.4,.5),('fixed_50D0_50S1',.5,.5),('fixed_65D0_35S1',.35,.5)]
  for name,a,b in policies:
   candidates.append({'arm':arm,'policy':name,'ws':a,'wc':b,
      'fit_metrics':metric(y,(1-a)*dp+a*p,(1-b)*dc+b*c,np.ones(len(y)))})
 write(OUT/'POLICY_FROZEN.json',{'candidates':candidates,'source_sha256':pins,'locked_panels_used_for_weight_fit':False})
 panels={}
 for role in PRIMARY:
  ds=[]
  for op in OPS:
   name=role+'__'+op;old=load(BASE/'evaluation/predictions'/f'{name}.npz');new=load(CLEAN/'evaluation/predictions'/f'{name}.npz')
   assert np.array_equal(old['sample_id'],new['sample_id'])and np.array_equal(old['target_share'],new['target_share'])
   j=list(old['model_order']).index('D0F1')
   ds.append({'y':old['target_share'],'d0p':old['balanced_share'][:,j],'d0c':old['balanced_classifier'][:,j],
              'sp':new['balanced_share'],'sc':new['balanced_classifier'],'w':old['sample_weight']})
  panels[role]={k:np.concatenate([a[k]for a in ds])for k in ds[0]}
 for suffix in('hf_high_negative','st_francis_dense_negative'):
  name='native12_VS_'+suffix;old=load(BASE/'evaluation/predictions'/f'{name}.npz');new=load(CLEAN/'evaluation/predictions'/f'{name}.npz')
  assert np.array_equal(old['sample_id'],new['sample_id'])and np.array_equal(old['target_share'],new['target_share'])
  j=list(old['model_order']).index('D0F1')
  panels[name]={'y':old['target_share'],'d0p':old['balanced_share'][:,j],'d0c':old['balanced_classifier'][:,j],
               'sp':new['balanced_share'],'sc':new['balanced_classifier'],'w':old['sample_weight']}
 rows=[];gates=[]
 for cand in candidates:
  j=ARMS.index(cand['arm']);checks=[]
  for name,a in panels.items():
   p=(1-cand['ws'])*a['d0p']+cand['ws']*a['sp'][:,j];c=(1-cand['wc'])*a['d0c']+cand['wc']*a['sc'][:,j]
   m=metric(a['y'],p,c,a['w']);b=metric(a['y'],a['d0p'],a['d0c'],a['w'])
   rows.append({'arm':cand['arm'],'policy':cand['policy'],'panel':name,'ws':cand['ws'],'wc':cand['wc'],**m})
   if name in PRIMARY:
    checks.append({'name':name,'passed':m['rmse']<=b['rmse']+max(.005,.05*b['rmse'])and m['auc_classifier']>=b['auc_classifier']-.02,
                   'rmse':m['rmse'],'rmse_max':b['rmse']+max(.005,.05*b['rmse']),'auc':m['auc_classifier'],'auc_min':b['auc_classifier']-.02})
   else:
    limit=.11 if'hf_high'in name else .12
    checks.append({'name':name,'passed':m['rmse']<=limit and m['auc_classifier']>=.75,'rmse':m['rmse'],'rmse_max':limit,'auc':m['auc_classifier']})
    checks.append({'name':name+'_negative','passed':m['negative']['share_mean']<=b['negative']['share_mean']+.01 and m['negative']['classifier_mean']<=b['negative']['classifier_mean']+.02,
                    'share_mean':m['negative']['share_mean'],'share_max':b['negative']['share_mean']+.01,'score_mean':m['negative']['classifier_mean'],'score_max':b['negative']['classifier_mean']+.02})
  gates.append({'arm':cand['arm'],'policy':cand['policy'],'passed':all(a['passed']for a in checks),'checks':checks})
 write(OUT/'RESULTS.json',{'status':'completed','candidates':candidates,'rows':rows,'gates':gates,'any_gate_passed':any(g['passed']for g in gates),
                           'elapsed_seconds':time.perf_counter()-began,'new_inference':False,'new_training':False,'Test_used':False})
 for g in gates:print(json.dumps({'candidate':g['arm']+'_'+g['policy'],'passed':g['passed'],'failed':[c['name']for c in g['checks']if not c['passed']]},ensure_ascii=False))
 print(json.dumps({'elapsed_seconds':time.perf_counter()-began,'fit_weights':[(c['arm'],c['ws'],c['wc'])for c in candidates if c['policy']=='fit_only']}))

if __name__=='__main__':main()
