"""Freeze both expert policies, evaluate, then create one local candidate CSV.

Never trains, uses Test labels, fits on locked panels, or uploads a file.
"""
from pathlib import Path
import csv,gc,html,json,sys,time
import numpy as np
from pipeline.r16_multisupport_data import ROOT,OUT,REGIMES,KINDS,PARENT,read,write,sha,save,load,role
from pipeline.training_r16_multisupport import MODELS,codes,verify,features,tree_features,metric

REPORT=ROOT/'reports/r16_multisupport_full_r1'
FINAL=MODELS/'final'
MEMBERS=('tree5',*KINDS)
def aligned(data,ids,y):
 if not np.array_equal(data['sample_id'].astype(str),ids.astype(str))or not np.array_equal(data['target_share'],y):raise ValueError('Saved fit IDs/labels differ')
 return data
def fit_policy(x,c,y):
 from scipy.optimize import minimize
 from sklearn.metrics import roc_auc_score
 from pipeline.ensemble_r16_production import CLASSIFIER_CANDIDATES
 prior=np.asarray([.5,.125,.125,.125,.125]);ridge=.0001
 result=minimize(lambda w:np.mean((x@w-y)**2)+ridge*np.sum((w-prior)**2),prior,
  jac=lambda w:2*x.T@(x@w-y)/len(y)+2*ridge*(w-prior),method='SLSQP',bounds=[(.5,1.)]+[(0.,.25)]*4,
  constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(5)}],options={'ftol':1e-12,'maxiter':1000})
 if not result.success:raise ValueError('Convex capped policy fit failed')
 aucs={k:float(roc_auc_score(y>0,c@np.asarray(w)))for k,w in CLASSIFIER_CANDIDATES.items()};best=max(aucs,key=lambda k:(aucs[k],CLASSIFIER_CANDIDATES[k][0]))
 w=result.x;cw=np.asarray(CLASSIFIER_CANDIDATES[best])
 if w[0]<.5-1e-8 or np.any(w[1:]>.25+1e-8)or abs(w.sum()-1)>1e-8:raise ValueError('Cap contract failed')
 return {'regression_weights':w.tolist(),'independent_classifier_weights':cw.tolist(),'classifier_candidate':best,'classifier_fit_AUC_candidates':aucs,'score_share_weight':0.,'ridge_to_prior':ridge,'fit_metrics':metric(y,x@w,c@cw)}
def apply(policy,shares,probs):
 p=np.zeros(len(shares['D0']));c=p.copy()
 for regime in REGIMES:
  p+=.5*(shares[regime]@np.asarray(policy['experts'][regime]['regression_weights']))
  c+=.5*(probs[regime]@np.asarray(policy['experts'][regime]['independent_classifier_weights']))
 return {'share':np.clip(p,0,1),'score':np.clip(c,0,1)}
class FrozenPredictor:
 def __init__(self):
  import lightgbm as lgb
  self.trees={}
  for reg in REGIMES:
   self.trees[reg]=[(lgb.Booster(model_file=str(MODELS/'trees'/reg/f'fold_{f:02d}'/'regression.txt')),
                    lgb.Booster(model_file=str(MODELS/'trees'/reg/f'fold_{f:02d}'/'classification.txt')))for f in range(5)]
 def neural(self,reg,kind,x,w):
  import torch
  from pipeline.models_r16_multisupport import MultiSupportModel
  folder=MODELS/'neural'/reg/kind;model=MultiSupportModel(kind).eval();r=torch.load(folder/'best_regression.pt',map_location='cpu',weights_only=True)
  if r['architecture']!=model.architecture()or r['code_sha256']!=codes():raise ValueError('Neural architecture/source mismatch')
  model.load_state_dict(r['state_dict']);p=[];c=[]
  with torch.no_grad():
   for off in range(0,len(x),64):
    xx=torch.from_numpy(x[off:off+64]);ww=torch.from_numpy(w[off:off+64]);pp,_,q=model(xx,ww)
    if not torch.allclose(pp,(q*ww).sum((1,2)),atol=1e-7,rtol=0):raise ValueError('Inference pooling mismatch')
    p.append(pp.numpy())
  cl=torch.load(folder/'best_classifier.pt',map_location='cpu',weights_only=True)
  if cl['architecture']!=model.architecture():raise ValueError('Independent classifier architecture differs')
  model.classifier.load_state_dict({k.removeprefix('classifier.'):v for k,v in cl['state_dict'].items()if k.startswith('classifier.')})
  with torch.no_grad():
   for off in range(0,len(x),128):c.append(model.classifier(torch.from_numpy(x[off:off+128])).sigmoid().numpy())
  del model,r,cl;gc.collect();return np.concatenate(p),np.concatenate(c)
 def predict(self,db):
  x,w=features(db);tr=tree_features(db,'regression');tc=tree_features(db,'classification');shares={};probs={}
  for reg in REGIMES:
   ps=[np.mean([np.clip(a.predict(tr,num_threads=2),0,1)for a,b in self.trees[reg]],axis=0)]
   cs=[np.mean([b.predict(tc,num_threads=2)for a,b in self.trees[reg]],axis=0)]
   for kind in KINDS:
    p,c=self.neural(reg,kind,x,w);ps.append(p);cs.append(c)
   shares[reg]=np.column_stack(ps);probs[reg]=np.column_stack(cs)
  if any(not np.isfinite(a).all()or((a<0)|(a>1)).any()for a in[*shares.values(),*probs.values()]):raise ValueError('Invalid frozen prediction')
  return shares,probs
def prerequisites():
 pins=codes();verify(pins)
 for n in('ops/finalize_r16_multisupport_full.py','ops/predict_r16_production_nn.py','ops/finalize_r16_production.py','pipeline/ensemble_r16_production.py'):
  pins[n]=sha(ROOT/n)
 for p in(MODELS/'ALL_TREES_COMPLETED.json',MODELS/'ALL_NEURAL_COMPLETED.json'):
  d=read(p)
  if d['status']!='completed'or d['code_sha256']!=codes()or d['data_manifest_sha256']!=sha(OUT/'manifest.json'):raise ValueError('Complete same-version models required')
  pins[str(p.relative_to(ROOT))]=sha(p)
 for typ,reg,names in [(t,r,[f'fold_{f:02d}'for f in range(5)]if t=='trees'else KINDS)for t in('trees','neural')for r in REGIMES]:
  for name in names:
   folder=MODELS/typ/reg/name;s=read(folder/'summary.json')
   if s['status']!='completed'or typ=='neural'and s['epochs']!=100:raise ValueError('Incomplete full model')
   for k,h in s['artifact_sha256'].items():
    if sha(folder/k)!=h:raise ValueError('Model bytes changed')
    pins[str((folder/k).relative_to(ROOT))]=h
   pins[str((folder/'summary.json').relative_to(ROOT))]=sha(folder/'summary.json')
 return pins
def finalize():
 import torch
 from ops.predict_r16_production_nn import guard_state,read_raw_csv
 from ops.finalize_r16_production import write_submission,_stats,descriptive_comparison
 from ops.r16_support_temporal_data import load_panel,load_long_panel
 from ops.r16_forward_data import OPERATORS
 state,audit_guard=guard_state();sys.addaudithook(audit_guard);torch.set_num_threads(2)
 pins=prerequisites()
 if FINAL.exists():raise FileExistsError('Fresh finalization only')
 FINAL.mkdir();REPORT.mkdir(exist_ok=True,parents=True);start=time.perf_counter();fit=role(0,'D0','selection');y=fit['share'];ids=fit['sample_id'];ps={};cs={};policies={}
 for regime in REGIMES:
  trees=[aligned(load(MODELS/'trees'/regime/f'fold_{f:02d}'/'selection.npz'),ids,y)for f in range(5)]
  a=[np.mean([d['share']for d in trees],0)];b=[np.mean([d['classifier_prob']for d in trees],0)]
  for kind in KINDS:
   folder=MODELS/'neural'/regime/kind
   a.append(aligned(load(folder/'selection_regression.npz'),ids,y)['share']);b.append(aligned(load(folder/'selection_classification.npz'),ids,y)['classifier_prob'])
  ps[regime]=np.column_stack(a);cs[regime]=np.column_stack(b);policies[regime]=fit_policy(ps[regime],cs[regime],y)
 policy={'format':'r16_multisupport_50_50_capped_independent_v1','experts':policies,'regimes':list(REGIMES),'member_order':list(MEMBERS),
  'regime_regression_weights':[.5,.5],'regime_classifier_weights':[.5,.5],'fit_role':'local_fit only, exposed development2400/unique400 backgrounds',
  'fit_data_sha256':read(OUT/'manifest.json')['selection']['sha256'],'disclosed_pilot_gate_miss':{'native12_HF_RMSE':.1175534730485378,'criterion':.11,'user_authorized_proceed':True},
  'postprocessing':'clip to [0,1] only; no Test distribution calibration','Test_used_for_policy':False}
 write(FINAL/'policy.json',policy);pins[str((FINAL/'policy.json').relative_to(ROOT))]=sha(FINAL/'policy.json')
 write(FINAL/'MODEL_REGISTRY.json',{'status':'completed','artifact_sha256':pins});registry=FINAL/'MODEL_REGISTRY.json'
 freeze=FINAL/'GLOBAL_FREEZE.json';write(freeze,{'status':'completed','all_models_frozen':True,'ensemble_policy_frozen':True,'data_manifest_sha256':sha(PARENT),
  'artifact_sha256':{str(registry.relative_to(ROOT)):sha(registry),str((FINAL/'policy.json').relative_to(ROOT)):sha(FINAL/'policy.json')},'Test_used':False})
 verify(pins);pred=FrozenPredictor();rows=[]
 rows.append({'panel':'local_fit','model':'blend','development_only':True,**metric(y,**{'p':apply(policy,ps,cs)['share'],'c':apply(policy,ps,cs)['score']})})
 def one(name,d,weights=None):
  s,c=pred.predict(d['db']);e=apply(policy,s,c);payload={'sample_id':d['sample_id'],'target_share':d['share'],'share':e['share'],'score':e['score'],'weights':np.ones(len(d['share']))if weights is None else weights}
  for reg in REGIMES:
   payload['share_'+reg]=s[reg];payload['independent_classifier_'+reg]=c[reg]
   a=policy['experts'][reg];p=s[reg]@a['regression_weights'];cp=c[reg]@a['independent_classifier_weights']
   rows.append({'panel':name,'model':reg,**metric(d['share'],p,cp,weights)})
   for j,k in enumerate(MEMBERS):rows.append({'panel':name,'model':reg+'/'+k,**metric(d['share'],s[reg][:,j],c[reg][:,j],weights)})
  if 'site'in d:payload['site']=d['site']
  save(FINAL/'evaluation'/f'{name}.npz',payload);row={'panel':name,'model':'blend',**metric(d['share'],e['share'],e['score'],weights)}
  if 'site'in d:
   row['site_metrics']={str(site):metric(d['share'][d['site']==site],e['share'][d['site']==site],e['score'][d['site']==site])for site in np.unique(d['site'])}
   row['site_macro_RMSE']=float(np.mean([a['rmse']for a in row['site_metrics'].values()]))
  for label,mask in [('negative',d['share']==0),('positive',d['share']>0)]:
   if mask.any():row[label]=metric(d['share'][mask],e['share'][mask],e['score'][mask])
  rows.append(row);write(FINAL/'progress.json',{'stage':'frozen_evaluation','panel':name,'metrics':row,'elapsed_seconds':time.perf_counter()-start});print(json.dumps(row),flush=True)
  return payload
 for name in('local_gate','local_report','domain_holdout'):
  parts=[load_panel(PARENT,name,op,freeze_receipt=freeze)for op in OPERATORS];keys=set.intersection(*(set(p)for p in parts));d={k:np.concatenate([p[k]for p in parts])for k in keys}
  payload=one(name,d)
  for j,op in enumerate(OPERATORS):
   a=sum(len(p['share'])for p in parts[:j]);b=a+len(parts[j]['share']);rows.append({'panel':name+'__'+op,'model':'blend',**metric(d['share'][a:b],payload['share'][a:b],payload['score'][a:b])})
 positive=[load_long_panel(PARENT,'native12_positive',operator=op,freeze_receipt=freeze)for op in OPERATORS];keys=set.intersection(*(set(d)for d in positive));positive={k:np.concatenate([d[k]for d in positive])for k in keys}
 for name in('hf_high_negative','st_francis_dense_negative'):
  negative=load_long_panel(PARENT,name,freeze_receipt=freeze);keys=set(positive)&set(negative);d={k:np.concatenate([positive[k],negative[k]])for k in keys}
  weights=np.r_[np.full(len(positive['share']),.5/len(positive['share'])),np.full(len(negative['share']),.5/len(negative['share']))];one('native12_VS_'+name,d,weights)
 # Own-fold Tree CV uses only the corresponding fold model. A Tree5 mean on
 # these rows would include models trained on them and is never called OOF.
 cvrows=[]
 for reg in REGIMES:
  ys=[];pps=[];ccs=[]
  for f in range(5):
   d=role(f,reg,'cv');a,b=pred.trees[reg][f];p=np.clip(a.predict(tree_features(d['db'],'regression'),num_threads=2),0,1);c=b.predict(tree_features(d['db'],'classification'),num_threads=2)
   ys.append(d['share']);pps.append(p);ccs.append(c);cvrows.append({'regime':reg,'fold':f,**metric(d['share'],p,c)})
  cvrows.append({'regime':reg,'fold':'pooled_ownfold_only',**metric(np.concatenate(ys),np.concatenate(pps),np.concatenate(ccs))})
 write(FINAL/'EVALUATION.json',{'status':'completed','rows':rows,'tree_ownfold_CV':cvrows,'hybrid_all5fold_OOF':False,'primary_unique_backgrounds_per_pooled_panel':400,'selection_or_recipe_changed_after_freeze':False})
 verify(pins)
 query=(ROOT/'data/raw/Test.csv').resolve();state['allowed_query']=query
 qids,db,qreceipt=read_raw_csv(query);s,c=pred.predict(db);e=apply(policy,s,c)
 path=ROOT/'submissions/r16_multisupport_full_r1/submission_r16.csv';receipt=write_submission(path,qids,e)
 save(FINAL/'query_predictions.npz',{'sample_id':qids,**e,**{'share_'+r:s[r]for r in REGIMES},**{'classifier_'+r:c[r]for r in REGIMES}})
 stats={'ensemble':{k:_stats(v)for k,v in e.items()},'historical_comparison':descriptive_comparison(qids,e),'query_receipt':qreceipt,'submission':receipt,
  'expert_models':{r:{name:{'share':_stats(s[r][:,j]),'independent_classifier':_stats(c[r][:,j])}for j,name in enumerate(MEMBERS)}for r in REGIMES},'Test_labels_used':False,'policy_changed_after_query':False}
 write(FINAL/'TEST_PREDICTION_AUDIT.json',stats);verify(pins)
 text=['# R16 D0F1+H0 본체 50:50 결과','',
  '장시간/HF 소형 결합 RMSE 0.117553이 기준0.11에 미달했음을 명시한 채 진행하도록 사용자가 승인한 실행이다. 각 조건 Tree5-Fold와 신경망4종 Fold0(각100epoch/Adam/noES)를 새로 학습했다. 기존8채널 체크포인트를 재사용하지 않았다.','',
  'A′75클립/13컴포넌트 내 원천분리, B19/C′25 유지, 배경은 기존R9 Fold0 TRAIN78869개 안에서 nested5-Fold/900초support+24hpurge를 계승한다. 전체509개 독립CV나 신경망전체5-Fold/최종hybrid OOF가 아니다. 장시간 native≥15 컴포넌트는 하나이고 Tree fold4에서는 TRAIN에서 홀드아웃된다.','',
  '## 고정 교차 평가','', '| 패널 | RMSE | 독립 score AUC | Bias |','|---|---:|---:|---:|']
 for row in rows:
  if row['model']=='blend'and'__'not in row['panel']:text.append(f"| {row['panel']} | {row['rmse']:.6f} | {row['classifier_auc']:.6f} | {row['bias']:.6f} |")
 text+=['','## 고정 결합','', 'D0:H0 회귀·분류 비율은 각각50:50이다. 아래 조건 내 비율만 local_fit2400건에서 정했다. 회귀 Tree≥50%, 각 신경망 아키텍처≤25%; score에는 share 또는 마스크contrast가 들어가지 않는다.','', '| 조건 | 모델군 | 회귀 비중 | 독립 분류 비중 |','|---|---|---:|---:|']
 for reg in REGIMES:
  a=policy['experts'][reg]
  for k,w,cw in zip(MEMBERS,a['regression_weights'],a['independent_classifier_weights']):text.append(f'| {reg} | {k} | {w:.6f} | {cw:.6f} |')
 text+=['','## 후보 CSV','', '[submission_r16.csv](../../submissions/r16_multisupport_full_r1/submission_r16.csv)','',
  '| 출력 | 평균 | RMS | 중앙값 | p90 |','|---|---:|---:|---:|---:|']
 for k,a in stats['ensemble'].items():text.append(f"| {k} | {a['mean']:.6f} | {a['rms']:.6f} | {a['p50']:.6f} | {a['p90']:.6f} |")
 text+=['','후처리는[0,1]clip만 적용했다. 동결 후 Test1500개를 추론했으며, Test 통계에 맞춘 재보정·공식 업로드는 하지 않았다. 공식 RMSE/AUC 및 이전 Public0.782464 초과 여부는 미측정이다.','',
  '[전체 평가](../../models/r16_multisupport_full_r1/final/EVALUATION.json) · [예측 분포/R9·R14 비교](../../models/r16_multisupport_full_r1/final/TEST_PREDICTION_AUDIT.json) · [동결 정책](../../models/r16_multisupport_full_r1/final/policy.json)']
 (REPORT/'R16_MULTISUPPORT_FULL_RESULTS.md').write_text('\n'.join(text)+'\n')
 (REPORT/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>R16 본체 결합 결과</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:36px auto;padding:24px}pre{white-space:pre-wrap}a{color:#1259a7}</style><h1>R16 본체 결합 결과</h1><p><a href="../../submissions/r16_multisupport_full_r1/submission_r16.csv">후보 CSV</a> · <a href="R16_MULTISUPPORT_FULL_RESULTS.md">전체 MD</a></p><pre>'+html.escape('\n'.join(text))+'</pre></html>')
 write(FINAL/'COMPLETED.json',{'status':'completed','submission':receipt,'elapsed_seconds':time.perf_counter()-start,'official_upload':False,'policy_sha256':sha(FINAL/'policy.json'),'report':str((REPORT/'index.html').relative_to(ROOT))})
