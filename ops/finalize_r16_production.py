"""Finalize completed R16 Tree5 + NN Fold0, then frozen evaluation/query.

No learner training. Policy weights use only the exact fixed local_fit arrays.
A completed model/policy receipt is verified before locked evaluation or query.
This script does not open query files on import, readiness, or synthetic QA.
"""
from pathlib import Path
import argparse,csv,datetime,gc,hashlib,html,json,os,sys,tempfile,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
KINDS=('bg_cnn','fdy','itransformer','tfgrid_fdy_conformer')
MANIFEST='reports/r16_cross_operator_r1/data/manifest.json'
TREE_REGISTRY='models/r16_production_trees_r1/ALL_FIVEFOLD_TREES_FROZEN.json'
NN_ROOT='models/r16_production_nn_fold0_r1'
OUTPUT='models/r16_production_final_r1'
SUBMISSION='submissions/r16_main_r1/submission_r16.csv'
PRIMARY_PYTHON='/Users/hongtaeyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
CODE_FILES=('ops/finalize_r16_production.py','ops/predict_r16_production_nn.py',
 'pipeline/ensemble_r16_production.py','pipeline/trees_r16_production.py',
 'pipeline/training_r16_production.py','pipeline/models_r16_production.py',
 'pipeline/r16_production_data.py','ops/r16_forward_data.py')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def safe(p):
 p=Path(p);p=(ROOT/p).resolve()if not p.is_absolute()else p.resolve()
 if not p.is_relative_to(ROOT):raise ValueError('Project-contained paths required')
 return p
def relative(p):return str(safe(p).relative_to(ROOT))
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
 q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def verify(pins):
 for n,h in pins.items():
  if sha(safe(n))!=h:raise ValueError('Frozen bytes changed: '+n)
def npz(path):
 with np.load(path,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def save_npz(path,**arrays):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb')as f:np.savez_compressed(f,**arrays)
 tmp.replace(path)
def prerequisites(nn_root=NN_ROOT,tree_registry=TREE_REGISTRY):
 """Completion checks happen before output creation, labels, or query access."""
 nn=safe(nn_root);tr=safe(tree_registry)
 if not tr.is_file()or not(nn/'ALL_COMPLETED.json').is_file():raise PermissionError('Completed Tree5 and all four 100epoch NN models required')
 t=read(tr);a=read(nn/'ALL_COMPLETED.json')
 if t.get('status')!='completed'or t.get('all_models_frozen')is not True or set(t.get('models',{}))!=set(map(str,range(5))):raise PermissionError('Exactly five completed Tree folds required')
 if a.get('status')!='completed'or a.get('fold')!=0 or a.get('models')!=list(KINDS)or(nn/'FAILED.json').exists():raise PermissionError('Completed clean NN Fold0 only')
 for k in KINDS:
  s=read(nn/k/'summary.json')
  if s.get('status')!='completed'or s.get('epochs_run')!=100 or s.get('kind')!=k:raise PermissionError('All architectures must finish 100 epochs')
 verify(t['artifact_sha256']);return nn,tr,t

def _checked_saved(d,ids,y,*,kind):
 if not np.array_equal(np.asarray(d['sample_id']).astype(str),ids):raise ValueError('Saved selection ID/order mismatch: '+kind)
 if 'target_share'in d and not np.array_equal(np.asarray(d['target_share'],float),y):raise ValueError('Saved selection labels changed: '+kind)
 if 'target_presence'in d and not np.array_equal(np.asarray(d['target_presence'],bool),y>0):raise ValueError('Saved selection presence labels changed: '+kind)
 return d

def selection_predictions(nn,tr):
 from pipeline.ensemble_r16_production import local_fit_contract
 from pipeline.training_r16_production import restore_completed_model
 from pipeline.trees_r16_production import PhysicsLGBM
 binding,ids,y=local_fit_contract();shares={};clf={};tree_p=[];tree_c=[]
 for f in range(5):
  folder=safe(tr['models'][str(f)]['model_dir']);m=PhysicsLGBM.load(folder)
  d=_checked_saved(npz(folder/'selection_predictions.npz'),ids,y,kind='tree'+str(f));tree_p.append(d['share']);tree_c.append(d['classifier_prob']);del m;gc.collect()
 shares['tree5']=np.mean(tree_p,axis=0);clf['tree5']=np.mean(tree_c,axis=0)
 for k in KINDS:
  m,_=restore_completed_model(nn/k,'cpu');del m;gc.collect()
  d=_checked_saved(npz(nn/k/'best_regression_selection_predictions.npz'),ids,y,kind=k)
  c=_checked_saved(npz(nn/k/'best_classifier_selection_predictions.npz'),ids,y,kind=k+' classifier')
  shares[k]=d['share'];clf[k]=c['classifier_prob']
 return shares,clf,y,ids,binding

def model_pins(nn,registry,tree_record,policy):
 pins={**tree_record['artifact_sha256'],relative(registry):sha(registry),relative(policy):sha(policy)}
 for f in CODE_FILES:pins[f]=sha(ROOT/f)
 for name in('execution_identity.json','ALL_COMPLETED.json'):pins[relative(nn/name)]=sha(nn/name)
 for k in KINDS:
  folder=nn/k;s=read(folder/'summary.json');pins[relative(folder/'summary.json')]=sha(folder/'summary.json')
  for name,h in s['artifact_sha256'].items():pins[relative(folder/name)]=h
  pins.update(s['source_code_sha256'])
 pins[MANIFEST]=sha(ROOT/MANIFEST);verify(pins);return pins

def _stats(a):
 a=np.asarray(a,float)
 return{'n':len(a),'mean':float(a.mean()),'std':float(a.std()),'rms':float(np.sqrt(np.mean(a*a))),
  'p01':float(np.quantile(a,.01)),'p10':float(np.quantile(a,.1)),'p25':float(np.quantile(a,.25)),
  'p50':float(np.quantile(a,.5)),'p75':float(np.quantile(a,.75)),'p90':float(np.quantile(a,.9)),
  'p99':float(np.quantile(a,.99)),'fraction_ge090':float(np.mean(a>=.9))}
def _metrics(y,p,c,site=None):
 from pipeline.training_r16_production import metrics
 d=metrics(y,p,c)
 if site is not None:
  site=np.asarray(site).astype(str);s={v:float(np.sqrt(np.mean((np.asarray(p)[site==v]-np.asarray(y)[site==v])**2)))for v in np.unique(site)}
  d.update(site_RMSE=s,site_macro_RMSE=float(np.mean(list(s.values()))))
 return d

def _frozen_predictions(db,nn,tree_record):
 from pipeline.training_r16_production import predict_completed_model
 from pipeline.trees_r16_production import PhysicsLGBM
 p=[];c=[];tree_parts={}
 for f in range(5):
  m=PhysicsLGBM.load(safe(tree_record['models'][str(f)]['model_dir']));q=m.predict(db)
  tree_parts[str(f)]=q;p.append(q['share']);c.append(q['classifier_prob']);del m;gc.collect()
 shares={'tree5':np.mean(p,axis=0)};cls={'tree5':np.mean(c,axis=0)}
 for k in KINDS:
  q=predict_completed_model(nn/k,db,batch_size=64,device='cpu')
  if any(not np.isfinite(a).all()or((a<0)|(a>1)).any()for a in q.values()):raise ValueError('Invalid frozen neural probabilities: '+k)
  if float(abs(q['share']-(q['cell_mask']*q['weights']).sum((1,2))).max())>3e-6:raise ValueError('Frozen physical pooling differs: '+k)
  shares[k]=q['share'];cls[k]=q['score'];del q;gc.collect()
 return shares,cls,tree_parts

def write_submission(path,ids,pred):
 path=safe(path)
 if path.exists():raise FileExistsError('Never overwrite an existing submission candidate')
 ids=np.asarray(ids).astype(str);p=np.asarray(pred['share']);c=np.asarray(pred['score'])
 if p.shape!=(len(ids),)or c.shape!=p.shape or len(set(ids))!=len(ids)or not np.isfinite(p).all()or not np.isfinite(c).all()or((p<0)|(p>1)|(c<0)|(c>1)).any():raise ValueError('Invalid frozen submission predictions')
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('x',newline='')as f:
  w=csv.writer(f);w.writerow(['ID','share_added_vessel','score_added_vessel'])
  for i,a,b in zip(ids,p,c):w.writerow([i,format(float(a),'.17g'),format(float(b),'.17g')])
 with tmp.open(newline='')as f:
  rows=list(csv.reader(f))
 if rows[0]!=['ID','share_added_vessel','score_added_vessel']or [r[0]for r in rows[1:]]!=ids.tolist():raise ValueError('CSV schema/ID roundtrip differs')
 if not np.array_equal(np.asarray([float(r[1])for r in rows[1:]]),p.astype(float))or not np.array_equal(np.asarray([float(r[2])for r in rows[1:]]),c.astype(float)):raise ValueError('CSV numeric roundtrip differs')
 tmp.replace(path);return{'path':relative(path),'sha256':sha(path),'rows':len(ids),'ID_order_preserved':True,'numeric_roundtrip_exact':True,'official_upload':False}

def descriptive_comparison(ids,pred):
 """Old prediction files only, after the policy is frozen; no model selection."""
 from scipy.stats import spearmanr
 result={}
 for label,name in [('R9_Fold0_all5','submissions/r9_fold0_all5_cpu_r1/submission.csv'),('R14_step1','submissions/r14_main_r2/submission_r14_step1.csv')]:
  path=ROOT/name
  if not path.exists():result[label]={'status':'unavailable'};continue
  with path.open(newline='')as f:rows=list(csv.DictReader(f))
  if [v['ID']for v in rows]!=np.asarray(ids).astype(str).tolist():raise ValueError('Historical prediction IDs differ')
  out={'file_sha256':sha(path),'path':name}
  for key,col in [('share','share_added_vessel'),('score','score_added_vessel')]:
   old=np.asarray([float(v[col])for v in rows]);new=np.asarray(pred[key],float)
   out[key]={'previous':_stats(old),'R16_minus_previous_mean':float((new-old).mean()),'mean_absolute_difference':float(abs(new-old).mean()),
    'Pearson':float(np.corrcoef(new,old)[0,1])if new.std()>0 and old.std()>0 else None,
    'Spearman':float(spearmanr(new,old).statistic)if new.std()>0 and old.std()>0 else None}
  result[label]=out
 return result

def render_completed_report(out,policy,rows,audit):
 """Human-readable derived report; never affects model/policy selection."""
 from pipeline.ensemble_r16_production import MEMBERS
 headline=[v for v in rows if v['operator']=='all'and v['model']=='ensemble']
 table='| 평가 | N | RMSE | MAE | Bias | 독립 score AUC | Site-macro RMSE |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for r in headline:
  table+='| '+r['role']+' | '+str(r['n'])+' | '+' | '.join('—'if r.get(k)is None else f'{r[k]:.6f}'for k in('rmse','mae','bias','classifier_auc','site_macro_RMSE'))+' |\n'
 weight='| 모델군 | share 가중치 | 독립 분류 가중치 |\n|---|---:|---:|\n'
 for k,w,c in zip(MEMBERS,policy['regression_weights'],policy['classifier_weights']):weight+=f'| {k} | {w:.6f} | {c:.6f} |\n'
 csvlink=os.path.relpath(safe(audit['submission']['path']),out)
 summary=('시행 범위: PhysicsLightGBM 회귀·분류 5-Fold와 신경망 4종의 Fold 0, 각 100 epoch입니다. '
  '신경망 전체 5-Fold나 최종 hybrid OOF가 아닙니다. local_fit은 checkpoint·앙상블 선택용 개발 자료이며 성능 추정치로 해석하지 않습니다. '
  '고정 교차 검증은 별도 참조·크롭, 미관측 사이트, 독립 저·고주파 채널 변형을 우선하며 R11의 인위적 장시간 스트레스만으로 레시피를 바꾸지 않았습니다.')
 text='# R16 본체 Fold 0 + Tree 5-Fold 실행 결과\n\n'+summary+'\n\n## 고정 교차 검증\n\n'+table+'\n## 동결 앙상블\n\n'+weight
 text+='\n회귀 Tree 비중은 최소 50%, 개별 신경망은 최대 25%입니다. score의 share 비중은 0이며 독립 분류 확률만 사용합니다.\n\n'
 text+='## 제출 후보와 예측 기술 통계\n\n[submission_r16.csv]('+csvlink+')\n\n'
 for k,v in audit['ensemble'].items():text+=f"- {k}: 평균 {v['mean']:.6f}, RMS {v['rms']:.6f}, 중앙값 {v['p50']:.6f}, p90 {v['p90']:.6f}\n"
 text+='\n이 통계는 동결 후 생성한 예측값의 설명입니다. Test 라벨을 알 수 없으므로 공식 RMSE·AUC나 간극 해소를 보장하지 않습니다. 공식 업로드는 수행하지 않았습니다.\n\n'
 text+='## 상세 증거\n\n- [전체 모델·연산자별 교차 평가](CROSS_VALIDATION_SUMMARY.json)\n- [R9/R14 대비 예측 분포·상관](TEST_PREDICTION_AUDIT.json)\n- [동결 정책](policy.json)\n- [모델·정책 동결](GLOBAL_MODEL_POLICY_FREEZE.json)\n- [고정 R11 소형 스트레스 진단](../../reports/r16_r11_stress_r1/SUMMARY.md)\n'
 md=out/'R16_PRODUCTION_RESULTS.md';md.write_text(text)
 h='<html lang="ko"><meta charset="utf-8"><title>R16 본체 실행 결과</title><style>body{font:16px system-ui;max-width:1100px;margin:36px auto;padding:0 24px;color:#17252a;background:#f7faf9}pre{white-space:pre-wrap;background:white;padding:24px;line-height:1.6}a{color:#086b58}</style><h1>R16 본체 실행 결과</h1><p><a href="'+html.escape(csvlink,quote=True)+'">submission_r16.csv 내려받기</a> · <a href="CROSS_VALIDATION_SUMMARY.json">전체 교차 평가</a> · <a href="TEST_PREDICTION_AUDIT.json">예측 분포 감사</a></p><pre>'+html.escape(text)+'</pre></html>'
 hp=out/'index.html';hp.write_text(h)
 return {relative(p):sha(p)for p in(md,hp)}

def finalize(output=OUTPUT,nn_root=NN_ROOT,tree_registry=TREE_REGISTRY,submission=SUBMISSION,expected_ready=None):
 from ops.predict_r16_production_nn import global_freeze,guard_state,read_raw_csv,OPERATORS
 from pipeline.ensemble_r16_production import fit_policy,apply_policy
 from pipeline.training_r16_production import atomic_json
 import torch
 state,guard=guard_state();sys.addaudithook(guard);torch.set_num_threads(2);torch.set_num_interop_threads(1)
 if expected_ready:
  ready=read(safe(expected_ready))
  if ready.get('status')!='ready'or ready.get('official_upload')is not False:raise PermissionError('Valid finalizer ready contract required')
  verify(ready['artifact_sha256'])
  if ready['script_sha256']!=sha(__file__):raise ValueError('Finalizer differs from launch contract')
 nn,registry,trees=prerequisites(nn_root,tree_registry);out=safe(output)
 if out.exists():raise FileExistsError('Fresh finalization folder required; failed partials are never auto reused')
 if safe(submission).exists():raise FileExistsError('Fresh submission output required')
 out.mkdir(parents=True);began=time.perf_counter();freeze=None
 try:
  write(out/'progress.json',{'status':'fitting_policy','role':'local_fit only'})
  shares,clf,y,ids,binding=selection_predictions(nn,trees)
  policy=fit_policy(shares,clf,y,sample_ids=ids,binding=binding);policy_path=out/'policy.json';write(policy_path,policy)
  save_npz(out/'selection_predictions.npz',sample_id=ids,target_share=y,**{'share__'+k:v for k,v in shares.items()},**{'classifier__'+k:v for k,v in clf.items()})
  pins=model_pins(nn,registry,trees,policy_path);pins[relative(out/'selection_predictions.npz')]=sha(out/'selection_predictions.npz')
  freeze=out/'GLOBAL_MODEL_POLICY_FREEZE.json'
  certificate={'status':'completed','all_models_frozen':True,'ensemble_policy_frozen':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
   'data_manifest_sha256':sha(ROOT/MANIFEST),'production_data_manifest_sha256':read(nn/'execution_identity.json')['data_manifest_sha256'],
   'artifact_sha256':pins,'nn_models':{k:relative(nn/k)for k in KINDS},'ensemble_policy_path':relative(policy_path),
   'tree_registry_path':relative(registry),'query_source_allowlist':{},'policy_fit_role':'local_fit exact2400 only',
   'Tree_folds':[0,1,2,3,4],'NN_folds':[0],'full_hybrid_5fold_OOF':False,'Test_read_before_this_freeze':False,'official_upload':False}
  write(freeze,certificate);global_freeze(freeze,ROOT/MANIFEST,nn)
  final_fit=apply_policy(policy,shares,clf);metrics_rows=[{'role':'local_fit','operator':'all','model':'ensemble',**_metrics(y,final_fit['share'],final_fit['score'])}]
  from ops.r16_forward_data import load_panel
  for role in('local_gate','local_report','domain_holdout'):
   global_freeze(freeze,ROOT/MANIFEST,nn);acc={k:[]for k in('share','score','y','site')}
   for op in OPERATORS:
    write(out/'progress.json',{'status':'frozen_evaluation','role':role,'operator':op})
    data=load_panel(ROOT/MANIFEST,role,op,freeze_receipt=freeze);s,c,tp=_frozen_predictions(data['db'],nn,trees);e=apply_policy(policy,s,c)
    payload={'sample_id':data['sample_id'],'target_share':data['share'],**{'share__'+k:v for k,v in s.items()},**{'classifier__'+k:v for k,v in c.items()},'share':e['share'],'score':e['score']}
    save_npz(out/'evaluation'/f'{role}__{op}.npz',**payload)
    for name in s:metrics_rows.append({'role':role,'operator':op,'model':name,**_metrics(data['share'],s[name],c[name],data['site'])})
    metrics_rows.append({'role':role,'operator':op,'model':'ensemble',**_metrics(data['share'],e['share'],e['score'],data['site'])})
    for k,v in [('share',e['share']),('score',e['score']),('y',data['share']),('site',data['site'])]:acc[k].append(v)
    print(json.dumps({'event':'frozen_panel_completed','role':role,'operator':op,'ensemble_metrics':metrics_rows[-1]}),flush=True)
   v={k:np.concatenate(x)for k,x in acc.items()};metrics_rows.append({'role':role,'operator':'all','model':'ensemble',**_metrics(v['y'],v['share'],v['score'],v['site'])})
  write(out/'CROSS_VALIDATION_SUMMARY.json',{'status':'completed','metrics':metrics_rows,'primary_axes':['separate_reference_crop','unseen_site','independent_LF_HF_channel'],
   'R11_small_fixed_diagnostic':'reports/r16_r11_stress_r1/RESULTS.json','full_hybrid_5fold_OOF':False,'fit_or_reselection_after_global_freeze':False})
  # Only now, after completed-model/policy verification, permit raw query bytes.
  global_freeze(freeze,ROOT/MANIFEST,nn);query=safe('data/raw/Test.csv');state['allowed_query']=query
  query_sha=sha(query);qfreeze=out/'GLOBAL_QUERY_ALLOWLIST_FREEZE.json'
  qcertificate={**certificate,'query_source_allowlist':{relative(query):query_sha},'prior_model_policy_freeze_path':relative(freeze),'prior_model_policy_freeze_sha256':sha(freeze)}
  write(qfreeze,qcertificate);global_freeze(qfreeze,ROOT/MANIFEST,nn)
  qids,db,qreceipt=read_raw_csv(query);write(out/'progress.json',{'status':'frozen_query_inference','rows':len(qids)})
  s,c,tp=_frozen_predictions(db,nn,trees);e=apply_policy(policy,s,c)
  save_npz(out/'query_predictions.npz',sample_id=qids,**{'share__'+k:v for k,v in s.items()},**{'classifier__'+k:v for k,v in c.items()},**e)
  csvreceipt=write_submission(submission,qids,e)
  audit={'status':'completed','model_stats':{k:{'share':_stats(s[k]),'independent_classifier':_stats(c[k])}for k in s},
   'ensemble':{k:_stats(e[k])for k in e},'historical_comparison':descriptive_comparison(qids,e),
   'query_receipt':qreceipt,'submission':csvreceipt,'Test_labels_available':False,'policy_reselected_after_query':False}
  write(out/'TEST_PREDICTION_AUDIT.json',audit);global_freeze(qfreeze,ROOT/MANIFEST,nn)
  report_pins=render_completed_report(out,policy,metrics_rows,audit)
  result={'status':'completed','submission':csvreceipt,'policy':relative(policy_path),'policy_sha256':sha(policy_path),'report_sha256':report_pins,
   'freeze':relative(qfreeze),'freeze_sha256':sha(qfreeze),'elapsed_seconds':time.perf_counter()-began,'official_upload':False,
   'full_hybrid_5fold_OOF':False,'Tree_folds':list(range(5)),'NN_folds':[0]}
  write(out/'COMPLETED.json',result);write(out/'progress.json',result);return result
 except BaseException as e:
  write(out/'FAILED.json',{'status':'failed','error':repr(e),'no_automatic_partial_reuse':True,'model_policy_frozen':freeze is not None});raise

def fixture(output):
 from pipeline.ensemble_r16_production import fixture as ensemble_fixture
 out=safe(output)
 if out.exists():raise FileExistsError('Fresh fixture receipt required')
 numeric=ensemble_fixture();rng=np.random.default_rng(16);ids=np.asarray(['fixture_'+str(i)for i in range(8)])
 with tempfile.TemporaryDirectory(dir='/private/tmp',prefix='r16-finalizer-')as temp:
  # CSV checks do not require models or query data. Keep test output project-contained.
  path=ROOT/'reports'/('.r16_finalizer_csv_fixture_'+str(os.getpid())+'.csv')
  try:r=write_submission(path,ids,{'share':rng.random(8).astype(np.float32),'score':rng.random(8).astype(np.float32)})
  finally:
   if path.exists():path.unlink()
 denied=False
 try:prerequisites('models/__r16_fixture_absent_nn__',TREE_REGISTRY)
 except PermissionError:denied=True
 if not denied:raise AssertionError('Missing completed models were accepted')
 result={'status':'passed','ensemble_fixture':numeric,'CSV_roundtrip_exact':r['numeric_roundtrip_exact'],
  'incomplete_models_rejected_before_labels_query_outputs':denied,'actual_completed_R16_inference':'not_measured_until_NN100_all4_complete',
  'Test_opened':False,'learner_training':False,'script_sha256':sha(__file__)}
 write(out,result);return result

def ready(output,launch_receipt=None):
 out=safe(output);pins={n:sha(ROOT/n)for n in CODE_FILES}
 target=safe(launch_receipt)if launch_receipt else out
 command=[PRIMARY_PYTHON,'ops/finalize_r16_production.py','run','--output',OUTPUT,'--nn-root',NN_ROOT,'--tree-registry',TREE_REGISTRY,'--submission',SUBMISSION,'--expected-ready',relative(target)]
 result={'status':'ready','command':command,'cwd':str(ROOT),'script_sha256':sha(__file__),'artifact_sha256':pins,
  'environment':{'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'.:/private/tmp/underwater-ml-deps:/private/tmp/underwater-eda-deps','DYLD_LIBRARY_PATH':'/opt/homebrew/opt/libomp/lib','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'},
  'minimum_prerequisites':'Tree5 frozen plus NN4 Fold0 each100epochs ALL_COMPLETED; incomplete models rejected',
  'no_actual_prediction_policyfit_or_Test_access_performed_by_ready':True,'official_upload':False}
 write(out,result);return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['fixture','ready','run']);p.add_argument('--output',default=OUTPUT)
 p.add_argument('--nn-root',default=NN_ROOT);p.add_argument('--tree-registry',default=TREE_REGISTRY);p.add_argument('--submission',default=SUBMISSION);p.add_argument('--expected-ready');a=p.parse_args()
 if a.stage=='fixture':result=fixture(a.output)
 elif a.stage=='ready':result=ready(a.output,a.expected_ready)
 else:result=finalize(a.output,a.nn_root,a.tree_registry,a.submission,a.expected_ready)
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
