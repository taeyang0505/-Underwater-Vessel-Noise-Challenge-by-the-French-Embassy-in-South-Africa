"""R21 fixed, CPU-only physical-background feature candidates.

No predictive model training, Test access, model-weight changes or new feature
columns in a frozen LightGBM. TRAIN-only statistical coefficients and
deterministic within-window robust summaries are explicitly distinguished.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import datetime as dt, hashlib, json, sys, warnings
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/r21_clean_val_zero_training_r1/quiet'
R20=ROOT/'reports/r20_integrated_cpu_diagnostics_r1/quiet_features'
FINAL=ROOT/'reports/r18_production_r1/finalization'
CANDIDATES=('R20_commonK3_control','bandwise_lowK5_TRAIN_ratio','timevarying_median5_signed','robust_linear_db_signed')
PINS={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic)else str(x))+'\n')
def pin(p):
 p=Path(p);PINS[str(p.relative_to(ROOT))]=sha(p);return p
def guard(event,args):
 if event=='open'and args and isinstance(args[0],(str,bytes,os.PathLike)):
  n=Path(os.fsdecode(args[0])).name.lower()
  if n in ('test.csv','query.npz','query_predictions.npz')or'submission'in n:raise PermissionError('R21 physical candidates forbid Test/query/submission reads')
def load(p):
 with np.load(pin(p),allow_pickle=False)as z:return {k:z[k]for k in z.files}
def observed(db):
 d=np.asarray(db,np.float64)
 if d.ndim!=3 or d.shape[1:]!=(15,25)or np.isinf(d).any()or not np.isfinite(d).any((1,2)).all():raise ValueError('Finite/NaN 15x25 db windows required')
 # Preserve already-quantized float32 observations byte-numerically. Casting
 # then rounding them again creates/merges rank ties and changes quiet indices.
 # Only genuinely unquantized inputs are rounded at this top boundary.
 finite=np.isfinite(d)
 if np.any(abs(d[finite]-np.round(d[finite],1))>1e-4):d=np.round(d,1)
 o=np.isfinite(d);off=np.nanmax(d,(1,2))
 e=np.where(o,10**((d-off[:,None,None])/10),0.)
 return d,o,e,off
def bandwise_quiet(d,k=5):
 _,o,e,_=observed(d)
 a=np.sort(np.where(o,e,np.inf),axis=1)[:,:k]
 out=a.mean(1);out[o.sum(1)<k]=np.nan
 return out
def share_from_bg(e,o,b):
 """Signed sum, then final clipping only; no per-cell physical upper cap."""
 total=e.sum((1,2));bg=np.where(o,b,0).sum((1,2))
 return np.clip(1-bg/total,0,1)
def rolling_median(d,o):
 baseline=np.empty_like(d)
 for t in range(15):baseline[:,t]=np.nanmedian(d[:,max(0,t-2):min(15,t+3)],axis=1)
 return baseline
def huber_trend(d,o):
 # This is a deterministic per-window robust affine dB summary, not a model
 # learned across labelled examples. Missing clock slots retain true indices.
 t=(np.arange(15,dtype=float)-7)/7;t=t[None,:,None]
 init=np.nanmedian(d,axis=1);intercept=np.where(np.isfinite(init),init,0);slope=np.zeros_like(intercept)
 weights=o.astype(float);clean=np.where(o,d,0)
 for _ in range(5):
  pred=intercept[:,None,:]+slope[:,None,:]*t
  residual=np.where(o,d-pred,np.nan)
  mad=1.4826*np.nanmedian(abs(residual-np.nanmedian(residual,axis=1)[:,None,:]),axis=1)
  scale=np.maximum(mad,1.) # fixed 1 dB floor; never fit to y or test.
  r=abs(np.where(o,residual,0))/scale[:,None,:]
  weights=o*np.minimum(1.,1.345/np.maximum(r,1e-12))
  sw=weights.sum(1);st=(weights*t).sum(1);stt=(weights*t*t).sum(1)
  sy=(weights*clean).sum(1);sty=(weights*t*clean).sum(1);det=sw*stt-st*st
  slope=np.divide(sw*sty-st*sy,det,out=np.zeros_like(sw),where=det>1e-12)
  intercept=np.divide(sy-slope*st,sw,out=intercept.copy(),where=sw>0)
 return intercept[:,None,:]+slope[:,None,:]*t
def extract_candidates(db):
 """Returns fixed candidate share arrays using db/mask only; no truth inputs.

 `bandwise_lowK5` is undefined when any observed band has <5 samples. It is
 left NaN; the parent gate must fail closed or report common-support coverage.
 """
 from ops.r20_quiet_background_features import quiet
 d,o,e,off=observed(db)
 c20=json.loads((R20/'COEFFICIENT_FREEZE.json').read_text())['coefficients']['3']
 c21=json.loads((OUT/'COEFFICIENT_FREEZE.json').read_text())['bandwise_lowK5_coefficient']
 qb,*_=quiet(d,3);b0=qb*np.asarray(c20)
 b1=bandwise_quiet(d,5)*np.asarray(c21)
 b2=10**((rolling_median(d,o)-off[:,None,None])/10)
 b3=10**((huber_trend(d,o)-off[:,None,None])/10)
 return {CANDIDATES[0]:np.clip(1-(b0*o.sum(1)).sum(1)/e.sum((1,2)),0,1),
  CANDIDATES[1]:np.clip(1-(b1*o.sum(1)).sum(1)/e.sum((1,2)),0,1),
  CANDIDATES[2]:share_from_bg(e,o,b2),CANDIDATES[3]:share_from_bg(e,o,b3)}
def extract_features(db):
 """Physical diagnostic features for later research; NOT added to a model."""
 d,o,e,_=observed(db);out=extract_candidates(d)
 total=e.sum((1,2));out=dict(out)
 for name,ix in [('LF',slice(0,9)),('MF',slice(9,20)),('HF',slice(20,25)),('safe1to5k',slice(15,23))]:
  p=e[:,:,ix].sum(2);obs=o[:,:,ix].all(2);v=np.where(obs,10*np.log10(np.maximum(p,1e-300)),np.nan)
  out[name+'_energy_fraction']=e[:,:,ix].sum((1,2))/total
  out[name+'_bump_db']=np.nanmax(v,1)-np.nanmedian(v,1)
 return pd.DataFrame(out)
def metrics(y,p,w):
 m=np.isfinite(y)&np.isfinite(p)&np.isfinite(w)&(w>0)
 if not m.any():return {'n':0,'rmse':None,'bias':None,'positive_bias':None,'mae':None}
 y=y[m];p=p[m];w=w[m]/w[m].sum();err=p-y;bias=float(w@err)
 return {'n':len(y),'rmse':float(np.sqrt(w@(err*err))),'bias':bias,'positive_bias':max(bias,0.),'mae':float(w@abs(err))}
def old_b1(panel,d):
 vals=[]
 for f in ('lightgbm','dinov2_small','dasheng_base'):
  path=FINAL/'predictions/old'/f/(panel+'.npz');z=load(path)
  receipt=json.loads(pin(path.with_suffix('.json')).read_text())
  if receipt['input_sha256']!=sha(FINAL/'inputs'/(panel+'.npz'))or receipt['prediction_sha256']!=sha(path):raise ValueError('Saved prediction pin differs')
  if not np.array_equal(z['sample_id'].astype(str),d['sample_id'].astype(str)):raise ValueError('ID mismatch')
  vals.append(.5*(z['share_D0']+z['share_H0']))
 return sum(w*x for w,x in zip((.6,.25,.15),vals))
def main():
 sys.addaudithook(guard);warnings.filterwarnings('ignore',category=RuntimeWarning)
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'PROTOCOL.json').exists():raise FileExistsError('Immutable output requires a fresh version')
 protocol={'created_KST':dt.datetime.now().astimezone().isoformat(),'registered_before_new_coefficient_fit_and_label_reads':True,'candidates':list(CANDIDATES),
  'new_predictive_model_training':False,'new_inference':False,'GPU_MPS':False,'Test_read':False,'new_LightGBM_features_added':False,
  'candidate_formulas':{'R20_commonK3_control':'R20 frozen common quietK3 and TRAIN median correction',
  'bandwise_lowK5_TRAIN_ratio':'Per-band lowest5 observed linear powers, TRAIN median fullmean/quietmean correction, signed final share clipping',
  'timevarying_median5_signed':'Per-band rolling5 median dB at actual clock slots, convert to energy, sum signed E-B then final clip',
  'robust_linear_db_signed':'Per-band affine dB baseline at original t=0..14, five fixed Huber IRLS steps delta1.345 with MAD scale floor1dB, signed total excess'},
  'coefficient_scope':'same R20 pure SA TRAIN11700 sources only; no labels, no Test',
  'quantization':'Preserve existing 0.1dB float32-grid observations within1e-4dB tolerance to retain exact quiet-rank ties; otherwise round1decimal before preprocessing',
  'fixed_blend_lambdas':[0,.05,.10],'blend_rule':'(1-lambda)*saved old_B1 + lambda*fixed physical share; no fitted blend weights',
  'parameter_selection':False,'new_Clean_Val_selection':'parent separate gate only; these historical panels are descriptive diagnostics',
  'limitations':['No false-positive perfection guarantee; natural transients are also real background.',
   'Rolling/quiet/robust baseline can subtract a persistent true addition; constant-window nonidentifiability remains.',
   'A frozen LightGBM cannot consume new columns without retraining; these are standalone features/estimates and fixed blends only.'],
  'code_sha256':sha(__file__)}
 write(OUT/'PROTOCOL.json',protocol)
 from pipeline.dataset_r9 import R9Sources
 from pipeline.source_split_r6 import apply_minute_qc
 meta=pd.read_csv(pin(R20/'COEFFICIENT_TRAIN_SOURCE_ROWS.csv.gz'))
 src=R9Sources(verify_banks=False)
 db=np.asarray(src.background_bank()[meta.base_window_index.to_numpy(np.int64)])
 db=np.round(apply_minute_qc(db,meta.boundary_mask_bits.to_numpy(np.int64)),1);d,o,e,_=observed(db)
 qb=bandwise_quiet(d);full=np.divide(e.sum(1),o.sum(1),out=np.full_like(qb,np.nan),where=o.sum(1)>0)
 ratios=full/np.maximum(qb,1e-300);coef=np.nanmedian(ratios,axis=0)
 write(OUT/'COEFFICIENT_FREEZE.json',{'source_rows':len(meta),'source_groups':int(meta.background_group.nunique()),'statistical_TRAIN_only_fit':True,
  'bandwise_lowK5_coefficient':coef.tolist(),'source_row_sha256':sha(R20/'COEFFICIENT_TRAIN_SOURCE_ROWS.csv.gz'),
  'R20_commonK3_freeze_sha256':sha(R20/'COEFFICIENT_FREEZE.json'),'frozen_before_eval_labels':True,'source_identity':src.identity})
 pd.DataFrame({'band':np.arange(25),'frequency_hz':10**(1.5+.1*np.arange(25)),'coefficient_median':coef,'n':np.isfinite(ratios).sum(0),
  'ratio_p10':np.nanquantile(ratios,.1,axis=0),'ratio_p90':np.nanquantile(ratios,.9,axis=0)}).to_csv(OUT/'COEFFICIENTS.csv',index=False)
 del db,d,o,e,qb,full,ratios
 rows=[];rowdata=[]
 for panel in ('fresh_fit_D0','fresh_fit_H0','local_gate','domain_holdout','native12_VS_hf_high_negative','native12_VS_st_francis_dense_negative'):
  d=load(FINAL/'inputs'/(panel+'.npz'));y=d['share'].astype(float);w=np.asarray(d.get('weights',np.ones(len(y))),float)
  if w.ndim!=1:w=np.ones(len(y))
  baseline=old_b1(panel,d);p=extract_candidates(d['db']);slices={'all':np.ones(len(y),bool),'y=0':y==0,'0<y<=.10':(y>0)&(y<=.10),'positive_active>=10':(y>0)&(d['active_minutes']>=10)}
  candidates={'old_B1':baseline,**p}
  for name,v in p.items():
   for lam in (.05,.10):candidates[f'old_B1_blend_{name}_lambda{lam:g}']=(1-lam)*baseline+lam*v
  for name,v in candidates.items():
   for sl,mask in slices.items():rows.append({'panel':panel,'candidate':name,'slice':sl,**metrics(y[mask],v[mask],w[mask])})
  df=pd.DataFrame({'panel':panel,'sample_id':d['sample_id'].astype(str),'share_truth':y,'old_B1':baseline,**p})
  rowdata.append(df)
 pd.DataFrame(rows).to_csv(OUT/'HISTORICAL_FIXED_CANDIDATE_METRICS.csv',index=False)
 pd.concat(rowdata,ignore_index=True).to_csv(OUT/'HISTORICAL_FIXED_CANDIDATE_ROW_PREDICTIONS.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 # Meaningful invariant/edge-case checks; never use their results to retune.
 probe=np.linspace(50,90,15*25).reshape(1,15,25).repeat(3,0);probe[1,:2]=np.nan
 p0=extract_candidates(probe);p1=extract_candidates(probe+20)
 diff={k:float(np.nanmax(abs(p0[k]-p1[k])))for k in CANDIDATES}
 if max(diff.values())>1e-10:raise AssertionError('Common gain invariance fails')
 stationary=np.full((1,15,25),10*np.log10(200.));flat=extract_candidates(stationary)
 qa={'gain_plus20_max_abs':diff,'range_and_NaN_validated':True,'Nobs_lt5_is_explicit_NaN':bool(np.isnan(extract_candidates(np.concatenate([probe[:1,:3],np.full((1,12,25),np.nan)],axis=1))[CANDIDATES[1]][0])),
  'same_observed_E200_true_B200_A0_or_B100_A100':{'true_y':[0.,.5],'fixed_candidate_predictions':{k:float(v[0])for k,v in flat.items()}},
  'new_predictive_model_training':False,'code_sha256':sha(__file__)}
 if not qa['Nobs_lt5_is_explicit_NaN']:raise AssertionError('Missing-data guard fails')
 write(OUT/'NUMERIC_QA.json',qa)
 write(OUT/'INPUT_SHA256.json',PINS)
 f=pd.DataFrame(rows);small=f[(f['slice']=='all')&~f.candidate.str.contains('blend')]
 text='# R21 무학습 물리 배경 피처 후보 검산\n\n'
 text+='후보 4개와 소량 blend 가중치0/.05/.10을 평가 라벨 조회 전에 고정했다. 모델 재학습·신규 모델 추론·LightGBM 특징 추가·가중치 변경·GPU/MPS·Test 접근은 없다. 새 K5 보정계수만 기존 R20 TRAIN 순수 배경11,700창에서 통계 적합했다. Huber는 각 창 내부의 고정5회 robust 추세 통계이며 라벨 기반 학습이 아니다.\n\n'
 text+=small.to_markdown(index=False,floatfmt='.6f')+'\n\n'
 text+='이 표는 과거 진단 패널의 기술 대조이며 새 Clean-Val에서의 채택 결과가 아니다. 물리값이 실제 추가 에너지의 정답/상한이라는 보장도 없다. signed total excess를 마지막에만 clip하여 셀별 양수 오차 누적을 피하지만 자연 과도음과 장시간 추가 성분의 식별불가능성은 남는다. 무추가 false positive 완벽 제거를 보장하지 않는다. 기존 모델 출력을 수정해 배포하거나 제출 CSV를 생성하지 않았다.\n\n'
 text+='호출: `extract_candidates(db)`는 후보별 [N] share 배열 dict, `extract_features(db)`는 동일4추정+대역군 에너지/돌출을 담은 DataFrame을 돌려준다. 관측 NaN 유지·0.1dB 격자·전체 관측 에너지 분모를 보존한다. 입력 값 외 모델 예측/정답/추정 사이트를 보지 않는다. K5에5개 관측 분이 없으면 NaN이며 상위 게이트는 fail-closed 또는 명시적 공통지원 범위 비교가 필요하다.\n'
 (OUT/'R21_QUIET_FEATURE_CANDIDATES.md').write_text(text)
 for name,h in PINS.items():
  if sha(ROOT/name)!=h:raise ValueError('Pinned input changed '+name)
 files={str(p.relative_to(OUT)):sha(p)for p in sorted(OUT.rglob('*'))if p.is_file()and p.name not in ('ARTIFACT_SHA256.json','COMPLETED.json')}
 write(OUT/'ARTIFACT_SHA256.json',files)
 write(OUT/'COMPLETED.json',{'status':'completed','candidates':len(CANDIDATES),'historical_panels':6,'TRAIN_coefficient_rows':len(meta),'new_predictive_model_training':False,'Test_read':False,'files':len(files),'code_sha256':sha(__file__)})
 print(small.to_string(index=False),flush=True)
if __name__=='__main__':main()
