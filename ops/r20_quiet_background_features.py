"""R20 TRAIN-only quiet-background coefficients and descriptive residual audit.

No predictive model training, inference, Test access, policy selection or public
score fitting. The c[f,K] coefficients ARE a statistical fit on pure TRAIN SA
backgrounds; this is explicitly recorded. Correlation squared is descriptive,
not out-of-sample predictive R2 or an estimate of a Bayes error floor.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import datetime as dt, hashlib, json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/r20_integrated_cpu_diagnostics_r1/quiet_features'
FINAL=ROOT/'reports/r18_production_r1/finalization'
R19=ROOT/'reports/r19_multislice_law_transfer_r1'
KS=(2,3,5)
PANELS=('local_gate','domain_holdout','local_report','fresh_fit_D0','fresh_fit_H0',
 'native12_VS_hf_high_negative','native12_VS_st_francis_dense_negative')
PINS={}
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  n=Path(os.fsdecode(args[0])).name.lower()
  if n in ('test.csv','query.npz','query_predictions.npz') or 'submission' in n:
   raise PermissionError('R20 quiet features forbid challenge query/submission reads')
sys.addaudithook(guard)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def pin(p):
 p=Path(p);PINS[str(p.relative_to(ROOT))]=sha(p);return p
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def load(p):
 with np.load(pin(p),allow_pickle=False) as z:return {k:z[k] for k in z.files}
def now():return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec='seconds')
def mean(a,axis):
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning);return np.nanmean(a,axis=axis)
def corr(a,b,obs):
 n=obs.sum(1);ma=np.divide(np.where(obs,a,0).sum(1),n,out=np.zeros(len(a)),where=n>0)
 mb=np.divide(np.where(obs,b,0).sum(1),n,out=np.zeros(len(a)),where=n>0)
 aa=np.where(obs,a-ma[:,None],0);bb=np.where(obs,b-mb[:,None],0)
 den=np.sqrt((aa*aa).sum(1)*(bb*bb).sum(1))
 return np.divide((aa*bb).sum(1),den,out=np.full(len(a),np.nan),where=den>1e-12)
def energies(d):
 d=np.asarray(d,float);o=np.isfinite(d);offset=np.nanmax(d,axis=(1,2))
 e=np.where(o,10**((d-offset[:,None,None])/10),0)
 return e,o
def quiet(d,k):
 """Equal rank of relative multiband level and adjacent actual-clock change.

 Never compress missing clock minutes. If neither adjacent difference exists,
 change score is zero and the isolated-minute flag is retained. nobs<K rows
 receive NaN estimates, never silently a smaller K.
 """
 d=np.asarray(d,float);e,o=energies(d);ot=o.all(2)
 bandbase=np.nanmedian(d,axis=1)
 level=np.nanmedian(d-bandbase[:,None],axis=2)
 delta=np.full((len(d),15),np.nan)
 for t in range(15):
  pieces=[]
  for j in (t-1,t+1):
   if 0<=j<15:pieces.append(np.where(ot[:,t]&ot[:,j],np.nanmedian(abs(d[:,t]-d[:,j]),axis=1),np.nan))
  delta[:,t]=mean(np.stack(pieces),0)
 isolated=ot&~np.isfinite(delta);delta=np.where(isolated,0,delta)
 # rank 1..n with average ties; absent minutes are ranked last only for sorting.
 rl=pd.DataFrame(np.where(ot,level,np.nan)).rank(axis=1,method='average').to_numpy()
 rd=pd.DataFrame(np.where(ot,delta,np.nan)).rank(axis=1,method='average').to_numpy()
 cost=np.where(ot,(rl+rd)/2,np.inf);ix=np.argsort(cost,axis=1,kind='stable')[:,:k]
 q=e[np.arange(len(d))[:,None],ix].mean(1)
 q[ot.sum(1)<k]=np.nan
 return q,e,o,ix,isolated.sum(1)
def feature_table(d,coeff):
 d=np.asarray(d,float);e,o=energies(d);ot=o.all(2);nobs=ot.sum(1)
 total=e.sum((1,2));wf=e.sum(1)/total[:,None];out={}
 med=np.nanmedian(d,1);q10=np.nanquantile(d,.1,axis=1);q90=np.nanquantile(d,.9,axis=1)
 band={
  'energy_fraction':wf,'max_minus_median':np.nanmax(d,1)-med,
  'median_minus_min':med-np.nanmin(d,1),'q90_minus_q10':q90-q10,
  'time_std_db':np.nanstd(d,1),'mean_minus_median':mean(d,1)-med,
  'adjacent_abs_change':mean(np.where(o[:,1:]&o[:,:-1],abs(d[:,1:]-d[:,:-1]),np.nan),1),
  'median_relative_profile':med-np.nanmedian(d,axis=(1,2))[:,None]}
 for label,v in band.items():
  for f in range(25):out[f'band{f:02d}/{label}']=v[:,f]
 out['observation/nobs']=nobs;out['observation/first_missing']=~ot[:,0];out['observation/last_missing']=~ot[:,-1]
 curves={};region_ix={'LF':np.arange(9),'MF':np.arange(9,20),'HF':np.arange(20,25),'safe1to5k':np.arange(15,23),'all':np.arange(25)}
 for name,ix in region_ix.items():
  p=e[:,:,ix].sum(2);v=np.where(ot,10*np.log10(np.maximum(p,1e-300)),np.nan)
  curves[name]=v;baseline=np.nanmedian(v,1);mn=np.nanmin(v,1);peak=np.nanmax(v,1)
  s=p.sum(1);prob=np.divide(p,s[:,None],out=np.zeros_like(p),where=s[:,None]>0)
  event=ot&(v>baseline[:,None]+3);runs=np.zeros(len(d),int);current=np.zeros(len(d),int)
  for t in range(15):current=np.where(event[:,t],current+1,0);runs=np.maximum(runs,current)
  for key,val in {
   'energy_fraction':e[:,:,ix].sum((1,2))/total,'bump_db':peak-baseline,'median_minus_min_db':baseline-mn,
   'std_db':np.nanstd(v,1),'persistence_minutes':event.sum(1),'longest_run':runs,
   'time_entropy_normalized':-(np.where(prob>0,prob*np.log(np.maximum(prob,1e-300)),0)).sum(1)/np.log(np.maximum(nobs,2)),
   'peak_energy_fraction':prob.max(1),'edge_energy_fraction':prob[:,0]+prob[:,-1],
   'top2_energy_fraction':np.sort(prob,axis=1)[:,-2:].sum(1),
   'adjacent_abs_change_db':mean(np.where(ot[:,1:]&ot[:,:-1],abs(v[:,1:]-v[:,:-1]),np.nan),1),
   'active_to_quiet_energy_ratio':np.divide(np.nanmean(np.where(event,p,np.nan),1),np.nanmean(np.where(ot&~event,p,np.nan),1),out=np.full(len(d),np.nan),where=np.nanmean(np.where(ot&~event,p,np.nan),1)>0)
  }.items():out[f'{name}/{key}']=val
 for a,b in (('LF','MF'),('LF','HF'),('MF','HF'),('LF','safe1to5k'),('MF','safe1to5k')):
  out[f'corr/{a}/{b}/lag0']=corr(curves[a],curves[b],ot)
  for lag in (1,2,3):
   for aa,bb in ((a,b),(b,a)):
    out[f'corr/{aa}/{bb}/lag{lag}']=corr(curves[aa][:,:-lag],curves[bb][:,lag:],ot[:,:-lag]&ot[:,lag:])
 # All within-window zero-lag band pairs. TOL powers have no waveform phase.
 for f in range(25):
  for g in range(f+1,25):out[f'band_corr/{f:02d}/{g:02d}']=corr(d[:,:,f],d[:,:,g],o[:,:,f]&o[:,:,g])
 lf=wf[:,:9].sum(1);mf=wf[:,9:20].sum(1);hf=wf[:,20:].sum(1)
 out['ratio/HF_to_LF']=hf/np.maximum(lf,1e-15);out['ratio/MF_to_LF']=mf/np.maximum(lf,1e-15)
 for q in (.0,.05,.20,.50):
  base=np.nanquantile(d,q,axis=1);a=np.where(o,1-10**((base[:,None]-d)/10),0)
  out[f'naive_floor/q{int(q*100):02d}/uncertified_excess']=np.clip((e*np.maximum(a,0)).sum((1,2))/total,0,1)
 preds={}
 for k in KS:
  qb,_,_,_,isolated=quiet(d,k)
  for mode,c in [('uncorrected',np.ones(25)),('TRAIN_median_ratio',coeff[k])]:
   bh=qb*c;est=np.clip(1-(bh*o.sum(1)).sum(1)/total,0,1)
   out[f'quiet/K{k}/{mode}/share']=est;preds[k,mode]=est
   for name,ix in region_ix.items():out[f'quiet/K{k}/{mode}/{name}_background_ratio']=(bh[:,ix]*o[:,:,ix].sum(1)).sum(1)/np.maximum(e[:,:,ix].sum((1,2)),1e-30)
  out[f'quiet/K{k}/isolated_observed_minutes']=isolated
 return pd.DataFrame(out),preds
def old_b1(panel,d):
 arr=[]
 for family in ('lightgbm','dinov2_small','dasheng_base'):
  p=FINAL/'predictions/old'/family/(panel+'.npz');r=json.loads(pin(p.with_suffix('.json')).read_text())
  if r['input_sha256']!=sha(FINAL/'inputs'/(panel+'.npz')) or r['prediction_sha256']!=sha(p):raise ValueError('Prediction receipt differs')
  z=load(p)
  if not np.array_equal(z['sample_id'].astype(str),d['sample_id'].astype(str)):raise ValueError('IDs differ')
  arr.append((z['share_D0']+z['share_H0'])/2)
 return sum(w*x for w,x in zip((.6,.25,.15),arr))
def weighted_stats(y,p,w):
 m=np.isfinite(y)&np.isfinite(p)&np.isfinite(w)&(w>0)
 if not m.any():return {'n':0,'rmse':None,'bias':None,'mae':None,'predictive_R2':None}
 a=y[m];b=p[m];ww=w[m]/w[m].sum();err=b-a;var=ww@((a-ww@a)**2)
 return {'n':int(m.sum()),'rmse':float(np.sqrt(ww@(err*err))),'bias':float(ww@err),'mae':float(ww@abs(err)),
  'predictive_R2':float(1-(ww@(err*err))/var) if var>1e-15 else None,'prediction_mean':float(ww@b),'target_mean':float(ww@a)}
def residual_scan(frame,y,p,panel):
 rows=[]
 for feature in frame:
  x=frame[feature].to_numpy(float)
  for target,t in (('truth_share',y),('residual_y_minus_old_B1',y-p)):
   m=np.isfinite(x)&np.isfinite(t)
   if m.sum()<20 or np.std(x[m])<1e-12 or np.std(t[m])<1e-12:continue
   pear=float(np.corrcoef(x[m],t[m])[0,1]);sp=float(spearmanr(x[m],t[m]).statistic)
   rows.append({'panel':panel,'target':target,'feature':feature,'n':int(m.sum()),'pearson':pear,'spearman':sp,
    'univariate_affine_in_sample_R2_equals_r_squared':pear*pear,'abs_spearman':abs(sp)})
 return rows
def source_purge_audit(train,evalmeta):
 rows=[]
 for name,e in evalmeta.items():
  common=set(train.background_group.astype(str))&set(e.background_group.astype(str))
  rows.append({'panel':name,'coefficient_TRAIN_windows':len(train),'eval_unique_windows':int(e.base_window_index.nunique()),
   'exact_index_overlap':int(train.base_window_index.isin(e.base_window_index).sum()),'group_overlap_count':len(common),
   'group_disjoint':not common,'vanilla_primary_source_exposure_remains':name=='vanilla_addition'})
 return rows
def main():
 start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
 protocol={'registered_KST':now(),'scope':'local CPU statistics only; no predictive-model training/inference/selection',
 'K':list(KS),'quiet_score':'0.5 * (within-window rank of median_f(db-median_t(db)) + rank of adjacent actual-clock median_f abs delta); average ties, earliest clock ties',
 'missing':'Only complete observed minutes; absent clock intervals never compressed; isolated change=0 with flag; Nobs<K estimates NaN',
 'coefficient_fit':'Pure R18 TRAIN SA backgrounds; exclude every exact R19 vanilla background index; c[f,K]=median(TRAIN full observed band-mean E / quiet-K band-mean E)',
 'coefficient_sample_weight':'Each TRAIN window weight1; overlapping source windows are NOT independent recordings',
 'no_selection':'K2/3/5 all reported; no y/residual/public/Test-based K or coefficient tuning',
 'residual_features':'Predefined band, regional, quiet, entropy, persistence and zero/lag envelope correlations; no physical waveform phase available',
 'correlation_R2':'Pearson r^2 descriptive same-sample affine R2; NOT held-out trained-model R2 or evidence of causation',
 'predictive_R2':'Direct fixed quiet-estimator 1-SSE/SST; undefined for constant-y slices',
 'Test_read':False,'GPU_MPS':False,'submission_generated':False,'source_file_sha256':sha(Path(__file__))}
 pp=OUT/'PROTOCOL.json'
 if pp.exists():raise FileExistsError('Fresh immutable audit output required')
 write(pp,protocol)
 from pipeline.dataset_r9 import R9Sources
 from pipeline.source_split_r6 import apply_minute_qc
 src=R9Sources(verify_banks=True)
 tp=ROOT/'data/synthetic/r18_production_v1/train_backgrounds.parquet';vp=R19/'law_panels/paired_backgrounds.parquet'
 original=pd.read_parquet(pin(tp));vm=pd.read_parquet(pin(vp));train=original[~original.base_window_index.isin(vm.base_window_index)].copy()
 train.to_csv(OUT/'COEFFICIENT_TRAIN_SOURCE_ROWS.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 db=apply_minute_qc(np.asarray(src.background_bank()[train.base_window_index.to_numpy(np.int64)]),train.boundary_mask_bits.to_numpy(np.int64))
 db=np.round(db,1);coeff={};cr=[]
 for k in KS:
  qb,e,o,_,_=quiet(db,k);full=np.divide(e.sum(1),o.sum(1),out=np.full_like(qb,np.nan),where=o.sum(1)>0)
  ratios=full/np.maximum(qb,1e-300);coeff[k]=np.nanmedian(ratios,0)
  for f in range(25):
   a=ratios[:,f];a=a[np.isfinite(a)];cr.append({'K':k,'band':f,'frequency_hz':10**(1.5+.1*f),'n':len(a),'c_median':float(coeff[k][f]),'ratio_p10':float(np.quantile(a,.1)),'ratio_p90':float(np.quantile(a,.9)),'ratio_arithmetic_mean_diagnostic_only':float(a.mean())})
 pd.DataFrame(cr).to_csv(OUT/'TRAIN_QUIET_COEFFICIENTS.csv',index=False)
 write(OUT/'COEFFICIENT_FREEZE.json',{'frozen_KST':now(),'TRAIN_rows':len(train),'excluded_exact_vanilla_rows':len(original)-len(train),
  'TRAIN_groups':int(train.background_group.nunique()),'coefficients':{str(k):v.tolist() for k,v in coeff.items()},
  'source_identity':src.identity,'metadata_sha256':{str(tp.relative_to(ROOT)):sha(tp),str(vp.relative_to(ROOT)):sha(vp)},
  'coefficient_csv_sha256':sha(OUT/'TRAIN_QUIET_COEFFICIENTS.csv'),'Test_or_evaluation_labels_used':False})
 del db
 metrics=[];scans=[];exposure=[];rowoutputs=[];featurecount=None
 for panel in (*PANELS,'vanilla_addition'):
  if panel=='vanilla_addition':
   d=load(R19/'law_panels/vanilla_addition.npz')
   rows=pd.read_csv(pin(R19/'inference/LAW_TRANSFER_ROW_PREDICTIONS.csv'));rows=rows[rows.law.eq(panel)]
   if not np.array_equal(rows.sample_id.astype(str),d['sample_id'].astype(str)):raise ValueError('Vanilla IDs differ')
   pred=rows.old_B1_share.to_numpy();evalmeta=vm
  else:
   d=load(FINAL/'inputs'/(panel+'.npz'));pred=old_b1(panel,d)
   evalmeta=pd.DataFrame({'base_window_index':d['base_window_index'],'background_group':d['background_group']}).drop_duplicates()
  exposure+=source_purge_audit(train,{panel:evalmeta})
  y=d['share'].astype(float)
  # R19 raw panels use weights for CELL energy pooling, whereas compact
  # native panels store a 1-D class-balanced SAMPLE weight. Never mix units.
  candidate_w=np.asarray(d.get('weights',np.ones(len(y))),float)
  w=candidate_w if candidate_w.ndim==1 else np.ones(len(y),float)
  if w.shape!=(len(y),):raise ValueError('Sample weights must be one-dimensional')
  x,est=feature_table(d['db'],coeff);featurecount=len(x.columns)
  slices={'all':np.ones(len(y),bool),'y=0':y==0,'0<y<=0.03':(y>0)&(y<=.03),'0.03<y<=0.10':(y>.03)&(y<=.1),'0<y<=0.10':(y>0)&(y<=.1),'0.10<y<=0.30':(y>.1)&(y<=.3),'y>0.30':y>.3,'positive_active>=10':(y>0)&(d['active_minutes']>=10)}
  for label,p in [('old_B1_saved',pred),*[(f'quiet_K{k}_{m}',v) for (k,m),v in est.items()]]:
   for s,ix in slices.items():metrics.append({'panel':panel,'estimator':label,'slice':s,**weighted_stats(y[ix],p[ix],w[ix])})
  scans+=residual_scan(x,y,pred,panel)
  base=pd.DataFrame({'panel':panel,'sample_id':d['sample_id'].astype(str),'y':y,'old_B1':pred,'residual':y-pred,'weight':w})
  for (k,m),v in est.items():base[f'quiet_K{k}_{m}']=v
  rowoutputs.append(base)
  write(OUT/'progress.json',{'panel_completed':panel,'elapsed_seconds':time.perf_counter()-start,'updated_KST':now()})
 met=pd.DataFrame(metrics);met.to_csv(OUT/'QUIET_ESTIMATOR_METRICS_ALL_SLICES.csv',index=False)
 cf=pd.DataFrame(scans);cf.to_csv(OUT/'ALL_PHYSICAL_FEATURE_CORRELATIONS.csv',index=False)
 top=cf[cf.target.eq('residual_y_minus_old_B1')].sort_values('abs_spearman',ascending=False).groupby('panel',sort=False).head(30)
 top.to_csv(OUT/'TOP30_RESIDUAL_FEATURES_BY_PANEL.csv',index=False)
 pd.concat(rowoutputs,ignore_index=True).to_csv(OUT/'QUIET_ESTIMATOR_ROW_OUTPUTS.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 pd.DataFrame(exposure).to_csv(OUT/'COEFFICIENT_SOURCE_EXPOSURE.csv',index=False)
 def table(df):return df.to_markdown(index=False,floatfmt='.6f')
 overview=met[(met['slice']=='all')&met.estimator.isin(['old_B1_saved',*['quiet_K'+str(k)+'_TRAIN_median_ratio' for k in KS]])]
 text='# R20 조용한 분 기반 배경 추정 및 물리 특징 잔차 감사\n\n'
 text+='모델 재학습·추가 추론·Test 접근은 없다. 다만 순수 TRAIN 배경에서 보정계수 c를 통계 적합했으므로 ‘어떤 적합도 없음’이라고 부르지 않는다. K나 계수는 검증 라벨·Public 점수로 선택하지 않았다.\n\n'
 text+=f'계수 TRAIN {len(train):,}창({train.background_group.nunique()} 원천그룹), vanilla exact 중복 {len(original)-len(train)}창 제외. 특징 {featurecount}개. 관측시계 분 간 차이를 보존했다. K보다 관측 분이 적으면 해당 추정은 미측정이며 성적표 n이 달라진다.\n\n'
 text+='## 직접 물리 추정 대 저장 old_B1\n\n'+table(overview)+'\n\n'
 text+='조용한 분에도 장시간 추가 성분이 남거나 실제 배경이 시간에 따라 변하면 Bhat은 틀릴 수 있다. 이 추정치는 물리적 정답이나 상한이 아니며 자동 후처리에 채택하지 않았다. 상대 공통 이득에는 불변이지만 임의 대역별 변화에는 불변이 아니다.\n\n'
 text+='## 원천 노출\n\n'+table(pd.DataFrame(exposure))+'\n\n'
 text+='gate/report/domain/native/fresh-fit은 계수 TRAIN과 원천그룹 분리가 확인된 경우에만 독립 원천으로 표시한다. vanilla는 primary 학습 풀에서 생성되어 exact 중복을 제거해도 인근 시간·그룹 노출 및 기존 모델의 학습 노출이 남는다. 통계 표본 수는 독립 녹음 수가 아니다.\n\n'
 text+='## 각 패널 잔차 상관 상위 특징\n\n'+table(top.groupby('panel',sort=False).head(5)[['panel','feature','n','pearson','spearman','univariate_affine_in_sample_R2_equals_r_squared']])+'\n\n'
 text+='전체 상위30/전체 스캔/구간별 오차는 별도 CSV에 보존했다. Pearson r²는 같은 표본의 단변량 절편 포함 선형 설명력이며 향후 모델의 검증 R²가 아니다. 수백 특징 탐색과 원천 재사용 때문에 인과나 일반화 개선을 증명하지 않는다. 25대역 TOL 전력은 파형 위상 정보가 없으므로 ‘위상 지수’를 계산하지 않고 envelope 동조·lag만 계산했다.\n'
 (OUT/'R20_QUIET_BACKGROUND_FEATURES.md').write_text(text)
 for p,h in list(PINS.items()):
  if sha(ROOT/p)!=h:raise ValueError('Source changed '+p)
 write(OUT/'INPUT_SHA256.json',PINS)
 artifacts={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('ARTIFACT_SHA256.json','COMPLETED.json')}
 write(OUT/'ARTIFACT_SHA256.json',artifacts)
 write(OUT/'COMPLETED.json',{'status':'completed','created_KST':now(),'seconds':time.perf_counter()-start,'TRAIN_coefficient_rows':len(train),'feature_count':featurecount,'panels':len(rowoutputs),'artifacts':len(artifacts),'statistical_TRAIN_coefficient_fit':True,'model_optimizer_steps':0,'Test_read':False})
 print(json.dumps({'status':'completed','elapsed_seconds':time.perf_counter()-start,'output':str(OUT)},ensure_ascii=False),flush=True)
if __name__=='__main__':
 warnings.filterwarnings('ignore',category=RuntimeWarning)
 main()
