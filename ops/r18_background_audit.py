"""R18 TRAIN-only background restoration controls; no challenge query IO.

All SA rows are inventoried descriptively. Priors use only the original
Fold0 TRAIN background sources, purged from every evaluated source role.
True source background is a training target / evaluation truth, never an
estimator input. Metadata-conditioned templates are explicitly oracle-metadata
diagnostics because Test does not provide these known source identifiers.
"""
from pathlib import Path
import datetime, gc, hashlib, json, os, sys, time, warnings
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/r18_fundamental_audit_r1/background'
MODELS=ROOT/'models/r18_background_audit_r1'
SEED=2026103008

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k]for k in z.files}
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).name.lower()
  if p in ('test.csv','query_predictions.npz','sample_submission.csv'):
   raise PermissionError('R18 background controls prohibit Test/query access')
def note(**v):
 print(json.dumps(v,ensure_ascii=False),flush=True);save(OUT/'progress.json',v)
def bands(db):
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  return {k:np.asarray(fn(db,axis=1),float) for k,fn in [('q05',lambda a,axis:np.nanquantile(a,.05,axis=axis)),('q20',lambda a,axis:np.nanquantile(a,.2,axis=axis)),('median',np.nanmedian),('mean',np.nanmean),('max',np.nanmax),('std',np.nanstd)]}
def energy(db):
 return np.where(np.isfinite(db),10**(np.nan_to_num(db,nan=0.)/10),0.)
def background_truth(d,bank):
 # Access permitted only as a supervised TRAIN target / scoring truth.
 bg=np.asarray(bank[np.asarray(d['base_window_index'],int)],float)
 return np.where(np.isfinite(d['db']),bg,np.nan)
def features(db):
 s=bands(db);o=np.isfinite(db);n=o.sum((1,2))
 rel=s['q20']-np.nanmedian(s['q20'],axis=1)[:,None]
 e=energy(db);et=e.sum((1,2));w=e/et[:,None,None]
 # No labels, metadata, q, reconstructed component proxies or target support.
 x=np.column_stack([s[k]for k in ('q05','q20','median','mean','max','std')]+[rel,
     np.nan_to_num(db-s['median'][:,None,:]).reshape(len(db),-1),
     np.log10(et/np.maximum(n,1)),w.sum(1),o.any(2).astype(float)])
 return np.nan_to_num(x).astype(np.float32)
def inventory():
 p=ROOT/'data/processed/sa_row_manifest.parquet';m=pd.read_parquet(p)
 a=np.load(ROOT/'data/processed/sa_rows_db.npy',mmap_mode='r');assert a.shape==(len(m),25)
 assert len(m)==1715779
 m['export_month']=pd.to_datetime(m.t).dt.month
 m['export_hour']=pd.to_datetime(m.t).dt.hour
 counts=[];spectral=[]
 for site,ix in m.groupby('site').indices.items():
  ix=np.asarray(ix);r=m.iloc[ix];good=r.eligible_for_minute_window.to_numpy(bool);v=np.asarray(a[ix[good]],float)
  counts.append({'site':site,'rows':len(r),'eligible_rows':int(good.sum()),'recorders':r.recorder.nunique(),'sensor_groups':r.sensor_group.nunique(),
       'first_export_clock':str(r.t.min()),'last_export_clock':str(r.t.max()),'unique_months':r.export_month.nunique(),'unique_hours':r.export_hour.nunique(),
       'nominal_sample_rates':','.join(map(str,sorted(r.sample_rate_hz.unique()))),'export_clock_UTC_verified':False})
  if len(v):
   quant=np.nanquantile(v,[.1,.5,.9],axis=0)
   spectral.extend({'site':site,'band':j,'frequency_hz':float(10**(1.5+.1*j)),'db_p10':quant[0,j],'db_p50':quant[1,j],'db_p90':quant[2,j]}for j in range(25))
 pd.DataFrame(counts).to_csv(OUT/'all_SA_rows_by_site.csv',index=False)
 pd.DataFrame(spectral).to_csv(OUT/'all_SA_eligible_band_db_quantiles.csv',index=False)
 for c in ('recorder','export_month','export_hour'):
  m.groupby(['site',c],dropna=False).agg(rows=('row_id','size'),eligible_rows=('eligible_for_minute_window','sum')).reset_index().to_csv(OUT/f'all_SA_rows_by_{c}.csv',index=False)
 rec={'rows':len(m),'eligible_rows':int(m.eligible_for_minute_window.sum()),'sites':m.site.nunique(),'recorders':m.recorder.nunique(),'sensor_groups':m.sensor_group.nunique(),
      'signal_shape':list(a.shape),'raw_row_bank_sha256':sha(ROOT/'data/processed/sa_rows_db.npy'),'metadata_sha256':sha(p),
      'scope':'Full corpus descriptive inventory only; no evaluation backgrounds enter prior fitting','clock':'export clock; UTC/local timezone is not verified'}
 save(OUT/'SA_FULL_INVENTORY.json',rec);del m;gc.collect();return rec
def datasets():
 out={}
 for role in ('train','earlystop'):
  for reg in ('D0','H0'):out[role+'_'+reg]=load(ROOT/f'data/synthetic/r17_clean_v1/{role}_{reg}.npz')
 for reg in ('D0','H0'):out['fresh_'+reg]=load(ROOT/f'data/synthetic/r17_clean_freshfit_v1/ensemble_fit_fresh_{reg}.npz')
 from ops.r17_audit_common import panel_data
 for p in ('local_gate','local_report','domain_holdout'):out[p]=panel_data(p)
 return out
def prior_fit(bank,wm,trainmeta):
 from pipeline.source_split_r6 import apply_minute_qc
 summaries=[]
 for st in range(0,len(trainmeta),512):
  f=trainmeta.iloc[st:st+512]
  bg=apply_minute_qc(np.asarray(bank[f.base_window_index.to_numpy(int)]),f.boundary_mask_bits.to_numpy(int))
  summaries.append(10*np.log10(energy(bg).sum(1)/np.maximum(np.isfinite(bg).sum(1),1)))
 profiles=np.concatenate(summaries);global_template=np.median(profiles,axis=0)
 # PCA has only TRAIN background spectra, not labels or evaluation examples.
 centered=profiles-np.median(profiles,axis=1)[:,None]
 mu=centered.mean(0);_,_,v=np.linalg.svd(centered-mu,full_matrices=False);basis=v[:3]
 meta=trainmeta.copy();meta['month']=pd.to_datetime(meta.start_export_clock).dt.month
 meta['hourbin']=pd.to_datetime(meta.start_export_clock).dt.hour//6
 templates={}
 for keys in [('site',),('recorder',),('recorder','month','hourbin')]:
  group={}
  for key,ix in meta.groupby(list(keys)).indices.items():
   if len(ix)>=50:group[tuple(key) if isinstance(key,tuple) else(key,)]=np.median(profiles[np.asarray(ix)],axis=0)
  templates[keys]=group
 np.savez_compressed(MODELS/'background_priors.npz',global_template=global_template,pca_mean=mu,pca_basis=basis)
 save(OUT/'PRIOR_FIT.json',{'train_unique_backgrounds':len(trainmeta),'source_groups':trainmeta.background_group.nunique(),
      'site_counts':trainmeta.site.value_counts().to_dict(),'templates':{'+'.join(k):len(v)for k,v in templates.items()},
      'min_template_count':50,'PCA_components':3,'fit_uses_Test_or_eval_sources':False})
 return global_template,mu,basis,templates
def evaluate(d,bg,bhat,name,panel,rows,condition_rows):
 o=np.isfinite(d['db']);E=energy(d['db']);B=energy(bg);den=E.sum((1,2));trueb=B.sum((1,2));y=d['share']
 bh=np.broadcast_to(np.asarray(bhat,float),bg.shape);BH=energy(np.where(o,bh,np.nan));btotal=BH.sum((1,2))
 q=np.maximum(0,1-np.divide(BH,E,out=np.zeros_like(E),where=o));q=np.where(o,q,0)
 pcell=(q*E).sum((1,2))/den
 pintegral=np.clip(1-btotal/den,0,1)
 db_err=(bh-bg)[o];totaldb=10*np.log10(np.maximum(btotal,1e-200)/trueb)
 for variant,p in [('cell_positive_part',pcell),('integral_clip',pintegral)]:
  err=p-y;row={'panel':panel,'estimator':name,'pooling':variant,'n':len(y),'rmse':float(np.sqrt(np.mean(err**2))),'bias':float(err.mean()),'prediction_mean':float(p.mean()),
       'target_mean':float(y.mean()),'B_cell_db_rmse':float(np.sqrt(np.mean(db_err**2))),'B_integral_db_rmse':float(np.sqrt(np.mean(totaldb**2))),
       'B_integral_relative_RMSE':float(np.sqrt(np.mean((btotal/trueb-1)**2))),'Bhat_gt_E_cell_fraction':float((BH[o]>E[o]).mean())}
  rows.append(row)
  for s in np.unique(d['site']):
   ix=d['site']==s;condition_rows.append({**row,'site':str(s),'n':int(ix.sum()),'rmse':float(np.sqrt(np.mean(err[ix]**2))),'bias':float(err[ix].mean())})
def main():
 sys.addaudithook(guard);OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
 if (OUT/'COMPLETED.json').exists():raise FileExistsError('Immutable completed background audit')
 plan={'status':'fixed_before_fitting','seed':SEED,'all_SA_rows_descriptive_only':1715779,'prior_scope':'R16 original Fold0 TRAIN sources only',
       'conditional_predictor':'LightGBM log mean B total target; 500 max trees, ES40 on designated earlystop; no target q/y features',
       'template_candidates':['window_q20','TRAIN_global','TRAIN_site_metadata','TRAIN_recorder_metadata','TRAIN_recorder_month_hour_metadata','TRAIN_spectral_PCA3_q20','learned_totalB_raw_summary'],
       'evaluation':['fresh_D0','fresh_H0','local_gate','local_report','domain_holdout'],'no_hyperparameter_selection_on_evaluation':True,'Test_used':False,
       'code_sha256':sha(__file__)}
 save(OUT/'PROTOCOL.json',plan);inventory_summary=inventory();note(phase='full_SA_inventory_completed')
 from pipeline.r16_production_data import _crossrole
 trainmeta=pd.read_parquet(ROOT/'data/synthetic/r16_multisupport_v1/fold_00/train_backgrounds.parquet')
 checks={'earlystop':pd.read_parquet(ROOT/'data/synthetic/r17_clean_v1/earlystop_backgrounds.parquet'),
     'fresh_fit':pd.read_parquet(ROOT/'data/synthetic/r17_clean_freshfit_v1/ensemble_fit_fresh_backgrounds.parquet')}
 checks.update({p:pd.read_parquet(ROOT/f'reports/r16_support_temporal_r1/data/{p}_backgrounds.parquet')for p in ('local_gate','local_report','domain_holdout')})
 proof=_crossrole(trainmeta,checks);save(OUT/'BACKGROUND_SOURCE_PURGE.json',proof)
 bank=np.load(ROOT/'data/processed/sa_windows_db.npy',mmap_mode='r');wm=pd.read_parquet(ROOT/'data/processed/sa_window_manifest.parquet')
 global_template,mu,basis,templates=prior_fit(bank,wm,trainmeta);note(phase='TRAIN_background_priors_fitted',n=len(trainmeta))
 data=datasets();rows=[];cond=[]
 # Supervised background-integral predictor is trained on known components in
 # TRAIN only, with a distinct purged ES source. It does not see query truth.
 import lightgbm as lgb
 trdb=np.concatenate([data['train_D0']['db'],data['train_H0']['db']]);esdb=np.concatenate([data['earlystop_D0']['db'],data['earlystop_H0']['db']])
 trbg=np.concatenate([background_truth(data['train_D0'],bank),background_truth(data['train_H0'],bank)])
 esbg=np.concatenate([background_truth(data['earlystop_D0'],bank),background_truth(data['earlystop_H0'],bank)])
 def btarget(bg):return np.log10(energy(bg).sum((1,2))/np.isfinite(bg).sum((1,2)))
 model=lgb.LGBMRegressor(n_estimators=500,learning_rate=.04,num_leaves=31,min_child_samples=80,n_jobs=2,random_state=SEED,verbosity=-1,deterministic=True,force_col_wise=True)
 x=features(trdb);v=features(esdb);history={};model.fit(x,btarget(trbg),eval_set=[(v,btarget(esbg))],callbacks=[lgb.early_stopping(40,verbose=False),lgb.record_evaluation(history)])
 model.booster_.save_model(str(MODELS/'log_mean_background_power.txt'));save(OUT/'B_INTEGRAL_LEARNER.json',{'train_n':len(x),'ES_n':len(v),'features':x.shape[1],'best_iteration':model.best_iteration_,'history':history,'B_truth_used_as_input':False})
 del x,v,trdb,esdb,trbg,esbg;gc.collect()
 precision=[]
 for panel in plan['evaluation']:
  d=data[panel];bg=background_truth(d,bank);s=bands(d['db']);n=len(bg)
  evaluate(d,bg,bg,'oracle_true_source_B',panel,rows,cond)
  evaluate(d,bg,s['q20'][:,None,:],'window_q20',panel,rows,cond)
  evaluate(d,bg,global_template[None,None,:],'TRAIN_global',panel,rows,cond)
  # Project a query-only q20 spectral vector onto TRAIN background shape basis,
  # retaining the observed level. No exact source retrieval or heldout fitting.
  level=np.median(s['q20'],axis=1)[:,None];shape=s['q20']-level
  projection=mu+(shape-mu)@basis.T@basis+level
  evaluate(d,bg,projection[:,None,:],'TRAIN_spectral_PCA3_q20',panel,rows,cond)
  source=wm.iloc[d['base_window_index'].astype(int)].reset_index(drop=True)
  source['month']=pd.to_datetime(source.start_export_clock).dt.month;source['hourbin']=pd.to_datetime(source.start_export_clock).dt.hour//6
  for keys,name in [(('site',),'TRAIN_site_metadata'),(('recorder',),'TRAIN_recorder_metadata'),(('recorder','month','hourbin'),'TRAIN_recorder_month_hour_metadata')]:
   values=[];fallback=0
   for _,r in source.iterrows():
    key=tuple(r[k]for k in keys);t=templates[keys].get(key)
    if t is None:fallback+=1;t=global_template
    values.append(t)
   evaluate(d,bg,np.asarray(values)[:,None,:],name,panel,rows,cond)
   precision.append({'panel':panel,'metadata_template':name,'fallback_global_fraction':fallback/n,'true_metadata_unavailable_in_Test':True})
  # Same q20 spectrum, scaled only by a TRAIN-learned raw-input B integral.
  predmean=10**model.predict(features(d['db']));base=energy(np.broadcast_to(s['q20'][:,None,:],bg.shape))
  obs=np.isfinite(d['db']);base=np.where(obs,base,0.);desired=predmean*obs.sum((1,2));scale=desired/base.sum((1,2))
  estimated=s['q20'][:,None,:]+10*np.log10(scale)[:,None,None]
  evaluate(d,bg,estimated,'learned_totalB_raw_summary',panel,rows,cond)
  Btotal=energy(bg).sum((1,2));Etotal=energy(d['db']).sum((1,2));sensitivity=[]
  for err in (.01,.03,.05,.1):
   sensitivity.append({'target_share_error':err,'B_error_over_E':err,'B_relative_error_needed_p50':float(np.median(err*Etotal/Btotal)),
        'positive_total_B_db_tolerance_p50':float(np.median(10*np.log10(1+err*Etotal/Btotal)))})
  for r in sensitivity:precision.append({'panel':panel,**r})
  note(phase='background_estimators_evaluated',panel=panel,n=n)
 pd.DataFrame(rows).to_csv(OUT/'background_reconstruction_metrics.csv',index=False)
 pd.DataFrame(cond).to_csv(OUT/'background_reconstruction_by_site.csv',index=False)
 pd.DataFrame(precision).to_csv(OUT/'background_precision_and_metadata_access.csv',index=False)
 metrics=pd.DataFrame(rows);main=metrics[metrics.pooling.eq('integral_clip')]
 text='# R18 원본 SA 배경 복원 감사\n\n'
 text+='전체 SA 1,715,779행을 기술 통계로 집계했지만, 평가 배경을 priors에 넣어서는 독립 복원 성능을 주장할 수 없다. 배경 사전분포는 원래 Fold0 TRAIN 배경 '+str(len(trainmeta))+'개만 사용했다. TRAIN과 ES/fresh-fit/gate/report는 원천·그룹 분리 및 24시간 purge를 실제 재검사했다.\n\n'
 text+='`site/recorder/month/hour` 템플릿은 로컬 정답 원천 메타데이터를 알려준 진단 대조군이다. Test의 추정 사이트와 같지 않으며 실제 배포 가능한 성능으로 부르지 않는다. 달/시간은 export clock이고 UTC·현지 시각이 확인되지 않았다.\n\n'
 text+='## 전체 원본 자료\n\n'+pd.read_csv(OUT/'all_SA_rows_by_site.csv').to_markdown(index=False)+'\n\n'
 text+='## 실제 복원 결과: 적분 에너지 방식\n\n'+main[['panel','estimator','n','rmse','bias','B_cell_db_rmse','B_integral_db_rmse']].to_markdown(index=False,floatfmt='.6f')+'\n\n'
 text+='관측 E는 0.1dB 반올림값이다. 원래 배경 B는 정확한 SA 원천에서 검산용으로만 가져왔다. oracle_true_source_B는 배경을 이미 안다는 비현실적 대조이고, 그 잔여 오차는 반올림과 raw target 차이를 포함한다. 다른 추정기는 q/y/원래 B를 입력으로 받지 않는다.\n\n'
 text+='`integral_clip = clip(1-sum(Bhat)/sum(E),0,1)`과 `cell_positive_part = sum(max(E-Bhat,0))/sum(E)`는 다르다. 후자는 개별 셀의 B 과대추정과 과소추정을 비대칭 처리하며 실제 물리 상한 증명이 아니다. 두 결과를 CSV에 모두 보존했다.\n\n'
 text+='## 식별성과 목표 오차의 의미\n\n'
 text+='공통 이득 g를 B/A/E에 함께 곱해도 share는 변하지 않는다. 따라서 절대 dB를 지웠다는 사실만으로 0.07 Bayes 오차 하한이 생긴다고 할 수 없다. 반대로 같은 E=200이라도 (B,A)=(200,0)과 (100,100)은 share=0/0.5로 다르다. raw dB 자체도 성분 식별성을 보장하지 않는다.\n\n'
 text+='정확한 y와 추정 y의 차이는 (Btotal-Bhat_total)/Etotal이다(clip 전). RMSE .033을 달성하려면 배경 적분 오차의 E-normalized RMS가 대략 .033 수준이어야 한다. 이것은 1등 방법이나 반드시 배경을 명시적으로 복원했다는 증거가 아니다. background_precision_and_metadata_access.csv에 .01/.03/.05/.1 share 오차에 해당하는 B 정밀도를 기록했다.\n\n'
 text+='## 제한\n\n평가 패널은 반복 사용한 개발 진단이며 최종 blind 성능이 아니다. 고정 패널에는 유사 참조/녹음일 관계가 남는다. 새 소형 배경 적분 회귀기는 D0/H0 합성 24,000행을 사용했고, 거대 신경망 재학습이나 Test 적합은 하지 않았다. 이 결과만으로 .02~.03 달성 또는 공개 LB와의 정렬을 약속하지 않는다.\n'
 (OUT/'R18_BACKGROUND_RECONSTRUCTION.md').write_text(text)
 pins={str(p.relative_to(ROOT)):sha(p)for p in sorted(OUT.rglob('*'))if p.is_file()and p.name not in ('COMPLETED.json','SHA256SUMS.json','progress.json')}
 pins.update({str(p.relative_to(ROOT)):sha(p)for p in MODELS.glob('*')if p.is_file()});pins[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
 save(OUT/'SHA256SUMS.json',pins);save(OUT/'COMPLETED.json',{'status':'completed','Test_used':False,'source_purge_verified':True,'all_SA_inventory':inventory_summary,'artifact_count':len(pins),'report_sha256':sha(OUT/'R18_BACKGROUND_RECONSTRUCTION.md')})
 note(phase='completed',artifacts=len(pins))

if __name__=='__main__':main()
