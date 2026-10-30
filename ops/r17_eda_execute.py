"""Actual xarray/hvPlot/Evidently execution. Frozen inputs, no fitting on Test."""
import os
os.environ['DO_NOT_TRACK']='1'
import sys,json,warnings,argparse,gc,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.r17_audit_common import *
from pipeline.r16_multisupport_data import role
import xarray as xr

NAMES=['selection','local_gate','local_report','domain_holdout','native12_VS_hf_high_negative','native12_VS_st_francis_dense_negative']
def add_meta(f,d,label):
 f=f.copy();f['dataset']=label
 for k in ['sample_id','share','reference_index','base_window_index','site','background_group','operator','family','active_minutes']:
  if k in d and len(d[k])==len(f):f[k]=d[k]
 if 'cell_share' in d:
  q=d['cell_share'];f['support_minutes']=(q>0).any(2).sum(1)
  o=np.isfinite(d['db']);base=np.nanmax(d['db'],axis=(1,2),keepdims=True);E=np.where(o,10**((d['db']-base)/10),0)
  f['rounded_weight_pool_error']=(E*q).sum((1,2))/E.sum((1,2))-d['share']
  f['q05_contaminated_band_fraction']=((np.where(o,q,np.nan)>0).sum(1)>np.maximum(0,o.sum(1)*.95)).mean(1)
  f['q20_contaminated_band_fraction']=((np.where(o,q,np.nan)>0).sum(1)>np.maximum(0,o.sum(1)*.80)).mean(1)
  f['support_on_first']=np.any(q[:,0]>0,1);f['support_on_last']=np.any(q[:,-1]>0,1)
 return f
def build_tables():
 out=OUT/'eda';out.mkdir(parents=True,exist_ok=True);tables={};registry=[]
 # Every row of both actual Fold0 training arrays is examined.
 for reg in ['D0','H0']:
  d=role(0,reg,'train');f=add_meta(batches_features(d['db']),d,'TRAIN_'+reg);f.to_parquet(out/('TRAIN_'+reg+'.parquet'),index=False);tables['TRAIN_'+reg]=f
  registry.append(dict(dataset='TRAIN_'+reg,n=len(f),unique_backgrounds=f.base_window_index.nunique(),labels=True,scope='all stored Fold0 TRAIN rows'))
  print('FEATURES',reg,len(f),flush=True);del d;gc.collect()
 # Public SA backgrounds correspond exactly to the admitted train pool and QC masks.
 from pipeline.dataset_r9 import R9Sources
 s=R9Sources(verify_banks=False);meta=pd.read_parquet(ROOT/'data/synthetic/r16_multisupport_v1/fold_00/train_backgrounds.parquet');parts=[]
 for off in range(0,len(meta),512):
  db=np.stack([s.background(row)for _,row in meta.iloc[off:off+512].iterrows()]);parts.append(physical_features(db))
 f=pd.concat(parts,ignore_index=True)
 for k in ['site','recorder','base_window_index','background_group']:f[k]=meta[k].values
 f['dataset']='SA_BACKGROUND';f.to_parquet(out/'SA_BACKGROUND.parquet',index=False);tables['SA_BACKGROUND']=f
 registry.append(dict(dataset='SA_BACKGROUND',n=len(f),unique_backgrounds=f.base_window_index.nunique(),labels=False,scope='all admitted Fold0 background windows after boundary-minute QC'))
 refmeta=meta[['base_window_index','recorder']].drop_duplicates('base_window_index').set_index('base_window_index').recorder.to_dict()
 # Include all public background recorder lookups, without Test site identification.
 bm=pd.read_parquet(s.bg_dir/'window_manifest.parquet') if (s.bg_dir/'window_manifest.parquet').exists() else None
 if bm is not None:refmeta.update(bm.set_index('base_window_index').recorder.to_dict())
 from pipeline.r16_multisupport_data import PARENT
 for mp in PARENT.parent.glob('*background*.parquet'):
  z=pd.read_parquet(mp)
  if {'base_window_index','recorder'}<=set(z):refmeta.update(z.drop_duplicates('base_window_index').set_index('base_window_index').recorder.to_dict())
 for reg in ['D0','H0']:
  tables['TRAIN_'+reg]['recorder']=tables['TRAIN_'+reg].base_window_index.map(refmeta)
  tables['TRAIN_'+reg].to_parquet(out/('TRAIN_'+reg+'.parquet'),index=False)
 for name in NAMES:
  d=panel_data(name);f=add_meta(batches_features(d['db']),d,name)
  pred=load(ROOT/f'models/r16_v1_fourway_r2/predictions/{name}.npz');assert np.array_equal(d['sample_id'],pred['sample_id'])
  for j,m in enumerate(MEMBERS):
   f['pred_'+m]=.5*(pred['share_D0'][:,j]+pred['share_H0'][:,j]);f['score_'+m]=.5*(pred['classifier_D0'][:,j]+pred['classifier_H0'][:,j]);f['error_'+m]=f['pred_'+m]-f['share']
  f['recorder']=f.base_window_index.map(refmeta).fillna(f['background_group'].astype(str))
  f.to_parquet(out/(name+'.parquet'),index=False);tables[name]=f;registry.append(dict(dataset=name,n=len(f),unique_backgrounds=f.base_window_index.nunique(),labels=True,scope='all frozen panel rows'))
  print('FEATURES',name,len(f),flush=True)
  if name=='selection':
   for nm,key in [('ensemble_fit','ensemble_fit_row_indices'),('checkpoint_selection','checkpoint_selection_row_indices')]:
    z=f.iloc[pred[key]].copy().reset_index(drop=True);z['dataset']=nm;z.to_parquet(out/(nm+'.parquet'),index=False);tables[nm]=z
  del d;gc.collect()
 # Test: descriptive only. No target, source matching, domain classifier, threshold or policy fitting.
 candidates=list((ROOT/'data').rglob('Test.csv'));p=next((z for z in candidates if 'raw' in str(z)),candidates[0]);t=pd.read_csv(p);cols=[c for c in t if c!='ID'];assert len(cols)==375
 # Use the already verified official input parser, not guessed flattened order.
 from ops.predict_r16_production_nn import read_raw_csv
 ids,db,receipt=read_raw_csv(p)
 f=batches_features(db);f['sample_id']=ids;f['dataset']='Test_descriptive_only';f.to_parquet(out/'Test_descriptive_only.parquet',index=False)
 registry.append(dict(dataset='Test_descriptive_only',n=len(f),unique_backgrounds=None,labels=False,scope='descriptive only; excluded from Evidently reference/current fitting and all policy selection'))
 return tables,registry

def load_tables():return {p.stem:pd.read_parquet(p)for p in (OUT/'eda').glob('*.parquet')}
def summaries():
 out=OUT/'eda';tables=load_tables();rows=[];group=[]
 from scipy.stats import ks_2samp,wasserstein_distance
 numeric=[c for c in tables['TRAIN_D0'].select_dtypes(include='number') if c not in ['reference_index','base_window_index','share','active_minutes','support_minutes'] and not c.startswith('rounded_weight')]
 for name,f in tables.items():
  for c in numeric:
   if c not in f:continue
   a=f[c].dropna().to_numpy(float)
   if len(a):rows.append(dict(dataset=name,feature=c,n_valid=len(a),missing_fraction=1-len(a)/len(f),p10=np.quantile(a,.1),p50=np.quantile(a,.5),p90=np.quantile(a,.9),mean=a.mean()))
 for ref,cur in [('SA_BACKGROUND','TRAIN_D0'),('SA_BACKGROUND','TRAIN_H0'),('TRAIN_D0','TRAIN_H0'),('TRAIN_D0','ensemble_fit'),('TRAIN_D0','local_gate'),('TRAIN_D0','domain_holdout')]:
  for c in numeric:
   if c not in tables[cur] or c not in tables[ref]:continue
   a=tables[ref][c].dropna().to_numpy(float);b=tables[cur][c].dropna().to_numpy(float)
   if len(a) and len(b):group.append(dict(reference=ref,current=cur,feature=c,ks=ks_2samp(a,b).statistic,wasserstein=wasserstein_distance(a,b),median_delta=np.median(b)-np.median(a)))
 csv(out/'all_feature_quantiles.csv',rows);csv(out/'all_group_KS.csv',group)
 cond=[]
 for name,f in tables.items():
  for dim in ['site','recorder','n_obs','operator']:
   if dim not in f:continue
   for key,z in f.groupby(dim,dropna=False):
    for c in ['bump_high_db','bump_low_db','corr_low_high','LF_energy_fraction','HF_longest_bump_run']:
     if c in z:cond.append(dict(dataset=name,condition=dim,value=str(key),n=len(z),feature=c,p10=z[c].quantile(.1),median=z[c].median(),p90=z[c].quantile(.9)))
 csv(out/'conditional_feature_quantiles.csv',cond)
 jsave(out/'feature_dictionary.json',{'db':'0.1dB quantized minute-band observations, NaN kept; no calibration invented','mask':'O=isfinite(db); missing values not treated as quiet audio','n_obs':'count of minutes with at least one observed band','bump_low_db':'max-minus-median of log summed linear band power, bands0..12 (31.6..501.2Hz)','bump_high_db':'same, bands20..24 (3162..7943Hz)','LF_energy_fraction':'rounded observation energy ratio bands0..8 (31.6..199.5Hz), dimensionless','persistence':'number/longest consecutive minutes > temporal q20+3dB, NaN breaks runs','lag':'Pearson on original clock offsets; paired observed cells only; constant/insufficient sequences return NaN','bandXX_relative_median':'band temporal median minus window median across band medians; offset-invariant, retains relative static profile','q05/q20_contaminated':'synthetic known-added q>0 fraction exceeds95%/80% of observed times; conservative occupancy proxy, not proof quantile equals pure B','no_inference_from_Test':'Test descriptive outputs are never reference/current input to fitted thresholds or policy selection','ESS':'inverse squared source-use proportions, concentration proxy not independent samples'})

def evidently_reports():
 from evidently import Report,Dataset,DataDefinition
 from evidently.presets import DataDriftPreset
 out=OUT/'eda/evidently';out.mkdir(parents=True,exist_ok=True);tables=load_tables();records=[]
 cols=['n_obs','missing_first','missing_last','longest_observed_run','bump_high_db','bump_low_db','bump_safe_db','std_broadband_db','corr_low_high','corr_mid_high','lag1_mid_high','lag2_mid_high','LF_energy_fraction','MF_energy_fraction','HF_energy_fraction','HF_longest_bump_run']+[f'band{i:02d}_max_minus_median'for i in range(25)]
 def emit(a,b,label):
  if min(len(a),len(b))<20:return
  cs=[c for c in cols if c in a and c in b and a[c].notna().any() and b[c].notna().any()]
  aa=a[cs].astype(float);bb=b[cs].astype(float)
  definition=DataDefinition(numerical_columns=cs)
  r=Report([DataDriftPreset(method='ks',threshold=.05)]).run(current_data=Dataset.from_pandas(bb,data_definition=definition),reference_data=Dataset.from_pandas(aa,data_definition=definition))
  r.save_html(str(out/(label+'.html')));(out/(label+'.json')).write_text(r.json());records.append(dict(report=label,reference_rows=len(a),current_rows=len(b),columns=len(cs),threshold='KS pvalue<0.05; large N sensitive, use separate KS effect size CSV',Test_used=False));print('EVIDENTLY',label,flush=True)
 for a,b in [('SA_BACKGROUND','TRAIN_D0'),('SA_BACKGROUND','TRAIN_H0'),('TRAIN_D0','TRAIN_H0'),('TRAIN_D0','ensemble_fit'),('TRAIN_D0','local_gate'),('TRAIN_D0','domain_holdout')]:emit(tables[a],tables[b],a+'__'+b)
 # Full conditional tables, separate reports for each common site, recorder and density.
 for dim in ['site','recorder','n_obs']:
  a=tables['SA_BACKGROUND'];b=pd.concat([tables['TRAIN_D0'],tables['TRAIN_H0']],ignore_index=True)
  if dim not in a or dim not in b:continue
  for i,key in enumerate(sorted(set(a[dim].dropna())&set(b[dim].dropna()),key=str)):
   emit(a[a[dim]==key],b[b[dim]==key],f'conditional_{dim}_{i:02d}')
 for i,key in enumerate(sorted(set(tables['TRAIN_D0'].operator)&set(tables['TRAIN_H0'].operator))):
  emit(tables['TRAIN_D0'][tables['TRAIN_D0'].operator==key],tables['TRAIN_H0'][tables['TRAIN_H0'].operator==key],f'conditional_operator_{i:02d}')
 # Prediction drift is saved-output distribution, not a performance or truth test.
 q=pd.read_csv(OUT/'task1/rows_Test_descriptive_only.csv');r16=pd.read_csv(ROOT/'submissions/r16_multisupport_tree5_r1/submission_r16_tree5.csv') if (ROOT/'submissions/r16_multisupport_tree5_r1/submission_r16_tree5.csv').exists() else None
 from evidently.presets import DataDriftPreset
 for label,key in [('R9_vs_R16_LGB','share_lightgbm5'),('R9_vs_R16_V1','share_A_original')]:
  a=pd.DataFrame({'saved_share':q['R9_share'],'saved_score':q['R9_score']});b=pd.DataFrame({'saved_share':q[key],'saved_score':q['score_'+key.removeprefix('share_')]})
  # No DataDrift fit or significance test on Test: explicit descriptive CSV only.
  csv(OUT/'eda'/(label+'_descriptive_only.csv'),[dict(output=c,reference_mean=a[c].mean(),current_mean=b[c].mean(),pearson=a[c].corr(b[c]),reference_p50=a[c].median(),current_p50=b[c].median()) for c in a])
 csv(out/'report_index.csv',records);jsave(out/'execution.json',{'DO_NOT_TRACK':'1','actual_library':'evidently','report_count':len(records),'all_rows_in_each_comparison':True,'test_drift_fit':False,'Test_prediction_descriptive_CSV_only':True})

def source_evidently():
 from evidently import Report,Dataset,DataDefinition
 from evidently.presets import DataDriftPreset
 out=OUT/'source';f=pd.read_csv(out/'reference_509_all.csv');cols=['duration_seconds','native_minutes','LF_fraction','MF_fraction','HF_fraction','broadband_temporal_cv','band_cv_median']
 for col,label in [('A_prime75','A75_vs_excluded434'),('Fold0_TRAIN60','TRAIN60_vs_other449')]:
  definition=DataDefinition(numerical_columns=cols);a=f[~f[col]][cols];b=f[f[col]][cols]
  r=Report([DataDriftPreset(method='ks',threshold=.05)]).run(Dataset.from_pandas(b,data_definition=definition),Dataset.from_pandas(a,data_definition=definition));r.save_html(str(out/(label+'.evidently.html')));(out/(label+'.evidently.json')).write_text(r.json())
 # Complete clip status table, including reserved validation versus excluded source screening.
 aa=pd.read_parquet(ROOT/'data/synthetic/r16_multisupport_v1/reference_A_folds.parquet');r4=pd.read_parquet(ROOT/'data/versions/vtuad_reference_r4_v1/reference_manifest.parquet');excluded=pd.read_parquet(ROOT/'data/versions/vtuad_reference_r4_v1/excluded_manifest.parquet')
 reason=excluded.set_index('clip_id').r4_exclusion_reason.to_dict();known=set(r4.clip_id);asplit=aa.set_index('clip_id').full_cv_fold.to_dict()
 f['r16_status']=[('Fold0_TRAIN60'if x in asplit and asplit[x]!=0 else'Fold0_CV15')if x in asplit else('B_or_C_reserved44'if x in known else'outside_selected119')for x in f.clip_id]
 f['exclusion_reason']=[reason.get(x,'not admitted to selected119 by legacy split/category/graph screening; exact prior reason not inferred here')if x not in known else('B/C reference reservation'if x not in asplit else'Fold0 held reference group'if asplit[x]==0 else'used')for x in f.clip_id]
 csv(out/'reference_509_status_and_reason.csv',f)
 csv(out/'reference_status_counts.csv',f.groupby('r16_status',as_index=False).size())

def gallery():
 import hvplot.xarray, hvplot.pandas, holoviews as hv
 from bokeh.embed import file_html
 from bokeh.resources import INLINE
 from bokeh.models import CustomJS,Select,Div
 from bokeh.layouts import column
 out=OUT/'explorer';out.mkdir(parents=True,exist_ok=True);tables=load_tables();cases=[];arrays=[];truth=[];dataset=[];index=[]
 def append(d,i,label,extra=''):
  cases.append(dict(case=len(cases),label=label,source_index=int(i),sample_id=str(d.get('sample_id',np.arange(len(d['db'])))[i]),site=str(d.get('site',['unknown']*len(d['db']))[i]),share=float(d['share'][i])if 'share'in d else None,description=extra))
  arrays.append(d['db'][i]);truth.append(d['cell_share'][i]if 'cell_share'in d else np.full((15,25),np.nan));dataset.append(label);index.append(i)
 for name in ['ensemble_fit','domain_holdout']:
  f=tables[name];eCat=abs(f.error_catboost5);eDino=abs(f.error_dinov2_small)
  rank=(eDino-eCat if name=='ensemble_fit'else eCat-eDino).sort_values(ascending=False).index[:50]
  d=panel_data('selection'if name=='ensemble_fit'else name)
  sourceidx=load(ROOT/'models/r16_v1_fourway_r2/predictions/selection.npz')['ensemble_fit_row_indices']if name=='ensemble_fit'else np.arange(len(f))
  for i in rank:append(d,int(sourceidx[i]),name,f'|Cat error|={eCat[i]:.5f}; |DINO error|={eDino[i]:.5f}; difference={(eDino[i]-eCat[i]):.5f}')
 for reg in ['D0','H0']:
  d=role(0,reg,'train');positive=d['share']>0
  for title,ix in [('short',positive&(d['active_minutes']<=3)),('long',positive&(d['active_minutes']>=10)),('edge',positive&((d['window_start']<0)|(d['window_start']+d['selected_minutes']>15))),('no_source',~positive)]:
   ids=np.flatnonzero(ix)
   if len(ids):append(d,int(ids[0]),'TRAIN_'+reg+'_'+title,'Original stored acoustic mixture; no new synthesis')
 # Public original/QC pair is explicit; observed artifacts are not silently erased.
 from pipeline.dataset_r9 import R9Sources
 s=R9Sources(verify_banks=False);meta=pd.read_parquet(ROOT/'data/synthetic/r16_multisupport_v1/fold_00/train_backgrounds.parquet')
 for _,row in pd.concat([meta.iloc[:1],meta[meta.boundary_mask_bits>0].iloc[:1]]).iterrows():
  for stage,db in [('raw',s.background_bank()[int(row.base_window_index)]),('minute_QC',s.background(row))]:append({'db':db[None],'site':np.array([row.site]),'sample_id':np.array([row.window_id])},0,'SA_'+stage)
 for i in [0,25,100,300,508]:
  m=s.reference_minutes[i];p=np.full((15,25),np.nan);k=min(len(m),15);p[:k]=10*np.log10(np.maximum(m[:k],1e-30));append({'db':p[None],'sample_id':np.array([s.references.iloc[i].clip_id])},0,'VTUAD_native','uncalibrated raw reference, first up to15 native minutes; remainder NaN')
 ds=xr.Dataset({'db':(('case','minute','frequency_hz'),np.stack(arrays)),'O':(('case','minute','frequency_hz'),np.isfinite(arrays).astype(np.int8)),'true_q':(('case','minute','frequency_hz'),np.stack(truth))},coords={'case':np.arange(len(cases)),'minute':np.arange(1,16),'frequency_hz':FREQ},attrs={'note':'Known q is added-recording ratio, not naturally occurring ship truth. Saved scalar predictions cannot reconstruct neural q_hat.'})
 ds.to_netcdf(out/'representative_windows.xarray.nc',engine='scipy');np.savez_compressed(out/'representative_windows.npz',db=np.stack(arrays),true_q=np.stack(truth),dataset=np.array(dataset),source_index=np.array(index));jsave(out/'cases.json',cases)
 if os.environ.get('R17_PREPARE_ONLY')=='1':print('GALLERY_PREPARED',len(cases),flush=True);return
 # hvPlot actually builds each view. HoloMap selector works fully offline, no server.
 views={}
 masks=load(out/'frozen_neural_masks.npz')if (out/'frozen_neural_masks.npz').exists() else {}
 for i,c in enumerate(cases):
  v=ds.isel(case=i);heat=v.db.hvplot.quadmesh(x='minute',y='frequency_hz',logy=True,clim=(float(np.nanmin(v.db)),float(np.nanmax(v.db))),cmap='Viridis',width=390,height=280,title='db / dB (NaN remains missing)')
  om=v.O.hvplot.quadmesh(x='minute',y='frequency_hz',logy=True,clim=(0,1),width=390,height=280,title='O / observation mask')
  tq=v.true_q.hvplot.quadmesh(x='minute',y='frequency_hz',logy=True,clim=(0,1),cmap='Plasma',width=390,height=280,title='known added cell_share q')
  curves=pd.DataFrame({'minute':np.arange(1,16),**{nm:band_curve(np.array(arrays[i])[None],lo,hi)[0] for nm,lo,hi in [('LF',0,9),('MF',9,20),('HF',20,25)]}}).hvplot.line(x='minute',y=['LF','MF','HF'],width=390,height=280,title='LF / MF / HF log summed power')
  charts=[heat,om,tq,curves]
  for key in ['dinov2_small_D0','dinov2_small_H0','dasheng_base_D0','dasheng_base_H0']:
   if key in masks:
    q=xr.DataArray(np.where(np.isfinite(arrays[i]),masks[key][i],np.nan),name='q_hat',dims=['minute','frequency_hz'],coords={'minute':np.arange(1,16),'frequency_hz':FREQ})
    charts.append(q.hvplot.quadmesh(x='minute',y='frequency_hz',logy=True,clim=(0,1),cmap='Plasma',width=390,height=280,title=key+' q_hat (CPU FP32 replay)'))
  views[f'{i:03d} | {c["label"]} | {c["sample_id"]} | {c["site"]}']=hv.Layout(charts).cols(2)
 hmap=hv.HoloMap(views,kdims=['case']);renderer=hv.renderer('bokeh');renderer.save(hmap,str(out/'window_explorer'),resources=INLINE)
 # Panel otherwise adds two cosmetic remote icons even with inline JS/CSS.
 import re
 p=out/'window_explorer.html';text=p.read_text();text=re.sub(r'<link\b[^>]*href="https://cdn\.holoviz\.org/[^\"]*(?:apple-touch-icon\.png|favicon\.ico)"[^>]*>','',text);p.write_text(text)
 print('GALLERY_DONE',len(cases),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['tables','summaries','evidently','gallery','source_evidently']);args=p.parse_args()
 if args.phase=='tables':
  ts,registry=build_tables();csv(OUT/'eda/dataset_registry.csv',registry)
 elif args.phase=='summaries':summaries()
 elif args.phase=='evidently':evidently_reports()
 elif args.phase=='source_evidently':source_evidently()
 else:gallery()
