"""Predeclared R20 CPU-only hard-panel diagnostics; zero fitting/training.

Only local SA/VTUAD numeric sources are read. Dedicated ambient addition is
positive for the official whole-added-recording energy target. Non-vessel
counterfactual labels are separately named and never called official truth.
"""
from pathlib import Path
import datetime as dt, hashlib, json, os, sys, time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/r20_integrated_cpu_diagnostics_r1/hard_panels'
PANELS=OUT/'panels';SEED=202610302043
NAMES=('hardneg1_natural','hardneg2_ambient','hardneg3_full15','scale_laws')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n');t.replace(p)
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  if Path(os.fsdecode(args[0])).name.lower() in ('test.csv','query.npz','query_predictions.npz','sample_submission.csv'):
   raise PermissionError('R20 hard panels prohibit challenge query/Test access')
def phys(db):
 from ops.r19_vanilla_law_panels import phys as oldphys
 d=oldphys(db);a=np.asarray(db,float);o=np.isfinite(a)
 e=np.where(o,10**((a-np.nanmax(a,axis=(1,2),keepdims=True))/10),0.)
 p=e[:,:,18:25].sum(2);v=np.where(o[:,:,18:25].all(2),10*np.log10(np.maximum(p,1e-300)),np.nan)
 d['hf_2to8_bump_db']=np.nanmax(v,1)-np.nanmedian(v,1)
 return d
def pick(frame,n,tag):
 if len(frame)<n:raise ValueError(f'{tag}: insufficient eligible rows {len(frame)} < {n}')
 key=[hashlib.sha256(f'{SEED}|{tag}|{v}'.encode()).hexdigest() for v in frame.base_window_index]
 return frame.iloc[np.argsort(key,kind='stable')[:n]].reset_index(drop=True)
def main():
 sys.addaudithook(guard)
 from pipeline.dataset_r9 import R9Sources
 from pipeline.source_split_r6 import minute_masks,apply_minute_qc,nearest_gap
 from pipeline.sampler_r9 import native_minutes,energy_weights
 from ops.r16_forward_data import pack_components
 if (OUT/'PANELS_FROZEN.json').exists():raise FileExistsError('Frozen panel bundle already exists')
 OUT.mkdir(parents=True,exist_ok=True);PANELS.mkdir(exist_ok=True)
 protocol={'status':'predeclared_before_generation_and_model_predictions','seed':SEED,'total_rows':4000,
  'panel_rows':dict.fromkeys(NAMES,1000),'no_training':True,'no_optimizer_or_policy_fit':True,'no_Test_read':True,
  'background':'original domain_holdout only, Nobs>=13 after original minute QC; no training-background ID/group/recorder/sensor overlap across all five old folds and new R18',
  'hardneg1':{'negative':500,'positive':500,'negative_criteria':'HF2..8 bump>=12 OR LF31.6..501 bump>=8 OR LF<=200 energy fraction<=.65','positive':'same B, reserved44 measured reference, natural full <=15min / contiguous15 crop, peak SNR uniform[-10,10], no EQ/fade/warp'},
  'hardneg2':{'normal_ship_added':250,'dedicated_ambient16_added':250,'weak_background22_added':250,'SA_no_addition':250,
   'geometry':'matched first min(3,all three reference lengths) native minutes and common insertion start',
   'share_prior':'paired prescribed Beta(1.4,6.6), population mean .175; no Test/LB fitting',
   'labels':'physical_share=sumOA/sumOE and physical_presence=(physical_share>0); separate unverified counterfactual_vessel_presence only for sensitivity'},
  'hardneg3':{'positive':500,'negative':500,'positive_background':'domain_holdout HF2..8 bump<2, Nobs>=13',
   'positive':'constant15 measured-reference mean spectrum; prescribed Beta(1.4,6.6) whole-window energy fraction; artificial nonidentifiability stress only',
   'negative':'same 500 natural-active hardneg1 backgrounds, no added recording',
   'no_selection_by_prediction_or_targetAUC':True,'final_bump_lt2':'measured and reported, not a post-generation rejection gate'},
  'scale_laws':{'paired_sources':300,'peak_rows':300,'integrated_rows':300,'HF_SNR_rows':300,'SA_zero':100,
   'peak':'scalar alpha=10^(U[-10,10]/10)*median_observed_B_broadband/max_native_R_broadband',
   'integrated':'scalar alpha from known B/R to prescribed Beta(1.4,6.6) target; population mean .175',
   'HF_SNR':'one scalar alpha for all25bands from observed 2..8k mean added/background SNR U[5,20]dB; no EQ'},
  'references':'reserved44 preferred for normal/scale; old60 reference exposure audited, ambient/weak background exposure explicitly reported',
  'db':'0.1dB round; physical q/y unrounded components; w from rounded db',
  'known_limitations':['All VTUAD recordings share a transitive corpus connected component; clip/hash holdout is not strict hull/day independence.',
   'Natural SA ship/biological truth unknown, but added-recording share=0 is exact for no-addition rows.',
   'Full15 stationary panel deliberately removes within-window contrast; not asserted to represent hidden Test.',
   'Hard-negative preselection changes diagnostic prevalence; AUC/RMSE cannot predict official LB.']}
 write(OUT/'PROTOCOL_BEFORE_GENERATION.json',protocol)
 source=R9Sources(verify_banks=False)
 srcpaths=['data/versions/quality_v2_native/background_manifest.parquet','data/processed/sa_row_manifest.parquet','reports/data_quality_boundary_flags.parquet',
 'data/synthetic/r17_clean_v1/reserved_B19_Cprime25.parquet','data/versions/vtuad_full_r9/reference_manifest.parquet','data/versions/vtuad_full_r9/ambient_control_manifest.parquet']
 allold=[];exposure=[]
 for f in range(5):
  p=f'data/synthetic/r16_multisupport_v1/fold_{f:02d}/train_backgrounds.parquet';srcpaths.append(p)
  q=pd.read_parquet(ROOT/p);allold.append(q);exposure.append({'model':'old_LGB5','fold':f,'train_backgrounds':len(q)})
 old=pd.concat(allold).drop_duplicates('base_window_index');newpath='data/synthetic/r18_production_v1/train_backgrounds.parquet';srcpaths.append(newpath)
 new=pd.read_parquet(ROOT/newpath);alltrain=pd.concat([old,new]).drop_duplicates('base_window_index')
 orig=pd.read_parquet(ROOT/srcpaths[0]);domain=orig[orig.split.eq('domain_holdout')].copy().reset_index(drop=True)
 bits,_,_=minute_masks(domain,pd.read_parquet(ROOT/srcpaths[1]),pd.read_parquet(ROOT/srcpaths[2]))
 domain['boundary_mask_bits']=bits
 db=apply_minute_qc(np.asarray(source.background_bank()[domain.base_window_index.to_numpy(int)]),bits)
 domain['qc_observed_minutes']=np.isfinite(db).all(2).sum(1)
 features=phys(db);domain=pd.concat([domain,features.add_prefix('physical_')],axis=1)
 for column in ['base_window_index','site','recorder','sensor_group']:
  overlap=set(domain[column])&set(alltrain[column])
  if overlap:raise AssertionError(f'Domain source overlap {column}: {overlap}')
 if 'background_group'not in domain:domain['background_group']='domain_sensor_'+domain.sensor_group.astype(str)
 # Different recorder/sensor components imply no shared chronological support.
 # Explicitly calculate nearest support gaps whenever a metadata sensor matches.
 gaps=[]
 for sensor,g in domain.groupby('sensor_group'):
  tr=alltrain[alltrain.sensor_group.eq(sensor)]
  if len(tr):
   seconds=lambda s:pd.to_datetime(s).to_numpy(dtype='datetime64[ns]').astype('int64')//10**9
   gap=np.maximum(nearest_gap(seconds(g.start_export_clock),seconds(tr.start_export_clock))-900,0)
   gaps.extend(gap.tolist())
 if gaps and min(gaps)<86400:raise AssertionError('24h purge failed')
 dense=domain[domain.qc_observed_minutes.ge(13)].copy()
 hardmask=(dense.physical_hf_2to8_bump_db.ge(12)|dense.physical_bump_low_db.ge(8)|dense.physical_LF_energy_fraction.le(.65))
 negatives=pick(dense[hardmask],500,'natural-hardneg')
 quiet=pick(dense[dense.physical_hf_2to8_bump_db.lt(2)&~dense.base_window_index.isin(negatives.base_window_index)],500,'full15-quiet')
 negatives.to_parquet(OUT/'natural_negative_backgrounds.parquet',index=False);quiet.to_parquet(OUT/'full15_backgrounds.parquet',index=False)
 domain[['base_window_index','site','recorder','sensor_group','qc_observed_minutes']].to_csv(OUT/'domain_eligible_identity.csv',index=False)
 proof={'old_LGB5_union_backgrounds':len(old),'old_per_fold':exposure,'new_R18_backgrounds':len(new),'domain_total':len(domain),
  'domain_dense_Nobs_ge13':len(dense),'natural_active_eligible':int(hardmask.sum()),'quiet_HFbump_lt2_eligible':int(dense.physical_hf_2to8_bump_db.lt(2).sum()),
  'overlap_by_model':[], 'source_disjoint_columns':['base_window_index','site','recorder','sensor_group'],
  'minimum_shared_sensor_support_gap_hours':min(gaps)/3600 if gaps else None,'purge':'no common recorder/sensor metadata component; chronological overlap impossible within admitted identities',
  'identity_limit':'Metadata recorder/sensor identities do not prove serial-number hardware independence.',
  'selected_natural_site':negatives.site.value_counts().to_dict(),'selected_quiet_site':quiet.site.value_counts().to_dict(),
  'natural_negative_criteria_counts':{'HF_ge12':int(negatives.physical_hf_2to8_bump_db.ge(12).sum()),'LFbump_ge8':int(negatives.physical_bump_low_db.ge(8).sum()),'LF_fraction_le065':int(negatives.physical_LF_energy_fraction.le(.65).sum())}}
 for name,tr in [(f'old_fold{f}',v)for f,v in enumerate(allold)]+[('new',new)]:
  proof['overlap_by_model'].append({'model':name,**{c:len(set(domain[c])&set(tr[c]))for c in ['base_window_index','site','recorder','sensor_group']}})
 write(OUT/'SOURCE_HOLDOUT_AUDIT.json',proof)
 reserved=pd.read_parquet(ROOT/'data/synthetic/r17_clean_v1/reserved_B19_Cprime25.parquet')
 weak=source.references[source.references.is_weak_background_label_in_vessel_archive].copy()
 ship=reserved[~reserved.weak_class.eq('background')].copy();shipids=ship.reference_row_index.to_numpy(int)
 assert len(reserved)==44 and len(weak)==22 and len(shipids)>0
 amb=pd.read_parquet(source.ref_dir/'ambient_control_manifest.parquet').sort_values('ambient_row_index');bank=np.load(source.reference_path,mmap_mode='r')
 ambminutes=[]
 for row in amb.itertuples(index=False):
  part=np.asarray(bank[int(row.segment_start):int(row.segment_start)+int(row.segment_count)])
  if hashlib.sha256(np.ascontiguousarray(part).tobytes()).hexdigest()!=row.expected_segment_slice_sha256:raise ValueError('Ambient cache SHA')
  ambminutes.append(native_minutes(part))
 assert len(ambminutes)==16
 refmeta=pd.concat([source.references.assign(source_class='vessel_archive'),amb.assign(source_class='dedicated_ambient')],ignore_index=True)
 refmeta['reserved44']=refmeta.sha256.isin(reserved.sha256)
 oldrefs=pd.read_parquet(ROOT/'data/synthetic/r16_multisupport_v1/fold_00/train_reference_pool.parquet') if (ROOT/'data/synthetic/r16_multisupport_v1/fold_00/train_reference_pool.parquet').exists() else None
 # Exact archive/hash identities and cache exposure; natural class is not an added-energy label.
 refmeta[['clip_id','sha256','name','source_class','archive_role','weak_class','sample_rate_hz','native_complete_minute_count','duration_seconds','licence','source_url','reserved44','segment_start','segment_count','expected_segment_slice_sha256']].to_csv(OUT/'REFERENCE_IDENTITY_AUDIT.csv',index=False)
 write(OUT/'AMBIENT_AUDIT.json',{'dedicated_ambient':len(amb),'weak_background_in_vessel_archive':len(weak),
  'byte_hash_overlap':len(set(amb.sha256)&set(weak.sha256)),'ambient_vs_all_vessel_overlap':len(set(amb.sha256)&set(source.references.sha256)),
  'ambient_sample_rates':{str(k):int(v)for k,v in amb.sample_rate_hz.value_counts().items()},
  'weak_sample_rates':{str(k):int(v)for k,v in weak.sample_rate_hz.value_counts().items()},
  'ambient_complete_minutes_min':int(min(map(len,ambminutes))),'ambient_minutes_total':int(sum(map(len,ambminutes))),
  'old_training_exposure':'dedicated ambient16 excluded; weak-background22 were eligible in all509 R9 archive and audited per current manifest',
  'new_training_exposure':'reserved44 excluded but weak-background class is not necessarily excluded; exact sampled references recorded per row',
  'class_semantics':'ambient/weak label does not prove isolated ship absence; physical added-recording target remains positive',
  'license':sorted(amb.licence.unique().tolist()),'power_cache':str(source.reference_path.relative_to(ROOT)),
  'cache_source_bank_sha256':source.identity['reference_bank_sha256'],'cache_all16_slice_SHA_verified':True,
  'Nyquist_note':'all actual headers/cache metadata audited; no 6.31/7.94k use beyond verified header Nyquist; SA URN_8321 not present in strict domain sample'})
 rng=np.random.default_rng(SEED);rows={name:[]for name in NAMES};receipts=[]
 negbg=[source.background(row)for _,row in negatives.iterrows()]
 def add(name,bgrow,bg,A,reference,kind,law,alpha=0.,cf=None,requested=None):
  d,rec=pack_components(bg,A,{'operator':law,'reference_index':int(reference),'kind':kind})
  i=len(rows[name]);d.update(sample_id=f'R20_{name}_{i:04d}',base_window_index=int(bgrow.base_window_index),site=str(bgrow.site),recorder=str(bgrow.recorder),
   reference_index=int(reference),kind=kind,law=law,alpha=float(alpha),counterfactual_vessel_presence=int(d['presence']if cf is None else cf),
   counterfactual_share=float(d['share']if cf is None or cf else 0.),requested_value=float(requested)if requested is not None else np.nan,
   active_minutes=int(np.any(d['cell_share']>0,1).sum()),weights=energy_weights(d['db']))
  rows[name].append(d);receipts.append({'sample_id':d['sample_id'],**rec})
 def normalizedB(bg):
  return np.where(np.isfinite(bg),10**((bg-float(np.nanmax(bg)))/10),0.)
 def placement(native):
  n=min(15,len(native));begin=int(rng.integers(len(native)-n+1));start=int(rng.integers(16-n));A=np.zeros((15,25));A[start:start+n]=native[begin:begin+n];return A
 def alpha_target(bg,A,y):return float(y)*normalizedB(bg).sum()/((1-float(y))*np.where(np.isfinite(bg),A,0.).sum())
 for i,(r,bg) in enumerate(zip(negatives.itertuples(index=False),negbg)):
  ri=int(shipids[i%len(shipids)]);native=source.reference_minutes[ri];A=placement(native);snr=float(rng.uniform(-10,10))
  alpha=10**(snr/10)*np.median(normalizedB(bg).sum(1)[np.isfinite(bg).all(1)])/native.sum(1).max()
  add(NAMES[0],r,bg,np.zeros((15,25)),-1,'natural_no_addition','no_addition')
  add(NAMES[0],r,bg,alpha*A,ri,'reserved_ship_added','native_peak',alpha,requested=snr)
  add(NAMES[2],r,bg,np.zeros((15,25)),-1,'natural_active_negative','no_addition')
 for i,r in enumerate(quiet.itertuples(index=False)):
  bg=source.background(r._asdict());ri=int(shipids[i%len(shipids)]);A=np.tile(source.reference_minutes[ri].mean(0),(15,1));y=float(rng.beta(1.4,6.6));alpha=alpha_target(bg,A,y)
  add(NAMES[2],r,bg,alpha*A,ri,'artificial_stationary15_positive','constant_full15',alpha,requested=y)
 for i in range(250):
  r=negatives.iloc[i];bg=negbg[i];ri=int(shipids[i%len(shipids)]);wi=int(weak.reference_row_index.iloc[i%len(weak)]);ai=i%16
  triple=[source.reference_minutes[ri],ambminutes[ai],source.reference_minutes[wi]];n=min(3,*map(len,triple));start=int(rng.integers(16-n));y=float(rng.beta(1.4,6.6))
  for native,index,kind,cf in zip(triple,[ri,-1000-ai,wi],['ship_added','ambient16_added','weak_background22_added'],[1,0,0]):
   A=np.zeros((15,25));A[start:start+n]=native[:n];alpha=alpha_target(bg,A,y)
   add(NAMES[1],r,bg,alpha*A,index,kind,'paired_integrated_target',alpha,cf=cf,requested=y)
  add(NAMES[1],r,bg,np.zeros((15,25)),-1,'SA_no_addition','no_addition',cf=0)
 for i in range(300):
  r=negatives.iloc[i];bg=negbg[i];ri=int(shipids[i%len(shipids)]);native=source.reference_minutes[ri];A=placement(native);B=normalizedB(bg);O=np.isfinite(bg)
  snr=float(rng.uniform(-10,10));y=float(rng.beta(1.4,6.6));hsnr=float(rng.uniform(5,20))
  laws=[('native_peak',10**(snr/10)*np.median(B.sum(1)[O.all(1)])/native.sum(1).max(),snr),
   ('integrated_target',alpha_target(bg,A,y),y),
   ('HF_SNR_scalar',10**(hsnr/10)*B[:,18:].sum()/np.where(O,A,0.)[:,18:].sum(),hsnr)]
  for law,alpha,req in laws:add(NAMES[3],r,bg,alpha*A,ri,'reserved_ship_added',law,alpha,requested=req)
  if i<100:add(NAMES[3],r,bg,np.zeros((15,25)),-1,'SA_no_addition','no_addition')
 metadata=[];ph=[]
 for name,records in rows.items():
  assert len(records)==1000
  keys=['sample_id','base_window_index','site','recorder','reference_index','kind','law','alpha','counterfactual_vessel_presence','counterfactual_share','requested_value','active_minutes',
   'db','cell_share','share','presence','observed_mask','weights','background_observed_power','added_full_power','added_observed_power','mixture_observed_power']
  arr={k:np.asarray([r[k]for r in records])for k in keys};p=PANELS/(name+'.npz');np.savez_compressed(p,**arr)
  assert np.array_equal(arr['presence']>0,arr['share']>0)
  for v in records:
   B=v['background_observed_power'];A=v['added_observed_power'];E=v['mixture_observed_power']
   assert abs(A.sum()/E.sum()-v['share'])<1e-12 and np.allclose(B+A,E)
  feat=phys(arr['db']);meta=pd.DataFrame({k:arr[k]for k in keys[:12]});meta['share']=arr['share'];meta['presence']=arr['presence'];meta['panel']=name
  feat=pd.concat([meta,feat],axis=1);ph.append(feat);metadata.append({'panel':name,'n':len(records),'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
  feat.to_csv(OUT/(name+'_physical.csv'),index=False)
 allphys=pd.concat(ph,ignore_index=True);allphys.to_csv(OUT/'ALL_PANEL_PHYSICAL_ROWS.csv',index=False)
 summaries=[]
 for (panel,kind,law),g in allphys.groupby(['panel','kind','law']):
  r={'panel':panel,'kind':kind,'law':law,'n':len(g)}
  for c in ['share','n_obs','LF_energy_fraction','MF_energy_fraction','HF_energy_fraction','bump_high_db','bump_low_db','hf_2to8_bump_db']:
   r[c+'_mean']=float(g[c].mean());r[c+'_p50']=float(g[c].median());r[c+'_p90']=float(g[c].quantile(.9))
  summaries.append(r)
 pd.DataFrame(summaries).to_csv(OUT/'PANEL_PHYSICAL_SUMMARY.csv',index=False)
 write(OUT/'GENERATION_RECIPES.json',receipts)
 pins={p:sha(ROOT/p)for p in srcpaths};pins.update({'ops/r20_hard_panels.py':sha(__file__),'ops/r16_forward_data.py':sha(ROOT/'ops/r16_forward_data.py'),'pipeline/sampler_r9.py':sha(ROOT/'pipeline/sampler_r9.py')})
 source.assert_unchanged(verify_banks=False)
 write(OUT/'PANELS_FROZEN.json',{'status':'frozen_before_predictions','created_KST':dt.datetime.now().astimezone().isoformat(),'panels':metadata,
  'source_identity':source.identity,'input_sha256':pins,'total_rows':4000,'no_Test_read':True,'optimizer_steps':0,
  'stationary_positive_bump_lt2_fraction':float(allphys[allphys.kind.eq('artificial_stationary15_positive')].hf_2to8_bump_db.lt(2).mean()),
  'no_parameter_search':True,'source_holdout_audit_sha256':sha(OUT/'SOURCE_HOLDOUT_AUDIT.json')})
 print(json.dumps({'status':'panels_frozen','path':str(OUT),'total_rows':4000,'panel_sha256':{m['panel']:m['sha256']for m in metadata}},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
