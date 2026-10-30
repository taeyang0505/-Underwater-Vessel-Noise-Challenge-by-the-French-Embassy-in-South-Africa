"""Frozen, no-training Clean-Val-R20 rare natural HF stress panel.

No Test arrays, model outputs, model fitting, or target-AUC search. Full SA
clock-window candidate search retains original basic row eligibility and masks
known boundary candidate minutes. An input-only threshold defines a stress
cohort, NOT a proven representative official-Test-negative distribution.
"""
from pathlib import Path
import datetime as dt, hashlib, json, os, sys, time, warnings
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/r21_clean_val_zero_training_r1/data'
SEED=202610302121
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(2**20),b''):h.update(v)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def now():return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec='seconds')
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  n=Path(os.fsdecode(args[0])).name.lower()
  if 'query' in n or 'submission' in n or n=='test.csv':raise PermissionError('No challenge query/submission reads in R21 panel generation')
def hash_order(s,tag):return hashlib.sha256(f'{SEED}|{tag}|{s}'.encode()).hexdigest()
def clock_minutes(s):return pd.to_datetime(s).to_numpy(dtype='datetime64[m]').astype(np.int64)
def nearest(starts,other):
 if not len(other):return np.full(len(starts),np.inf)
 other=np.sort(other);p=np.searchsorted(other,starts)
 return np.minimum(abs(starts-other[np.clip(p-1,0,len(other)-1)]),abs(starts-other[np.clip(p,0,len(other)-1)])).astype(float)
def block_stats(f):
 byrec=[]
 for (site,rec),g in f.groupby(['site','recorder'],sort=True):
  starts=np.sort(g.clock_start_minute.to_numpy(int));end=None;blocks=0;support=0;kept=[]
  for st in starts:
   if end is None or st>=end:blocks+=1;support+=15;end=int(st)+15;kept.append(int(st))
   else:support+=max(0,int(st)+15-end);end=max(end,int(st)+15)
  # Maximum-cardinality non-overlapping fixed-length windows is greedy by start.
  last=None;nonoverlap=0
  for st in starts:
   if last is None or st>=last:nonoverlap+=1;last=int(st)+15
  byrec.append({'site':site,'recorder':rec,'windows':len(g),'interval_union_blocks':blocks,'union_support_minutes':support,'nonoverlap_window_count':nonoverlap,'days':g.clock_date.nunique()})
 return pd.DataFrame(byrec)
def main():
 warnings.filterwarnings('ignore',category=RuntimeWarning);sys.addaudithook(guard)
 from pipeline.dataset_r9 import R9Sources
 from pipeline.sampler_r9 import energy_weights
 from ops.r16_forward_data import pack_components
 from ops.r19_vanilla_law_panels import phys
 if (OUT/'manifest.json').exists():raise FileExistsError('R21 immutable panel already built')
 OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 protocol={'created_KST':now(),'version':'Clean-Val-R20_natural_HF_stress_r1','seed':SEED,
  'negative_criteria':'Rounded0.1dB observed SA: sum-linear-power HF3162.3..7943.3Hz max-minus-time-median >=15dB AND LF31.6..199.5Hz fraction<=.65 AND Nobs>=3',
  'prevalence':'Exactly one no-addition row and one added row per unique selected background; 50% y=0',
  'source_search':'All original SA 15-minute, stride5 clock windows with all15 original row-eligible; known flagged boundary minutes NaN before threshold. Original fixed domain pool audited separately.',
  'priority':'Strict unseen site+recorder+sensor metadata versus all5 oldTRAIN union+newTRAIN; time-purge-only extra candidates reported separately, never mixed silently.',
  'TRAIN_purge':'Conservative15-minute support gap>=1440minutes for any common recorder OR sensor. Different recorder/sensor has no shared chronology; serial hardware independence not established.',
  'max_negative_rows':1000,'sampling':'All strict rare candidates if<=1000; otherwise deterministic SHA order. No duplication or threshold relaxation.',
  'positive_recipe':'Reserved B19/Cprime25 non-background measured references; full native <=15min / contiguous15crop if longer; random observed valid placement, no EQ/warp/repeat/fade/flat. One scalar alpha to predeclared y prior.',
  'positive_target_prior':'First alternating half U(.01,.10), remaining U(.10,.50), labels recomputed from raw A/B/E. No challenge fitting.',
  'validation_fit':'No validation fitting regardless N; fixed candidates only. Source groups/day/interval overlap explicitly reported.',
  'limitations':['Stress selection is not evidence of official negative distribution.','Matched positive and negative share a background and are not independent rows.','Overlapping15-minute windows are not independent recordings.','All local analysis pools have been examined before; this is not untouched final confirmation.'],
  'new_training':False,'GPU_MPS':False,'Test_read':False,'production_policy_changed':False,'script_sha256':sha(__file__)}
 save(OUT/'PROTOCOL_BEFORE_GENERATION.json',protocol)
 paths=['data/processed/sa_row_manifest.parquet','data/processed/sa_rows_db.npy','reports/data_quality_boundary_flags.parquet',
 'data/synthetic/r17_clean_v1/reserved_B19_Cprime25.parquet']
 old=[]
 for f in range(5):
  p=f'data/synthetic/r16_multisupport_v1/fold_{f:02d}/train_backgrounds.parquet';paths.append(p);old.append(pd.read_parquet(ROOT/p))
 newpath='data/synthetic/r18_production_v1/train_backgrounds.parquet';paths.append(newpath)
 tr=pd.concat([*old,pd.read_parquet(ROOT/newpath)],ignore_index=True).drop_duplicates('base_window_index')
 tr['clock_minute']=clock_minutes(tr.start_export_clock)
 tr_sites=set(tr.site.astype(str));tr_recs=set(tr.recorder.astype(str));tr_sensors=set(tr.sensor_group.astype(str))
 rec_clock={str(k):g.clock_minute.to_numpy(int) for k,g in tr.groupby('recorder',sort=False)}
 sensor_clock={str(k):g.clock_minute.to_numpy(int) for k,g in tr.groupby('sensor_group',sort=False)}
 meta=pd.read_parquet(ROOT/paths[0]);raw=np.load(ROOT/paths[1],mmap_mode='r',allow_pickle=False)
 flags=set(pd.read_parquet(ROOT/paths[2],columns=['row_id']).row_id.astype(int))
 candidates=[];arrays=[];row_ids=[];counts=[];original_eligible=0
 for (site,rec),h in meta.groupby(['site','recorder'],sort=False,observed=True):
  h=h.copy();h['minute']=clock_minutes(h.t);h['duration_distance']=abs(h.dur-60)
  h=h.sort_values(['duration_distance','t','row_id']).drop_duplicates('minute').sort_values('minute')
  minute=h.minute.to_numpy(int);ids=h.row_id.to_numpy(int);elig=h.eligible_for_minute_window.to_numpy(bool)
  sen=h.sensor_group.astype(str).to_numpy();flag=np.array([i in flags for i in ids])
  starts=np.unique(np.concatenate([(minute//5)*5-z for z in (0,5,10)]));localgood=0;localjoint=0
  for off in range(0,len(starts),1024):
   st=starts[off:off+1024];grid=st[:,None]+np.arange(15);ix=np.searchsorted(minute,grid);safe=np.minimum(ix,len(minute)-1)
   good=((ix<len(minute))&(minute[safe]==grid)).all(1)&elig[safe].all(1)
   original_eligible+=int(good.sum());localgood+=int(good.sum())
   if not good.any():continue
   st=st[good];safe=safe[good];x=np.array(raw[ids[safe]],copy=True);bound=flag[safe];x[bound]=np.nan;x=np.round(x,1)
   ft=phys(x);take=(ft.bump_high_db>=15)&(ft.LF_energy_fraction<=.65)&(ft.n_obs>=3)
   localjoint+=int(take.sum())
   for j in np.flatnonzero(take):
    sensors=sorted(set(sen[safe[j]]));sensor=str(sensors[0]) if len(sensors)==1 else '|'.join(sensors)
    strict=str(site)not in tr_sites and str(rec)not in tr_recs and not(set(sensors)&tr_sensors)
    gaps=[max(0,float(nearest(np.array([st[j]]),rec_clock.get(str(rec),np.array([],int)))[0])-15)]
    gaps.extend(max(0,float(nearest(np.array([st[j]]),sensor_clock.get(s,np.array([],int)))[0])-15) for s in sensors)
    gap=min(gaps);purged=gap>=1440
    identity=f'{site}|{rec}|{int(st[j])}';date=str(pd.to_datetime(int(st[j]),unit='m').date())
    r={'clock_window_id':identity,'site':str(site),'recorder':str(rec),'sensor_group':sensor,'clock_start_minute':int(st[j]),'clock_date':date,
      'start_export_clock':str(pd.to_datetime(int(st[j]),unit='m')),'end_export_clock':str(pd.to_datetime(int(st[j])+15,unit='m')),
      'n_obs':int(ft.n_obs.iloc[j]),'boundary_masked_minutes':int(bound[j].sum()),'HF_bump_db':float(ft.bump_high_db.iloc[j]),'LF_energy_fraction':float(ft.LF_energy_fraction.iloc[j]),
      'strict_unseen_site_recorder_sensor':strict,'TRAIN_support_gap_minutes':None if np.isinf(gap) else gap,'TRAIN_24h_purged':purged,
      'source_file_count':int(h.iloc[safe[j]].src_file.nunique()),'background_group':f'{site}|{rec}|{date}','candidate_array_index':len(arrays)}
    candidates.append(r);arrays.append(x[j]);row_ids.append(ids[safe[j]])
   save(OUT/'progress.json',{'site':str(site),'recorder':str(rec),'original_eligible_windows_examined':original_eligible,'joint_candidates':len(candidates),'elapsed_seconds':time.perf_counter()-start})
  counts.append({'site':str(site),'recorder':str(rec),'original_eligible15_windows':localgood,'after_boundary_QC_joint_candidates':localjoint})
 f=pd.DataFrame(candidates);f.to_csv(OUT/'ALL_SA_STRICT_PHYSICAL_CANDIDATES.csv',index=False)
 pd.DataFrame(counts).to_csv(OUT/'SA_WINDOW_SEARCH_BY_RECORDER.csv',index=False)
 assert original_eligible==256029,original_eligible
 strict=f[f.strict_unseen_site_recorder_sensor&f.TRAIN_24h_purged].copy()
 temporal=f[f.TRAIN_24h_purged&~f.strict_unseen_site_recorder_sensor].copy()
 strict=strict.assign(_rank=strict.clock_window_id.map(lambda x:hash_order(x,'selection'))).sort_values('_rank').head(1000).drop(columns='_rank').reset_index(drop=True)
 if not len(strict):raise ValueError('No strict unseen natural HF negatives; cannot silently relax')
 strict.to_parquet(OUT/'selected_negative_backgrounds.parquet',index=False)
 temporal.to_csv(OUT/'TEMPORAL_PURGE_ONLY_EXTRA_CANDIDATES_NOT_IN_MAIN.csv',index=False)
 bs=block_stats(strict);bs.to_csv(OUT/'INTERVAL_INDEPENDENCE_BY_RECORDER.csv',index=False)
 save(OUT/'SOURCE_AUDIT.json',{'old_TRAIN_union_windows':len(pd.concat(old).drop_duplicates('base_window_index')),'old_plus_new_TRAIN_union_windows':len(tr),
  'raw_SA_observation_rows':len(meta),'original_eligible15_windows':original_eligible,'all_joint_QC_candidates':len(f),'strict_unseen_joint_candidates':len(strict),
  'temporal_purge_only_extra_candidates':len(temporal),'selected_negatives':len(strict),'selected_unique_recorder':strict.recorder.nunique(),'selected_unique_site':strict.site.nunique(),
  'selected_unique_recorder_day':strict.background_group.nunique(),'interval_union_blocks':int(bs.interval_union_blocks.sum()),'nonoverlap_window_count':int(bs.nonoverlap_window_count.sum()),
  'known_boundary_minutes_masked':int(strict.boundary_masked_minutes.sum()),'sourceclock_is_not_verified_UTC':True,'strict_metadata_train_overlap':{'site':len(set(strict.site)&tr_sites),'recorder':len(set(strict.recorder)&tr_recs),'sensor_group':len(set(strict.sensor_group)&tr_sensors)},
  'no_fit_split':'No fitting, all candidates predeclared. Reused window support assessed via interval blocks, not inflated n.',
  'HF_ge12_zero_slice_duplicates_all_zero_slice':True,'all_HF_negative_group_median_ge15':float(strict.HF_bump_db.median())>=15})
 source=R9Sources(verify_banks=False);reserved=pd.read_parquet(ROOT/paths[3]);ref=reserved[~reserved.weak_class.eq('background')].copy()
 ref=ref.iloc[np.argsort([hash_order(x,'ref')for x in ref.clip_id.astype(str)])];refids=ref.reference_row_index.to_numpy(int)
 rng=np.random.default_rng(SEED);records=[];recipes=[];samplemeta=[]
 for i,r in enumerate(strict.itertuples(index=False)):
  bg=arrays[int(r.candidate_array_index)];ids=row_ids[int(r.candidate_array_index)];ri=int(refids[i%len(refids)]);native=source.reference_minutes[ri]
  length=min(15,len(native));begin=int(rng.integers(len(native)-length+1));part=native[begin:begin+length]
  legal=[j for j in range(16-length) if np.where(np.isfinite(bg)[j:j+length],part,0.).sum()>0]
  if not legal:raise ValueError('No nonzero observed support for reserved reference')
  placement=int(rng.choice(legal));A=np.zeros((15,25));A[placement:placement+length]=part
  target=float(rng.uniform(.01,.10)if i%2==0 else rng.uniform(.10,.50))
  O=np.isfinite(bg);B=np.where(O,10**((bg-np.nanmax(bg))/10),0.);alpha=target*B.sum()/((1-target)*np.where(O,A,0.).sum())
  for kind,added in [('no_addition',np.zeros((15,25))),('reserved_ship_added',alpha*A)]:
   sid=f'CleanValR20_{i:04d}_'+('zero'if kind=='no_addition'else'pos')
   d,receipt=pack_components(bg,added,{'kind':kind,'requested_target':0. if kind=='no_addition'else target,'alpha':0. if kind=='no_addition'else alpha,
    'reference_index':-1 if kind=='no_addition'else ri,'native_minutes':len(native),'crop_begin':begin,'selected_minutes':length,'placement':placement})
   d.update(sample_id=sid,base_window_index=-1,source_clock_id=r.clock_window_id,background_group=r.background_group,site=r.site,recorder=r.recorder,
    sensor_group=r.sensor_group,kind=kind,law='no_addition'if kind=='no_addition'else'vanilla_target_scalar',reference_index=-1 if kind=='no_addition'else ri,
    active_minutes=int(np.any(d['cell_share']>0,axis=1).sum()),weights=energy_weights(d['db']),raw_row_ids=ids)
   records.append(d);recipes.append({'sample_id':sid,**receipt});samplemeta.append({'sample_id':sid,'background_pair':i,'source_clock_id':r.clock_window_id,
    'site':r.site,'recorder':r.recorder,'sensor_group':r.sensor_group,'background_group':r.background_group,'kind':kind,'share':float(d['share']),'reference_index':int(d['reference_index']),
    'active_minutes':int(d['active_minutes']),'requested_target':0. if kind=='no_addition'else target})
 data={k:np.asarray([r[k] for r in records])for k in records[0]};np.savez_compressed(OUT/'clean_val_r20.npz',**data)
 pd.DataFrame(samplemeta).to_csv(OUT/'SAMPLE_METADATA.csv',index=False)
 physical=phys(data['db']);physical=pd.concat([pd.DataFrame(samplemeta),physical],axis=1);physical.to_csv(OUT/'CLEAN_VAL_PHYSICAL_ROWS.csv',index=False)
 assert len(records)==2*len(strict) and (data['share']==0).sum()==len(strict)
 assert np.array_equal(data['presence']>0,data['share']>0)
 assert ((physical.kind=='no_addition')&((physical.bump_high_db<15)|(physical.LF_energy_fraction>.65))).sum()==0
 save(OUT/'GENERATION_RECIPES.json',recipes)
 # Reuse inventory based ONLY on exact input observations; no model values read.
 known={}
 oldpanels=ROOT/'reports/r20_integrated_cpu_diagnostics_r1/hard_panels/panels'
 for p in sorted(oldpanels.glob('*.npz')):
  with np.load(p,allow_pickle=False)as z:
   for sid,x in zip(z['sample_id'].astype(str),z['db']):known.setdefault(np.nan_to_num(x,nan=-1e20).astype(np.float32).tobytes(),[]).append((p.name,sid))
 reuse=[]
 for sid,x in zip(data['sample_id'].astype(str),data['db']):
  hits=known.get(np.nan_to_num(x,nan=-1e20).astype(np.float32).tobytes(),[])
  reuse.append({'sample_id':sid,'exact_R20_input_reusable':bool(hits),'R20_panel_names':'|'.join(a for a,b in hits),'R20_sample_ids':'|'.join(b for a,b in hits)})
 pd.DataFrame(reuse).to_csv(OUT/'SAVED_R20_EXACT_INPUT_REUSE_INVENTORY.csv',index=False)
 inputs={p:sha(ROOT/p)for p in paths};inputs['data/processed/sa_rows_db.npy']=sha(ROOT/'data/processed/sa_rows_db.npy')
 save(OUT/'INPUT_SHA256.json',inputs)
 hashes={p.name:sha(p)for p in OUT.iterdir()if p.is_file()and p.name not in ('manifest.json','COMPLETED.json')}
 save(OUT/'manifest.json',{'status':'frozen_before_model_inference','created_KST':now(),'panel':'clean_val_r20','rows':len(records),'zero_rows':len(strict),'positive_rows':len(strict),
  'weak_positive_rows':int(((data['share']>0)&(data['share']<=.1)).sum()),'unique_backgrounds':len(strict),'files':hashes,'script_sha256':sha(__file__),
  'no_training':True,'no_Test':True,'source_identity':source.identity,'target_AUC_selection':False,'elapsed_seconds':time.perf_counter()-start,
  'input_fields_for_model':'db only; true A/B/E/q and raw_row_ids are diagnostic truth, never model inputs','validation_representative_of_Test_negative_claim':False})
 save(OUT/'COMPLETED.json',{'status':'completed','rows':len(records),'manifest_sha256':sha(OUT/'manifest.json'),'elapsed_seconds':time.perf_counter()-start})
 print(json.dumps({'status':'completed','rows':len(records),'strict_negative':len(strict),'temporal_only_extra':len(temporal),'union_blocks':int(bs.interval_union_blocks.sum()),'nonoverlap_windows':int(bs.nonoverlap_window_count.sum()),'seconds':time.perf_counter()-start},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
