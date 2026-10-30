"""Before completed model evaluation, extend R21 full-clock stress coverage.

Preserve the already frozen 82-row bundle; add eligible original fixed-window
domain partial-clock backgrounds. No model outputs or challenge data read.
"""
from pathlib import Path
import json,hashlib,os,sys,datetime as dt,time,warnings
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/r21_clean_val_zero_training_r1/data_multisite'
V1=ROOT/'reports/r21_clean_val_zero_training_r1/data'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item()if isinstance(x,np.generic)else str(x))+'\n')
def guard(e,args):
 if e=='open'and args and isinstance(args[0],(str,bytes,os.PathLike)):
  n=Path(os.fsdecode(args[0])).name.lower()
  if n=='test.csv'or'query'in n or'submission'in n:raise PermissionError('No Test/query in R21 multisite source extension')
def main():
 warnings.filterwarnings('ignore',category=RuntimeWarning);sys.addaudithook(guard)
 from pipeline.dataset_r9 import R9Sources
 from pipeline.source_split_r6 import minute_masks,apply_minute_qc
 from pipeline.sampler_r9 import energy_weights
 from ops.r19_vanilla_law_panels import phys
 from ops.r16_forward_data import pack_components
 from ops.r21_clean_val_data import SEED,hash_order,clock_minutes,nearest
 if (OUT/'manifest.json').exists():raise FileExistsError('Immutable v2 panel already frozen')
 OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 protocol={'created_KST':dt.datetime.now().astimezone().isoformat(),'scope_extension':'Original v1 full15-original-eligibility clock panel remains immutable. Add fixed original domain partial-clock/QC backgrounds satisfying identical post-round physical thresholds before any completed model evaluation or policy selection.',
  'negative_criteria':'HF3162..7943 bump>=15dB AND LF31..200 fraction<=.65 AND Nobs>=3, after original boundary-minute QC and0.1dB rounding',
  'source_disjointness':'All old5fold TRAIN+newTRAIN metadata site/recorder/sensor disjoint; conservative24h support gap within common identity; no threshold relaxation/duplication.',
  'positives':'Same v1 reserved natural reference scalar-y addition; one per negative; alternating weakU(.01,.10)/mediumU(.10,.50). No EQ/warp/repeat/flat.',
  'new_fitting':False,'model_output_access':False,'Test_access':False,'validation_fitting':False,'script_sha256':sha(__file__),'v1_manifest_sha256':sha(V1/'manifest.json')}
 save(OUT/'PROTOCOL_BEFORE_GENERATION.json',protocol)
 with np.load(V1/'clean_val_r20.npz',allow_pickle=False)as z:base={k:z[k]for k in z.files}
 bmeta=pd.read_parquet(V1/'selected_negative_backgrounds.parquet');baseclock={s:i for i,s in enumerate(bmeta.clock_window_id)}
 selected=bmeta.copy();selected['source_scope']='full_clock15_original_eligibility';selected['domain_base_window_index']=-1
 meta=pd.read_parquet(ROOT/'data/versions/quality_v2_native/background_manifest.parquet');domain=meta[meta.split.eq('domain_holdout')].copy().reset_index(drop=True)
 rows=pd.read_parquet(ROOT/'data/processed/sa_row_manifest.parquet');flags=pd.read_parquet(ROOT/'reports/data_quality_boundary_flags.parquet')
 bits,_,_=minute_masks(domain,rows,flags);source=R9Sources(verify_banks=False)
 x=apply_minute_qc(np.asarray(source.background_bank()[domain.base_window_index.to_numpy(int)]),bits);x=np.round(x,1);ft=phys(x)
 take=(ft.bump_high_db>=15)&(ft.LF_energy_fraction<=.65)&(ft.n_obs>=3)
 old=[]
 for f in range(5):old.append(pd.read_parquet(ROOT/f'data/synthetic/r16_multisupport_v1/fold_{f:02d}/train_backgrounds.parquet'))
 train=pd.concat([*old,pd.read_parquet(ROOT/'data/synthetic/r18_production_v1/train_backgrounds.parquet')],ignore_index=True).drop_duplicates('base_window_index')
 trainclock=clock_minutes(train.start_export_clock);site=set(train.site.astype(str));rec=set(train.recorder.astype(str));sens=set(train.sensor_group.astype(str))
 allrows=[];extra=[];conflicts=[]
 for j in np.flatnonzero(take):
  r=domain.iloc[j];st=int(clock_minutes(pd.Series([r.start_export_clock]))[0]);ident=f'{r.site}|{r.recorder}|{st}'
  strict=str(r.site)not in site and str(r.recorder)not in rec and str(r.sensor_group)not in sens
  shared=train.recorder.astype(str).eq(str(r.recorder))|train.sensor_group.astype(str).eq(str(r.sensor_group))
  gap=max(0,float(nearest(np.array([st]),trainclock[shared])[0])-15)if shared.any()else np.inf
  record={'clock_window_id':ident,'domain_base_window_index':int(r.base_window_index),'site':str(r.site),'recorder':str(r.recorder),'sensor_group':str(r.sensor_group),
   'clock_start_minute':st,'start_export_clock':str(r.start_export_clock),'end_export_clock':str(pd.to_datetime(st+15,unit='m')),'clock_date':str(pd.to_datetime(st,unit='m').date()),
   'n_obs':int(ft.n_obs.iloc[j]),'boundary_masked_minutes':int(int(bits[j]).bit_count()),'HF_bump_db':float(ft.bump_high_db.iloc[j]),'LF_energy_fraction':float(ft.LF_energy_fraction.iloc[j]),
   'strict_unseen_site_recorder_sensor':strict,'TRAIN_support_gap_minutes':None if np.isinf(gap)else gap,'TRAIN_24h_purged':gap>=1440,'background_group':f'{r.site}|{r.recorder}|'+str(pd.to_datetime(st,unit='m').date()),
   'source_scope':'original_fixed_domain_partial_clock_QC','domain_position':int(j)}
  allrows.append(record)
  if not strict or gap<1440:continue
  if ident in baseclock:
   olddb=base['db'][2*baseclock[ident]];same=np.array_equal(olddb,x[j],equal_nan=True)
   conflicts.append({'clock_window_id':ident,'same_exact_observation':same,'action':'retained_frozen_v1_observation','domain_position':int(j)})
  else:extra.append(record)
 pd.DataFrame(allrows).to_csv(OUT/'ORIGINAL_DOMAIN_STRICT_PHYSICAL_CANDIDATES.csv',index=False)
 pd.DataFrame(conflicts).to_csv(OUT/'DOMAIN_FULLCLOCK_IDENTITY_DEDUP.csv',index=False)
 extra=sorted(extra,key=lambda r:hash_order(r['clock_window_id'],'extension'))
 reserved=pd.read_parquet(ROOT/'data/synthetic/r17_clean_v1/reserved_B19_Cprime25.parquet');ref=reserved[~reserved.weak_class.eq('background')].copy()
 ref=ref.iloc[np.argsort([hash_order(s,'ref')for s in ref.clip_id.astype(str)])];refids=ref.reference_row_index.to_numpy(int)
 rng=np.random.default_rng(SEED+1);records=[];recipes=[];metarows=[]
 rmeta=rows.copy();rmeta['clock_minute']=clock_minutes(rmeta.t)
 for local,r in enumerate(extra):
  i=len(bmeta)+local;j=r['domain_position'];bg=x[j];ri=int(refids[i%len(refids)]);native=source.reference_minutes[ri];length=min(15,len(native));begin=int(rng.integers(len(native)-length+1));part=native[begin:begin+length]
  legal=[s for s in range(16-length)if np.where(np.isfinite(bg)[s:s+length],part,0.).sum()>0]
  if not legal:raise ValueError('No valid positive observed support')
  placement=int(rng.choice(legal));A=np.zeros((15,25));A[placement:placement+length]=part;target=float(rng.uniform(.01,.1)if i%2==0 else rng.uniform(.1,.5))
  O=np.isfinite(bg);B=np.where(O,10**((bg-np.nanmax(bg))/10),0.);alpha=target*B.sum()/((1-target)*np.where(O,A,0.).sum())
  rid=np.full(15,-1,np.int64);matches=rmeta[rmeta.site.astype(str).eq(r['site'])&rmeta.recorder.astype(str).eq(r['recorder'])&rmeta.clock_minute.between(r['clock_start_minute'],r['clock_start_minute']+14)]
  matches=matches.assign(duration_distance=abs(matches.dur-60)).sort_values(['duration_distance','t','row_id']).drop_duplicates('clock_minute')
  for z in matches.itertuples():rid[int(z.clock_minute-r['clock_start_minute'])]=int(z.row_id)
  for kind,added in [('no_addition',np.zeros((15,25))),('reserved_ship_added',alpha*A)]:
   sid=f'CleanValR20_{i:04d}_'+('zero'if kind=='no_addition'else'pos')
   d,recipe=pack_components(bg,added,{'kind':kind,'requested_target':0. if kind=='no_addition'else target,'alpha':0. if kind=='no_addition'else alpha,'reference_index':-1 if kind=='no_addition'else ri,
    'native_minutes':len(native),'selected_minutes':length,'placement':placement,'crop_begin':begin})
   d.update(sample_id=sid,base_window_index=int(r['domain_base_window_index']),source_clock_id=r['clock_window_id'],background_group=r['background_group'],site=r['site'],recorder=r['recorder'],sensor_group=r['sensor_group'],kind=kind,
    law='no_addition'if kind=='no_addition'else'vanilla_target_scalar',reference_index=-1 if kind=='no_addition'else ri,active_minutes=int(np.any(d['cell_share']>0,1).sum()),weights=energy_weights(d['db']),raw_row_ids=rid)
   records.append(d);recipes.append({'sample_id':sid,**recipe});metarows.append({'sample_id':sid,'background_pair':i,'source_clock_id':r['clock_window_id'],'site':r['site'],'recorder':r['recorder'],'sensor_group':r['sensor_group'],'background_group':r['background_group'],
    'kind':kind,'share':float(d['share']),'reference_index':int(d['reference_index']),'active_minutes':int(d['active_minutes']),'requested_target':0. if kind=='no_addition'else target})
 arrays={k:np.concatenate([base[k],np.asarray([v[k]for v in records])])if records else base[k]for k in base}
 np.savez_compressed(OUT/'clean_val_r20.npz',**arrays)
 sm=pd.concat([pd.read_csv(V1/'SAMPLE_METADATA.csv'),pd.DataFrame(metarows)],ignore_index=True);sm.to_csv(OUT/'SAMPLE_METADATA.csv',index=False)
 selected=pd.concat([selected,pd.DataFrame(extra)],ignore_index=True);selected.to_parquet(OUT/'selected_negative_backgrounds.parquet',index=False)
 pf=pd.concat([sm,phys(arrays['db'])],axis=1);pf.to_csv(OUT/'CLEAN_VAL_PHYSICAL_ROWS.csv',index=False)
 n=len(selected);assert len(arrays['share'])==2*n and (arrays['share']==0).sum()==n and len(set(arrays['sample_id']))==2*n
 assert ((pf.kind=='no_addition')&((pf.bump_high_db<15)|(pf.LF_energy_fraction>.65)|(pf.n_obs<3))).sum()==0
 assert selected.clock_window_id.nunique()==n
 stats=[]
 for (site,rec),g in selected.groupby(['site','recorder']):
  starts=np.sort(g.clock_start_minute.to_numpy(int));end=None;blocks=0;total=0;nonover=0;last=None
  for st in starts:
   if end is None or st>end:blocks+=1;total+=15
   else:total+=max(0,int(st)+15-end)
   end=max(end if end is not None else int(st),int(st)+15)
   if last is None or st>=last:nonover+=1;last=int(st)+15
  stats.append({'site':site,'recorder':rec,'windows':len(g),'connected_support_blocks':blocks,'union_support_minutes':total,'nonoverlap15_window_count':nonover,'recorder_days':g.clock_date.nunique()})
 pd.DataFrame(stats).to_csv(OUT/'INTERVAL_INDEPENDENCE_BY_RECORDER.csv',index=False)
 # Exact old R20 reuse inventory by input array bytes, never prediction values.
 known={}
 for p in (ROOT/'reports/r20_integrated_cpu_diagnostics_r1/hard_panels/panels').glob('*.npz'):
  with np.load(p)as z:
   for sid,v in zip(z['sample_id'].astype(str),z['db']):known.setdefault(np.nan_to_num(v,nan=-1e20).astype(np.float32).tobytes(),[]).append((p.name,sid))
 reuse=[]
 for sid,v in zip(arrays['sample_id'].astype(str),arrays['db']):
  hits=known.get(np.nan_to_num(v,nan=-1e20).astype(np.float32).tobytes(),[]);reuse.append({'sample_id':sid,'exact_R20_input_reusable':bool(hits),'R20_panel_names':'|'.join(a for a,b in hits),'R20_sample_ids':'|'.join(b for a,b in hits)})
 pd.DataFrame(reuse).to_csv(OUT/'SAVED_R20_EXACT_INPUT_REUSE_INVENTORY.csv',index=False)
 save(OUT/'GENERATION_RECIPES_EXTENSION.json',recipes)
 save(OUT/'SOURCE_AUDIT.json',{'v1_backgrounds':len(bmeta),'original_domain_qualifying':len(allrows),'same_clock_as_v1':len(conflicts),'additional_domain_partial_backgrounds':len(extra),
  'selected_backgrounds':n,'rows':2*n,'zero_fraction':.5,'weak_positive_rows':int(((arrays['share']>0)&(arrays['share']<=.1)).sum()),'site_counts':selected.site.value_counts().to_dict(),
  'nobs_counts':selected.n_obs.value_counts().sort_index().to_dict(),'sites':selected.site.nunique(),'recorders':selected.recorder.nunique(),'recorder_days':selected.background_group.nunique(),
  'connected_support_blocks':sum(r['connected_support_blocks']for r in stats),'nonoverlap15_window_count':sum(r['nonoverlap15_window_count']for r in stats),'source_fit':False,
  'strict_TRAIN_overlap':{'site':len(set(selected.site)&set(train.site)),'recorder':len(set(selected.recorder)&set(train.recorder)),'sensor_group':len(set(selected.sensor_group.astype(str))&set(train.sensor_group.astype(str)))},
  'no_Test_representativeness_claim':True,'HF_ge12_zero_axis_duplicate':True})
 inputs=[V1/'manifest.json',V1/'clean_val_r20.npz',ROOT/'data/versions/quality_v2_native/background_manifest.parquet',ROOT/'data/processed/sa_row_manifest.parquet',ROOT/'reports/data_quality_boundary_flags.parquet',ROOT/'data/synthetic/r17_clean_v1/reserved_B19_Cprime25.parquet']
 save(OUT/'INPUT_SHA256.json',{str(p.relative_to(ROOT)):sha(p)for p in inputs})
 hashes={p.name:sha(p)for p in OUT.iterdir()if p.is_file()and p.name not in ('manifest.json','COMPLETED.json')}
 save(OUT/'manifest.json',{'status':'frozen_before_completed_model_evaluation','created_KST':dt.datetime.now().astimezone().isoformat(),'panel':'clean_val_r20','rows':2*n,'zero_rows':n,'positive_rows':n,'weak_positive_rows':int(((arrays['share']>0)&(arrays['share']<=.1)).sum()),
  'files':hashes,'script_sha256':sha(__file__),'new_training':False,'no_Test':True,'source_identity':source.identity,'v1_immutable':True,'elapsed_seconds':time.perf_counter()-start})
 save(OUT/'COMPLETED.json',{'status':'completed','manifest_sha256':sha(OUT/'manifest.json'),'rows':2*n,'seconds':time.perf_counter()-start})
 print(json.dumps({'rows':2*n,'backgrounds':n,'new_domain':len(extra),'sites':selected.site.value_counts().to_dict(),'nobs':selected.n_obs.value_counts().sort_index().to_dict(),'seconds':time.perf_counter()-start},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
