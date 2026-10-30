"""Paired, small, no-training R19 vanilla versus six frozen-law diagnostic.

Reads admitted SA / VTUAD numeric caches only. This is a counterfactual law
sensitivity experiment sharing source backgrounds with past training; it is
not an independent generalization test or a reconstruction of hidden Test.
"""
from pathlib import Path
import hashlib,json,sys,time,datetime,os,warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/r19_multislice_law_transfer_r1/law_panels'
SEED=202610301943;N=2000

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n');t.replace(p)
def note(**v):
 print(json.dumps(v,ensure_ascii=False),flush=True);save(OUT/'progress.json',v)
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0]))
  if p.name.lower() in ('test.csv','sample_submission.csv','query.npz','query_predictions.npz'):
   raise PermissionError('R19 law panel generation prohibits Test inputs/predictions')
def npz(p,a):
 p=Path(p);t=p.with_suffix('.tmp')
 with t.open('wb')as f:np.savez_compressed(f,**a)
 t.replace(p)
def stats(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)]
 return {'n':len(x),'mean':float(x.mean()) if len(x)else None,'rms':float(np.sqrt(np.mean(x*x))) if len(x)else None,**{f'p{int(q*100):02d}':float(np.quantile(x,q))if len(x)else None for q in (.1,.5,.9)}}
def proportional_sample(frame,n,rng):
 # Preserve site x recorder x n_obs composition rather than site balancing.
 groups=list(frame.groupby(['site','recorder','qc_observed_minutes'],sort=True,dropna=False))
 expected=np.array([len(g)*n/len(frame) for _,g in groups]);counts=np.floor(expected).astype(int)
 order=np.argsort(-(expected-counts),kind='stable');counts[order[:n-counts.sum()]]+=1
 picks=[];proof=[]
 for (key,g),k in zip(groups,counts):
  if k:picks.extend(rng.choice(g.index.to_numpy(),int(k),replace=False).tolist())
  proof.append(dict(zip(['site','recorder','n_obs'],key),source_count=len(g),selected_count=int(k),expected_selected=float(len(g)*n/len(frame))))
 selected=frame.loc[picks].iloc[rng.permutation(n)].reset_index(drop=True)
 return selected,pd.DataFrame(proof)
def phys(db):
 a=np.asarray(db,float);o=np.isfinite(a);off=np.nanmax(a,axis=(1,2),keepdims=True);e=np.where(o,10**((a-off)/10),0)
 out={'n_obs':o.all(2).sum(1),'LF_energy_fraction':e[:,:,:9].sum((1,2))/e.sum((1,2)),
      'MF_energy_fraction':e[:,:,9:20].sum((1,2))/e.sum((1,2)), 'HF_energy_fraction':e[:,:,20:].sum((1,2))/e.sum((1,2))}
 for name,lo,hi in [('bump_high_db',20,25),('bump_low_db',0,13)]:
  p=e[:,:,lo:hi].sum(2);v=np.where(o[:,:,lo:hi].all(2),10*np.log10(np.maximum(p,1e-300)),np.nan)
  out[name]=np.nanmax(v,axis=1)-np.nanmedian(v,axis=1)
 return pd.DataFrame(out)
def bins(y):
 return np.select([y==0,(y>0)&(y<=.05),(y>.05)&(y<=.15),y>.15],['y=0','0<y<=0.05','0.05<y<=0.15','y>0.15'],default='invalid')
def main():
 sys.addaudithook(guard)
 from pipeline.dataset_r9 import R9Sources
 from pipeline.sampler_r9 import smooth_eq,energy_weights
 from ops.r16_forward_data import OPERATORS,generate_operator,pack_components,channel_zero_mean
 from ops.r16_support_temporal_data import natural_geometry
 if (OUT/'manifest.json').exists():raise FileExistsError('Completed immutable output; use new version')
 OUT.mkdir(parents=True,exist_ok=True)
 start=time.perf_counter()
 poolpath=ROOT/'data/synthetic/r17_clean_v1/reference_pool_465.parquet';reservedpath=ROOT/'data/synthetic/r17_clean_v1/reserved_B19_Cprime25.parquet'
 code=['ops/r19_vanilla_law_panels.py','ops/r16_forward_data.py','pipeline/sampler_r9.py','pipeline/dataset_r9.py','ops/r16_support_temporal_data.py']
 protocol={'version':'r19_paired_vanilla_law_panel_v1','fixed_before_generation':True,'seed':SEED,'vanilla_rows':N,'custom_law_quotas':dict(zip(OPERATORS,[335,335,335,335,330,330])),'laws':list(OPERATORS)+['vanilla_addition'],
  'total_rows':N*2,'unique_paired_source_rows':N,'zero_fraction_per_law':.2,'positive_fraction_per_law':.8,'custom_total_rows':N,'scope_revision_before_inference':True,
  'background_sampling':'proportional largest-remainder site/recorder/n_obs strata from primary78869, unique without replacement',
  'reference_sampling':'uniform cyclic shuffle of allowed465, reserved44 excluded; train exposure diagnostic, no independent reference claim',
  'geometry':'natural contiguous whole native <=15 minutes; random contiguous15 if longer; valid observed placement enumeration, no warp/repeat/fade',
  'shared_random_draws':'same B,A,crop,position,O,requestedSNR,baseEQ,postEQ,subbandSNR between eachcustomrow and corresponding vanillaID; eachsource assigned exactlyonecustomlaw',
  'requested_SNR_uniform_db':[-10,10],
  'vanilla_alpha':'10**(snr/10)*median_observed_B_broadband / max_whole_native_A_broadband; noEQ; one scalar acrossall25bands',
  'custom_eq':'frozen six-law code; baseEQ smooth ±6dB; postEQ zero-log-mean ±6dB same as oldTRAIN/R18TRAIN',
  'target_share_forcing':False,'Test_used':False,'new_training':False,'new_model_inference_in_this_script':False,
  'physical_label':'known energy of whole measured added recording incl ambient, not isolated vessel truth',
  'q_y_raw_unrounded':True,'db_rounding_decimals':1,'w_from_rounded_db':True,
  'read_inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (poolpath,reservedpath)},
  'code_sha256':{p:sha(ROOT/p)for p in code},
  'warning':'Paired source-training-exposed sensitivity probe; no Test-distribution fitting or hidden-label inference'}
 save(OUT/'PROTOCOL_BEFORE_GENERATION.json',protocol)
 note(stage='load_verified_sources',elapsed_seconds=time.perf_counter()-start)
 source=R9Sources(verify_banks=False);primary=source.role_metadata(0,'train')
 assert len(primary)==78869
 pool=pd.read_parquet(poolpath);reserved=pd.read_parquet(reservedpath)
 assert len(pool)==465 and not(set(pool.sha256)&set(reserved.sha256))
 selected,quota=proportional_sample(primary,N,np.random.default_rng([SEED,0x4247]))
 selected.to_parquet(OUT/'paired_backgrounds.parquet',index=False);quota.to_csv(OUT/'background_strata_sampling.csv',index=False)
 zero=np.zeros(N,bool);zero[np.random.default_rng([SEED,0x5A]).choice(N,N//5,replace=False)]=True
 refids=pool.reference_row_index.to_numpy(int);rng=np.random.default_rng([SEED,0x5246])
 seq=np.concatenate([rng.permutation(refids) for _ in range((int((~zero).sum())+len(refids)-1)//len(refids))])[:int((~zero).sum())]
 refs=np.full(N,-1,np.int64);refs[~zero]=seq
 # Fixed 2000 source draws. Each law receives exactly these sources and masks.
 common={'sample_id':np.asarray([f'R19_LAW_{i:05d}' for i in range(N)]),'base_window_index':selected.base_window_index.to_numpy(np.int64),
  'site':selected.site.to_numpy(str),'recorder':selected.recorder.to_numpy(str),'reference_index':refs,'presence':(~zero).astype(np.int64),
  'explicit_no_source':zero,'requested_snr_db':np.empty(N,np.float64),'selected_minutes':np.zeros(N,np.int64),
  'crop_begin':np.zeros(N,np.int64),'window_start':np.zeros(N,np.int64),'family':np.full(N,'no_source',dtype='<U40')}
 bg=np.stack([source.background(row) for _,row in selected.iterrows()]);common['observed_mask']=np.isfinite(bg)
 baseeq=np.zeros((N,25));posteq=baseeq.copy();subsnr=np.zeros((N,2));geom=[]
 for i in range(N):
  r=np.random.default_rng([SEED,i,0x4348]);g=np.random.default_rng([SEED,i,0x4745])
  common['requested_snr_db'][i]=r.uniform(-10,10);subsnr[i]=r.uniform(-10,10,2)
  if zero[i]:geom.append(None);continue
  native=source.reference_minutes[refs[i]];length=min(15,len(native));begin=int(g.integers(len(native)-length+1));part=native[begin:begin+length]
  legal=[s for s in range(16-length) if np.where(np.isfinite(bg[i,s:s+length]),part,0).sum()>0]
  geometry={'crop_begin_minute':begin,'selected_minutes':length,'window_start_minute':int(g.choice(legal)),
            'family':'natural_whole_native'if len(native)==length else'natural_contiguous15','full_native_minutes':len(native),'time_warp':False,'source_repetition':False,'added_bell_or_fade':False}
  geom.append(geometry)
  for k,j in [('selected_minutes','selected_minutes'),('crop_begin','crop_begin_minute'),('window_start','window_start_minute'),('family','family')]:common[k][i]=geometry[j]
  baseeq[i]=smooth_eq(r,6);posteq[i]=channel_zero_mean(r,6)
 npz(OUT/'background_only.npz',{'db':bg,**common});npz(OUT/'shared_random_draws.npz',{'base_eq_db':baseeq,'post_eq_db':posteq,'subband_snr_db':subsnr,**common})
 # Early preview for CPU inference timing; no model inputs other than admitted SA.
 npz(OUT/'preview128.npz',{'db':bg[:128].astype(np.float32),'share':np.zeros(128),'presence':np.zeros(128,np.int64),'sample_id':common['sample_id'][:128]})
 note(stage='preview_ready',preview='preview128.npz',elapsed_seconds=time.perf_counter()-start)
 oldpath=ROOT/'data/synthetic/r16_multisupport_v1/fold_00/train_backgrounds.parquet';newpath=ROOT/'data/synthetic/r18_production_v1/train_backgrounds.parquet'
 oldbg=pd.read_parquet(oldpath);newbg=pd.read_parquet(newpath)
 oldrefpath=ROOT/'data/synthetic/r16_multisupport_v1/fold_00/reference_pool.parquet'
 exposure={'background_pool_primary':len(primary),'selected_unique_backgrounds':N,'old_R16_train_background_overlap':int(selected.base_window_index.isin(oldbg.base_window_index).sum()),
 'R18_train_background_overlap':int(selected.base_window_index.isin(newbg.base_window_index).sum()),'R18_positive_references_all_seen':True,
 'source_background_SHA':source.bg_manifest['background_bank']['sha256'],'source_reference_SHA':source.ref_manifest['bank']['sha256'],'identity':source.identity,
 'metadata_SHA':{str(p.relative_to(ROOT)):sha(p)for p in (oldpath,newpath)}}
 save(OUT/'SOURCE_EXPOSURE.json',exposure)
 bgf=phys(bg);bgf.to_csv(OUT/'background_only_physical_features.csv',index=False)
 bg_rows=[]
 for label,path,selector in [('primary78869',ROOT/'reports/r18_public_gap_audit_r1/background/R9_allowed100965_QC_features.csv.gz',None)]:
  cache=pd.read_csv(path)
  if 'in_primary78869'in cache:cache=cache[cache.in_primary78869.eq(True)]
  elif 'base_window_index'in cache:cache=cache[cache.base_window_index.isin(primary.base_window_index)]
  else:raise ValueError('Cannot align primary feature cache')
  assert len(cache)==78869
  for c in ['n_obs','LF_energy_fraction','bump_high_db','bump_low_db']:
   s=stats(cache[c]);b=stats(bgf[c]);bg_rows.append({'feature':c,'primary_n':len(cache),'sample_n':len(bgf),'primary_p50':s['p50'],'sample_p50':b['p50'],'primary_p90':s['p90'],'sample_p90':b['p90'],'KS':float(ks_2samp(cache[c],bgf[c]).statistic)})
 pd.DataFrame(bg_rows).to_csv(OUT/'background_distribution_preservation.csv',index=False)
 assignment_rng=np.random.default_rng([SEED,0x4C4157]);posids=assignment_rng.permutation(np.flatnonzero(~zero));negids=assignment_rng.permutation(np.flatnonzero(zero));subsets={};ps=ns=0
 for law,quota_n in zip(OPERATORS,[335,335,335,335,330,330]):
  pcount=quota_n*4//5;ncount=quota_n//5;subsets[law]=np.sort(np.r_[posids[ps:ps+pcount],negids[ns:ns+ncount]]);ps+=pcount;ns+=ncount
 subsets['vanilla_addition']=np.arange(N)
 assert len(set(np.concatenate([subsets[k] for k in OPERATORS])))==N
 pd.DataFrame({'sample_id':common['sample_id'],'custom_law':[next(k for k in OPERATORS if i in subsets[k])for i in range(N)]}).to_csv(OUT/'custom_law_assignment_locked_before_inference.csv',index=False)
 artifacts={};summary=[];rows=[];checks={'zero_same_across_laws':True,'O_same_across_laws':True,'raw_identity_max':0.,'rounded_pool_gap_max':0.,'vanilla_scalar_invariance_max':0.,'vanilla_matches_control_eq0_max':0.}
 reflookup=pool.set_index('reference_row_index');receipts=[];baseline_zero=None
 for law in list(OPERATORS)+['vanilla_addition']:
  selected_ids=subsets[law];lawzero=zero[selected_ids];nk=len(selected_ids)
  data={k:np.array(v,copy=True)for k,v in common.items()};data.update(law=np.full(N,law,dtype='<U32'),db=np.empty((N,15,25),np.float32),cell_share=np.empty((N,15,25),np.float32),
   share=np.empty(N,np.float64),weights=np.empty((N,15,25),np.float64),alpha=np.full(N,np.nan),active_minutes=np.zeros(N,np.int64))
  # Store physical components in raw relative power units for independent replay.
  for name in ['background_observed_power','added_observed_power','added_full_power','mixture_observed_power']:data[name]=np.empty((N,15,25),np.float64)
  lfref=np.full(N,np.nan);mfref=lfref.copy();hfref=lfref.copy();lfadd=lfref.copy();mfadd=lfref.copy();hfadd=lfref.copy()
  for i in selected_ids:
   native=None if zero[i]else source.reference_minutes[refs[i]]
   if law=='vanilla_addition':
    a=np.zeros((15,25));alpha=0.
    if not zero[i]:
     g=geom[i];background=np.asarray(bg[i],np.float64);off=float(np.nanmax(background));b=np.where(np.isfinite(background),10**((background-off)/10),0)
     alpha=float(10**(common['requested_snr_db'][i]/10)*np.median(b.sum(1)[np.isfinite(bg[i]).all(1)])/native.sum(1).max())
     part=native[g['crop_begin_minute']:g['crop_begin_minute']+g['selected_minutes']];s=g['window_start_minute'];a[s:s+len(part)]=alpha*part
     denom=part.sum();vanilla_ratios=np.asarray([a[:,:9].sum(),a[:,9:20].sum(),a[:,20:].sum()])/a.sum();part_ratios=np.asarray([part[:,:9].sum(),part[:,9:20].sum(),part[:,20:].sum()])/denom
     checks['vanilla_scalar_invariance_max']=max(checks['vanilla_scalar_invariance_max'],float(abs(vanilla_ratios-part_ratios).max()))
    value,receipt=pack_components(bg[i],a,{'operator':law,'alpha':alpha,'explicit_no_source':bool(zero[i]),'reference_peak_scope':'whole native before crop; no EQ'})
    if not zero[i] and i<12:
     v,_=generate_operator(bg[i],native,geom[i],np.zeros(25),np.zeros(25),common['requested_snr_db'][i],subsnr[i],'control_eq6',zero=False)
     checks['vanilla_matches_control_eq0_max']=max(checks['vanilla_matches_control_eq0_max'],float(abs(v['added_full_power']-a).max()))
   else:value,receipt=generate_operator(bg[i],native,geom[i],baseeq[i],posteq[i],common['requested_snr_db'][i],subsnr[i],law,zero=bool(zero[i]))
   for k in ['db','cell_share','share','background_observed_power','added_observed_power','added_full_power','mixture_observed_power']:data[k][i]=value[k]
   data['weights'][i]=energy_weights(value['db']);data['active_minutes'][i]=np.any(value['cell_share']>0,axis=1).sum();data['alpha'][i]=receipt.get('alpha')if receipt.get('alpha')is not None else np.nan
   checks['raw_identity_max']=max(checks['raw_identity_max'],float(abs(value['mixture_observed_power']-value['background_observed_power']-value['added_observed_power']).max()))
   checks['rounded_pool_gap_max']=max(checks['rounded_pool_gap_max'],float(receipt['rounded_pool_gap']))
   if not zero[i]:
    g=geom[i];part=native[g['crop_begin_minute']:g['crop_begin_minute']+g['selected_minutes']];lfref[i],mfref[i],hfref[i]=[part[:,:9].sum()/part.sum(),part[:,9:20].sum()/part.sum(),part[:,20:].sum()/part.sum()]
    afull=value['added_full_power'];lfadd[i],mfadd[i],hfadd[i]=[afull[:,:9].sum()/afull.sum(),afull[:,9:20].sum()/afull.sum(),afull[:,20:].sum()/afull.sum()]
   receipts.append({'sample_id':common['sample_id'][i],'law':law,'receipt':json.dumps(receipt,ensure_ascii=False,allow_nan=False)})
  data={k:v[selected_ids]for k,v in data.items()}
  assert np.array_equal(data['share']>0,~lawzero)
  assert np.array_equal(np.isfinite(data['db']),common['observed_mask'][selected_ids])
  assert np.array_equal(data['db'][lawzero],np.round(bg[selected_ids][lawzero],1).astype(np.float32),equal_nan=True)
  assert np.all(data['cell_share'][~common['observed_mask'][selected_ids]]==0)
  assert checks['vanilla_scalar_invariance_max']<1e-12 and checks['vanilla_matches_control_eq0_max']<1e-12, checks
  f=phys(data['db']);f['share']=data['share'];f['presence']=data['presence'];f['target_bin']=bins(data['share']);f['sample_id']=common['sample_id'][selected_ids];f['law']=law;f['active_minutes']=data['active_minutes']
  f['reference_LF_fraction']=lfref[selected_ids];f['reference_MF_fraction']=mfref[selected_ids];f['reference_HF_fraction']=hfref[selected_ids]
  f['added_full_LF_fraction']=lfadd[selected_ids];f['added_full_MF_fraction']=mfadd[selected_ids];f['added_full_HF_fraction']=hfadd[selected_ids]
  for c in ['base_window_index','site','recorder','reference_index','selected_minutes','window_start','requested_snr_db']:f[c]=data[c]
  f.to_csv(OUT/(law+'_physical_features.csv'),index=False)
  for label,g in [('all',f),*list(f.groupby('target_bin',sort=True))]:
   for c in ['share','LF_energy_fraction','MF_energy_fraction','HF_energy_fraction','bump_high_db','bump_low_db','active_minutes']:
    summary.append({'law':law,'target_bin':label,'feature':c,**stats(g[c])})
  path=OUT/(law+'.npz');npz(path,data)
  artifacts[law]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'n':nk,'zero':int(lawzero.sum()),'positive':int((~lawzero).sum()),'bytes':path.stat().st_size,'share':stats(data['share'])}
  note(stage='law_ready',law=law,artifact=artifacts[law],elapsed_seconds=time.perf_counter()-start)
 pd.DataFrame(summary).to_csv(OUT/'law_target_stratified_physical_summary.csv',index=False)
 pd.DataFrame(receipts).to_csv(OUT/'numeric_recipe_receipts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 reuse=pd.Series(refs[~zero]).value_counts();save(OUT/'NUMERIC_VERIFICATION.json',{**checks,'zero_rows':int(zero.sum()),'vanilla_positive_rows':1600,'allcustom_positive_rows':1600,'reference_unique':len(reuse),'ref_reuse_min':int(reuse.min()),'ref_reuse_max':int(reuse.max()),'reference_Kish_ESS':float(reuse.sum()**2/(reuse*reuse).sum())})
 # Same-target-bin summaries are descriptive conditional slices, not paired matched y.
 report='# R19 변형 없는 원음 합성과 6개 법칙의 대조 패널\n\n'
 report+='재학습 없음, Test 접근 없음. vanilla 2,000행(무추가400·추가1,600) +6개법칙 합계2,000행(335/335/335/335/330/330)=총4,000행이다. 모든법칙양성률80%. 같은2,000배경/참조/위치/SNR를 공유하며각원천ID에custom1개와vanilla1개를배정했다. 독립4,000원천이아니다. 입력마스킹·서포트·라벨도 검사했다.\n\n'
 report+='primary78869의 사이트×녹음기×Nobs 비율을 최대잔여법으로 근사보존하여2000고유배경을 비복원 추출했다. 학습원천과 일부/전부가 공유되는 법칙민감도 진단이며 독립 일반화 성적이 아니다. 원시 음원은 기존 검산된 minute_band_power 캐시를 쓰며 추가EQ가 없는vanilla에서도 scalar SNR alpha와 자연crop/placement는 가정으로 남는다. 공식숨은생성기라고 주장하지 않는다.\n\n'
 report+='원음은15분이하일때 전체완전분을 사용하고 초과시 자연연속15분을 자른다. 시간신축·반복·평탄화·fade 없음. 6법칙과vanilla 사이에서모든geometry를고정했다. 법칙할당은추론전seed고정. 개별custom법칙은vanilla의해당ID부분집합과paired비교해야하며vanilla전체2,000과는source구성차이도섞인다. 각법칙의share가달라지는것자체가효과이므로share구간요약은같은정답쌍매칭이아니다.\n\n'
 report+='## 배경분포 보존\n\n'+pd.DataFrame(bg_rows).to_markdown(index=False,floatfmt='.6f')+'\n\n'
 toplines=pd.DataFrame(summary).query("target_bin=='all' and feature in ['share','LF_energy_fraction','bump_high_db','bump_low_db']")
 report+='## 법칙별 전체 분포\n\n'+toplines.to_markdown(index=False,floatfmt='.6f')+'\n\n'
 report+='## 보존 검사\n\n'+json.dumps(checks,ensure_ascii=False,indent=2)+'\n\n전체 y구간분포는 law_target_stratified_physical_summary.csv, alpha/EQ/수식은 numeric_recipe_receipts.csv.gz, 실제성분A/B/E와정답q/y는각법칙NPZ에저장했다. vanilla의스칼라곱은원음의대역비중을보존하지만관측O·crop·분모B가최종혼합물비중을바꿀수있다. LF/MF/HF기준은각각31.6~199.5Hz/251.2~2511.9Hz/3162.3~7943.3Hz이며출처별다른범위통계와혼합하지않는다.\n'
 (OUT/'R19_VANILLA_PANEL_DIAGNOSTICS.md').write_text(report)
 assert all(sha(ROOT/p)==h for p,h in protocol['code_sha256'].items())
 manifest={**protocol,'status':'completed','source_identity':source.identity,'artifacts':artifacts,'elapsed_seconds':time.perf_counter()-start,'created_KST':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')}
 save(OUT/'manifest.json',manifest);save(OUT/'readiness.json',{'ready':True,'manifest_sha256':sha(OUT/'manifest.json')})
 outputs={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(OUT.iterdir())if p.is_file() and p.name not in ['OUTPUT_SHA256.json','COMPLETED.json']}
 save(OUT/'OUTPUT_SHA256.json',outputs);save(OUT/'COMPLETED.json',{'status':'completed','files':len(outputs),'manifest_sha256':sha(OUT/'manifest.json'),'seconds':time.perf_counter()-start,'new_training':False,'new_inference':False,'Test_used':False})
 print(json.dumps({'status':'completed','manifest':str(OUT/'manifest.json'),'files':len(outputs),'seconds':time.perf_counter()-start},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
