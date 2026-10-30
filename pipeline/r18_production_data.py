"""R18 production source-balanced, length-stratified natural-support data.

No Test access, time warp, waveform repeat, artificial flat-long resampling or
target-share forcing. Every source background has one zero and four positive
examples. Long H0 examples use real contiguous 12--15-minute recordings.
"""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/synthetic/r18_production_v1'
REPORT=ROOT/'reports/r18_production_r1/data'
SEED=202610301823
LONG_FRACTION=.135
ROWS={'train':60000,'clean_ES':2000,'fresh_fit':2000}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def write(path,v):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.item() if isinstance(a,np.generic) else str(a))+'\n')
    temp.replace(path)

def progress(**v):
    write(REPORT/'progress.json',v);print(json.dumps(v,ensure_ascii=False),flush=True)

def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
        if Path(args[0]).name.lower() in ('test.csv','sample_submission.csv','query_predictions.npz'):
            raise PermissionError('R18 production data prohibits query inputs')

def balanced(frame,n,salt):
    frame=frame.reset_index(drop=True)
    pools={s:list(g.index[np.argsort([hashlib.sha256(f'{SEED}|{salt}|{int(frame.at[i,"base_window_index"])}'.encode()).digest() for i in g.index])])
           for s,g in frame.groupby('site',sort=True)}
    result=[]
    while len(result)<n:
        advanced=False
        for s in sorted(pools):
            if pools[s] and len(result)<n:result.append(pools[s].pop());advanced=True
        if not advanced:raise ValueError('Insufficient background pool '+salt)
    out=frame.loc[result].copy()
    return out.iloc[np.random.default_rng([SEED,len(salt)]).permutation(n)].reset_index(drop=True)

def frames():
    from pipeline.r16_production_data import _crossrole
    original=ROOT/'data/synthetic/r16_multisupport_v1/fold_00/train_backgrounds.parquet'
    stop=ROOT/'data/synthetic/r17_clean_v1/earlystop_backgrounds.parquet'
    fit=ROOT/'data/synthetic/r17_clean_freshfit_v1/ensemble_fit_fresh_backgrounds.parquet'
    tr=pd.read_parquet(original)
    sparse=tr[tr.qc_observed_minutes.lt(12)].copy()
    sparse=balanced(sparse,min(1200,len(sparse)),'train_sparse_cap')
    eligible=pd.concat([tr[tr.qc_observed_minutes.ge(12)],sparse],ignore_index=True)
    chosen={'train':balanced(eligible,12000,'production_train'),
            'clean_ES':pd.read_parquet(stop),'fresh_fit':pd.read_parquet(fit)}
    for role,f in chosen.items():
        if f.base_window_index.duplicated().any() or len(f)*5!=ROWS[role]:raise AssertionError('Source cardinality')
        if role!='train' and f.qc_observed_minutes.min()<12:raise AssertionError('High-density validation required')
        f.to_parquet(OUT/(role+'_backgrounds.parquet'),index=False)
    proof=[]
    for j,(name,f) in enumerate(chosen.items()):
        proof.extend({'primary':name,**r} for r in _crossrole(f,dict(list(chosen.items())[j+1:])))
    fixed={n:pd.read_parquet(ROOT/f'reports/r16_support_temporal_r1/data/{n}_backgrounds.parquet')
           for n in ('local_gate','local_report','domain_holdout')}
    for name,f in chosen.items():proof.extend({'primary':name,**r} for r in _crossrole(f,fixed))
    oldfit=pd.read_parquet(ROOT/'reports/r16_support_temporal_r1/data/local_fit_backgrounds.parquet')
    proof.extend({'primary':'fresh_fit',**r}for r in _crossrole(chosen['fresh_fit'],{'legacy_local_fit':oldfit}))
    rows=[]
    for role,f in chosen.items():
        for (site,rec,nobs),g in f.groupby(['site','recorder','qc_observed_minutes']):
            rows.append({'role':role,'site':site,'recorder':rec,'n_obs':int(nobs),'backgrounds':len(g),'rows_per_regime':5*len(g)})
    pd.DataFrame(rows).to_csv(REPORT/'BACKGROUND_SITE_DENSITY.csv',index=False)
    write(REPORT/'BACKGROUND_SOURCE_PURGE.json',proof)
    return chosen,{str(p.relative_to(ROOT)):sha(p)for p in (original,stop,fit)}

def reference_pool():
    path=ROOT/'data/synthetic/r17_clean_v1/reference_pool_465.parquet'
    reserved_path=ROOT/'data/synthetic/r17_clean_v1/reserved_B19_Cprime25.parquet'
    pool=pd.read_parquet(path);reserved=pd.read_parquet(reserved_path)
    if len(pool)!=465 or pool.sha256.nunique()!=465 or len(reserved)!=44:
        raise AssertionError('465 unique +44 reservations required')
    if set(pool.sha256)&set(reserved.sha256):raise AssertionError('Reserved reference included')
    pool.to_parquet(OUT/'reference_pool_465.parquet',index=False)
    reserved.to_parquet(OUT/'reserved_B19_Cprime25.parquet',index=False)
    hist=pool.native_complete_minute_count.value_counts().sort_index()
    nlong=int(pool.native_complete_minute_count.ge(12).sum())
    if nlong!=15:raise AssertionError('Length feasibility changed')
    counts={'short_1to4':int(pool.native_complete_minute_count.le(4).sum()),
            'medium_5to11':int(pool.native_complete_minute_count.between(5,11).sum()),'long_ge12':nlong}
    detail=[]
    for role,n in ROWS.items():
        pos=n*4//5;long_n=int(round(pos*LONG_FRACTION));nonlong=pos-long_n
        short_n=int(round(nonlong*counts['short_1to4']/(465-nlong)));middle_n=nonlong-short_n
        quotas={'short_1to4':short_n,'medium_5to11':middle_n,'long_ge12':long_n}
        for name,total in quotas.items():detail.append({'role':role,'stratum':name,'references':counts[name],'positive_assignments':total,
            'per_clip_cap':int(np.ceil(total/counts[name])),'uniform465_cap_incompatible_with_long_quota':int(np.ceil(pos/465))<np.ceil(long_n/nlong)})
    pd.DataFrame(detail).to_csv(REPORT/'REFERENCE_LENGTH_QUOTAS_AND_CAPS.csv',index=False)
    feasibility={'eligible_clips':465,'reserved_clips':44,'native_minute_histogram':{str(int(k)):int(v)for k,v in hist.items()},
        'length_strata':counts,'H0_positive_ge12_target':LONG_FRACTION,
        'train_positive_rows':48000,'train_long_uses':6480,'minimum_long_per_clip_cap':432,
        'uniform_per_clip_cap_104_cannot_meet_long_quota':True,
        'no_warp_or_repeat_or_artificial_flat_support':True,
        'long_support_conditioning':'Long strata assigned only fully observed backgrounds; same positive prevalence80% in all O strata.',
        'strict_unseen_reference_claimed':False,'reserved_shared_hull_and_day_relationships_remain':True,
        'weak_background22_in_vessel_archive_semantics':'Known energy of whole added measured recording, not isolated vessel truth.',
        'input_sha256':{str(path.relative_to(ROOT)):sha(path),str(reserved_path.relative_to(ROOT)):sha(reserved_path)}}
    write(REPORT/'REFERENCE_CAP_FEASIBILITY.json',feasibility)
    return pool,feasibility

def balanced_draw(ids,count,rng):
    out=np.concatenate([rng.permutation(ids)for _ in range((count+len(ids)-1)//len(ids))])[:count]
    rng.shuffle(out);return out

def schedule(pool,frame,role):
    from ops.r16_forward_data import OPERATORS
    n=len(frame);rng=np.random.default_rng([SEED,sum(role.encode()),0x524546]);zero=np.arange(n)%5==0
    positive=np.flatnonzero(~zero);nl=int(round(len(positive)*LONG_FRACTION))
    full=positive[frame.qc_observed_minutes.to_numpy()[positive]==15]
    if len(full)<nl:raise ValueError('Insufficient full-O positive slots for natural long quota')
    long_slots=rng.choice(full,nl,replace=False);other=np.setdiff1d(positive,long_slots);rng.shuffle(other)
    short_ids=pool.loc[pool.native_complete_minute_count.le(4),'reference_row_index'].to_numpy(int)
    middle_ids=pool.loc[pool.native_complete_minute_count.between(5,11),'reference_row_index'].to_numpy(int)
    long_ids=pool.loc[pool.native_complete_minute_count.ge(12),'reference_row_index'].to_numpy(int)
    ns=int(round(len(other)*len(short_ids)/(len(short_ids)+len(middle_ids))))
    assignments=[('short_1to4',other[:ns],short_ids),('medium_5to11',other[ns:],middle_ids),('long_ge12',long_slots,long_ids)]
    refs=np.full(n,-1,np.int64);strata=np.full(n,'zero',dtype='<U20');caps={}
    for name,slots,ids in assignments:
        refs[slots]=balanced_draw(ids,len(slots),rng);strata[slots]=name;caps[name]=int(np.ceil(len(slots)/len(ids)))
    operators=np.full(n,'control_eq6',dtype='<U32');ops=np.resize(np.asarray(OPERATORS),len(positive));rng.shuffle(ops);operators[positive]=ops
    return zero,refs,strata,operators,caps

def long_geometry(native,rng):
    length=int(rng.integers(12,min(15,len(native))+1));begin=int(rng.integers(len(native)-length+1));start=int(rng.integers(16-length))
    return {'crop_begin_minute':begin,'selected_minutes':length,'window_start_minute':start,'family':'natural_long_contiguous12to15',
            'full_native_minutes':len(native),'source_repetition':False,'time_warp':False,'added_bell_or_fade':False}

def mi(mask,z):
    tab=pd.crosstab(mask,z).to_numpy(float);p=tab/tab.sum();ind=p.sum(1)[:,None]*p.sum(0)[None,:];ok=p>0
    return float(np.sum(p[ok]*np.log(p[ok]/ind[ok])))

def materialize(source,pool,backgrounds,role,regime):
    from ops.r16_forward_data import generate_operator,choose_geometry,channel_zero_mean
    from ops.r16_support_temporal_data import natural_geometry,mask_bits
    from pipeline.sampler_r9 import smooth_eq
    frame=backgrounds.iloc[np.repeat(np.arange(len(backgrounds)),5)].reset_index(drop=True);n=len(frame)
    zero,refs,strata,operators,caps=schedule(pool,frame,role)
    a={'db':np.empty((n,15,25),np.float32),'cell_share':np.empty((n,15,25),np.float32),'share':np.empty(n,np.float64),
       'presence':(~zero).astype(np.int64),'sample_id':np.asarray([f'R18P_{role}_{i:06d}'for i in range(n)]),
       'reference_index':refs,'support_stratum':strata,'operator':operators,'explicit_no_source':zero,
       'pair_background_id':np.repeat(np.arange(len(backgrounds)),5),'base_window_index':frame.base_window_index.to_numpy(np.int64),
       'site':frame.site.to_numpy(dtype=str),'recorder':frame.recorder.to_numpy(dtype=str),'sensor_group':frame.sensor_group.to_numpy(np.int64),
       'background_group':frame.background_group.to_numpy(dtype=str),'active_minutes':np.zeros(n,np.int64),
       'selected_minutes':np.zeros(n,np.int64),'crop_begin':np.zeros(n,np.int64),'window_start':np.zeros(n,np.int64),
       'family':np.full(n,'no_source',dtype='<U40'),'reference_weak_class':np.full(n,'no_source',dtype='<U32')}
    lookup=pool.set_index('reference_row_index');max_raw=max_round=0.;started=time.perf_counter()
    for i,row in frame.iterrows():
        bg=np.asarray(source.background(row),float);rng=np.random.default_rng([SEED,sum(role.encode()),i,0x4348]);grng=np.random.default_rng([SEED,sum(role.encode()),i,0x4745])
        native=None if zero[i] else source.reference_minutes[int(refs[i])]
        if zero[i]:geometry=None
        elif regime=='H0':geometry=long_geometry(native,grng)if strata[i]=='long_ge12'else natural_geometry(bg,native,grng)
        else:geometry=choose_geometry(bg,native,grng)
        eq=np.zeros(25)if zero[i]else smooth_eq(rng,6.)
        post=np.zeros(25)if zero[i]else channel_zero_mean(rng,6. if role=='train'else 12.)
        snr=float(rng.uniform(-10,10));sub=rng.uniform(-10,10,2)
        value,receipt=generate_operator(bg,native,geometry,eq,post,snr,sub,str(operators[i]),zero=bool(zero[i]))
        E,A,B=value['mixture_observed_power'],value['added_observed_power'],value['background_observed_power']
        if not np.array_equal(E,B+A)or abs(A.sum()/E.sum()-value['share'])>1e-12:raise AssertionError('Raw physical identity')
        if bool(value['presence'])!=bool(not zero[i]):raise AssertionError('Observed positive vanished')
        for k in ('db','cell_share','share'):a[k][i]=value[k]
        active=int(np.any(value['cell_share']>0,axis=1).sum());a['active_minutes'][i]=active
        if regime=='H0' and not zero[i] and ((active>=12)!=(strata[i]=='long_ge12')):raise AssertionError('Exact long support quota')
        if geometry:
            for k,g in [('selected_minutes','selected_minutes'),('crop_begin','crop_begin_minute'),('window_start','window_start_minute'),('family','family')]:a[k][i]=geometry[g]
            a['reference_weak_class'][i]=str(lookup.loc[int(refs[i])].weak_class)
        max_raw=max(max_raw,float(receipt['raw_pool_identity_error']));max_round=max(max_round,float(receipt['rounded_pool_gap']))
        if i and i%10000==0:progress(stage='materialize',role=role,regime=regime,rows=i,total=n,seconds=time.perf_counter()-started)
    a['effective_mask_bits']=mask_bits(a['db']);a['n_obs']=np.isfinite(a['db']).all(2).sum(1)
    table=pd.DataFrame({'site':a['site'],'recorder':a['recorder'],'O':a['effective_mask_bits'],'n_obs':a['n_obs'],'z':a['presence']})
    rates=table.groupby(['site','recorder','O','n_obs']).z.agg(['size','sum']).reset_index()
    if not np.all(rates['sum']*5==rates['size']*4):raise AssertionError('Observation/site/recorder prevalence differs')
    information=mi(a['effective_mask_bits'],a['presence'])
    if abs(information)>1e-14 or not np.array_equal(a['share']>0,a['presence'].astype(bool)):raise AssertionError('Presence/MI')
    reuse=pd.DataFrame({'reference_index':refs[~zero],'stratum':strata[~zero]}).value_counts().rename('uses').reset_index()
    if len(reuse)!=465 or any(reuse[reuse.stratum.eq(s)].uses.max()>cap for s,cap in caps.items()):raise AssertionError('Cap/coverage')
    reuse.to_csv(REPORT/f'{role}_{regime}_REFERENCE_REUSE.csv',index=False);rates.to_csv(REPORT/f'{role}_{regime}_MASK_RATES.csv',index=False)
    o=np.isfinite(a['db']);qerr=float(np.max(abs(a['db'][o]*10-np.rint(a['db'][o]*10))))
    if qerr>.0002 or np.any(a['cell_share'][~o]!=0):raise AssertionError('Rounded grid/targets')
    path=OUT/f'{role}_{regime}.npz'
    with path.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**a)
    path.with_suffix('.tmp').replace(path)
    with np.load(path,allow_pickle=False)as z:
        if not np.array_equal(z['share'],a['share'])or not np.array_equal(z['db'],a['db'],equal_nan=True):raise AssertionError('Saved roundtrip')
    counts=reuse.uses.to_numpy(float);pos=~zero;long_n=int((a['active_minutes'][pos]>=12).sum())
    receipt={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'n':n,'rows':n,'unique_backgrounds':len(backgrounds),
       'positive_rows':int(pos.sum()),'zero_rows':int(zero.sum()),'positive_fraction':float(pos.mean()),
       'reference_count':len(reuse),'reference_reuse_min':int(counts.min()),'reference_reuse_max':int(counts.max()),'stratum_caps':caps,
       'reference_concentration_Kish_ESS':float(counts.sum()**2/(counts*counts).sum()),'true_independent_recording_count_claimed':False,
       'positive_ge12_count':long_n,'positive_ge12_fraction':long_n/int(pos.sum()),'positive_active_minutes_mean':float(a['active_minutes'][pos].mean()),
       'positive_active_minutes_p50':float(np.median(a['active_minutes'][pos])),'share_mean':float(a['share'].mean()),'share_rms':float(np.sqrt(np.mean(a['share']**2))),
       'full_O_presence_MI_nats':information,'every_site_recorder_mask_positive_fraction':.8,'accidental_zero':0,'quantization_error_max':qerr,
       'raw_identity_error_max':max_raw,'rounded_pool_gap_max':max_round,'bytes':path.stat().st_size,'elapsed_seconds':time.perf_counter()-started}
    write(REPORT/f'{role}_{regime}_RECEIPT.json',receipt);return receipt

def main():
    from pipeline.dataset_r9 import R9Sources
    sys.addaudithook(guard);OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise FileExistsError('Immutable completed R18 production version')
    source=R9Sources(verify_banks=False);pool,feasibility=reference_pool();selected,input_pins=frames()
    codefiles=['pipeline/r18_production_data.py','ops/prepare_r18_production_data.py','ops/r16_forward_data.py','ops/r16_support_temporal_data.py','pipeline/sampler_r9.py']
    protocol={'version':'r18_production_v1','status':'fixed_before_generation','seed':SEED,'rows_per_regime':ROWS,
        'reference_feasibility':feasibility,'code_sha256':{n:sha(ROOT/n)for n in codefiles},'background_metadata_input_sha256':input_pins,
        'source_identity':source.identity,'Test_used':False,'KFold':False,'positive_fraction':.8,
        'long_H0_positive_fraction':LONG_FRACTION,'operators':'same six frozen R16 forward operators, balanced count',
        'reference_EVAL_reservations_preserved':44,'strict_reference_independence_claimed':False,
        'background_generation':'original SA background unchanged; QC missing minutes retained; no static spectrum/PCA/highshare forcing',
        'validation':'clean_ES/fresh_fit source highdensity>=12; oldlocalfitexcluded fromfreshfit;24h purge verified'}
    write(OUT/'PROTOCOL_BEFORE_GENERATION.json',protocol)
    roles={}
    for role,frame in selected.items():
        roles[role]={}
        for regime in ('D0','H0'):
            if (OUT/f'{role}_{regime}.npz').exists():raise FileExistsError('Partial data requires new version, do not reuse')
            roles[role][regime]=materialize(source,pool,frame,role,regime)
            progress(stage='dataset_complete',role=role,regime=regime,**roles[role][regime])
    if any(sha(ROOT/n)!=h for n,h in protocol['code_sha256'].items()):raise ValueError('Generator code changed')
    manifest={**protocol,'status':'completed','roles':roles,'created_KST':pd.Timestamp.now(tz='Asia/Seoul').isoformat()}
    write(OUT/'manifest.json',manifest)
    write(OUT/'readiness.json',{'ready':True,'version':'r18_production_v1','manifest_sha256':sha(OUT/'manifest.json')})
    contract=json.loads((OUT/'DATA_CONTRACT.json').read_text());contract.update(status='completed',roles=roles,manifest_sha256=sha(OUT/'manifest.json'))
    write(OUT/'DATA_CONTRACT.json',contract)
    rows=[{'role':role,'regime':regime,**rec}for role,rs in roles.items()for regime,rec in rs.items()]
    pd.DataFrame(rows).drop(columns=['stratum_caps']).to_csv(REPORT/'DATASET_SUMMARY.csv',index=False)
    report='# R18 본체 데이터 구축\n\n'
    report+='TRAIN은 체계별60,000행(12,000배경×음성1+양성4), clean_ES/fresh_fit은 각각2,000행이다. 모든O/사이트/녹음기층양성률은80%이며 MI는수치오차범위0이다.\n\n'
    report+='465개중실제12분이상참조는15개다. 양성48,000개의13.5%=6,480회를이15개로구성하려면클립당cap최소432가필요하므로 전체동일cap104는불가능하다. 길이층별cap을두고 실제ESS를별도보고한다. 인위적시간신축·반복·평탄15분변환은없다.\n\n'
    report+=pd.DataFrame(rows)[['role','regime','n','unique_backgrounds','reference_count','reference_reuse_max','reference_concentration_Kish_ESS','positive_ge12_fraction','share_mean']].to_markdown(index=False,floatfmt='.6f')+'\n\n'
    report+='장시간층은 관측15분이모두있는배경에배정하며, 모든O별양성률80%는동일하게유지한다. support와O의관계까지독립이라고주장하지않는다. 예약44개바이트를제외해도확대465풀에같은hull/day관계가남아엄격한미관측참조평가로부르지않는다.\n'
    (REPORT/'R18_PRODUCTION_DATA.md').write_text(report)
    pins={str(p.relative_to(ROOT)):sha(p)for p in OUT.iterdir()if p.is_file()}
    write(REPORT/'DATA_READY.json',{'status':'completed','manifest_path':str((OUT/'manifest.json').relative_to(ROOT)),'manifest_sha256':sha(OUT/'manifest.json'),
        'roles':roles,'artifact_sha256':pins,'Test_used':False,'new_KFold':False})
    progress(stage='all_completed',manifest_sha256=sha(OUT/'manifest.json'))

if __name__=='__main__':main()
