"""TRAIN-only paired channel-sensitivity prototype; never competition calibration.

Four fixed hypotheses, no Test access/comparison/selection, models or fitting.
Only new artifacts are written. Frozen R9/R14 sources remain unchanged.
"""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SEED=202610271515
FREQ=np.asarray([31.6,39.8,50.1,63.1,79.4,100,125.9,158.5,199.5,251.2,316.2,398.1,501.2,631,794.3,1000,1258.9,1584.9,1995.3,2511.9,3162.3,3981.1,5011.9,6309.6,7943.3])
CANDIDATES={'control_eq6':(0.,0.),'mfhf_shelf6':(6.,0.),'mfhf_shelf12':(12.,0.),'mfhf_shelf12_lfminus6':(12.,-6.)}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(json.dumps({'shape':list(a.shape),'dtype':str(a.dtype)},sort_keys=True).encode()+a.tobytes()).hexdigest()
def table(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def qstats(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    return {'n':len(a),'p10':float(np.quantile(a,.1)) if len(a) else None,
            'p50':float(np.quantile(a,.5)) if len(a) else None,'p90':float(np.quantile(a,.9)) if len(a) else None,
            'mean':float(a.mean()) if len(a) else None}
def refuse_test(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=str(args[0]).lower()
        if Path(p).name=='test.csv' or '/eda_deep/' in p or '/r15_band_comparison_' in p:
            raise PermissionError('No Test/cohort-derived artifact access in TRAIN sensitivity prototype')

def shelf_curve(boost,low):
    z=np.clip((np.log(FREQ)-np.log(200.))/(np.log(500.)-np.log(200.)),0.,1.)
    high=.5-.5*np.cos(np.pi*z)
    return low*(1-high)+boost*high

def choose_trajectory(background_db,native,rng):
    """One real contiguous crop containing measured broadband peak, no warp.

    Legal placements are enumerated once using known TRAIN observation
    support. This conditioning is explicit and is different from old R9's
    mask-independent geometry. There is no label/value/performance retry.
    """
    x=np.asarray(background_db,float);r=np.asarray(native,float)
    if x.shape!=(15,25) or r.ndim!=2 or r.shape[1]!=25 or not len(r):raise ValueError('Expected native minute power and15minute background')
    if not np.isfinite(r).all() or (r<0).any() or not (r.sum(1)>0).all():raise ValueError('Invalid measured reference')
    length=int(rng.integers(1,min(6,len(r))+1));peak=int(np.argmax(r.sum(1)))
    begins=np.arange(max(0,peak-length+1),min(peak,len(r)-length)+1)
    begin=int(rng.choice(begins));part=r[begin:begin+length];o=np.isfinite(x)
    starts=[]
    for start in range(16-length):
        if np.where(o[start:start+length],part,0.).sum()>0:starts.append(start)
    if not starts:raise ValueError('No observed legal placement; do not retry source/seed or alter target')
    start=int(rng.choice(starts))
    return {'crop_begin_minute':begin,'selected_minutes':length,'window_start_minute':start,
       'measured_full_source_peak_index':peak,'full_native_minutes':len(r),'legal_observed_placements':len(starts),
       'placement_conditioning':'known TRAIN observed support; enumerate feasible geometry once',
       'time_warp':False,'source_repetition':False,'added_bell_or_fade':False,'fixed_full15_component':False}

def synthesize_pair(background_db,native,selection,channel_db,snr_db,*,no_source=False):
    """Numerical oracle API. q/y are solved from raw components, never chosen.

    Power uses a common per-background offset. B=0 outside O is an observed
    integration convention; the real missing background power is unknown.
    A_full is retained outside O, so masked additions are not destroyed.
    """
    from pipeline.sampler_r9 import energy_weights
    from pipeline.relative_grid_r6 import quantization_pooling_bound
    x=np.asarray(background_db,np.float64);o=np.isfinite(x)
    if x.shape!=(15,25) or np.isinf(x).any() or (o.any(1)!=o.all(1)).any() or o.all(1).sum()<3:
        raise ValueError('Original full-band minute mask/min3 required')
    peak=float(np.nanmax(x));b=np.where(o,np.power(10.,(np.where(o,x,peak)-peak)/10.),0.)
    bmedian=float(np.median(b.sum(1)[o.all(1)]));added=np.zeros_like(b);alpha=0.;full_peak=None
    if not no_source:
        r=np.asarray(native,np.float64);curve=np.asarray(channel_db,float)
        if curve.shape!=(25,) or not np.isfinite(curve).all() or not np.isfinite(snr_db):raise ValueError('Finite prescribed channel/SNR required')
        transformed=r*np.power(10.,curve[None,:]/10.)
        full_peak=float(transformed.sum(1).max())
        alpha=float(10**(float(snr_db)/10.)*bmedian/full_peak)
        begin=int(selection['crop_begin_minute']);length=int(selection['selected_minutes']);start=int(selection['window_start_minute'])
        if not(0<=begin<=len(r)-length and 1<=length<=6 and 0<=start<=15-length):raise ValueError('Non-native crop/placement')
        added[start:start+length]=alpha*transformed[begin:begin+length]
    observed_added=np.where(o,added,0.);e=b+observed_added
    q=np.divide(observed_added,e,out=np.zeros_like(e),where=o);y=float(observed_added.sum()/e.sum())
    if not no_source and not y>0:raise ValueError('Positive request has no observed support')
    raw=np.full_like(e,np.nan);raw[o]=10*np.log10(e[o])+peak;rounded=np.round(raw,1).astype(np.float32)
    raw_pool=float((energy_weights(raw)*q).sum());rounded_pool=float((energy_weights(rounded)*q).sum())
    half_ulp=float(np.abs(np.spacing(rounded[o])).max()/2)
    bound=float(quantization_pooling_bound(.05+half_ulp+1e-12))
    if (abs(raw_pool-y)>1e-10 or abs(rounded_pool-y)>bound+2e-7 or not np.array_equal(np.isfinite(rounded),o)
          or not np.isfinite(q).all() or (q<0).any() or (q>1).any()):raise AssertionError('Physical components/mask/quantization identity failed')
    result={'db':rounded,'unrounded_db':raw,'cell_share':q,'share':np.float64(y),'presence':np.int64(y>0),
       'added_full_power':added,'added_observed_power':observed_added,'mixture_observed_power':e}
    receipt={'alpha':alpha,'full_native_peak_after_all_channels':full_peak,'background_median_observed_minute_power':bmedian,
       'requested_full_native_peak_SNR_db':None if no_source else float(snr_db),
       'reference_peak_scope':'entire measured native minute trajectory, after EQ, before crop/placement/O',
       'requested_no_source':bool(no_source),'observed_added_positive':bool(y>0),'raw_pool_identity_error':abs(raw_pool-y),
       'rounded_pool_gap':abs(rounded_pool-y),'rounded_pool_theoretical_bound':bound,'power_db_offset':peak,
       'component_interpretation':'whole measured reference recording added, not isolated vessel-only GT'}
    return result,receipt

def numeric_fixture():
    bg=np.full((15,25),90.);bg[:5]=np.nan;bg[10:]=np.nan
    native=np.ones((9,25));native[4]*=10
    selection=choose_trajectory(bg,native,np.random.default_rng(42))
    a,ar=synthesize_pair(bg,native,selection,np.zeros(25),2.)
    b,br=synthesize_pair(bg,native,selection,shelf_curve(12.,-6.),2.)
    z,zr=synthesize_pair(bg,None,None,np.zeros(25),None,no_source=True)
    assert a['share']>0 and b['share']>0 and z['share']==0 and not z['added_full_power'].any()
    assert all(np.isfinite(v).all() for v in (a['cell_share'],b['cell_share'],a['added_full_power']))
    for out in (a,b,z):
        reconstructed=np.power(10.,(np.where(np.isfinite(out['unrounded_db']),out['unrounded_db'],90.)-90.)/10.)
        assert np.allclose(reconstructed[np.isfinite(bg)],out['mixture_observed_power'][np.isfinite(bg)],rtol=1e-12,atol=1e-14)
    expected=10**(.2)
    assert abs(ar['alpha']*ar['full_native_peak_after_all_channels']/ar['background_median_observed_minute_power']-expected)<1e-12
    # Source bank dtype must not choose float32 power arithmetic for the
    # common stored B while the candidate oracle computes float64.
    bank=(bg+.123456).astype(np.float32);as64=np.asarray(bank,np.float64);o=np.isfinite(as64);peak=float(np.nanmax(as64))
    stored_b=np.where(o,np.power(10.,(np.where(o,as64,peak)-peak)/10.),0.)
    aa,_=synthesize_pair(bank,native,selection,np.zeros(25),2.)
    assert np.array_equal(aa['mixture_observed_power'],stored_b+aa['added_observed_power'])
    return {'status':'passed','observed_positive_support':True,'exact_zero':True,'raw_component_reconstruction':True,
       'post_channel_full_native_peak_SNR_identity':True,'max_raw_pool_error':max(ar['raw_pool_identity_error'],br['raw_pool_identity_error']),
       'float32_bank_to_float64_stored_background_identity':True,'Test_or_model_access':False}

def source_separation(source,train,chosen):
    import pandas as pd
    validations=[source.role_metadata(0,role) for role in ('selection','ensemble_fit','ensemble_gate','report')]
    held=pd.concat(validations,ignore_index=True)
    if set(train.base_window_index)&set(held.base_window_index) or set(train.background_group)&set(held.background_group):raise ValueError('TRAIN source overlaps heldout')
    a=pd.to_datetime(chosen.start_export_clock).astype('int64').to_numpy()/1e9
    mins=[]
    for sensor,ids in chosen.groupby('sensor_group').indices.items():
        other=held[held.sensor_group==sensor]
        if not len(other):continue
        b=np.sort(pd.to_datetime(other.start_export_clock).astype('int64').to_numpy()/1e9);v=a[ids];pos=np.searchsorted(b,v)
        distance=np.minimum(abs(v-b[np.clip(pos-1,0,len(b)-1)]),abs(v-b[np.clip(pos,0,len(b)-1)]))
        mins.extend((np.maximum(distance-900.,0)/3600).tolist())
    gap=min(mins) if mins else None
    if gap is not None and gap<24-1e-8:raise ValueError('Inherited900s support +24h purge not satisfied')
    return {'source_role':'R9 Fold0 TRAIN only','TRAIN_pool_rows':len(train),'chosen_unique':len(chosen),
        'heldout_base_index_intersection':0,'heldout_background_group_intersection':0,
        'minimum_same_sensor_exportclock_900s_support_gap_hours':gap,'exportclock_not_claimed_absolute_UTC':True,
        'four_heldout_roles_metadata_only_no_arrays_opened':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rows',type=int,default=4000);args=ap.parse_args()
    if args.rows!=4000:raise ValueError('Predeclared prototype4000 only')
    sys.addaudithook(refuse_test)
    out=ROOT/'reports/r15_forward_prototype_r1'
    if out.exists():raise FileExistsError('Fresh prototype path required; no partial reuse')
    out.mkdir(parents=True);started=time.perf_counter();code_sha=sha(__file__)
    plan={'version':'r15_TRAIN_only_channel_sensitivity_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'status':'predeclared_before_this_generation','rows_per_candidate':4000,'total_candidate_rows':16000,'seed':SEED,'fold':0,
       'source_config':'configs/r9_full_training_v1.json','source_config_sha256':sha(ROOT/'configs/r9_full_training_v1.json'),
       'code_sha256':code_sha,'reference_pool':'existing509 measured vessel-archive native minutes;16dedicatedambient excluded',
       'not_claimed_isolated_vessel_energy':True,'source_background_selection':'4000 unique Fold0 TRAIN rows, uniform without replacement',
       'no_source_fraction':.2,'no_source_count':800,'observed_positive_count':3200,
       'reference_schedule':'balanced permutations over all509, independent of Test/model metrics',
       'trajectory':'uniform1..min(6,native_minutes), contiguous crop containing original native broadband peak; no warp/repeat/bell/fade/full15',
       'placement':'uniform among inside-window starts with positive observed support, enumerated once; explicit change from R9 mask-independent placement',
       'SNR_db_uniform':[-10.,10.],'SNR_denominator':'whole native trajectory after channel EQ, beforecrop/placement/O; same sampledSNR eachcandidate',
       'common_random_smooth_EQ_max_db':6.,'channel_hypotheses':{n:{'HF_shelf_boost_db':v[0],'LF_change_db':v[1],
            'additional_curve_db':shelf_curve(*v).tolist()} for n,v in CANDIDATES.items()},
       'shelf':'raised-cosine in log-frequency from<=200Hz low value to>=500Hz HF value; referenceonly',
       'candidate_scope':'single-run TRAIN physical sensitivity challenge, not calibrated/competition-ready training dataset',
       'prior_exposure':'authors already viewed earlier Test descriptive audit; predeclared means locked before this generation, not genuinely blind',
       'no_Test_derived_target_or_profile_parameters':True,'Test_read_comparison_distance_optimization_and_candidate_selection':False,
       'official_rule_context':'Test inputs inference only; no generator tuning against Test distributions or adversarial-validation classifier on Test rows',
       'model_training_or_inference':False,'PCA_anchor_feature_transform':False,'highshare_forcing':False,
       'power_units':'common per-background dB offset; observedB integration zero outsideO, true missingB unknown',
       'outputs':'continuousfloat64 A_full/A_observed/B_observed/E/q/y plus rawdb,rounded0.1db and frozen recipe; no feature/model fitting'}
    plan['prior_failed_materialization']={'path':'reports/r15_forward_prototype_r1_FAILED_B_dtype','partial_reused':False,
        'correction':'common background power explicitlyfloat64, same seed/candidates/selection/physical laws'}
    dump(out/'plan.json',plan);dump(out/'NUMERIC_FIXTURE.json',numeric_fixture())
    from pipeline.dataset_r9 import R9Sources
    from pipeline.sampler_r9 import smooth_eq
    source=R9Sources();train=source.role_metadata(0,'train')
    rng=np.random.default_rng(SEED);pick=rng.choice(len(train),4000,replace=False);chosen=train.iloc[pick].reset_index(drop=True)
    separation=source_separation(source,train,chosen);dump(out/'SOURCE_SEPARATION.json',separation)
    source_pins={**source.code_sha256,'configs/r9_full_training_v1.json':sha(ROOT/'configs/r9_full_training_v1.json'),
       str((source.bg_dir/'manifest.json').relative_to(ROOT)):sha(source.bg_dir/'manifest.json'),
       str((source.ref_dir/'manifest.json').relative_to(ROOT)):sha(source.ref_dir/'manifest.json')}
    for role in ('train','selection','ensemble_fit','ensemble_gate','report'):
        p=source.bg_dir/'fold_00'/(role+'_manifest.parquet');source_pins[str(p.relative_to(ROOT))]=sha(p)
    zero=np.zeros(4000,bool);zero[rng.choice(4000,800,replace=False)]=True
    refs=np.full(4000,-1,np.int64);order=[]
    while len(order)<3200:order.extend(rng.permutation(509).tolist())
    refs[~zero]=np.asarray(order[:3200])
    common={'background_db':np.empty((4000,15,25),np.float64),'background_observed_power':np.empty((4000,15,25),np.float64),
       'observed_mask':np.empty((4000,15,25),bool),'power_db_offset':np.empty(4000),
       'base_window_index':chosen.base_window_index.to_numpy(np.int64),'reference_index':refs,'explicit_no_source':zero,
       'pair_id':np.asarray([f'r15_pair_{i:04d}' for i in range(4000)]),'site':chosen.site.astype(str).to_numpy(),
       'background_group':chosen.background_group.astype(str).to_numpy()}
    # npz may not contain pandas/object arrays.
    common['site']=common['site'].astype(str);common['background_group']=common['background_group'].astype(str)
    candidate_arrays={name:{'db':np.empty((4000,15,25),np.float32),
       **{k:np.empty((4000,15,25),np.float64) for k in ('unrounded_db','cell_share','added_full_power','added_observed_power','mixture_observed_power')},
       'share':np.empty(4000,np.float64),'presence':np.empty(4000,np.int64)} for name in CANDIDATES}
    recipes=[];max_raw=max_round=0.;audit_counts={n:{'max_raw_pool_error':0.,'max_rounded_pool_gap':0.,'max_rounding_bound':0.} for n in CANDIDATES}
    for i,row in chosen.iterrows():
        local=np.random.default_rng(np.random.SeedSequence([SEED,int(i),0x465744]))
        bg=np.asarray(source.background(row),np.float64);o=np.isfinite(bg);offset=float(np.nanmax(bg));b=np.where(o,np.power(10.,(np.where(o,bg,offset)-offset)/10.),0.)
        common['background_db'][i]=bg;common['background_observed_power'][i]=b;common['observed_mask'][i]=o;common['power_db_offset'][i]=offset
        native=None if zero[i] else source.reference_minutes[refs[i]]
        selection=None if zero[i] else choose_trajectory(bg,native,local)
        base_eq=np.zeros(25) if zero[i] else smooth_eq(local,6.)
        snr=None if zero[i] else float(local.uniform(-10.,10.))
        recipe={'pair_id':str(common['pair_id'][i]),'base_window_index':int(row.base_window_index),
            'boundary_mask_bits':int(row.boundary_mask_bits),'reference_index':int(refs[i]),
            'reference_clip_id':None if zero[i] else str(source.references.iloc[refs[i]].clip_id),
            'reference_SHA':None if zero[i] else str(source.references.iloc[refs[i]].sha256),
            'explicit_no_source':bool(zero[i]),'trajectory':selection,'common_random_eq_db':base_eq.tolist(),
            'same_sampled_SNR_db':snr,'seed_sequence':[SEED,int(i),0x465744],'candidates':{}}
        for name,extra in CANDIDATES.items():
            curve=base_eq+shelf_curve(*extra)
            result,receipt=synthesize_pair(bg,native,selection,curve,snr,no_source=bool(zero[i]))
            receipt['total_reference_channel_db']=curve.tolist();recipe['candidates'][name]=receipt
            for k,v in result.items():candidate_arrays[name][k][i]=v
            audit=audit_counts[name]
            for field,receipt_field in [('max_raw_pool_error','raw_pool_identity_error'),('max_rounded_pool_gap','rounded_pool_gap'),('max_rounding_bound','rounded_pool_theoretical_bound')]:audit[field]=max(audit[field],receipt[receipt_field])
        recipes.append(recipe)
        if (i+1)%500==0:print(json.dumps({'stage':'generated','pairs':i+1,'seconds':time.perf_counter()-started}),flush=True)
    np.savez_compressed(out/'common_backgrounds.npz',**common)
    dump(out/'recipes.json',recipes)
    chosen.to_parquet(out/'background_metadata.parquet',index=False)
    rows=[];stats={};zero_ref=None
    for name,arrays in candidate_arrays.items():
        arrays.update(pair_id=common['pair_id'],sample_id=np.asarray([str(v)+'__'+name for v in common['pair_id']]))
        if not np.array_equal(arrays['presence'].astype(bool),~zero):raise AssertionError('Observed positive/zero schedule changed')
        if not np.array_equal(np.isfinite(arrays['db']),common['observed_mask']):raise AssertionError('Observation mask changed')
        this_zero=array_sha(arrays['db'][zero])
        if zero_ref is None:zero_ref=this_zero
        elif this_zero!=zero_ref:raise AssertionError('Paired zero backgrounds changed across candidates')
        raw_y=arrays['added_observed_power'].sum((1,2))/arrays['mixture_observed_power'].sum((1,2))
        raw_q=np.divide(arrays['added_observed_power'],arrays['mixture_observed_power'],out=np.zeros_like(arrays['cell_share']),where=common['observed_mask'])
        if abs(raw_y-arrays['share']).max()>1e-12 or abs(raw_q-arrays['cell_share']).max()>1e-12:raise AssertionError('Saved component target oracle failed')
        if not np.array_equal(arrays['mixture_observed_power'],common['background_observed_power']+arrays['added_observed_power']):
            raise AssertionError('Saved E=B+A exactfloat64 sum oracle failed')
        for subset,take in [('all',np.ones(4000,bool)),('positive',~zero),('known_no_addition',zero)]:
            d=arrays['db'][take].astype(float);med=np.nanmedian(d,1);mx=np.nanmax(d,1);bump=mx-med
            for band in range(25):rows.append({'candidate':name,'subset':subset,'feature':'per_band_max_minus_median','frequency_hz':FREQ[band],**qstats(bump[:,band])})
        d=arrays['db'].astype(float);high=10*np.log10(np.where(np.isfinite(d[:,:,20:25]),10**(np.where(np.isfinite(d[:,:,20:25]),d[:,:,20:25],0.)/10),0.).sum(2).clip(np.finfo(float).tiny))
        high=np.where(common['observed_mask'].all(2),high,np.nan);high_bump=np.nanmax(high,1)-np.nanmedian(high,1)
        stats[name]={'known_share_all':qstats(arrays['share']),'known_share_positive':qstats(arrays['share'][~zero]),
           'HF_power_bump_all':qstats(high_bump),'HF_power_bump_positive':qstats(high_bump[~zero]),
           'no_source_count':int(zero.sum()),'observed_positive_count':int(arrays['presence'].sum()),
           'no_source_db_pair_sha256':this_zero,**audit_counts[name]}
        np.savez_compressed(out/(name+'.npz'),**arrays)
    table(out/'band25_TRAIN_sensitivity.csv',rows)
    for name,h in source_pins.items():
        if sha(ROOT/name)!=h:raise AssertionError('Frozen source changed:'+name)
    if sha(__file__)!=code_sha:raise AssertionError('Prototype code changed')
    # Verify every saved npz is readable without pickle, numerical/Unicode only.
    for file in out.glob('*.npz'):
        with np.load(file,allow_pickle=False) as z:
            if any(z[k].dtype.kind=='O' for k in z.files):raise AssertionError('Object array forbidden')
    lengths=[r['trajectory']['selected_minutes'] for r in recipes if r['trajectory'] is not None]
    refcounts=np.bincount(refs[refs>=0],minlength=509)
    manifest={'status':'completed_TRAIN_physics_sensitivity_only','competition_training_ready':False,'plan_sha256':sha(out/'plan.json'),
       'code_sha256':code_sha,'source_identity':source.identity,'source_pins':source_pins,'source_separation':separation,
       'pair_rows':4000,'candidate_count':4,'total_candidate_rows':16000,'source_backgrounds_unique':4000,
       'all509_references_used':bool((refcounts>0).all()),'reference_use_min':int(refcounts.min()),'reference_use_max':int(refcounts.max()),
       'selected_native_minutes':{str(int(k)):int(v) for k,v in zip(*np.unique(lengths,return_counts=True))},
       'reference_independent_holdout':False,'statistics':stats,'candidate_selection_or_Test_distance_computed':False,
       'Test_opened':False,'model_fit_or_inference_performed':False,'original_sources_unchanged':True,
       'elapsed_seconds':time.perf_counter()-started}
    lines=['# R15 TRAIN 전용 채널 민감도 프로토타입','',
      '이 결과는 고정된 물리 가설 4개를 한 번 비교한 진단용 프로토타입이다. 대회용으로 보정되거나 검증된 학습 자료가 아니다. Test 파일·기존 Test EDA·분포 비교표를 이 스크립트에서 읽지 않았고, Test와의 거리·근접성 통과 기준·최적 후보 선택을 계산하지 않았다. 이미 이전 Test 기술 통계를 보았으므로 사전 선언은 이번 생성 전 고정이라는 뜻이며 진정한 blind 실험을 뜻하지 않는다. 공식 규정상 Test 분포에 맞춘 생성기 조정은 하지 않는다.','',
      '## 원천·정답·변경 범위','',
      '기존 R9Sources의 Fold0 TRAIN 배경 4,000개를 무복원 추출했다. 기존 24시간 지원 구간 purge와 분리된 원천 ID를 검산했으며 heldout 자료는 메타데이터만 읽었다. VTUAD vessel-archive의 검증된 native minute 파워 509개를 모두 순환 사용했고, 전용 ambient16개는 사용하지 않았다. 참조는 전체 측정 녹음이며 자연 선박 유무나 고립 선박 에너지 정답이 아니다.','',
      '양성3,200개는 실제 연속1–6분을 native broadband peak를 포함하도록 잘라 사용했다. 길이 변환·반복·bell·강제 fade·15분 장시간 성분·highshare 강제·PCA/anchor 특징을 추가하지 않았다. 짧은 원천은 실제 사용 가능한 완전 분만 썼다. 원래 R9와 달리 알려진 TRAIN 관측 마스크와 겹치는 합법적인 창 내 시작점을 한 번 열거해 추출했다. 이는 양성의 관측 지원을 보장하기 위한 조건부 배치이며 mask-independent 원본 R9의 재현이라고 주장하지 않는다.','',
      '## 고정 후보와 물리식','',
      '같은 배경·원천·native crop·배치·SNR·random smooth EQ±6dB를 4개 후보에 공유했다. 추가 채널은 reference에만 적용한다. 200→500Hz log-frequency raised cosine shelf를 사용하며, 500Hz 이상 +6/+12dB, 마지막 후보는 200Hz 이하 −6dB를 더한다. 이 값은 감도 확인용 가설이며 Test의 숨은 채널 응답 추정치가 아니다. 총 채널 범위는 공통 ±6dB와 추가 곡선의 합이다.','',
      '각 후보마다 EQ 이후 전체 native trajectory의 분 broadband peak로 alpha=10^(SNR/10)×Bmedian/Rpeak를 다시 풀었다. SNR은 공통 U[−10,+10]dB다. crop/O 이전의 전체 궤적 peak를 썼으므로 tail boost 재정규화는 없다. y=ΣOA/ΣOE, q=OA/E를 raw continuous 성분에서 계산했고 특정 share 범위를 만들지 않았다.','',
      '파워는 창의 공통 dB offset을 뺀 상대 단위다. 관측하지 못한 실제 B는 알 수 없으며 B_observed=0은 관측 적분 규약이다. A_full은 결측 분에도 보존하고, A_observed만 마스크 적분에 쓴다. 각 NPZ는 unrounded dB, 0.1dB dB, float64 q/y/A/E를 저장한다. common_backgrounds.npz는 동일 B/O/원천 ID를 담는다.','',
      '## 전체 및 양성 정답·HF 변화','',
      '| 후보 | share 전체 p50 | share 양성 p50 | HF bump 전체 p50 | HF bump 양성 p50 | 최대 raw 항등식 오차 | 최대 rounded pooling gap |','|---|---:|---:|---:|---:|---:|---:|']
    for name,v in stats.items():lines.append(f"| {name} | {v['known_share_all']['p50']:.6f} | {v['known_share_positive']['p50']:.6f} | {v['HF_power_bump_all']['p50']:.4f} | {v['HF_power_bump_positive']['p50']:.4f} | {v['max_raw_pool_error']:.3g} | {v['max_rounded_pool_gap']:.6f} |")
    lines +=['','모든 후보에서 no-source800개/양성3200개이며 zero 입력은 후보 사이 bitwise 동일하다. 수치 변화는 저장한 TRAIN 물리 가설의 효과이며 실대회 정답 분포·일반화 성능·validation gap 해결을 증명하지 않는다. 모델을 학습하거나 추론하지 않았고 원래 생산 코드·자료·정책·학습 상태는 수정하지 않았다.','',
      '[사전 선언](plan.json) · [원천 분리](SOURCE_SEPARATION.json) · [25대역 통계](band25_TRAIN_sensitivity.csv) · [정확한 재현 recipe](recipes.json)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    manifest['artifact_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    dump(out/'manifest.json',manifest)
    print(json.dumps({'status':manifest['status'],'path':str(out.relative_to(ROOT)),'manifest_sha256':sha(out/'manifest.json'),
       'elapsed_seconds':manifest['elapsed_seconds'],'statistics':stats},ensure_ascii=False))

if __name__=='__main__':main()
