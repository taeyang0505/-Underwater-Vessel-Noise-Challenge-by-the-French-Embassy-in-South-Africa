"""R16 CPU cross-operator/source data experiment, no Test/model access or fit.

New hypotheses supplement the valid R15 fixed-broadband-SNR law; they are
not presented as proved corrections or reconstructions of a hidden process.
"""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/r16_cross_operator_r1/data';SEED=202610271616
OPERATORS=('control_eq6','mfhf_shelf6','mfhf_shelf12','mfhf_shelf12_lfminus6',
           'alpha_preEQ_inserted','independent_subband_snr')
LEGACY={OPERATORS[i]:v for i,v in enumerate(((0.,0.),(6.,0.),(12.,0.),(12.,-6.)))}
ROLES={'local_fit':('ensemble_fit','B'), 'local_gate':('ensemble_gate','B'),
       'local_report':('report','C'),'domain_holdout':('domain_holdout','C')}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        p=str(args[0]).lower()
        if Path(p).name in ('test.csv','sample_submission.csv') or '/eda_deep/' in p or '/r15_band_comparison_' in p:
            raise PermissionError('No Test/cohort artifact access in cross-operator experiment')
def safe(root,name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Project-relative artifact path required')
    q=(Path(root)/p).resolve()
    if not q.is_relative_to(Path(root).resolve()) or not q.is_file() or q.is_symlink():raise ValueError('Missing/unsafe artifact')
    return q
def qstats(a):
    a=np.asarray(a,float)
    return {'n':len(a),'mean':float(a.mean()),'rms':float(np.sqrt(np.mean(a*a))),
            'p10':float(np.quantile(a,.1)),'p50':float(np.quantile(a,.5)),'p90':float(np.quantile(a,.9))}

def load_panel(manifest_path,role,operator,*,freeze_receipt=None):
    """No model access; report/domain only after caller's global freeze.

    Required report gate: status=completed, all_models_frozen=True,
    ensemble_policy_frozen=True, data_manifest_sha256, artifact_sha256 map
    of real project-relative model-summary/policy files with exact SHA.
    """
    mp=Path(manifest_path);m=json.loads(mp.read_text())
    ready=json.loads((mp.parent/'readiness.json').read_text())
    if m.get('status')!='completed' or ready.get('ready')is not True or ready['manifest_sha256']!=sha(mp):raise ValueError('Complete immutable R16 bundle required')
    if role in ('local_report','domain_holdout'):
        if freeze_receipt is None:raise PermissionError('Global checkpoint/policy freeze required before report inputs')
        gate=json.loads(Path(freeze_receipt).read_text())
        if (gate.get('status')!='completed' or gate.get('all_models_frozen')is not True or gate.get('ensemble_policy_frozen')is not True
            or gate.get('data_manifest_sha256')!=sha(mp) or not gate.get('artifact_sha256')):raise PermissionError('Report freeze binding missing')
        for name,h in gate['artifact_sha256'].items():
            if sha(safe(ROOT,name))!=h:raise ValueError('Report frozen model/policy changed')
    if role=='train':e=m['training'][operator]
    else:e=m['panels'][role][operator]
    p=safe(mp.parent,e['path'])
    if sha(p)!=e['sha256']:raise ValueError('R16 panel checksum differs')
    with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    if len(d['share'])!=e['n'] or not np.array_equal(d['share']>0,d['presence'].astype(bool)):raise ValueError('Panel schema/physical labels differ')
    return d

def load_training(manifest_path,arm='control'):
    """Exactly4000 rows per arm, same paired background IDs, raw arrays only."""
    if arm not in ('control','mixed'):raise ValueError('Control/mixed arm required')
    return load_panel(manifest_path,'train',arm)

def channel_zero_mean(rng,maximum=12.):
    axis=np.linspace(-1.,1.,25)
    # Both basis functions have arithmetic25-band log-gain mean zero.
    curve=rng.uniform(-maximum,maximum)*axis+rng.uniform(-maximum/2,maximum/2)*(axis**2-(axis**2).mean())
    curve-=curve.mean();peak=abs(curve).max()
    if peak>maximum:curve*=maximum/peak
    curve-=curve.mean()
    return curve

def choose_geometry(bg,native,rng,*,report_crop=False):
    from ops.r15_forward_prototype import choose_trajectory
    n=len(native)
    if n<=15 and rng.random()<.25:
        part=native;starts=[s for s in range(16-n) if np.where(np.isfinite(bg)[s:s+n],part,0.).sum()>0]
        if not starts:raise ValueError('No observed native-full placement')
        return {'crop_begin_minute':0,'selected_minutes':n,'window_start_minute':int(rng.choice(starts)),
          'family':'natural_full','full_native_minutes':n,'legal_observed_placements':len(starts),
          'placement_conditioning':'known observed support, enumerate once','source_repetition':False,'time_warp':False,'added_bell_or_fade':False}
    if report_crop:
        length=int(rng.integers(1,min(6,n)+1));begin=int(rng.integers(n-length+1));part=native[begin:begin+length]
        starts=[s for s in range(16-length) if np.where(np.isfinite(bg)[s:s+length],part,0.).sum()>0]
        if not starts:raise ValueError('No observed natural crop placement')
        return {'crop_begin_minute':begin,'selected_minutes':length,'window_start_minute':int(rng.choice(starts)),
          'family':'natural_uniform_crop_1to6','full_native_minutes':n,'legal_observed_placements':len(starts),
          'placement_conditioning':'known observed support, enumerate once','source_repetition':False,'time_warp':False,'added_bell_or_fade':False}
    selection=choose_trajectory(bg,native,rng);selection['family']='natural_peaked_1to6'
    return selection

def pack_components(bg,added,recipe):
    from pipeline.sampler_r9 import energy_weights
    from pipeline.relative_grid_r6 import quantization_pooling_bound
    bg=np.asarray(bg,np.float64);o=np.isfinite(bg);offset=float(np.nanmax(bg))
    b=np.where(o,10.**((np.where(o,bg,offset)-offset)/10.),0.);oa=np.where(o,added,0.);e=b+oa
    y=float(oa.sum()/e.sum());q=np.divide(oa,e,out=np.zeros_like(e),where=o)
    raw=np.full_like(e,np.nan);raw[o]=10*np.log10(e[o])+offset;db=np.round(raw,1).astype(np.float32)
    rawgap=abs(float((energy_weights(raw)*q).sum())-y)
    gap=abs(float((energy_weights(db)*q).sum())-y)
    half=float(abs(np.spacing(db[o])).max()/2);bound=float(quantization_pooling_bound(.05+half+1e-12))
    if not np.array_equal(np.isfinite(db),o) or rawgap>1e-10 or gap>bound+2e-7:raise AssertionError('R16 physical/rounding identity failed')
    if np.isinf(bg).any() or not np.isfinite(q).all() or (q<0).any() or (q>1).any():raise ValueError('Invalid components')
    recipe.update(power_db_offset=offset,raw_pool_identity_error=rawgap,rounded_pool_gap=gap,rounding_bound=bound,
      physical_share=y,observed_positive=bool(y>0))
    return {'db':db,'unrounded_db':raw,'cell_share':q,'share':np.float64(y),'presence':np.int64(y>0),
      'observed_mask':o,'background_observed_power':b,'added_full_power':added,'added_observed_power':oa,
      'mixture_observed_power':e},recipe

def generate_operator(bg,native,geometry,base_eq,post_eq,snr,sub_snr,operator,*,zero=False):
    """All candidate laws use shared crop/placement/mask and prescribed draws."""
    from ops.r15_forward_prototype import shelf_curve
    bg=np.asarray(bg,float);o=np.isfinite(bg);added=np.zeros((15,25),float)
    offset=float(np.nanmax(bg));b=np.where(o,10**((np.where(o,bg,offset)-offset)/10),0.)
    rec={'operator':operator,'explicit_no_source':bool(zero),'alpha':None,'alpha_subbands':None,
         'common_random_eq_db':base_eq.tolist(),'zero_mean_post_eq_db':post_eq.tolist(),
         'whole_measured_recording_not_isolated_vessel_truth':True,'no_target_share_forcing':True}
    if not zero:
        begin=int(geometry['crop_begin_minute']);n=int(geometry['selected_minutes']);start=int(geometry['window_start_minute'])
        part=np.asarray(native[begin:begin+n],float);ob=o[start:start+n]
        if operator in LEGACY:
            curve=base_eq+shelf_curve(*LEGACY[operator]);transformed=native*10**(curve[None,:]/10.)
            peak=float(transformed.sum(1).max());bmedian=float(np.median(b.sum(1)[o.all(1)]))
            alpha=float(10**(snr/10)*bmedian/peak)
            added[start:start+n]=alpha*transformed[begin:begin+n]
            rec.update(alpha=alpha,channel_db=curve.tolist(),requested_SNR_db=float(snr),reference_peak=peak,
               reference_peak_scope='whole native afterEQ beforecrop/O; unchanged R15 normalization')
        elif operator=='alpha_preEQ_inserted':
            peak=float(np.where(ob,part,0.).sum(1).max());bmedian=float(np.median(b.sum(1)[o.all(1)]))
            alpha=float(10**(snr/10)*bmedian/peak)
            added[start:start+n]=alpha*part*10**(post_eq[None,:]/10.)
            rec.update(alpha=alpha,channel_db=post_eq.tolist(),requested_SNR_db=float(snr),reference_peak=peak,
              alpha_is_observation_mask_conditioned=True,
              reference_peak_scope='actually observed inserted native segment beforeEQ; no postEQ renormalization')
        elif operator=='independent_subband_snr':
            alpha=[];peaks=[];background=[]
            for j,ix in enumerate((np.arange(9),np.arange(9,25))):
                peak=float(np.where(ob[:,ix],part[:,ix],0.).sum(1).max())
                bmedian=float(np.median(b[:,ix].sum(1)[o.all(1)]))
                if peak<=0 or bmedian<=0:raise ValueError('Zero subband support; stop, no reseed or label change')
                a=float(10**(float(sub_snr[j])/10)*bmedian/peak);alpha.append(a);peaks.append(peak);background.append(bmedian)
                added[start:start+n,ix]=a*part[:,ix]*10**(post_eq[ix][None,:]/10.)
            rec.update(alpha_subbands=alpha,subband_indices=[list(range(9)),list(range(9,25))],subband_SNR_db=list(map(float,sub_snr)),
               subband_reference_peak=peaks,subband_background_median=background,channel_db=post_eq.tolist(),
               alpha_is_observation_mask_conditioned=True,
               reference_peak_scope='observed inserted segment preEQ peak independently <=200 and>200; no postEQ renormalization')
        else:raise ValueError('Unknown operator')
    d,rec=pack_components(bg,added,rec)
    if not zero and not d['presence']:raise AssertionError('Known positive lacks observed support')
    return d,rec

def reference_partition(source):
    import pandas as pd
    from pipeline.vtuad_reference_r4 import connected_components
    p=ROOT/'data/versions/vtuad_reference_r4_v1/reference_manifest.parquet';r=pd.read_parquet(p)
    rebuilt=connected_components(r)
    if rebuilt.reference_component.tolist()!=r.reference_component.tolist() or r.reference_component.nunique()!=22:
        raise ValueError('Original selected reference graph differs')
    groups=sorted(r.reference_component.unique(),key=lambda v:canonical([SEED,'referenceABC',str(v)]))
    mapping={g:'A' if i<12 else 'B' if i<17 else 'C' for i,g in enumerate(groups)}
    r['r16_reference_partition']=r.reference_component.map(mapping)
    full=source.references[['sha256','reference_row_index']]
    r=r.merge(full,on='sha256',how='left',validate='one_to_one')
    if len(r)!=119 or r.reference_row_index.isna().any():raise ValueError('119selected references missing from verified509cache')
    for col in ('hull_id','recording_day','sha256'):
        if r.groupby(col).r16_reference_partition.nunique().max()!=1:raise ValueError('Reference direct identity crosses A/B/C')
    pairs=0
    for i,a in r.iterrows():
        for _,b in r.iloc[i+1:].iterrows():
            if a.r16_reference_partition!=b.r16_reference_partition and pd.Timestamp(a.utc_begin)<pd.Timestamp(b.utc_end) and pd.Timestamp(b.utc_begin)<pd.Timestamp(a.utc_end):pairs+=1
    if pairs:raise ValueError('UTC support crosses reference partitions')
    proof={'source_manifest':str(p.relative_to(ROOT)),'source_sha256':sha(p),'selected_clips':119,'selected_graph_components':22,
       'full525_graph_components':1,'partition_rule':'stableSHA metadata order12A/5B/5C components, no acoustic/model results',
       'partition_counts':{k:{'clips':len(g),'components':g.reference_component.nunique()} for k,g in r.groupby('r16_reference_partition')},
       'cross_partition_hull_day_bytes_or_UTC_overlaps':0,'excluded_fullcorpustransitivebridges_unused':True,
       'new_models_reference_unseen_only':True,'old_R9_R14_models_saw_shared509':True,
       'conditional_selected_graph_independence_not_new_corpus':True}
    return r,proof

def domain_metadata(source):
    import pandas as pd
    from pipeline.source_split_r6 import minute_masks,apply_minute_qc
    p=ROOT/'data/versions/quality_v2_native/background_manifest.parquet';allmeta=pd.read_parquet(p)
    d=allmeta[allmeta.split.eq('domain_holdout')].copy()
    expected={'13 Saldhana Bay','9 Fish Hoek','10 Sea Point'}
    if set(d.site)!=expected:raise ValueError('Old domain site identity differs')
    rows=pd.read_parquet(ROOT/'data/processed/sa_row_manifest.parquet',columns=['row_id','site','recorder','t','window_included','eligible_for_minute_window'])
    flags=pd.read_parquet(ROOT/'reports/data_quality_boundary_flags.parquet');bits,masked,_=minute_masks(d,rows,flags)
    d['boundary_mask_bits']=bits
    counts=[];bank=source.background_bank()
    for start in range(0,len(d),1024):
        batch=d.iloc[start:start+1024];a=apply_minute_qc(np.asarray(bank[batch.base_window_index.to_numpy(int)]),batch.boundary_mask_bits.to_numpy(int))
        counts.extend(np.isfinite(a).all(2).sum(1).tolist())
    d['qc_observed_minutes']=counts;d=d[d.qc_observed_minutes>=3].copy()
    d['background_group']=['R16_DOMAIN_'+canonical([int(r.sensor_group),str(r.recording_week_start)])[:20] for r in d.itertuples()]
    return d.reset_index(drop=True),{'metadata_path':str(p.relative_to(ROOT)),'metadata_sha256':sha(p),'eligible_rows':len(d),
      'source_sites':sorted(expected),'QC_masked_source_minutes':len(masked),'previous_development_domain_has_been_evaluated':True,
      'unseen_site_relative_to_new_TRAIN_only_not_genuinely_new_blind_panel':True}

def choose_backgrounds(source,domain):
    import pandas as pd
    chosen={};full={'train':source.role_metadata(0,'train')}
    for role,(old,_) in ROLES.items():full[role]=domain if old=='domain_holdout' else source.role_metadata(0,old)
    for j,(role,frame) in enumerate(full.items()):
        rng=np.random.default_rng(np.random.SeedSequence([SEED,0x4247,j]));n=4000 if role=='train' else 400
        if role=='domain_holdout':
            pieces=[]
            for k,(_,g) in enumerate(frame.groupby('site',sort=True)):
                size=134 if k==0 else 133;pieces.append(g.iloc[rng.choice(len(g),size,replace=False)])
            pick=pd.concat(pieces).iloc[rng.permutation(n)]
        else:pick=frame.iloc[rng.choice(len(frame),n,replace=False)]
        chosen[role]=pick.reset_index(drop=True)
    proof=[]
    for i,(a,fa) in enumerate(chosen.items()):
        for b,fb in list(chosen.items())[i+1:]:
            if set(fa.base_window_index)&set(fb.base_window_index) or set(fa.background_group)&set(fb.background_group):raise ValueError('Background IDs/groups cross roles')
            gaps=[]
            for sensor,ga in fa.groupby('sensor_group'):
                gb=fb[fb.sensor_group==sensor]
                if not len(gb):continue
                x=pd.to_datetime(ga.start_export_clock).astype('int64').to_numpy()/1e9;y=np.sort(pd.to_datetime(gb.start_export_clock).astype('int64').to_numpy()/1e9)
                pos=np.searchsorted(y,x);distance=np.minimum(abs(x-y[np.clip(pos-1,0,len(y)-1)]),abs(x-y[np.clip(pos,0,len(y)-1)]))
                gaps.extend((np.maximum(distance-900,0)/3600).tolist())
            gap=min(gaps) if gaps else None
            if gap is not None and gap<24-1e-8:raise ValueError('Cross-role900s/24h support purge failed')
            proof.append({'a':a,'b':b,'base_index_overlap':0,'background_group_overlap':0,'minimum_support_gap_hours':gap,
                          'no_shared_sensor':not bool(gaps)})
    return chosen,proof

def allocate(n):
    return {**{k:np.empty((n,15,25),np.float64) for k in ('unrounded_db','cell_share','background_observed_power','added_full_power','added_observed_power','mixture_observed_power')},
       'db':np.empty((n,15,25),np.float32),'share':np.empty(n,np.float64),'presence':np.empty(n,np.int64),'observed_mask':np.empty((n,15,25),bool)}

def make_role(source,frame,refs,role,operators,out):
    from pipeline.sampler_r9 import smooth_eq
    rolecode={'train':0,'local_fit':1,'local_gate':2,'local_report':3,'domain_holdout':4}[role]
    n=len(frame);rng=np.random.default_rng(np.random.SeedSequence([SEED,rolecode,0x534348]));zero=np.zeros(n,bool)
    zero[rng.choice(n,n//5,replace=False)]=True;available=refs.reference_row_index.to_numpy(int);idx=np.full(n,-1,int);order=[]
    while len(order)<int((~zero).sum()):order.extend(rng.permutation(available).tolist())
    idx[~zero]=np.asarray(order[:int((~zero).sum())]);reflookup=refs.set_index('reference_row_index')
    mixed=np.empty(n,dtype='<U32');positive=np.flatnonzero(~zero);assignment=np.resize(np.asarray(OPERATORS),len(positive));rng.shuffle(assignment)
    mixed[:]='control_eq6';mixed[positive]=assignment
    arrays={op:allocate(n) for op in operators};recipes=[];families=[];components=[]
    maxima={op:{'max_raw_pool_error':0.,'max_rounded_pool_gap':0.,'max_rounding_bound':0.} for op in operators}
    for i,row in frame.iterrows():
        local=np.random.default_rng(np.random.SeedSequence([SEED,rolecode,i,0x4752]));bg=np.asarray(source.background(row),float)
        native=None if zero[i] else source.reference_minutes[int(idx[i])]
        geometry=None if zero[i] else choose_geometry(bg,native,local,report_crop=role in ('local_report','domain_holdout'))
        eq=np.zeros(25) if zero[i] else smooth_eq(local,6.);post=np.zeros(25) if zero[i] else channel_zero_mean(local,6. if role=='train' else 12.)
        snr=float(local.uniform(-10,10));sub=local.uniform(-10,10,2)
        component='' if zero[i] else str(reflookup.loc[int(idx[i])].reference_component)
        families.append('no_source' if zero[i] else geometry['family']);components.append(component)
        recipe={'pair_id':f'r16_{role}_{i:05d}','background_index':int(row.base_window_index),'reference_index':int(idx[i]),
            'reference_component':component,'geometry':geometry,'explicit_no_source':bool(zero[i]),'seed':[SEED,rolecode,i,0x4752],
            'mixed_operator':str(mixed[i]),'operator_receipts':{}}
        needed=OPERATORS if role=='train' else tuple(operators)
        for op in needed:
            d,receipt=generate_operator(bg,native,geometry,eq,post,snr,sub,op,zero=bool(zero[i]));recipe['operator_receipts'][op]=receipt
            destinations=([op] if op in arrays else [])+(['mixed'] if role=='train' and op==mixed[i] else [])
            for target in destinations:
                for key,value in d.items():arrays[target][key][i]=value
                audit=maxima[target]
                for field,key in [('max_raw_pool_error','raw_pool_identity_error'),('max_rounded_pool_gap','rounded_pool_gap'),('max_rounding_bound','rounding_bound')]:audit[field]=max(audit[field],receipt[key])
        recipes.append(recipe)
    entries={};pair_id=np.asarray([r['pair_id'] for r in recipes])
    for op,d in arrays.items():
        d.update(pair_id=pair_id,sample_id=np.asarray([p+'__'+op for p in pair_id]),
             base_window_index=frame.base_window_index.to_numpy(np.int64),reference_index=idx.astype(np.int64),
             reference_component=np.asarray(components),site=frame.site.astype(str).to_numpy(dtype=str),
             background_group=frame.background_group.astype(str).to_numpy(dtype=str),family=np.asarray(families),explicit_no_source=zero,
             operator=mixed if op=='mixed' else np.full(n,op,dtype='<U32'))
        if not np.array_equal(d['presence']>0,~zero) or any(v.dtype.kind=='O' for v in d.values()):raise ValueError('Observed positive/zero/schema mismatch')
        if not np.array_equal(d['mixture_observed_power'],d['background_observed_power']+d['added_observed_power']):raise AssertionError('ExactE=B+A failed')
        oracle=d['added_observed_power'].sum((1,2))/d['mixture_observed_power'].sum((1,2))
        if abs(oracle-d['share']).max()>1e-12:raise AssertionError('Saved share differs from components')
        folder=out/role;folder.mkdir(exist_ok=True);path=folder/(op+'.npz');np.savez_compressed(path,**d)
        with np.load(path,allow_pickle=False) as z:
            if any(z[k].dtype.kind=='O' for k in z.files):raise ValueError('No-pickle reload failed')
        entries['control' if role=='train' and op=='control_eq6' else op]={'path':str(path.relative_to(out)),'sha256':sha(path),'n':n,
             'statistics':{'share':qstats(d['share']),'zero_count':int((d['share']==0).sum()),'positive_count':int(d['presence'].sum()),**maxima[op]}}
    frame.to_parquet(out/(role+'_backgrounds.parquet'),index=False)
    dump(out/(role+'_recipes.json'),recipes)
    return entries

def fixture():
    rng=np.random.default_rng(32);bg=np.full((15,25),90.);bg[:5]=np.nan;bg[10:]=np.nan
    native=np.ones((9,25));native[4]=np.arange(1,26);geom=choose_geometry(bg,native,rng)
    eq=np.linspace(-6,6,25);post=channel_zero_mean(rng);curves=[]
    for op in OPERATORS:
        d,r=generate_operator(bg,native,geom,eq,post,1.,[-3.,6.],op)
        assert d['share']>0 and np.isfinite(d['cell_share']).all();curves.append(r)
    assert abs(post.mean())<1e-12
    # PreEQ law's alpha is independent of its following channel magnitude.
    _,a=generate_operator(bg,native,geom,eq,np.zeros(25),1.,[-3.,6.],'alpha_preEQ_inserted')
    _,b=generate_operator(bg,native,geom,eq,post,1.,[-3.,6.],'alpha_preEQ_inserted')
    assert a['alpha']==b['alpha']
    return {'status':'passed','operators':list(OPERATORS),'physical_q_y_mask_and_zero_mean':True,
            'preEQ_alpha_unchanged_after_channel_variation':True,'actual_models_or_Test_opened':False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['build','fixture']);a=ap.parse_args()
    sys.addaudithook(guard)
    if a.action=='fixture':print(json.dumps(fixture()));return
    if OUT.exists():raise FileExistsError('New R16 folder required; failed partial never reused')
    OUT.mkdir(parents=True);started=time.perf_counter();own=sha(__file__)
    plan={'version':'r16_cross_operator_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seed':SEED,
      'attachment':'1fc448b5-90b6-4ff8-b790-a20fe8ecf1ee','code_sha256':own,'operators':list(OPERATORS),
      'training':{'control':4000,'mixed':4000,'same_unique_backgrounds':True,'ref_partition':'A','mixed_equal_positive_operator_cycle':True},
      'panels':{k:{'backgrounds':400,'operators':list(OPERATORS),'reference_partition':v[1]} for k,v in ROLES.items()},
      'background_roles':ROLES,'reference_partition_rule':'119 selected/22 direct components;metadataSHAorder12A5B5C',
      'trajectory':'25percent naturalfull if native<=15; otherwise1..6real contiguousminutes; TRAIN/fit/gate crop includes preEQbroadbandpeak,report/domain uses uniform nativecropbegin; no warp/repeat/bell/highshare/full_smooth15',
      'zero_fraction':.2,'positive_support_conditioning':'enumerate valid insideplacements once, no source/seed/targetretry',
      'base_EQ_bound':6.,'postEQ_mean25_loggain':0.,'postEQ_bound':{'train':6.,'local_fit':12.,'local_gate':12.,'local_report':12.,'domain_holdout':12.},'SNR_uniform_db':[-10.,10.],
      'new_laws':'inserted-observedpreEQpeakscale thenzero-meanEQ no renormalization; independent <=200/>200 SNR preEQ scales thensameEQ',
      'legacy_law':'R15whole-native-afterEQ broadbandSNR intentionally retained; alternate law not proved hidden-generator correction',
      'domain':'oldpublicSA domain_holdout Saldhana/FishHoek/SeaPoint; previously development-exposed, sitesunseen onlyrelative newTRAIN',
      'report_access':'checkpoint+ensemblepolicyfreeze SHA gate before local_report/domain arrays',
      'Test_read_or_Test_based_generator_tuning':False,'prior_Test_EDA_exposure_disclosed':True,'blind_experiment_claimed':False,
      'mask_conditioning_confound':'positive placement and newoperatorpreEQalpha depend on observedO; declare rather than claim purechannelonlycontrast',
      'training_per_arm_rows':4000,'training_backgrounds_not_duplicated_within_arm':True,
      'model_fitting_by_data_builder':False,'natural_ship_GT_claimed':False}
    dump(OUT/'plan.json',plan);dump(OUT/'NUMERIC_FIXTURE.json',fixture())
    from pipeline.dataset_r9 import R9Sources
    source=R9Sources();references,refproof=reference_partition(source);references.to_parquet(OUT/'reference_partitions.parquet',index=False)
    domain,dproof=domain_metadata(source);backgrounds,bproof=choose_backgrounds(source,domain)
    dump(OUT/'SOURCE_AUDIT.json',{'reference':refproof,'background_crossrole':bproof,'domain':dproof})
    pins={**source.code_sha256,'ops/r15_forward_prototype.py':sha(ROOT/'ops/r15_forward_prototype.py'),
       refproof['source_manifest']:refproof['source_sha256'],dproof['metadata_path']:dproof['metadata_sha256'],
       'data/processed/sa_row_manifest.parquet':sha(ROOT/'data/processed/sa_row_manifest.parquet'),
       'reports/data_quality_boundary_flags.parquet':sha(ROOT/'reports/data_quality_boundary_flags.parquet')}
    for role in ('train','ensemble_fit','ensemble_gate','report'):
        p=source.bg_dir/'fold_00'/(role+'_manifest.parquet');pins[str(p.relative_to(ROOT))]=sha(p)
    manifest={'status':'completed','version':'r16_cross_operator_v1','plan_sha256':sha(OUT/'plan.json'),'code_sha256':own,
      'source_identity':source.identity,'source_pins':pins,'training':{},'panels':{},'reference_counts':refproof['partition_counts'],
      'report_rows_for_parameter_selection':False,'reference_unseen_for_old_models':False,'Test_accessed':False}
    for role,frame in backgrounds.items():
        part='A' if role=='train' else ROLES[role][1];refs=references[references.r16_reference_partition.eq(part)].copy()
        ops=('control_eq6','mixed') if role=='train' else OPERATORS
        entry=make_role(source,frame,refs,role,ops,OUT)
        if role=='train':manifest['training']=entry
        else:manifest['panels'][role]=entry
        print(json.dumps({'role':role,'n_backgrounds':len(frame),'operators':len(entry),'seconds':time.perf_counter()-started}),flush=True)
    for p,h in pins.items():
        if sha(ROOT/p)!=h:raise AssertionError('Frozen source changed')
    if sha(__file__)!=own:raise AssertionError('Builder source changed')
    manifest['elapsed_seconds']=time.perf_counter()-started
    manifest['artifact_sha256']={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    dump(OUT/'manifest.json',manifest)
    dump(OUT/'readiness.json',{'ready':True,'manifest_sha256':sha(OUT/'manifest.json'),'scope':'small CPU crossoperator experiment only',
       'source_checks_passed':True,'model_training_performed':False,'production_ready':False,'report_policy_freeze_required':True})
    print(json.dumps({'status':'completed','manifest_sha256':sha(OUT/'manifest.json'),'rows_train':8000,'panel_candidate_rows':9600,'source_audit':refproof,'seconds':manifest['elapsed_seconds']}))

if __name__=='__main__':main()
