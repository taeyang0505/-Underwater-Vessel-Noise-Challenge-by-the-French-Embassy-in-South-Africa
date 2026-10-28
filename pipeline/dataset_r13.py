"""Fold0 R13 paired TRAIN: preserve strictHF v2, change positive alpha only.

No validation/query/Test reader or model fitting. Baseline references the
immutable strict site/sensor/recorder/effective-mask matched R12 TRAIN.
Exactly25% of positive rows in long/standard x original-CNN-subset strata
receive LHS Uniform(.25,.85) observed-window energy targets. Every source,
mask, reference, EQ, trajectory and label presence is retained. A target
law is not per-row monotonic uplift; down/up counts are disclosed.

Known targets are added WHOLE-recording energy, not isolated-vessel or
naturally ship-absent truth. No hard share/minute-SNR cap is introduced.
Pure helpers below can be reused by a separately verified fullscale builder;
this module does NOT implement Fold1..4 or certify those source pools.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.dataset_r9 import R9Sources
from pipeline.dataset_r12 import validate_r12_bundle
from pipeline.relative_grid_r6 import quantization_pooling_bound

ROOT=Path(__file__).resolve().parents[1]
VERSION='r13_highshare_paired_train_v1'
DEFAULT_OUTPUT='data/synthetic/r13_cpu_correction_v1'
BASELINE='data/synthetic/r12_mask_matched_control_v2'
PARENT='data/synthetic/r12_cpu_ablation_v3'
MASTER='configs/r13_cpu_correction_v1.json'
SEED=2026102712
SHARE_BOUNDS=(.25,.85)
CODE=('pipeline/dataset_r13.py','ops/build_r13_cpu_data.py')


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(canonical({'shape':list(a.shape),'dtype':str(a.dtype)})+a.tobytes()).hexdigest()
def local(name):
    p=Path(name);p=(ROOT/p if not p.is_absolute() else p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Project-contained paths required')
    return p
def relative(name):
    parts=list(local(name).relative_to(ROOT).parts)
    if parts and parts[0]=='data.nosync' and (ROOT/'data').resolve()==ROOT/'data.nosync':parts[0]='data'
    return Path(*parts).as_posix()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify_pins(pins):
    for n,h in pins.items():
        if sha(local(n))!=h:raise ValueError('Immutable source changed: '+n)
def load_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def reconstruct_reference_from_recipe(native_minute_power,recipe):
    """Reconstruct original R12 EQ/crop/placement from recorded geometry.

    Source-I/O free and independent of variable-length RNG consumption.
    Returns pre-mask placed reference power[15,25]; never pads or repeats
    a waveform. Existing R12 long_smooth_bell is explicitly a statistical
    minute-power resampling, retained exactly rather than redefined here.
    """
    native=np.asarray(native_minute_power,np.float64)
    if native.ndim!=2 or native.shape[1]!=25 or not len(native) or not np.isfinite(native).all() or (native<0).any():raise ValueError('Native finite nonnegative minute powers required')
    if recipe.get('no_source') is not False:raise ValueError('Positive R12 recipe required')
    curve=np.asarray(recipe['reference_eq_db'],float);route=recipe['route']
    if curve.shape!=(25,) or not np.isfinite(curve).all():raise ValueError('Recorded EQ invalid')
    r=native*10.**(curve[None,:]/10.)
    length=int(route['rendered_trajectory_minutes']);crop=int(route['crop_begin_minute']);start=int(route['window_start_minute'])
    if length<1 or crop<0:raise ValueError('Invalid recorded trajectory')
    if route['requested_family']=='long_smooth_bell':
        if not 5<=len(r)<=14 or not 12<=length<=15:raise ValueError('Foreign statistical long source')
        coords=np.linspace(0,len(r)-1,length)
        part=np.stack([np.interp(coords,np.arange(len(r)),r[:,f]) for f in range(25)],1)
        part*=(.6+.4*np.sin(np.pi*(np.arange(length)+.5)/length)**2)[:,None]
    else:
        if crop+length>len(r):raise ValueError('Recorded crop exceeds actual native trajectory')
        part=r[crop:crop+length]
    peak=float((part if route['requested_family'].startswith('long_') else r).sum(1).max())
    if not np.isclose(peak,recipe['full_reference_peak_after_EQ'],rtol=1e-12,atol=0):raise ValueError('Recorded reference peak/source differs')
    out=np.zeros((15,25));lo,hi=max(0,start),min(15,start+len(part))
    if hi<=lo:raise ValueError('No observed-window geometric support')
    out[lo:hi]=part[lo-start:hi-start]
    return out


def components(background_db,placed_reference_power,alpha):
    """Known B,A,E in a stable gauge; restore original absolute dB level."""
    x=np.asarray(background_db,np.float64);r=np.asarray(placed_reference_power,np.float64);o=np.isfinite(x)
    if x.shape!=(15,25) or r.shape!=x.shape or np.isinf(x).any() or (o.any(1)!=o.all(1)).any() or o.all(1).sum()<3:
        raise ValueError('At least3 complete measured/NaN background minutes required')
    if not np.isfinite(r).all() or (r<0).any() or not np.isfinite(alpha) or alpha<0:raise ValueError('Finite nonnegative components required')
    top=float(np.nanmax(x));b=np.where(o,10.**((x-top)/10.),0.);a=np.where(o,float(alpha)*r,0.);e=b+a
    return b,a,e,o,top


def _render(b,a,e,o,top):
    q=np.divide(a,e,out=np.zeros_like(a),where=o);y=float(a.sum()/e.sum())
    raw=np.full((15,25),np.nan);raw[o]=10*np.log10(e[o])+top
    db=np.round(raw,1).astype(np.float32)
    grid=np.round(db.astype(np.float64),1)
    ew=np.where(o,10**((np.where(o,grid,top)-np.nanmax(grid))/10),0.);w=ew/ew.sum()
    raw_error=float(abs(np.sum(e/e.sum()*q)-y));rounded_gap=float(abs(np.sum(w*q)-y))
    bound=quantization_pooling_bound(.05+float(np.max(np.abs(np.spacing(db[o]))))/2+1e-12)
    if raw_error>1e-11 or rounded_gap>bound+2e-7 or not np.array_equal(np.isfinite(db),o):raise AssertionError('Known energy/mask/quantization invariant failed')
    if not np.isfinite(q).all() or ((q<0)|(q>1)).any():raise ValueError('Invalid exact component fraction')
    bt,at=b.sum(1),a.sum(1);active=o.any(1)&(at>0)
    snr=np.full(15,np.nan);snr[active]=10*np.log10(at[active]/bt[active])
    return {'db':db,'unrounded_db':raw,'cell_share':q.astype(np.float32),'share':np.float32(y),'presence':np.int64(y>0)}, {
        'raw_label_identity_error':raw_error,'raw_share_float64':y,'rounded_pool_gap':rounded_gap,'rounding_bound':bound,
        'raw_q_serialized_max_error':float(abs(q-q.astype(np.float32)).max()),
        'saved_share_abs_roundoff':float(abs(y-float(np.float32(y)))),
        'global_observed_snr_db':float(10*np.log10(a.sum()/b.sum())) if a.sum()>0 else None,
        'minute_snr_db':[None if not np.isfinite(v) else float(v) for v in snr],
        'observed_added_minutes':int(active.sum()),'observed_minutes':int(o.any(1).sum())}


def solve_observed_mean_mixture(background_db,placed_reference_power,target_share,*,receipt=None):
    """Pure positive mixture helper; alpha=u*sum(B)/((1-u)*sum(O*R)).

    Global sum ratios equal observed-minute MEAN ratios because the same
    observed support is used in both means. No source/time/EQ changes,
    alpha/share cap, or adaptation to query/Test statistics is performed.
    """
    u=float(target_share)
    if not np.isfinite(u) or not 0<u<1:raise ValueError('Explicit target strictly between zero and one required')
    b,_,_,o,top=components(background_db,placed_reference_power,0.)
    r=np.asarray(placed_reference_power,float);rsum=float(np.sum(np.where(o,r,0.)))
    if rsum<=0:raise ValueError('Positive observed reference support required; never retry/relabel')
    alpha=u*b.sum()/((1-u)*rsum);a=np.where(o,alpha*r,0.);e=b+a
    item,audit=_render(b,a,e,o,top)
    if abs(audit['raw_share_float64']-u)>1e-11:raise AssertionError('Requested observed share not achieved')
    audit.update(version=VERSION,target_share_requested=u,alpha=float(alpha),target_law='observed-mean energy fraction',
        cap_applied=False,source_time_mask_EQ_changed=False,interpretation='whole added recording energy, not isolated-vessel truth')
    if receipt is not None:audit['caller_receipt']=receipt
    return item,audit


def highshare_schedule(arrays,subset,seed=SEED):
    """Fixed25% positives per long/standard x original6000 membership."""
    n=len(arrays['share']);z=np.asarray(arrays['presence'])==0;long=np.char.startswith(arrays['family'].astype(str),'long_')&~z
    inside=np.zeros(n,bool);inside[np.asarray(subset,int)]=True
    if n!=12000 or z.sum()!=2400 or long.sum()!=2880 or len(subset)!=6000:raise ValueError('Original strict matched12000/CNN6000 contract differs')
    requested=np.full(n,np.nan);selected=np.zeros(n,bool);strata=[]
    for j,(is_long,in_subset) in enumerate([(True,True),(True,False),(False,True),(False,False)]):
        eligible=np.flatnonzero(~z&(long==is_long)&(inside==in_subset))
        if len(eligible)%4:raise ValueError('Each fixed source stratum must support exact quarter')
        take=len(eligible)//4
        ix=np.random.default_rng(np.random.SeedSequence([seed,13,j,1])).permutation(eligible)[:take]
        target_rng=np.random.default_rng(np.random.SeedSequence([seed,13,j,2]));u=(np.arange(take)+target_rng.random(take))/take
        targets=SHARE_BOUNDS[0]+(SHARE_BOUNDS[1]-SHARE_BOUNDS[0])*u
        targets=targets[np.random.default_rng(np.random.SeedSequence([seed,13,j,3])).permutation(take)]
        selected[ix]=True;requested[ix]=targets
        strata.append({'long':is_long,'in_CNN_subset':in_subset,'eligible':len(eligible),'selected':take,
                       'target_mean':float(targets.mean()),'target_min':float(targets.min()),'target_max':float(targets.max())})
    if selected.sum()!=2400 or (selected&long).sum()!=720 or (selected&inside).sum()!=1200:raise AssertionError('Predeclared highshare source schedule failed')
    return selected,requested,strata


def _stats(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    return {'n':len(x),'mean':float(x.mean()) if len(x) else None,'rms':float(np.sqrt(np.mean(x*x))) if len(x) else None,
        'min':float(x.min()) if len(x) else None,'max':float(x.max()) if len(x) else None,
        'quantiles':{str(p):float(np.quantile(x,p)) for p in [0,.01,.05,.25,.5,.75,.95,.99,1]} if len(x) else {}}


def audit_distribution(arrays,recipes):
    y=arrays['share'].astype(float);z=y==0;long=np.char.startswith(arrays['family'].astype(str),'long_')&~z
    active=np.asarray([r['observed_added_minutes'] for r in recipes]);obs=np.isfinite(arrays['db']).all(2).sum(1)
    peak=np.asarray([np.nan if r.get('requested_full_peak_snr_db') is None else r['requested_full_peak_snr_db'] for r in recipes])
    global_snr=np.full(len(y),np.nan);global_snr[~z]=10*np.log10(y[~z]/(1-y[~z]))
    bins=[('zero',z),('0_to_.03',(~z)&(y<=.03)),('.03_to_.08',(y>.03)&(y<=.08)),('.08_to_.2',(y>.08)&(y<=.2)),
        ('.2_to_.25',(y>.2)&(y<.25)),('.25_to_.5',(y>=.25)&(y<=.5)),('.5_to_.7',(y>.5)&(y<=.7)),('.7_to_.85',(y>.7)&(y<=.85)),('above_.85',y>.85)]
    rows={}
    for name,m in [('all',np.ones(len(y),bool)),('positive',~z),('long_positive',long),('standard_positive',~z&~long),
                   ('support_le2_positive',~z&(active<=2)),('support_ge12_positive',~z&(active>=12)),
                   ('Nobs5_positive',~z&(obs==5)),('Nobs15_positive',~z&(obs==15))]:
        rows[name]={'share':_stats(y[m]),'observed_added_minutes':_stats(active[m]),'observed_minutes':_stats(obs[m]),
            'requested_full_peak_snr_db':_stats(peak[m]),'global_observed_snr_from_saved_share_db':_stats(global_snr[m])}
    return {'rows':len(y),'explicit_zero_count':int(z.sum()),'positive_count':int((~z).sum()),'long_positive_count':int(long.sum()),
        'high_share_counts':{str(t):int((y>=t).sum()) for t in [.25,.5,.7,.85]},'share_bins':{k:int(v.sum()) for k,v in bins},
        'family_counts':{str(k):int(v) for k,v in pd.Series(arrays['family']).value_counts().items()},'strata':rows,
        'label_semantics':'known whole added recording fraction; A=0 is no controlled addition, not natural vessel absence'}


def baseline_inputs():
    from ops.build_r12_mask_matched_control_v2 import load_control
    arrays,m=load_control(local(BASELINE));folder=local(BASELINE)
    if m['version']!='r12_site_sensor_recorder_mask_matched_HF_control_v2' or m['status']!='completed':raise ValueError('Strict matched v2 required')
    recipes=read(folder/m['train']['recipes_path']);meta=pd.read_parquet(folder/m['train']['metadata_path'])
    subset=np.load(folder/m['cnn_subset']['path'],allow_pickle=False)
    if len(arrays['share'])!=12000 or len(recipes)!=12000 or not np.array_equal(meta.sample_id.astype(str),arrays['sample_id'].astype(str)):raise ValueError('Baseline TRAIN alignment differs')
    return arrays,m,recipes,meta,subset


def _references(parent):
    result={}
    for role,e in {**parent['common_validation'],'high_snr_long':parent['high_snr_long']}.items():
        if role=='high_snr_long' and not e['created']:result[role]=dict(e);continue
        result[role]={k:relative(local(PARENT)/v) if k.endswith('path') or k=='path' else v for k,v in e.items() if k in ('path','sha256','metadata_path','metadata_sha256','recipes_path','recipes_sha256','created')}
        result[role]['values_opened']=False
    return result


def build_r13(output=DEFAULT_OUTPUT):
    out=local(output)
    if out.exists():raise FileExistsError('Fresh R13 output; no failed partial reuse')
    out.mkdir(parents=True)
    master=read(local(MASTER))
    if master['seed']!=SEED or master['format']!='r13_prospective_cpu_correction_v1':raise ValueError('Frozen R13 master differs')
    old=read(local(BASELINE)/'manifest.json');parent=validate_r12_bundle(local(PARENT))
    pinfiles=[*CODE,MASTER,local(BASELINE)/'manifest.json',local(BASELINE)/'readiness.json',local(BASELINE)/old['train']['path'],
        local(BASELINE)/old['train']['metadata_path'],local(BASELINE)/old['train']['recipes_path'],local(BASELINE)/old['cnn_subset']['path'],
        local(PARENT)/'manifest.json',local(PARENT)/'readiness.json','pipeline/dataset_r12.py','pipeline/dataset_r9.py',
        'ops/build_r12_mask_matched_control_v2.py']
    pins={relative(p):sha(local(p)) for p in pinfiles}
    plan={'version':VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,'input_sha256':pins,
        'TRAIN_only':True,'arms':['baseline','highshare'],'positive_highshare_fraction':.25,'target_uniform_bounds':list(SHARE_BOUNDS),
        'selection':'fixed permutations within long/standard x originalCNNsubset; target LHS independent shuffle; no y/error/source-performance selection or seed retries',
        'highshare_counts':{'full_positive':2400,'long':720,'standard':1680,'CNN_positive':1200,'CNN_long':360,'CNN_standard':840},
        'unchanged':['explicitzero20percent','long30percentpositive','all sources/site/sensor/recorder/O/reference/EQ/route/rowIDs','other9600rows full NPZ bytes'],
        'target_not_monotonic_uplift':'selected y may decrease; preserve declared uniform law rather than max(oldy,target)',
        'new_alpha':'u*sum(observedB)/((1-u)*sum(observedR)); no cap. Raw known q/y recalculated, input round0.1 only.',
        'baseline_cap_claim':'No cause is assumed from y distribution; original source code uses peak-reference/median-background scale without share clipping.',
        'validation_and_basis':'immutable external R12 references only; no values opened or new fit','model_eval_Test_network':False}
    write(out/'predeclared_plan.json',plan)
    try:
        base,m,recipes,meta,subset=baseline_inputs();audit=audit_distribution(base,recipes);write(out/'baseline_distribution_audit.json',audit)
        selected,requested,strata=highshare_schedule(base,subset)
        sources=R9Sources();train=sources.role_metadata(0,'train').set_index('base_window_index',drop=False)
        if not set(base['base_window_index']).issubset(set(train.index)):raise ValueError('Source outside existing TRAIN')
        if sources.identity!=m['source_identity']:raise ValueError('Original numeric source identity changed')
        high={k:v.copy() for k,v in base.items()};highmeta=meta.copy();newrecipes=json.loads(json.dumps(recipes));qa_rows=[];minute_snr=[];replay_err=[]
        for i in range(12000):
            if base['presence'][i]==0:continue
            recipe=recipes[i];row=train.loc[int(base['base_window_index'][i])];bg=sources.background(row)
            placed=reconstruct_reference_from_recipe(sources.reference_minutes[int(base['reference_index'][i])],recipe)
            b,a,e,o,top=components(bg,placed,recipe['alpha']);replay,oldqa=_render(b,a,e,o,top)
            if not all(np.array_equal(replay[k],base[k][i],equal_nan=np.asarray(replay[k]).dtype.kind=='f') for k in replay):raise AssertionError('Original rawB/A replay differs at row '+str(i))
            replay_err.append(oldqa['raw_label_identity_error']);minute_snr.extend(v for v in oldqa['minute_snr_db'] if v is not None)
            if not selected[i]:continue
            item,qa=solve_observed_mean_mixture(bg,placed,float(requested[i]))
            for k,v in item.items():high[k][i]=v
            if not np.array_equal(np.isfinite(item['db']),np.isfinite(base['db'][i])):raise AssertionError('Highshare changed observation mask')
            highmeta.at[i,'share']=float(item['share'])
            newrecipes[i]={**recipe,**qa,'parent_R12_recipe':recipe,'parent_requested_full_peak_snr_db':recipe['requested_full_peak_snr_db'],
                'requested_full_peak_snr_db':None,'new_effective_full_peak_snr_db':float(10*np.log10(qa['alpha']*recipe['full_reference_peak_after_EQ']/recipe['background_median_observed_minute_power'])),
                'r13_highshare_augmented':True}
            qa_rows.append({'row_index':i,'source_id':str(base['source_id'][i]),'sample_id':str(base['sample_id'][i]),
                'family':str(base['family'][i]),'baseline_share':float(base['share'][i]),'requested_share':float(requested[i]),
                'saved_share':float(item['share']),'alpha_before':float(recipe['alpha']),'alpha_after':qa['alpha'],
                'raw_identity_error':qa['raw_label_identity_error'],'rounded_pool_gap':qa['rounded_pool_gap'],
                'global_observed_snr_db':qa['global_observed_snr_db'],'new_effective_full_peak_snr_db':newrecipes[i]['new_effective_full_peak_snr_db'],
                'support_before':recipe['observed_added_minutes'],'support_after':qa['observed_added_minutes']})
        # Stratum/selection metadata never enters prediction inputs.
        highmeta['r13_highshare_augmented']=selected;highmeta['r13_requested_share']=requested
        for k in base:
            if array_sha(base[k][~selected])!=array_sha(high[k][~selected]):raise AssertionError('Untouched row bytes changed: '+k)
        immutable=set(base)-{'db','unrounded_db','share','cell_share'}
        if any(array_sha(base[k])!=array_sha(high[k]) for k in immutable):raise AssertionError('Source/presence/trajectory metadata changed')
        if not np.array_equal(np.isfinite(base['db']),np.isfinite(high['db'])):raise AssertionError('Any effective mask changed')
        np.savez_compressed(out/'highshare_train.npz',**high);highmeta.to_parquet(out/'highshare_train_metadata.parquet',index=False)
        write(out/'highshare_train_recipes.json',newrecipes);pd.DataFrame(qa_rows).to_parquet(out/'highshare_schedule.parquet',index=False)
        highaudit=audit_distribution(high,newrecipes)
        change={'increased':int(np.sum(high['share'][selected]>base['share'][selected])),
                'decreased':int(np.sum(high['share'][selected]<base['share'][selected])),
                'unchanged':int(np.sum(high['share'][selected]==base['share'][selected]))}
        summary={'baseline':audit,'highshare':highaudit,'selection_strata':strata,'selected_target_changes':change,
            'baseline_actual_minute_snr_db':_stats(minute_snr),'raw_B_A_replay_positive_rows':len(replay_err),'raw_identity_error_max':max(replay_err),
            'new_selected_raw_identity_error_max':max(q['raw_identity_error'] for q in qa_rows),'unchanged_rows_bytes':9600,
            'exactzero_count':2400,'long_positive_count':2880,'highshare_positive_count':2400,
            'input_raw_components_reversible_from_source_recipe':True,'source_and_O_unchanged':True,
            'new_y_law_not_Test_or_LB_target_matching':True,'baseline_caps_not_assumed_cause':True}
        write(out/'distribution_and_physics_audit.json',summary)
        sources.assert_unchanged(verify_banks=True);verify_pins(pins)
        baseentry={k:relative(local(BASELINE)/v) if k.endswith('path') or k=='path' else v for k,v in m['train'].items()}
        highentry={k:relative(out/v) for k,v in {'path':'highshare_train.npz','metadata_path':'highshare_train_metadata.parquet','recipes_path':'highshare_train_recipes.json'}.items()}
        highentry.update(sha256=sha(out/'highshare_train.npz'),metadata_sha256=sha(out/'highshare_train_metadata.parquet'),recipes_sha256=sha(out/'highshare_train_recipes.json'))
        manifest={'version':VERSION,'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,
            'input_sha256':pins,'plan_sha256':sha(out/'predeclared_plan.json'),'arms':{'baseline':baseentry,'highshare':highentry},
            'path_contract':'all arm, validation and subset paths are project-relative; bundle file_sha256 keys are bundle-relative',
            'parent_R12_data_dir':PARENT,'parent_R12_manifest_sha256':sha(local(PARENT)/'manifest.json'),'parent_R12_data_hash':parent['data_hash'],
            'baseline_control_data_hash':m['data_hash'],'source_identity':sources.identity,'basis_training_binding':parent['basis_training_binding'],
            'cnn_subset':{**m['cnn_subset'],'path':relative(local(BASELINE)/m['cnn_subset']['path'])},'validation_references':_references(parent),
            'source_24h_purge':parent['source_audit']['role_separation'],
            'preserved_contract':{'all_primary_sources_in_original_R9_TRAIN':True,'same12000_rowIDs_and_sources':True,'site_sensor_recorder_mask_presence_unchanged':True,
                'strict_HF_matching_preserved':True,'raw_original_components_replay_all9600positives_exact':True,
                'explicitzero20percent':True,'long30percent_of_positive':True,'same_original_CNN6000subset':True,
                'validation_values_opened':False,'model_training_eval_Test_read':False,'full5fold_dynamic_implemented':False},
            'highshare_fraction_of_positive':.25,'target_uniform_bounds':list(SHARE_BOUNDS),'highshare_schedule':{'path':relative(out/'highshare_schedule.parquet'),'sha256':sha(out/'highshare_schedule.parquet')},
            'file_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
        manifest['data_hash']=digest(manifest);write(out/'manifest.json',manifest)
        independent=verify_r13_dataset(out,require_ready=False);write(out/'independent_verification.json',independent)
        write(out/'readiness.json',{'version':VERSION,'ready_to_train':True,'manifest_sha256':sha(out/'manifest.json'),
            'data_hash':manifest['data_hash'],'independent_verification_sha256':sha(out/'independent_verification.json'),
            'source_checks_passed':True,'paired_raw_target_mask_checks_passed':True,'actual_model_training_or_evaluation_executed':False})
        return manifest
    except BaseException as e:
        write(out/'FAILED.json',{'error':repr(e),'ready_to_train':False,'partial_reuse_allowed':False});raise


def validate_r13_bundle(folder,*,require_ready=True):
    folder=local(folder);m=read(folder/'manifest.json')
    if (m.get('version')!=VERSION or m.get('status')!='completed' or digest({k:v for k,v in m.items() if k!='data_hash'})!=m['data_hash'] or (folder/'FAILED.json').exists()):raise ValueError('R13 manifest invalid/failed')
    verify_pins(m['input_sha256'])
    for n,h in m['file_sha256'].items():
        if sha(folder/n)!=h:raise ValueError('R13 generated artifact changed')
    for entry in m['arms'].values():
        for k,hkey in [('path','sha256'),('metadata_path','metadata_sha256'),('recipes_path','recipes_sha256')]:
            if sha(local(entry[k]))!=entry[hkey]:raise ValueError('R13 TRAIN artifact differs')
    if require_ready:
        r=read(folder/'readiness.json');qa=read(folder/'independent_verification.json')
        if (r.get('ready_to_train') is not True or r['manifest_sha256']!=sha(folder/'manifest.json') or r['data_hash']!=m['data_hash']
                or r['independent_verification_sha256']!=sha(folder/'independent_verification.json') or qa.get('status')!='passed' or qa['data_hash']!=m['data_hash']):raise ValueError('R13 readiness/raw-physics QA failed')
    return m


def load_r13_arrays(folder,arm):
    """Load TRAIN only; baseline keys/dtypes/bytes match immutable load_control."""
    if arm not in ('baseline','highshare'):raise ValueError('Explicit TRAIN arm baseline|highshare required')
    m=validate_r13_bundle(folder);return load_npz(local(m['arms'][arm]['path']))


def verify_r13_dataset(folder,*,require_ready=True):
    """Independent saved-component checks; no validation/model/Test access."""
    folder=local(folder);m=validate_r13_bundle(folder,require_ready=require_ready)
    base=load_npz(local(m['arms']['baseline']['path']));high=load_npz(local(m['arms']['highshare']['path']))
    subset=np.load(local(m['cnn_subset']['path']),allow_pickle=False);schedule=pd.read_parquet(local(m['highshare_schedule']['path']))
    selected=np.zeros(12000,bool);selected[schedule.row_index.to_numpy(int)]=True
    expected,requested,strata=highshare_schedule(base,subset)
    if not np.array_equal(selected,expected) or not np.array_equal(schedule.requested_share.to_numpy(),requested[selected]):raise AssertionError('Exact predeclared schedule replay failed')
    if set(base)!=set(high):raise AssertionError('TRAIN dict schema changed')
    for k in base:
        if base[k].dtype!=high[k].dtype or array_sha(base[k][~selected])!=array_sha(high[k][~selected]):raise AssertionError('Unchanged-row/schema verification failed')
    for k in set(base)-{'db','unrounded_db','share','cell_share'}:
        if array_sha(base[k])!=array_sha(high[k]):raise AssertionError('Source/label-presence identity changed')
    o=np.isfinite(high['db']);q=high['cell_share'].astype(float);raw=high['unrounded_db'];y=high['share'].astype(float)
    if not np.array_equal(o,np.isfinite(base['db'])) or not np.isfinite(q).all() or ((q<0)|(q>1)).any() or q[~o].any():raise AssertionError('Saved physical masks/ranges')
    if not np.array_equal(np.round(raw,1).astype(np.float32),high['db'],equal_nan=True):raise AssertionError('Raw->round0.1 input mismatch')
    e=np.where(o,10**((np.where(o,raw,0)-np.nanmax(raw,axis=(1,2),keepdims=True))/10),0)
    rawpool=np.sum(q*e,(1,2))/e.sum((1,2));err=float(abs(rawpool-y).max())
    if err>1e-7 or np.max(abs(y[selected]-requested[selected]))>3.1e-8:raise AssertionError('Saved raw component labels altered')
    if np.count_nonzero(y==0)!=2400 or not np.array_equal(high['presence'],(y>0).astype(high['presence'].dtype)):raise AssertionError('Presence/zero count')
    if not np.array_equal((q.sum(2)>0).sum(1),(base['cell_share'].sum(2)>0).sum(1)):raise AssertionError('Observed addition support changed')
    # Independent B from the source; A=E-B recovered from saved raw mixture
    # and compared with qE. This does not call the alpha-solver under test.
    sources=R9Sources();train=sources.role_metadata(0,'train').set_index('base_window_index',drop=False)
    errors=[];ys=[]
    for i in np.flatnonzero(selected):
        bg=np.asarray(sources.background(train.loc[int(base['base_window_index'][i])]),np.float64);mask=np.isfinite(bg);peak=np.nanmax(bg)
        b=np.where(mask,10**((bg-peak)/10),0.);mixture=np.where(mask,10**((raw[i]-peak)/10),0.);a=mixture-b
        scale=max(float(mixture.max()),1e-300)
        if a.min()<-2e-12*scale:raise AssertionError('Negative added energy in independent raw reconstruction')
        errors.append(float(np.max(abs(a-q[i]*mixture))/scale));ys.append(float(a.sum()/mixture.sum()))
    if max(errors)>1e-7 or np.max(abs(np.asarray(ys)-requested[selected]))>1e-10:raise AssertionError('Independent saved E-B/A/q/y oracle failed')
    sources.assert_unchanged(verify_banks=True)
    return {'status':'passed','data_hash':m['data_hash'],'source_schema_and12000rows':True,'unchanged9600commonfield_bytes':True,
        'original2400zeros_unchanged':True,'source_site_sensor_recorder_mask_presence_same':True,'long_720_standard1680_highshare':True,
        'same_CNN6000_highshare1200_long360_standard840':True,'exact_schedule_and_LHS_replay':True,
        'independent_raw_E_minus_sourceB_oracle_rows':2400,'independent_A_vs_qE_relative_error_max':max(errors),
        'independent_raw_y_vs_requested_error_max':float(np.max(abs(np.asarray(ys)-requested[selected]))),
        'saved_float32q_rawpool_y_error_max':err,'no_PCA_model_training_eval_Test_or_validation_values_read':True}
