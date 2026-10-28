"""Predeclared, CPU-only development probes of frozen R9 Fold0 models.

Does not read Test/report targets, train a model, or change R9 artifacts.
The complete-observation restriction and support conditioning apply to EVERY
factorial arm. Effects are stress-test contrasts, not proof of memorization.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops.infer_r9_fold0_all5_pilot import (
    inputs, predict_members, combine, environment, local, sha, verify_pins,
    MEMBERS, DEFAULT_CONFIG, DEFAULT_BUNDLE, DEFAULT_TREE)

SEED = 2026102604
ARMS = ('r9_complete', 'mean_scale_only', 'smooth_trajectory_only',
        'eq12_only', 'combined_stress')

def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def metrics(y, pred):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y,dtype=float);p=np.asarray(pred['share'],dtype=float);z=y>0
    e=p-y
    auc=lambda v:float(roc_auc_score(z,v)) if len(np.unique(z))==2 else None
    return {'n':len(y),'positive':int(z.sum()),'rmse':float(np.sqrt(np.mean(e*e))),
            'mse':float(np.mean(e*e)),'mae':float(np.abs(e).mean()),'bias':float(e.mean()),
            'auc_score':auc(pred['score']),'auc_share':auc(p),
            'target_mean':float(y.mean()),'target_rms':float(np.sqrt(np.mean(y*y))),
            'constant_0_rmse':float(np.sqrt(np.mean(y*y))),
            'constant_01_rmse':float(np.sqrt(np.mean((y-.1)**2))),
            'constant_mean_rmse':float(y.std()),'prediction_mean':float(p.mean()),
            'prediction_rms':float(np.sqrt(np.mean(p*p)))}

def eq_curve(rng, maximum):
    import numpy as np
    axis=np.linspace(-1,1,25)
    v=rng.uniform(-maximum,maximum)*axis+rng.uniform(-maximum/2,maximum/2)*(axis*axis-np.mean(axis*axis))
    return v*min(1.,maximum/max(float(np.max(np.abs(v))),1e-12))

def render(bg, native, spec, arm, target=None):
    """Known added energy remains the target, including external ambient."""
    import numpy as np
    from pipeline.sampler_r9 import place_minutes, energy_weights
    rng=np.random.default_rng(spec['render_seed'])
    eq=eq_curve(rng,12. if arm in ('eq12_only','combined_stress') else 6.)
    r=np.asarray(native,dtype=np.float64)*10.**(eq[None,:]/10.)
    placement_rng=np.random.default_rng(spec['placement_seed'])
    placed,trajectory=place_minutes(r,placement_rng,'native_'+spec['family'])
    if arm in ('smooth_trajectory_only','combined_stress'):
        # A continuous bounded time resampling, followed by a nonzero-ended
        # broad passage envelope; never interpreted as the unknown Test law.
        speed=float(np.random.default_rng(spec['warp_seed']).uniform(.8,1.25))
        support=np.flatnonzero(placed.sum(1)>0)
        support_center=float((support[0]+support[-1])/2.)
        pos=np.clip(support_center+(np.arange(15)-support_center)*speed,0.,14.)
        placed=np.stack([np.interp(pos,np.arange(15),placed[:,j]) for j in range(25)],1)
        center=float(np.random.default_rng(spec['warp_seed']+1).uniform(-2.,17.))
        width=float(np.random.default_rng(spec['warp_seed']+2).uniform(3.,10.))
        envelope=.15+.85*np.exp(-.5*((np.arange(15)-center)/width)**2)
        placed*=envelope[:,None]
        trajectory={**trajectory,'diagnostic_warp_factor':speed,
                    'envelope_center':center,'envelope_width':width,'envelope_floor':.15}
    x=np.asarray(bg,dtype=np.float64);o=np.isfinite(x);peak=float(np.nanmax(x))
    b=np.where(o,10.**((np.where(o,x,peak)-peak)/10.),0.)
    active=placed.sum(axis=1)>0
    if not active.any():raise ValueError('Empty reference support')
    if target is not None:
        alpha=float(target)*b.sum()/((1.-float(target))*np.sum(placed*o))
    elif spec['no_source']:
        alpha=0.
    elif arm in ('mean_scale_only','combined_stress'):
        # Mean reference and background power over ALL observed minutes,
        # including zero-added minutes. This deliberately changes target scale.
        denom=float(np.mean(placed.sum(1)[o.any(1)]))
        alpha=10.**(spec['snr_db']/10.)*float(np.mean(b.sum(1)[o.any(1)]))/denom
    else:
        alpha=10.**(spec['snr_db']/10.)*float(np.median(b.sum(1)[o.any(1)]))/float(np.max(r.sum(1)))
    a=np.where(o,alpha*placed,0.);e=b+a
    q=np.divide(a,e,out=np.zeros_like(a),where=o&(e>0))
    y=float(a.sum()/e.sum())
    raw=np.full((15,25),np.nan);raw[o]=10.*np.log10(e[o])+peak
    db=np.round(raw,1).astype(np.float32);w=energy_weights(db)
    raw_w=e/e.sum();gap=float(abs(np.sum(raw_w*q)-y))
    rounded_gap=float(abs(np.sum(w*q)-y))
    if gap>1e-10 or not np.array_equal(np.isfinite(db),o):raise AssertionError('Energy or observation identity broke')
    if target is not None and abs(y-target)>1e-10:raise AssertionError('Paired target mismatch')
    if target is None and ((y==0)!=bool(spec['no_source'])):raise AssertionError('Explicit zero policy differs')
    return db,q.astype(np.float32),np.float32(y),{
        'arm':arm,'family':spec['family'],'snr_db':spec['snr_db'],'alpha':alpha,
        'eq_min_db':float(eq.min()),'eq_max_db':float(eq.max()),'trajectory':trajectory,
        'raw_pool_identity_error':gap,'rounded_pool_label_gap':rounded_gap}

def fixture():
    import numpy as np
    bg=np.full((15,25),65.,dtype=np.float32)
    r=np.ones((5,25));spec={'render_seed':1,'placement_seed':2,'warp_seed':3,'family':'edge','snr_db':2.,'no_source':False}
    gaps=[]
    for arm in ARMS:
        db,q,y,recipe=render(bg,r,spec,arm)
        if not (0<y<1) or not np.allclose(db*10,np.round(db*10),atol=1e-4):raise AssertionError('Grid/range')
        other=render(bg,r,spec,arm)
        if not np.array_equal(db,other[0]):raise AssertionError('Seed reproducibility')
        gaps.append(recipe['raw_pool_identity_error'])
        zero=render(bg,r,{**spec,'no_source':True},arm)
        if zero[2]!=0 or zero[1].any():raise AssertionError('No source')
        for target in [.01,.2,.8]:
            if abs(float(render(bg,r,spec,arm,target=target)[2])-target)>3e-8:raise AssertionError('Paired ratio')
    # Edge support must survive time resampling, including one-minute clips.
    count=0
    for length in [1,2,3,5,15,30]:
        for family in ['inside','short','edge']:
            for seed in range(30):
                z={**spec,'render_seed':seed,'placement_seed':seed,'warp_seed':seed,'family':family}
                for arm in ['smooth_trajectory_only','combined_stress']:
                    render(bg,np.ones((length,25)),z,arm);count+=1
    return {'passed':True,'max_raw_energy_identity_error':max(gaps),'arms':list(ARMS),'additional_edge_warp_support_cases':count}

def sample_sources(sources,n):
    import numpy as np,pandas as pd
    from pipeline.source_split_r6 import minute_masks,apply_minute_qc,nearest_gap
    oldpath=ROOT/'data/versions/quality_v2_native/background_manifest.parquet'
    orig=pd.read_parquet(oldpath);domain=orig[orig.split.eq('domain_holdout')].copy().reset_index(drop=True)
    rowpath=ROOT/'data/processed/sa_row_manifest.parquet'
    flagpath=ROOT/'reports/data_quality_boundary_flags.parquet'
    bits,flagged,_=minute_masks(domain,pd.read_parquet(rowpath),pd.read_parquet(flagpath))
    domain['boundary_mask_bits']=bits
    bank=sources.background_bank()
    db=apply_minute_qc(np.asarray(bank[domain.base_window_index.to_numpy(int)]),bits)
    domain['qc_observed_minutes']=np.isfinite(db).all(2).sum(1)
    train=sources.role_metadata(0,'train');seen=sources.role_metadata(0,'selection')
    if set(domain.base_window_index)&set(train.base_window_index):raise AssertionError('BG source overlap')
    # Explicit recorder/sensor/time-support test, not an assumption from names.
    gaps=[]
    for sensor,g in domain.groupby('sensor_group'):
        t=train[train.sensor_group.eq(sensor)]
        if len(t):
            secs=lambda a:pd.to_datetime(a).to_numpy(dtype='datetime64[ns]').astype('int64')//10**9
            gap=np.maximum(nearest_gap(secs(g.start_export_clock),secs(t.start_export_clock))-900,0)
            gaps.extend(gap.tolist())
            if np.any(gap<86400):raise AssertionError('Unseen BG has <24h support purge')
    shared_recorders=set(domain.recorder)&set(train.recorder)
    shared_sensors=set(domain.sensor_group)&set(train.sensor_group)
    if shared_recorders or shared_sensors:raise AssertionError('Expected new metadata site/recorder components')
    pools={'seen':seen[seen.qc_observed_minutes.eq(15)].copy(),
           'unseen':domain[domain.qc_observed_minutes.eq(15)].copy()}
    selected={}
    for j,(name,frame) in enumerate(pools.items()):
        if len(frame)<n:raise ValueError('Insufficient unique full-observation source windows')
        # Within-domain source distribution retained; no Test site proportions.
        ids=np.random.default_rng(SEED+j).choice(len(frame),size=n,replace=False)
        selected[name]=frame.iloc[ids].reset_index(drop=True)
    evidence={'unseen_full_source_windows':len(domain),'unseen_minute_flags_masked':int(sum(int(v).bit_count() for v in bits)),
              'unseen_source_counts_by_site':domain.site.value_counts().to_dict(),
              'eligible_complete_counts':{k:len(v) for k,v in pools.items()},
              'selected_site_counts':{k:v.site.value_counts().to_dict() for k,v in selected.items()},
              'selected_observed_minutes':15,'source_windows_within_arm_unique':True,
              'train_base_index_overlap':0,'train_recorder_string_overlap':[],
              'train_sensor_group_overlap':[],'same_sensor_minimum_support_gap_seconds':min(gaps) if gaps else None,
              'purge_result':'no common sensor component; source/recorder components disjoint',
              'identity_limit':'sensor_group and recorder strings are metadata identities, not verified physical hardware IDs',
              'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [oldpath,rowpath,flagpath]}}
    return selected,evidence

def infer_panel(name,db,y,meta,nn,tree,binding,weights,out,threads):
    import numpy as np
    folder=out/name;folder.mkdir()
    np.savez_compressed(folder/'samples.npz',db=db,share=y)
    write(folder/'metadata.json',meta)
    pred,timing=predict_members(nn,tree,db,binding,batch_size=128,threads=threads)
    pred['frozen_all5']=combine(pred,weights)
    np.savez_compressed(folder/'predictions.npz',**{
        k+'_'+v:np.asarray(p[v]) for k,p in pred.items() for v in ['share','score']})
    result={m:metrics(y,p) for m,p in pred.items()}
    write(folder/'metrics.json',result);write(folder/'timings.json',timing)
    print(json.dumps({'panel_completed':name,'ensemble_rmse':result['frozen_all5']['rmse'],
                      'ensemble_auc':result['frozen_all5']['auc_score']}),flush=True)
    return result,pred

def run(a):
    import numpy as np,pandas as pd
    from pipeline.dataset_r9 import R9Sources
    from pipeline.sampler_r9 import native_minutes
    start=time.perf_counter();out=local(a.output);out.mkdir(parents=True,exist_ok=False)
    plan={'version':'r10_factorial_transfer_audit_v1','created_unix':time.time(),'seed':SEED,
          'samples_per_factorial_cell':a.n,'ambient_pairs':a.pairs,'arms':list(ARMS),
          'backgrounds':['Fold0 selection source pool','original domain_holdout'],
          'observation_policy':'all 15 observed AFTER exact original boundary-minute QC; same restriction both backgrounds/all arms',
          'zero_policy':'exact 20% explicit zero in every arm; complete observations prevent mask-induced zero; not raw R9 population',
          'reference_policy':'all509 admitted R9 reference clips shared; ambient uses487 non-background vs16 dedicated ambient',
          'paired_targets':'independent of LB: Uniform(.01,.60); same B/support length/EQ and target for ship/ambient; target is whole added recording',
          'generator_notes':['baseline matches R9 on complete observations',
            'mean scale uses reference/background mean power over ALL observed minutes including zero-added minutes; target scale changes',
            'trajectory warp .8..1.25 with Gaussian envelope floor.15; stress only, not claimed Test recipe',
            'EQ12 is diagnostic stress only; production remains EQ6',
            'same reference index/family/SNR/random draws across backgrounds and related arms'],
          'effect_statistics':['MSE contrast','RMSE contrast','AUC difference','target moments and constant baselines'],
          'interpretation':'development stress probes; domain/target-scale changes and bundled changes do not isolate a causal memorization mechanism',
          'no_operations':['Test access','report target access','model fitting','weight fitting','policy edits','GPU','SSH','submission'],
          'script_sha256':sha(__file__)}
    write(out/'predeclared_plan.json',plan);write(out/'numeric_fixture.json',fixture())
    nn=json.loads((ROOT/'ops/r9_fold0_pilot_nn_map.json').read_text())
    pins,modelpins,binding,tree=inputs(nn,ROOT/DEFAULT_TREE,ROOT/DEFAULT_BUNDLE,ROOT/DEFAULT_CONFIG)
    policy=ROOT/'submissions/r9_fold0_all5_cpu_r1/frozen_policy.json'
    frozen=json.loads(policy.read_text());weights=frozen.get('weights')
    if weights is None:weights=frozen['selected_weights']
    if list(weights)!=[.1,.1,.1,.1,.6]:raise ValueError('Unexpected frozen policy')
    pins[str(policy.relative_to(ROOT))]=sha(policy)
    write(out/'input_pins.json',{'pins':pins,'model_pins':modelpins,'binding':binding})
    sources=R9Sources();frames,evidence=sample_sources(sources,a.n)
    write(out/'background_source_audit.json',evidence)
    for name,f in frames.items():f.to_parquet(out/(name+'_source_manifest.parquet'),index=False)
    rng=np.random.default_rng(SEED+10);families=np.array(['inside']*(a.n//2)+['short']*(a.n//4)+['edge']*(a.n-a.n//2-a.n//4))
    rng.shuffle(families);no=np.zeros(a.n,bool);no[rng.choice(a.n,a.n//5,replace=False)]=True
    refs=np.resize(np.arange(509),a.n);rng.shuffle(refs)
    specs=[{'render_seed':int(rng.integers(2**31)),'placement_seed':int(rng.integers(2**31)),
            'warp_seed':int(rng.integers(2**31)),'family':str(families[i]),
            'snr_db':float(rng.uniform(-10,10)),'no_source':bool(no[i]),'reference_index':int(refs[i])} for i in range(a.n)]
    write(out/'schedule.json',specs)
    results={};preds={}
    for bgname,frame in frames.items():
        backgrounds=[sources.background(row) for _,row in frame.iterrows()]
        for arm in ARMS:
            records=[render(bg,sources.reference_minutes[s['reference_index']],s,arm) for bg,s in zip(backgrounds,specs)]
            db=np.stack([v[0] for v in records]);y=np.asarray([v[2] for v in records])
            name=bgname+'__'+arm
            meta={'background_domain':bgname,'arm':arm,'recipes':[v[3] for v in records],
                  'base_window_indices':frame.base_window_index.astype(int).tolist(),
                  'reference_indices':refs.astype(int).tolist(),'no_source':no.tolist()}
            results[name],preds[name]=infer_panel(name,db,y,meta,nn,tree,binding,weights,out,a.threads)
    # A paired ambient sensitivity probe uses shared geometric support.
    amb=pd.read_parquet(sources.ref_dir/'ambient_control_manifest.parquet').sort_values('ambient_row_index')
    bank=np.load(sources.reference_path,mmap_mode='r',allow_pickle=False);ambminutes=[]
    for row in amb.itertuples(index=False):
        part=np.asarray(bank[int(row.segment_start):int(row.segment_start)+int(row.segment_count)])
        if hashlib.sha256(np.ascontiguousarray(part).tobytes()).hexdigest()!=row.expected_segment_slice_sha256:raise ValueError('Ambient slice SHA')
        ambminutes.append(native_minutes(part))
    shiprows=sources.references[~sources.references.is_weak_background_label_in_vessel_archive].reference_row_index.to_numpy(int)
    pairrng=np.random.default_rng(SEED+20);pairdb={'ship':[],'ambient':[]};pairy=[];pairmeta=[]
    for i in range(a.pairs):
        row=frames['seen'].iloc[i%a.n];bg=sources.background(row)
        shipidx=int(shiprows[i%len(shiprows)]);ambidx=i%len(ambminutes)
        ship=sources.reference_minutes[shipidx];ambient=ambminutes[ambidx]
        length=min(3,len(ship),len(ambient));ship=ship[:length];ambient=ambient[:length]
        spec={**specs[i%a.n],'family':'inside','no_source':False};target=float(pairrng.uniform(.01,.60))
        s=render(bg,ship,spec,'r9_complete',target=target);b=render(bg,ambient,spec,'r9_complete',target=target)
        pairdb['ship'].append(s[0]);pairdb['ambient'].append(b[0]);pairy.append(s[2])
        pairmeta.append({'background':int(row.base_window_index),'ship_reference':shipidx,'ambient_reference':ambidx,
                         'target':target,'matched_native_minutes':length,'ship_recipe':s[3],'ambient_recipe':b[3]})
    paired={}
    for kind,items in pairdb.items():
        results['paired_'+kind],paired[kind]=infer_panel('paired_'+kind,np.stack(items),np.asarray(pairy),
                {'pairs':pairmeta,'target_definition':'whole externally added energy; no relabeling'},nn,tree,binding,weights,out,a.threads)
    pair_effect={}
    for m in list(MEMBERS)+['frozen_all5']:
        delta=paired['ship'][m]['share']-paired['ambient'][m]['share']
        pair_effect[m]={'ship_minus_ambient_share_mean':float(delta.mean()),'share_mean_abs_difference':float(np.abs(delta).mean()),
                        'ship_minus_ambient_score_mean':float(np.mean(paired['ship'][m]['score']-paired['ambient'][m]['score'])),
                        'interpretation':'Equal prediction at equal known y is correct for this target, not proof of equipment memorization.'}
    write(out/'paired_ambient_effects.json',pair_effect)
    effects={}
    for arm in ARMS[1:]:
        effects[arm]={}
        for m in list(MEMBERS)+['frozen_all5']:
            v=[results[k][m] for k in ['seen__r9_complete','unseen__r9_complete','seen__'+arm,'unseen__'+arm]]
            effects[arm][m]={}
            for metric in ['mse','rmse','auc_score','auc_share']:
                a0,b0,a1,b1=[x[metric] for x in v]
                effects[arm][m][metric]={'background_contrast':b0-a0,'generator_contrast':a1-a0,
                    'interaction':b1-b0-a1+a0,'combined_contrast':b1-a0}
    write(out/'factorial_effects.json',effects)
    sources.assert_unchanged();verify_pins(pins)
    final={'plan_sha256':sha(out/'predeclared_plan.json'),'status':'completed','seconds':time.perf_counter()-start,
           'results':results,'effects_file':'factorial_effects.json','paired_file':'paired_ambient_effects.json',
           'original_source_policy_model_pins_unchanged':True,'no_Test_or_unused_report_labels':True}
    write(out/'RESULTS.json',final)
    print(json.dumps({'completed':str(out),'seconds':final['seconds']}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='reports/r10_diagnostics/factorial_transfer_r2');p.add_argument('--n',type=int,default=1000)
    p.add_argument('--pairs',type=int,default=500);p.add_argument('--threads',type=int,default=4);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    environment(a.threads)
    if a.self_test:print(json.dumps(fixture()));return
    run(a)
if __name__=='__main__':main()
