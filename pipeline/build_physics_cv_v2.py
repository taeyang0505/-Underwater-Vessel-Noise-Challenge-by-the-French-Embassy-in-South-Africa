"""Regenerate clean full-pass TRAIN mixtures; preserve parent heldout bytes.

The source fold assignment and 24h support purge are immutable. No challenge,
calibration, sealed or model predictions are read. This is data construction,
not task-model training. Use a new output path for every source/config change.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,time
import numpy as np
import pandas as pd
from pipeline.build_quality_v2 import sha256,nearest_other_gap_seconds
from pipeline.native_sampler_v2 import ReferenceRecord
from pipeline.physics_data_checks import audit_active_snr,audit_minute_snr

ROOT=Path(__file__).resolve().parents[1]

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))

def flags_for_background(frame,flag_file):
    flags=pd.read_parquet(flag_file)
    # Only rows that actually supplied frozen window values can contaminate
    # those arrays; excluded short export rows are not silently reintroduced.
    used=flags[flags.window_included].copy()
    used['window_start']=pd.to_datetime(used.t).dt.floor('15min')
    keys=set(zip(used.recorder,used.window_start))
    result=frame.copy()
    result['boundary_candidate_flag']=[(rec,t) in keys for rec,t in zip(result.recorder,pd.to_datetime(result.start_export_clock))]
    result['urn_nominal_band_warning']=result.recorder.eq('URN_8321_8321.1.12000')
    result['boundary_artifact_candidate']=result.boundary_candidate_flag
    result['urn_8321_nominal_warning']=result.urn_nominal_band_warning
    return result

def balanced_full_pass(frame,seed):
    rng=np.random.default_rng(seed);pools={}
    for site,g in frame[frame.split.eq('train')].groupby('site',sort=True):
        pools[site]=list(rng.permutation(g.index.to_numpy()).astype(int))
    result=[]
    # Interleave sites while all their pools are available; every index is
    # used once rather than repeatedly sampling small sites with replacement.
    while pools:
        sites=list(pools)
        for site in rng.permutation(sites):
            result.append(pools[site].pop())
            if not pools[site]:del pools[site]
    order=np.asarray(result,dtype=np.int64)
    assert len(order)==len(np.unique(order))==int(frame.split.eq('train').sum())
    return order

def reference_records(version,split):
    vi=json.loads((version/'manifest.json').read_text());b=pd.read_parquet(version/vi['background_manifest'])
    table=pd.read_parquet(version/vi['reference_manifest']);use=table[table.split.eq(split)&table.reference_eligible]
    refs=[];banks={}
    for r in use.itertuples():
        path=ROOT/r.segment_file
        if r.segment_file not in banks:banks[r.segment_file]=np.load(path,mmap_mode='r')
        s=banks[r.segment_file][int(r.segment_start):int(r.segment_start+r.segment_count)]
        refs.append(ReferenceRecord(r.clip_id,r.source,r.group_key,r.split,s,measured=bool(r.measured),
            verified=bool(r.verified),role=r.role,quality_full_band_good=bool(r.quality_full_band_good),provenance=r.provenance))
    return b,refs

def sampler_for(version,seed,config):
    from pipeline.physics_sampler_v3 import PhysicsAwareSampler
    vi=json.loads((version/'manifest.json').read_text());b,refs=reference_records(version,'train')
    db=np.load(ROOT/vi['base_background_array'],mmap_mode='r')
    return PhysicsAwareSampler(db,b,refs,split='train',seed=seed,config=config,require_native_long=True),b

def materialize_train(version,out,order,seed,config,batch):
    sampler,b=sampler_for(version,seed,config);arrays=[[],[],[],[]];recipes=[];start=time.perf_counter()
    for start_index in range(0,len(order),batch):
        selected=order[start_index:start_index+batch]
        values=sampler.draw(len(selected),background_indices=selected)
        for i in range(4):arrays[i].append(values[i])
        r=values[4]
        if isinstance(r.get('background_index'),np.ndarray):r['background_index']=r['background_index'].tolist()
        recipes.append(r)
        if start_index and start_index%(batch*40)==0:print('train generated',start_index,'/',len(order),flush=True)
    data={key:np.concatenate(values) for key,values in zip(['db','share','presence','cell_share'],arrays)}
    np.savez_compressed(out/'train.npz',**data)
    write_json(out/'train_recipes.json',{'seed':seed,'batch_size':batch,'batches':recipes,
        'background_order_file':'training_background_order.npy','background_coverage_policy':'one pass, each clean primary exactly once'})
    items=[item for recipe in recipes for item in recipe['items']]
    actual=np.asarray([i['background_index'] for i in items],dtype=np.int64)
    assert np.array_equal(actual,order)
    assert set(actual)==set(b.index[b.split.eq('train')])
    assert not b.loc[actual,'boundary_candidate_flag'].any()
    secondary=[i.get('secondary_background_index',-1) for i in items]
    secondary=[int(i) for i in secondary if i is not None and int(i)>=0]
    assert all(b.loc[i,'split']=='train' and not b.loc[i,'boundary_candidate_flag'] for i in secondary)
    x=data['db'].astype('float64');valid=np.isfinite(x);peak=np.nanmax(x,axis=(1,2),keepdims=True)
    energy=np.where(valid,10**((x-peak)/10),0);q=data['cell_share'];y=data['share']
    target=(q*energy).sum((1,2))/energy.sum((1,2));err=float(np.max(np.abs(target-y)))
    assert err<3e-6 and (q[~valid]==0).all()
    assert np.isfinite(y).all() and ((0<=y)&(y<=1)).all()
    assert np.isfinite(q).all() and ((0<=q)&(q<=1)).all()
    assert np.array_equal(data['presence'],(y>0).astype('float32'))
    assert (valid.all(2).sum(1)>=3).all()
    desired=np.asarray([i['desired_share'] for i in items]);effective=np.asarray([i['effective_share'] for i in items])
    requested=np.asarray([i['requested_share'] for i in items])
    assert np.allclose(y,desired,atol=1e-6) and np.allclose(y,effective,atol=1e-6)
    assert not ((requested>0)&(y==0)).any()
    active=(q>0).any(2)&valid.all(2);observed=valid.all(2).sum(1);support=active.sum(1)/observed
    recipe_support=np.asarray([i['support_coverage'] for i in items])
    assert np.allclose(support,recipe_support,atol=1e-6)
    short_cap_checks=0
    for j,item in enumerate(items):
        assert b.loc[item['background_index'],'base_window_index']==item['base_background_index']
        references=item['references']
        for reference in references:
            eq=np.asarray(reference['reference_eq_db'],dtype=float)
            assert eq.shape==(25,) and np.isfinite(eq).all() and np.abs(eq).max()<=4+1e-9
            parameters=reference['reference_eq_parameters']
            if parameters:
                axis=np.linspace(-1,1,25)
                expected=(parameters['tilt_coefficient_db']*axis+parameters['quadratic_coefficient_db']*(axis**2-np.mean(axis**2)))*parameters['bound_scale']
                assert np.allclose(eq,expected,atol=1e-10)
            assert reference['reference_eq_applied_before_alpha'] is True
        if len(references)==1 and references[0]['kind']!='native_contiguous_15min_crop' and active[j].sum()<=2:
            assert y[j]<=min(.98,.25+.73*support[j])+2e-6
            short_cap_checks+=1
        if item.get('cap_applied'):
            assert item['effective_share']<=item['support_cap']+1e-8
        finite_minutes=[v for v in item['minute_snr_db'] if v is not None]
        if finite_minutes:
            assert abs(max(finite_minutes)-item['max_minute_snr_db_actual'])<1e-8
        else:assert item['max_minute_snr_db_actual'] is None
        if y[j]>0:
            expected_snr=10*np.log10(float(y[j])/(1-float(y[j])))
            assert abs(expected_snr-item['global_snr_db'])<2e-3
    snr_check=audit_active_snr(data['db'],q,[i['active_snr_db'] for i in items])
    minute_snr_check=audit_minute_snr(data['db'],q,[i['minute_snr_db'] for i in items],config.max_minute_snr_db)
    replay,_=sampler_for(version,seed,config);first=replay.draw(min(batch,len(order)),background_indices=order[:batch])
    for key,array in zip(['db','share','presence','cell_share'],first[:4]):
        assert np.array_equal(array,data[key][:len(array)],equal_nan=True)
    train_meta=pd.DataFrame({'sample_id':[f'physics_v2_{out.name}_train_{i:07d}' for i in range(len(y))],
        'background_index':actual,'base_background_index':[i['base_background_index'] for i in items],
        'boundary_candidate_flag':False,'urn_nominal_band_warning':b.loc[actual,'urn_nominal_band_warning'].to_numpy(),
        'boundary_artifact_candidate':False,'urn_8321_nominal_warning':b.loc[actual,'urn_nominal_band_warning'].to_numpy(),
        'observed_minutes':observed,'requested_share':requested,'effective_share':effective,'support_coverage':support,
        'global_snr_db':[i['global_snr_db'] for i in items],'active_snr_db':[i['active_snr_db'] for i in items],
        'support_cap':[i['support_cap'] for i in items],'cap_applied':[i['cap_applied'] for i in items],
        'max_minute_snr_db_actual':[i['max_minute_snr_db_actual'] for i in items],
        'minute_snr_cap_applied':[i['minute_snr_cap_applied'] for i in items],
        'route':[i['route'] for i in items]})
    train_meta.to_parquet(out/'train_metadata.parquet',index=False)
    return {'examples':len(y),'input_shape':list(data['db'].shape),'npz_bytes':(out/'train.npz').stat().st_size,
        'max_saved_cell_target_error':err,'positive_draw_became_zero_count':0,'frozen_generation_replayed':True,
        'primary_background_unique_count':len(set(actual)),'clean_primary_coverage_fraction':1.,
        'boundary_contaminated_primary_count':0,'boundary_contaminated_secondary_count':0,
        'secondary_background_uses':len(secondary),'short_support_cap_independent_checks':short_cap_checks,
        'route_counts':pd.Series([i['route'] for i in items]).value_counts().to_dict(),
        'capped_examples':int(sum(bool(i.get('cap_applied')) for i in items)),
        'active_snr_float32_interval_check':snr_check,
        'minute_snr_check':minute_snr_check,
        'minute_snr_capped_examples':int(sum(bool(i['minute_snr_cap_applied']) for i in items)),
        'active_snr_db_quantiles':np.quantile([i['active_snr_db'] for i in items if i['active_snr_db'] is not None],[.5,.9,.99,1]).tolist(),
        'active_snr_above30db_examples':int(sum(i['active_snr_db'] is not None and i['active_snr_db']>30 for i in items)),
        'zero_target_fraction':float((y==0).mean()),'seconds':time.perf_counter()-start,
        'surrogate_envelopes_used':False,'audio_repetition_used':False}

def add_evaluation_flags(path,parent,b):
    flags=b.set_index('base_window_index')[['boundary_candidate_flag','urn_nominal_band_warning',
        'boundary_artifact_candidate','urn_8321_nominal_warning']]
    summaries={}
    for name in ['validation_metadata.parquet','inner_validation_metadata.parquet']:
        shutil.copy2(parent/name,path/name.replace('.parquet','_original.parquet'))
        md=pd.read_parquet(parent/name);joined=flags.loc[md.base_background_index]
        for column in flags:md[column]=joined[column].to_numpy()
        md['boundary_policy']='diagnostic only; frozen evaluation examples retained'
        md.to_parquet(path/name,index=False)
        summaries[name]={'examples':len(md),'boundary_candidate_examples':int(md.boundary_candidate_flag.sum()),
            'urn_nominal_band_warning_examples':int(md.urn_nominal_band_warning.sum()),'original_sample_order_preserved':True}
    return summaries

def active_gap(b):
    active=b[b.split.isin(['train','validation','outer_validation'])];gap=np.inf
    for _,g in active.groupby('sensor_group'):
        sec=pd.to_datetime(g.start_export_clock).to_numpy(dtype='datetime64[ns]').astype('int64')//10**9;labels=g.split.to_numpy()
        for label in set(labels):
            d=nearest_other_gap_seconds(sec[labels==label],sec[labels!=label])-900
            if len(d):gap=min(gap,float(d.min()))
    assert gap>=86400
    return gap/3600

def main(a):
    from pipeline.physics_sampler_v3 import PhysicsAugmentationConfig
    cfg=json.loads(a.config.read_text());out=ROOT/cfg['output_dir'];parent=ROOT/cfg['parent_cv_dir']
    if out.exists():raise FileExistsError('Use a fresh version path; never overwrite a data version')
    parent_manifest=parent/'cv_manifest.json';old=json.loads(parent_manifest.read_text());assert old['ready_to_train']
    code_hashes={n:sha256(ROOT/n) for n in ['pipeline/build_physics_cv_v2.py',
        'pipeline/physics_sampler_v3.py','pipeline/native_sampler_v2.py','pipeline/augmentation_v2.py','pipeline/physics_data_checks.py']}
    first_gate=json.loads((ROOT/old['folds'][0]['data_dir']/'readiness_manifest.json').read_text())
    first_source=json.loads((ROOT/first_gate['source_version_dir']/'manifest.json').read_text())
    for name,h in first_source['frozen_v1_hashes'].items():assert sha256(ROOT/'data/processed'/name)==h
    flag_file=ROOT/'reports/data_quality_boundary_flags.parquet';physics=PhysicsAugmentationConfig(**cfg.get('physics_config',{}))
    bg=flags_for_background(pd.read_parquet(parent/'background_outer_assignment.parquet'),flag_file)
    out.mkdir(parents=True);bg.to_parquet(out/'background_outer_assignment.parquet',index=False)
    shutil.copy2(parent/'reference_outer_assignment.parquet',out/'reference_outer_assignment.parquet')
    source_hash=canonical_hash({'parent_cv_manifest_sha256':sha256(parent_manifest),'config':cfg,
        'physics_config':vars(physics),'boundary_flags_sha256':sha256(flag_file),
        'background_assignment_sha256':sha256(out/'background_outer_assignment.parquet')})
    cv={**old,'version':cfg['version'],'augmentation_version':cfg['augmentation_version'],
        'ready_to_train':False,'source_hash':source_hash,'parent_cv_source_hash':old['source_hash'],
        'parent_cv_manifest_sha256':sha256(parent_manifest),'parent_cv_dir':cfg['parent_cv_dir'],
        'heldout_bytes_policy':'Validation/outer NPZ and recipes are byte-identical to parent; evaluation metadata receive diagnostic flags only.',
        'folds':[],'model_training_executed':False}
    write_json(out/'cv_manifest.json',cv)
    audit={'version':cfg['version'],'source_hash':source_hash,'parent_cv_manifest_sha256':sha256(parent_manifest),
        'augmentation_version':cfg['augmentation_version'],'physical_limit_assumption':cfg['physical_limit_assumption'],
        'parent_source_hash':old['source_hash'],'all_source_fold_assignments_preserved':True,
        'boundary_candidate_physical_cause_verified':False,'boundary_policy':cfg['boundary_policy'],
        'URN_band_removal_applied':False,'calibration_sealed_Test_arrays_accessed':False,'actual_training_executed':False,
        'physics_config':vars(physics),'folds':[],'ready_to_train':False}
    for f in old['folds']:
        number=f['fold'];previous=ROOT/f['data_dir'];path=out/f'fold_{number:02d}';path.mkdir();v=path/'sources';v.mkdir()
        old_ready=json.loads((previous/'readiness_manifest.json').read_text());old_source=ROOT/old_ready['source_version_dir']
        si=json.loads((old_source/'manifest.json').read_text())
        b=flags_for_background(pd.read_parquet(old_source/'background_manifest.parquet'),flag_file)
        b['parent_split']=b.split;b['boundary_exclusion_applied']=b.split.eq('train')&b.boundary_candidate_flag
        b.loc[b.boundary_exclusion_applied,'split']='excluded_boundary_candidate'
        assert np.array_equal(b.base_window_index,bg.base_window_index)
        b.to_parquet(v/'background_manifest.parquet',index=False)
        shutil.copy2(old_source/'reference_manifest.parquet',v/'reference_manifest.parquet')
        refs=pd.read_parquet(v/'reference_manifest.parquet')
        assert refs.groupby('reference_component').split.nunique().max()==1
        assert refs.groupby('sha256').split.nunique().max()==1
        gap=active_gap(b)
        fold_hash=canonical_hash({'parent_source_manifest_sha256':sha256(old_source/'manifest.json'),
            'background_sha256':sha256(v/'background_manifest.parquet'),'reference_sha256':sha256(v/'reference_manifest.parquet'),
            'physics_config':vars(physics),'new_cv_source_hash':source_hash})
        source={**si,'version':cfg['version'],'source_hash':fold_hash,'ready_to_materialize':True,
            'metadata_sha256':{n:sha256(v/n) for n in ['background_manifest.parquet','reference_manifest.parquet']},
            'parent_source_version_manifest_sha256':sha256(old_source/'manifest.json'),
            'augmentation_version':cfg['augmentation_version'],'boundary_flags_sha256':sha256(flag_file)}
        write_json(v/'manifest.json',source)
        gate={**old_ready,'version':cfg['version'],'augmentation_version':cfg['augmentation_version'],
            'ready_to_train':False,'source_hash':fold_hash,'parent_cv_source_hash':source_hash,
            'parent_cv_manifest_sha256':sha256(parent_manifest),'parent_fold_readiness_sha256':sha256(previous/'readiness_manifest.json'),
            'source_version_dir':str(v.relative_to(ROOT)),'source_version_manifest_sha256':sha256(v/'manifest.json'),
            'physics_config':vars(physics),'minimum_cross_partition_support_gap_hours':gap,
            'heldout_arrays_and_recipes_frozen_from_parent':True,'augmentation_scope':'train only; old heldout recipe distribution deliberately unchanged for paired comparison',
            'model_training_executed':False,'partitions':{},'augmentation_config':vars(physics)}
        for key in ['file_sha256','files_sha256','data_package_hash','generator_code_sha256']:gate.pop(key,None)
        write_json(path/'readiness_manifest.json',gate)
        order=balanced_full_pass(b,cfg['permutation_seed']+number*10007);np.save(path/'training_background_order.npy',order)
        summary=materialize_train(v,path,order,cfg['generation_seed']+number*10007,physics,cfg['batch_size'])
        gate['partitions']['train']=summary;copied={}
        for split in ['validation','outer_validation']:
            for suffix in ['.npz','_recipes.json']:
                name=split+suffix;shutil.copy2(previous/name,path/name)
                assert sha256(path/name)==old_ready['file_sha256'][name]
                copied[name]=sha256(path/name)
            gate['partitions'][split]={**old_ready['partitions'][split],
                'generation_origin':'byte-identical parent frozen holdout, not regenerated',
                'parent_npz_sha256':copied[split+'.npz']}
        os.link(path/'validation.npz',path/'inner_validation.npz')
        eval_flags=add_evaluation_flags(path,previous,b)
        gate['frozen_heldout_file_sha256']=copied;gate['evaluation_diagnostic_flags']=eval_flags
        gate['file_sha256']={p.name:sha256(p) for p in path.iterdir() if p.is_file() and p.name not in ['readiness_manifest.json','manifest.json']}
        gate['files_sha256']=gate['file_sha256']
        gate['data_package_hash']=hashlib.sha256(json.dumps(gate['file_sha256'],sort_keys=True).encode()).hexdigest()
        assert all(sha256(ROOT/n)==h for n,h in code_hashes.items()),'Generator code changed during materialization'
        gate['generator_code_sha256']=code_hashes
        gate['ready_to_train']=True
        for name in ['readiness_manifest.json','manifest.json']:write_json(path/name,gate)
        cv['folds'].append({**f,'data_dir':str(path.relative_to(ROOT)),'data_package_hash':gate['data_package_hash'],
            'source_hash':fold_hash,'train_examples':summary['examples'],'heldout_arrays_byte_identical':True})
        audit['folds'].append({'fold':number,'background_counts':b.split.value_counts().to_dict(),
            'clean_train_primary_backgrounds':summary['examples'],'excluded_boundary_train_windows':int(b.boundary_exclusion_applied.sum()),
            'URN_flagged_clean_train_windows':int(b[b.split.eq('train')].urn_nominal_band_warning.sum()),
            'minimum_cross_split_support_gap_hours':gap,'training_checks':summary,'evaluation_flags':eval_flags,
            'frozen_heldout_file_sha256':copied,'data_package_hash':gate['data_package_hash']})
        write_json(out/'cv_manifest.json',cv);write_json(ROOT/'reports/group_cv_physics_data_audit.json',audit)
    assert sha256(parent_manifest)==cv['parent_cv_manifest_sha256']
    assert all(sha256(ROOT/n)==h for n,h in code_hashes.items())
    cv['ready_to_train']=True;cv['training_examples_total']=sum(f['train_examples'] for f in cv['folds'])
    audit['ready_to_train']=True;audit['training_examples_total']=cv['training_examples_total']
    audit['all_fold_examples_total']=cv['training_examples_total']+30000
    write_json(out/'cv_manifest.json',cv);write_json(ROOT/'reports/group_cv_physics_data_audit.json',audit)
    print('READY',out,'train',cv['training_examples_total'],'hash',source_hash,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/group_cv_physics_v2_data.json');main(p.parse_args())
