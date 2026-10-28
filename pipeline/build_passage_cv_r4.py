"""Build a NEW VT-only source CV package; never copy legacy evaluation arrays."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,time
import numpy as np
import pandas as pd
from pipeline.passage_sampler_r4 import PassageConfigR4,PassageReference,PassageSamplerR4,make_passage_schedule,schedule_slice
from pipeline.build_physics_cv_v2 import active_gap
from ops.verify_passage_cv_r4 import verify_arrays,check_reference_isolation,matched_distributions,verify_package,numeric_fixtures,sha

ROOT=Path(__file__).resolve().parents[1]
VERSION='quality_v4_vtuad_passage_group_cv_prebuilt'
PROTOCOL='source_group_nested5_vtuad_passage_r4'
SPLITS=['train','validation','outer_validation']
KEYS=['db','share','presence','cell_share']

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))

def load_records(table,split):
    records=[];banks={};checks={}
    for row in table[table.split.eq(split)].itertuples(index=False):
        metadata=row._asdict();path=ROOT/metadata['segment_file']
        if metadata['segment_file'] not in banks:banks[metadata['segment_file']]=np.load(path,mmap_mode='r')
        begin=int(metadata['segment_start']);end=begin+int(metadata['segment_count'])
        segments=banks[metadata['segment_file']][begin:end]
        ref=PassageReference(metadata,segments);ref.validate();records.append(ref)
        checks[metadata['clip_id']]=hashlib.sha256(np.ascontiguousarray(segments).tobytes()).hexdigest()
    return records,checks

def records_for_source(source,split):
    manifest=json.loads((source/'manifest.json').read_text())
    b=pd.read_parquet(source/'background_manifest.parquet');r=pd.read_parquet(source/'reference_manifest.parquet')
    records,checks=load_records(r,split)
    db=np.load(ROOT/manifest['base_background_array'],mmap_mode='r')
    return db,b,records,checks

def materialize(source,folder,split,indices,seed,cfg,batch,fold):
    db,b,refs,segment_hashes=records_for_source(source,split)
    sampler=PassageSamplerR4(db,b,refs,split,seed,cfg);values=[[],[],[],[]];recipes=[];items=[];started=time.perf_counter()
    schedule=make_passage_schedule(len(indices),seed+170003,cfg)
    for begin in range(0,len(indices),batch):
        selection=indices[begin:begin+batch];draw=sampler.draw(len(selection),selection,schedule_slice(schedule,begin,begin+len(selection)))
        for i in range(4):values[i].append(draw[i])
        recipes.append(draw[4]);items.extend(draw[4]['items'])
        if begin and begin%(batch*50)==0:print('R4',fold,split,begin,'/',len(indices),flush=True)
    arrays={k:np.concatenate(v) for k,v in zip(KEYS,values)};summary=verify_arrays(arrays)
    assert np.array_equal(np.asarray([i['background_index'] for i in items]),indices)
    assert not ((np.asarray([i['requested_share'] for i in items])>0)&(arrays['share']==0)).any()
    assert np.allclose([i['desired_share'] for i in items],arrays['share'],atol=1e-7)
    # Real data replay is limited to the first fixed batch, with identical RNG
    # state and exact known source indices. No draw-dependent seed retries.
    replay=PassageSamplerR4(db,b,refs,split,seed,cfg).draw(min(batch,len(indices)),indices[:batch],schedule_slice(schedule,0,min(batch,len(indices))))
    for k,v in zip(KEYS,replay[:4]):assert np.array_equal(v,arrays[k][:len(v)],equal_nan=True)
    assert replay[4]==recipes[0]
    np.savez_compressed(folder/(split+'.npz'),**arrays)
    np.savez_compressed(folder/(split+'_schedule.npz'),requested_share=schedule['requested_share'],rendered_segments=schedule['rendered_segments'])
    # Compact JSON still retains the complete deterministic component recipe.
    (folder/(split+'_recipes.json')).write_text(json.dumps({'version':'passage_r4','seed':seed,'batch_size':batch,'batches':recipes,
        'schedule_seed':schedule['schedule_seed'],'schedule_file':split+'_schedule.npz','schedule_scheme':schedule['scheme']},separators=(',',':'),allow_nan=False))
    metadata=[]
    for n,item in enumerate(items):
        bb=b.iloc[item['background_index']];rr=item['references'];r=rr[0] if rr else None
        duration=r['derived_duration_seconds'] if r else 0
        fine=-1 if r is None else min(int(np.searchsorted([120,300,600,900,1380],duration,side='right')-1),3)
        metadata.append({'sample_id':f'passage_r4_f{fold:02d}_{split}_{n:07d}','fold':fold,'validation_id':n,
            'background_index':item['background_index'],'base_background_index':item['base_background_index'],
            'background_group':str(bb.background_group),'reference_group':json.dumps([r['reference_component']] if r else []),
            'reference_component':r['reference_component'] if r else '',
            'reference_clip_id':r['clip_id'] if r else '', 'reference_count':int(r is not None),
            'duration_bin':r['duration_bin'] if r else -1,'fine_duration_bin':fine,
            'used_source_seconds':r['measured_used_seconds'] if r else 0,'rendered_duration_seconds':duration,
            'timewarp_ratio':r['timewarp_ratio'] if r else 0,
            'observed_minutes':item['observed_minutes'],'active_ref_minutes':item['active_ref_minutes'],
            'support_coverage':item['support_coverage'],'effective_share':item['desired_share'],'requested_share':item['requested_share'],
            'desired_share':item['desired_share'],'site':str(bb.site),'sensor_group':str(bb.sensor_group),
            'boundary_artifact_candidate':bool(bb.boundary_candidate_flag),'boundary_candidate_flag':bool(bb.boundary_candidate_flag),
            'urn_8321_nominal_warning':bool(bb.urn_nominal_band_warning),'urn_nominal_band_warning':bool(bb.urn_nominal_band_warning),
            'is_outer_oof':split=='outer_validation','label_semantics':'known added VT passage energy, including recorded ambience'})
    md=pd.DataFrame(metadata);md.to_parquet(folder/(split+'_sample_metadata.parquet'),index=False)
    summary.update({'positive_draw_became_zero_count':0,'frozen_generation_replayed':True,'replayed_examples':len(replay[0]),
        'schedule_seed':schedule['schedule_seed'],'scheduled_zero_count':schedule['zero_count'],'schedule_scheme':schedule['scheme'],
        'npz_bytes':(folder/(split+'.npz')).stat().st_size,'seconds':time.perf_counter()-started,
        'reference_components':int(md.loc[md.reference_count>0,'reference_component'].nunique()),
        'reference_clips':int(md.loc[md.reference_count>0,'reference_clip_id'].nunique()),
        'one_passage_per_positive':True,'natural_mask_only':True,'reference_segment_slice_sha256':segment_hashes})
    return summary,md

def main(args):
    config=json.loads(args.config.read_text());assert config['protocol']==PROTOCOL and config['version']==VERSION
    output=ROOT/config['output_dir'];parent=ROOT/config['parent_cv_dir'];references=ROOT/config['reference_version_dir']
    if output.exists():raise FileExistsError('R4 output already exists; choose a NEW version, never overwrite/delete')
    parent_digest=sha(parent/'cv_manifest.json')
    parent_cv=json.loads((parent/'cv_manifest.json').read_text());assert parent_cv['ready_to_train'] and parent_cv['n_splits']==5
    reference_manifest=references/'manifest.json';reference_digest=sha(reference_manifest);rv=json.loads(reference_manifest.read_text())
    if rv.get('ready_to_materialize') is not True:raise ValueError('VT reference source audit not ready')
    for name,h in rv['file_sha256'].items():assert sha(references/name)==h,('Changed frozen VT reference metadata',name)
    cfg=PassageConfigR4(**config.get('passage_config',{}));numeric=numeric_fixtures()
    names=['pipeline/passage_sampler_r4.py','pipeline/build_passage_cv_r4.py','ops/verify_passage_cv_r4.py']
    code={n:sha(ROOT/n) for n in names};output.mkdir(parents=True)
    shutil.copy2(parent/'background_outer_assignment.parquet',output/'background_outer_assignment.parquet')
    global_ref=references/'reference_manifest.parquet';shutil.copy2(global_ref,output/'reference_outer_assignment.parquet')
    source_hash=digest({'parent_cv_manifest_sha256':sha(parent/'cv_manifest.json'),'reference_version_manifest_sha256':sha(reference_manifest),
        'reference_assignment_sha256':sha(global_ref),'config':config,'code':code})
    cv={'version':VERSION,'protocol':PROTOCOL,'ready_to_train':False,'n_splits':5,'source_hash':source_hash,'folds':[],
        'source_pool':'VT-only eligible original train+validation; reserved/excluded/HMS/NOAA forbidden',
        'nested_early_stopping':True,'outer_validation_used_for_early_stopping':False,
        'challenge_test_accessed':False,'sealed_arrays_accessed':False,'model_training_executed':False,
        'legacy_evaluation_arrays_loaded':False,'same_transformation_policy_all_splits':True,
        'passage_config':cfg.to_dict(),'distribution_tolerances':config['distribution_tolerances']}
    write(output/'cv_manifest.json',cv)
    audit={'version':VERSION,'protocol':PROTOCOL,'source_hash':source_hash,'ready_to_train':False,
        'numeric_fixtures':numeric,'config':config,'code_sha256':code,'folds':[],
        'natural_source_dependence_limit':'Source isolation applies to retained hull/day/overlap/bytes; one shared VT hydrophone remains.',
        'simulator_limit':'Declared controlled fade/channel/energy rules; not a reverse-engineered official generator.'}
    outer_ids=set();outer_groups=set()
    for f in parent_cv['folds']:
        number=int(f['fold']);old=ROOT/f['data_dir'];folder=output/f'fold_{number:02d}';folder.mkdir();source=folder/'sources';source.mkdir()
        parent_gate=json.loads((old/'readiness_manifest.json').read_text());parent_source=ROOT/parent_gate['source_version_dir']
        si=json.loads((parent_source/'manifest.json').read_text())
        b=pd.read_parquet(parent_source/'background_manifest.parquet')
        shutil.copy2(parent_source/'background_manifest.parquet',source/'background_manifest.parquet')
        reference_file=references/f'fold_{number:02d}_reference_manifest.parquet';r=pd.read_parquet(reference_file)
        reference_check=check_reference_isolation(r)
        shutil.copy2(reference_file,source/'reference_manifest.parquet')
        assert sha(source/'reference_manifest.parquet')==rv['file_sha256'][reference_file.name]
        assert sha(source/'background_manifest.parquet')==si['metadata_sha256']['background_manifest.parquet']
        gap=active_gap(b);assert gap>=24
        fh=digest({'global_source_hash':source_hash,'fold':number,'background_sha256':sha(source/'background_manifest.parquet'),
            'reference_sha256':sha(source/'reference_manifest.parquet')})
        source_info={'version':VERSION,'source_hash':fh,'ready_to_materialize':True,'input_quality_gate_passed':True,
            'sealed_holdout_locked':True,'background_manifest':'background_manifest.parquet','reference_manifest':'reference_manifest.parquet',
            'base_background_array':si['base_background_array'],'parent_background_manifest_sha256':sha(parent_source/'background_manifest.parquet'),
            'reference_version_manifest_sha256':sha(reference_manifest),'metadata_sha256':{name:sha(source/name) for name in ['background_manifest.parquet','reference_manifest.parquet']}}
        write(source/'manifest.json',source_info)
        gate={'version':VERSION,'protocol':PROTOCOL,'cv_protocol':PROTOCOL,'fold':number,'ready_to_train':False,
            'source_hash':fh,'parent_cv_source_hash':source_hash,'source_version_dir':str(source.relative_to(ROOT)),
            'source_version_manifest_sha256':sha(source/'manifest.json'),'source_checks_passed':True,
            'reference_group_and_duplicate_isolation_passed':True,'background_week_and_purge_isolation_passed':True,
            'sealed_holdout_locked':True,'challenge_test_accessed':False,'sealed_arrays_accessed':False,
            'existing_sealed_NOAA_accessed':False,'sealed_holdout_materialized':False,'frozen_v1_data_modified':False,
            'model_evaluated_on_sealed':False,'calibration_accessed':False,
            'model_training_executed':False,'nested_early_stopping':True,'validation_role':'inner_early_stopping',
            'outer_validation_role':'outer_oof_reporting_only','feature_version':3,'preprocessing_version':3,
            'minimum_cross_partition_support_gap_hours':gap,'requires_fresh_task_fit':True,'passage_config':cfg.to_dict(),
            'same_transformation_policy_all_splits':True,'distribution_tolerances':config['distribution_tolerances'],'partitions':{}}
        gate['generation_seed_base']=config['generation_seed']
        write(folder/'readiness_manifest.json',gate)
        order=np.load(old/'training_background_order.npy',allow_pickle=False)
        assert len(np.unique(order))==len(order)==int(b.split.eq('train').sum())
        assert set(order)==set(np.flatnonzero(b.split.eq('train'))) and not b.iloc[order].boundary_candidate_flag.any()
        shutil.copy2(old/'training_background_order.npy',folder/'training_background_order.npy')
        tables={}
        for k,split in enumerate(SPLITS):
            seed=int(config['generation_seed'])+number*10007+k*1000003
            indices=order if split=='train' else np.random.default_rng(seed+19).choice(np.flatnonzero(b.split.eq(split)),size=int(config['evaluation_examples']),replace=True)
            summary,md=materialize(source,folder,split,indices,seed,cfg,int(config['batch_size']),number)
            gate['partitions'][split]=summary;tables[split]=md
        os.link(folder/'validation.npz',folder/'inner_validation.npz')
        shutil.copy2(folder/'validation_sample_metadata.parquet',folder/'inner_validation_metadata.parquet')
        shutil.copy2(folder/'outer_validation_sample_metadata.parquet',folder/'validation_metadata.parquet')
        shutil.copy2(folder/'outer_validation_sample_metadata.parquet',folder/'outer_validation_metadata.parquet')
        assert outer_ids.isdisjoint(tables['outer_validation'].sample_id);outer_ids.update(tables['outer_validation'].sample_id)
        assigned=set(r.loc[r.split.eq('outer_validation'),'reference_component']);assert outer_groups.isdisjoint(assigned);outer_groups.update(assigned)
        distribution=matched_distributions(tables,config['distribution_tolerances']);gate['matched_distribution_quality_gate']=distribution
        gate['file_sha256']={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name not in ['manifest.json','readiness_manifest.json']}
        gate['files_sha256']=gate['file_sha256'];gate['data_package_hash']=hashlib.sha256(json.dumps(gate['file_sha256'],sort_keys=True).encode()).hexdigest()
        gate['generator_code_sha256']=code
        write(folder/'readiness_manifest.json',gate)
        independent=verify_package(folder,require_ready=False)
        assert all(sha(ROOT/n)==h for n,h in code.items()),'R4 generation code changed during build'
        gate['independent_data_verification_passed']=True;gate['ready_to_train']=True
        write(folder/'readiness_manifest.json',gate);write(folder/'manifest.json',gate)
        cv['folds'].append({'fold':number,'data_dir':str(folder.relative_to(ROOT)),'train_examples':len(order),
            'validation_examples':len(tables['outer_validation']),'inner_validation_examples':len(tables['validation']),
            'outer_validation_file':'outer_validation.npz','validation_metadata_file':'validation_metadata.parquet',
            'outer_validation_metadata_file':'outer_validation_metadata.parquet','data_package_hash':gate['data_package_hash'],'source_hash':fh})
        audit['folds'].append({'fold':number,'source':reference_check,'minimum_gap_hours':gap,
            'parent_background_manifest_byte_identical':True,'training_primary_order_byte_identical':True,
            'partitions':gate['partitions'],'matched_distribution_quality_gate':distribution,'independent_verification':independent})
        write(output/'cv_manifest.json',cv);write(ROOT/'reports/R4_DATA_AUDIT.json',audit)
    assert sha(parent/'cv_manifest.json')==parent_digest and sha(reference_manifest)==reference_digest
    assert all(sha(references/name)==h for name,h in rv['file_sha256'].items())
    assert all(sha(ROOT/n)==h for n,h in code.items())
    cv['ready_to_train']=True;cv['training_examples_total']=sum(f['train_examples'] for f in cv['folds']);cv['outer_oof_examples']=len(outer_ids)
    audit['ready_to_train']=True;audit['training_examples_total']=cv['training_examples_total'];audit['outer_oof_examples']=len(outer_ids)
    write(output/'cv_manifest.json',cv);write(ROOT/'reports/R4_DATA_AUDIT.json',audit)
    print(json.dumps({'ready_to_train':True,'cv_dir':str(output),'source_hash':source_hash,'training_examples':cv['training_examples_total']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/group_cv_passage_r4_data.json');main(p.parse_args())
