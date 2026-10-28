"""Create a validated prebuilt train/validation/calibration package, no training.

The new sealed source is intentionally not materialized here. All generation
parameters are fixed independently of challenge Test inputs and leaderboard
scores. Use a fresh output path for any recipe or source change.
"""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import pandas as pd
from pipeline.native_sampler_v2 import NativeFirstSampler,ReferenceRecord,AugmentationConfig
from pipeline.build_quality_v2 import sha256
ROOT=Path(__file__).resolve().parents[1]

def make_sampler(version,split,seed,config):
    info=json.loads((version/'manifest.json').read_text())
    bmeta=pd.read_parquet(version/info['background_manifest'])
    db=np.load(ROOT/info['base_background_array'],mmap_mode='r')
    table=pd.read_parquet(version/info['reference_manifest'])
    # Never even open the sealed source array in a development sampler.
    use=table[(table.split==split)&(table.reference_eligible|table.role.eq('ambient_control'))]
    banks={};refs=[]
    for r in use.itertuples():
        path=ROOT/r.segment_file
        if r.segment_file not in banks:banks[r.segment_file]=np.load(path,mmap_mode='r')
        array=banks[r.segment_file]
        segments=array[int(r.segment_start):int(r.segment_start+r.segment_count)]
        refs.append(ReferenceRecord(r.clip_id,r.source,r.group_key,r.split,segments,
            measured=bool(r.measured),verified=bool(r.verified),role=r.role,
            quality_full_band_good=bool(r.quality_full_band_good),provenance=r.provenance))
    return NativeFirstSampler(db,bmeta,refs,split,seed,config,require_native_long=True),use

def materialize(version,out,split,n,seed,config,batch_size):
    sampler,refs=make_sampler(version,split,seed,config)
    arrays=[[],[],[],[]];recipes=[];start_time=time.perf_counter()
    for start in range(0,n,batch_size):
        values=sampler.draw(min(batch_size,n-start))
        for i in range(4):arrays[i].append(values[i])
        recipe=values[4];recipe['background_index']=recipe['background_index'].tolist();recipes.append(recipe)
        if start and start%(batch_size*20)==0:print(split,'generated',start,'/',n,flush=True)
    data={k:np.concatenate(v) for k,v in zip(['db','share','presence','cell_share'],arrays)}
    np.savez_compressed(out/f'{split}.npz',**data)
    (out/f'{split}_recipes.json').write_text(json.dumps({'seed':seed,'batch_size':batch_size,'batches':recipes},separators=(',',':')))
    # Audit saved quantized arrays rather than only the pre-save computation.
    with np.load(out/f'{split}.npz') as saved:
        x=saved['db'].astype('float64');valid=np.isfinite(x);peak=np.nanmax(x,axis=(1,2),keepdims=True)
        e=np.where(valid,10**((x-peak)/10),0)
        recovered=(saved['cell_share']*e).sum((1,2))/e.sum((1,2))
        error=float(np.max(np.abs(recovered-saved['share'])))
        assert error<3e-6
        assert np.array_equal(saved['presence'],(saved['share']>0).astype('float32'))
        assert saved['db'].shape==(n,15,25) and saved['cell_share'].shape==(n,15,25)
        assert np.isfinite(saved['share']).all() and ((saved['share']>=0)&(saved['share']<=1)).all()
        assert (valid.all(2).sum(1)>=3).all()
    items=sum([r['items'] for r in recipes],[])
    desired=np.array([r['desired_share'] for r in items])
    assert np.allclose(data['share'],desired,atol=1e-6)
    degenerate=int(((desired>0)&(data['share']==0)).sum());assert degenerate==0
    used=[q for r in items for q in r['references']]
    sources=pd.Series([q['source'] for q in used]).value_counts().to_dict()
    native=sum(q['kind']=='native_contiguous_15min_crop' for q in used)
    known=set(refs[refs.reference_eligible].clip_id)
    assert all(q['clip_id'] in known for q in used)
    # Replaying generation with the same RNG/source package must reproduce
    # the first saved batch, including the observation mask, bit for bit.
    replay,_=make_sampler(version,split,seed,config)
    first=replay.draw(min(batch_size,n))
    assert np.array_equal(first[0],data['db'][:len(first[0])],equal_nan=True)
    assert np.array_equal(first[1],data['share'][:len(first[1])])
    summary={'examples':n,'input_shape':list(data['db'].shape),'seconds':time.perf_counter()-start_time,
        'npz_bytes':(out/f'{split}.npz').stat().st_size,'max_saved_cell_target_error':error,
        'positive_draw_became_zero_count':degenerate,'frozen_generation_replayed':True,
        'reference_uses_by_source':sources,'native_long_reference_uses':native,
        'native_long_fraction_of_reference_uses':native/max(len(used),1),
        'unique_reference_clips_used':len({q['clip_id'] for q in used}),
        'reference_source_groups_used':len({(q['source'],q['group_key']) for q in used}),
        'zero_target_fraction':float((data['share']==0).mean()),
        'surrogate_envelopes_used':False,'audio_repetition_used':False}
    print(split,summary,flush=True)
    return summary

def main(a):
    if a.output.exists():raise FileExistsError('Use a fresh versioned output path')
    version=a.version;info=json.loads((version/'manifest.json').read_text())
    assert info['input_quality_gate_passed'] and info['sealed_holdout_locked']
    for file,h in info['frozen_v1_hashes'].items():assert sha256(ROOT/'data/processed'/file)==h
    ref=pd.read_parquet(version/'reference_manifest.parquet')
    active=ref[ref.reference_eligible & ref.split.isin(['train','validation','calibration','sealed_holdout'])]
    assert active.groupby(['source','group_key']).split.nunique().max()==1
    assert active.groupby('sha256').split.nunique().max()==1
    b=pd.read_parquet(version/'background_manifest.parquet')
    assert b[b.split.isin(['train','validation','calibration'])].groupby(['sensor_group','recording_week_start']).split.nunique().max()==1
    config=AugmentationConfig(background_blend_probability=a.background_blend_probability,
        measured_ambient_probability=a.ambient_noise_probability,common_response_range_db=a.receiver_response_db)
    a.output.mkdir(parents=True)
    # False is written first. An interrupted run can never look train-ready.
    report={'version':'quality_v2_native_prebuilt','ready_to_train':False,'source_hash':info['source_hash'],
        'source_version_dir':str(version.relative_to(ROOT)),'source_version_manifest_sha256':sha256(version/'manifest.json'),
        'source_checks_passed':True,'reference_group_and_duplicate_isolation_passed':True,
        'background_week_and_purge_isolation_passed':True,'sealed_holdout_locked':True,
        'sealed_holdout_materialized':False,'model_evaluated_on_sealed':False,
        'challenge_test_accessed':False,'existing_sealed_NOAA_accessed':False,'model_training_executed':False,
        'frozen_v1_data_modified':False,'augmentation_config':vars(config),'partitions':{}}
    readiness=a.output/'readiness_manifest.json';readiness.write_text(json.dumps(report,indent=2))
    for split,n,seed in [('train',a.n_train,a.seed),('validation',a.n_val,a.eval_seed+1009),
                         ('calibration',a.n_cal,a.eval_seed+2018)]:
        report['partitions'][split]=materialize(version,a.output,split,n,seed,config,a.batch)
        readiness.write_text(json.dumps(report,indent=2))
    # The contract fingerprints the actual sources, recipe configuration and
    # exact generated arrays; it is independent of the number of GPU epochs.
    report['file_sha256']={p.name:sha256(p) for p in a.output.iterdir() if p.name not in ['readiness_manifest.json','manifest.json']}
    report['files_sha256']=report['file_sha256']
    report['generator_code_sha256']={name:sha256(ROOT/name) for name in [
        'pipeline/augmentation_v2.py','pipeline/native_sampler_v2.py','pipeline/materialize_quality_v2.py',
        'pipeline/build_quality_v2.py','pipeline/attach_native_v2.py']}
    report['data_package_hash']=hashlib.sha256(json.dumps(report['file_sha256'],sort_keys=True).encode()).hexdigest()
    report['source_counts']=active.groupby(['source','split']).agg(clips=('clip_id','size'),
        source_groups=('group_key','nunique'),recording_days=('recording_day','nunique'),
        native_long_clips=('native_long','sum')).reset_index().to_dict('records')
    report['ready_to_train']=True
    report['requires_fresh_v2_task_model_or_context']=True
    report['vendor_original_pretrained_weights_may_be_reused']=True
    report['limitations']=['Synthetic examples reuse a small number of independent source groups.',
        'Annotations do not establish isolated natural vessel energy or independent physical hulls.',
        'Provider full-object MD5 was not recomputed for pinned-range NOAA acquisitions.',
        'No model has yet been trained or selected on this v2 package.']
    readiness.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (a.output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('READY',a.output,'package hash',report['data_package_hash'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',type=Path,default=ROOT/'data/versions/quality_v2_native')
    p.add_argument('--output',type=Path,default=ROOT/'data/synthetic/quality_v2_native_pilot')
    p.add_argument('--n-train',type=int,default=16000);p.add_argument('--n-val',type=int,default=3000)
    p.add_argument('--n-cal',type=int,default=3000);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-seed',type=int,default=20261024);p.add_argument('--batch',type=int,default=256)
    p.add_argument('--background-blend-probability',type=float,default=0.)
    p.add_argument('--ambient-noise-probability',type=float,default=0.)
    p.add_argument('--receiver-response-db',type=float,default=0.);main(p.parse_args())
