"""Independent prebuilt package checks without challenge or sealed inputs."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
from pipeline.build_quality_v2 import sha256, nearest_other_gap_seconds
from pipeline.materialize_quality_v2 import make_sampler
from pipeline.native_sampler_v2 import AugmentationConfig

ROOT = Path(__file__).resolve().parents[1]

def verify(package):
    r = json.loads((package/'readiness_manifest.json').read_text())
    assert r['ready_to_train'] and not r['model_training_executed']
    assert r == json.loads((package/'manifest.json').read_text())
    assert r['file_sha256'] == r['files_sha256']
    for name, expected in r['files_sha256'].items():
        assert sha256(package/name) == expected, name
    assert hashlib.sha256(json.dumps(r['file_sha256'], sort_keys=True).encode()).hexdigest() == r['data_package_hash']
    version = ROOT/r['source_version_dir']
    assert sha256(version/'manifest.json') == r['source_version_manifest_sha256']
    v = json.loads((version/'manifest.json').read_text())
    assert v['ready_to_materialize'] and v['input_quality_gate_passed']
    for name, expected in v['frozen_v1_hashes'].items():
        assert sha256(ROOT/'data/processed'/name) == expected, name
    for name, expected in r['generator_code_sha256'].items():
        assert sha256(ROOT/name) == expected, name
    b = pd.read_parquet(version/'background_manifest.parquet')
    dev = b[b.split.isin(['train','validation','calibration'])]
    assert dev.groupby(['sensor_group','recording_week_start']).split.nunique().max() == 1
    min_gap = np.inf
    for _, group in dev.groupby('sensor_group'):
        seconds = pd.to_datetime(group.start_export_clock).to_numpy(dtype='datetime64[ns]').astype('int64')//10**9
        labels = group.split.to_numpy()
        for split in set(labels):
            gaps = nearest_other_gap_seconds(seconds[labels==split], seconds[labels!=split])-900
            if len(gaps): min_gap=min(min_gap,float(gaps.min()))
    assert min_gap >= 86400
    ref = pd.read_parquet(version/'reference_manifest.parquet')
    eligible = ref[ref.reference_eligible]
    assert eligible.groupby(['source','group_key']).split.nunique().max() == 1
    assert eligible.groupby('sha256').split.nunique().max() == 1
    assert not (package/'sealed_holdout.npz').exists()
    assert not (package/'domain_holdout.npz').exists()
    result={'passed':True,'data_package_hash':r['data_package_hash'],
        'source_manifest_sha256':sha256(version/'manifest.json'),
        'minimum_cross_split_same_sensor_support_gap_hours':min_gap/3600,
        'source_and_generator_hashes_verified':True,'challenge_test_accessed':False,
        'sealed_source_arrays_accessed':False,'partitions':{}}
    for split in ['train','validation','calibration']:
        recipe = json.loads((package/f'{split}_recipes.json').read_text())
        items=[item for batch in recipe['batches'] for item in batch['items']]
        known=set(ref[(ref.split==split)&ref.reference_eligible].clip_id)
        with np.load(package/f'{split}.npz') as values:
            x=values['db'].astype('float64'); finite=np.isfinite(x)
            e=np.where(finite,10**((x-np.nanmax(x,axis=(1,2),keepdims=True))/10),0)
            y=values['share'];cell=values['cell_share']
            error=float(np.max(np.abs((cell*e).sum((1,2))/e.sum((1,2))-y)))
            assert error<3e-6 and np.isfinite(y).all() and ((0<=y)&(y<=1)).all()
            assert np.isfinite(cell).all() and ((0<=cell)&(cell<=1)).all()
            assert (cell[~finite]==0).all()
            assert np.array_equal(values['presence'],(y>0).astype('float32'))
            assert (finite.all(2).sum(1)>=3).all()
            desired=np.array([i['desired_share'] for i in items])
            assert np.allclose(y,desired,atol=1e-6)
            assert not np.any((desired>0)&(y==0))
            for i in items:
                row=b.iloc[i['background_index']]
                assert row.split==split and row.base_window_index==i['base_background_index']
                assert all(q['clip_id'] in known for q in i['references'])
                assert len({q['clip_id'] for q in i['references']})==len(i['references'])
            sampler,_=make_sampler(version,split,recipe['seed'],AugmentationConfig(**r['augmentation_config']))
            replay=sampler.draw(min(recipe['batch_size'],len(y)))
            for key,array in zip(['db','share','presence','cell_share'],replay[:4]):
                assert np.array_equal(array,values[key][:len(array)],equal_nan=True)
            result['partitions'][split]={'examples':len(y),'max_energy_target_error':error,
                'positive_draws_became_zero':0,'exact_first_batch_replay':True}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=ROOT/'data/synthetic/quality_v2_native_pilot')
    a=p.parse_args(); report=verify(a.data_dir)
    (ROOT/'reports/data_quality_prebuilt_v2_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
