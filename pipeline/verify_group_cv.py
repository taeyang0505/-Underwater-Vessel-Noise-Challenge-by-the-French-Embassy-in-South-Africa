"""Audit nested source CV packages, independently of any fitted model."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from pipeline.build_quality_v2 import sha256,nearest_other_gap_seconds
from pipeline.materialize_quality_v2 import make_sampler
from pipeline.native_sampler_v2 import AugmentationConfig
ROOT=Path(__file__).resolve().parents[1]

def verify(directory):
    cv=json.loads((directory/'cv_manifest.json').read_text());assert cv['ready_to_train']
    allb=pd.read_parquet(directory/'background_outer_assignment.parquet')
    allr=pd.read_parquet(directory/'reference_outer_assignment.parquet')
    assert set(allb.original_split)<=set(['train','validation'])
    assert set(allr.split)<=set(['train','validation'])
    assert allr.groupby('reference_component').outer_fold.nunique().max()==1
    assert allr.groupby(['source','original_group_key']).outer_fold.nunique().max()==1
    assert allr.groupby('sha256').outer_fold.nunique().max()==1
    assert allr[allr.source=='VTUAD_v1.3.2'].groupby('recording_day').outer_fold.nunique().max()==1
    assert allb.groupby(['sensor_group','recording_week_start']).outer_fold.nunique().max()==1
    assert not allr.sealed.any()
    ids=set();held_b=set();held_r=set();rows=[]
    for f in cv['folds']:
        path=ROOT/f['data_dir'];ready=json.loads((path/'readiness_manifest.json').read_text())
        assert ready==json.loads((path/'manifest.json').read_text())
        assert ready['ready_to_train'] and ready['nested_early_stopping']
        assert ready['source_hash']==f['source_hash'] and ready['parent_cv_source_hash']==cv['source_hash']
        assert ready['file_sha256']==ready['files_sha256']
        assert hashlib.sha256(json.dumps(ready['file_sha256'],sort_keys=True).encode()).hexdigest()==ready['data_package_hash']==f['data_package_hash']
        for name,h in ready['file_sha256'].items():assert sha256(path/name)==h,name
        for name,h in ready['generator_code_sha256'].items():assert sha256(ROOT/name)==h,name
        version=ROOT/ready['source_version_dir'];assert sha256(version/'manifest.json')==ready['source_version_manifest_sha256']
        vi=json.loads((version/'manifest.json').read_text())
        for name,h in vi['metadata_sha256'].items():assert sha256(version/name)==h,name
        for name,h in vi['frozen_v1_hashes'].items():assert sha256(ROOT/'data/processed'/name)==h,name
        b=pd.read_parquet(version/'background_manifest.parquet');r=pd.read_parquet(version/'reference_manifest.parquet')
        assert r.groupby('reference_component').split.nunique().max()==1
        assert r.groupby('sha256').split.nunique().max()==1
        assert b.groupby('background_group').split.nunique().max()<=2 # active + purged edge support
        active=b[b.split!='purged_temporal_boundary'];gap=np.inf
        assert active.groupby('background_group').split.nunique().max()==1
        for _,g in active.groupby('sensor_group'):
            sec=pd.to_datetime(g.start_export_clock).to_numpy(dtype='datetime64[ns]').astype('int64')//10**9
            labels=g.split.to_numpy()
            for label in set(labels):
                d=nearest_other_gap_seconds(sec[labels==label],sec[labels!=label])-900
                if len(d):gap=min(gap,float(d.min()))
        assert gap>=86400
        bs=set(b[b.split=='outer_validation'].background_group);rs=set(r[r.split=='outer_validation'].reference_component)
        assert held_b.isdisjoint(bs) and held_r.isdisjoint(rs);held_b.update(bs);held_r.update(rs)
        native=r[r.native_long].groupby('split').reference_component.nunique().to_dict()
        assert all(native.get(s,0)>0 for s in ['train','validation','outer_validation'])
        result={'fold':f['fold'],'minimum_support_gap_hours':gap/3600,'partitions':{},'native_components':native}
        for split in ['train','validation','outer_validation']:
            recipes=json.loads((path/f'{split}_recipes.json').read_text());items=[i for batch in recipes['batches'] for i in batch['items']]
            with np.load(path/f'{split}.npz') as data:
                x=data['db'].astype('float64');o=np.isfinite(x);e=np.where(o,10**((x-np.nanmax(x,axis=(1,2),keepdims=True))/10),0)
                error=float(np.max(np.abs((data['cell_share']*e).sum((1,2))/e.sum((1,2))-data['share'])))
                assert error<3e-6 and (data['cell_share'][~o]==0).all()
                assert np.array_equal(data['presence'],(data['share']>0).astype('float32'))
                desired=np.array([i['desired_share'] for i in items]);assert np.allclose(desired,data['share'],atol=1e-6)
                assert not ((desired>0)&(data['share']==0)).any()
                known=set(r[r.split==split].clip_id)
                for item in items:
                    row=b.iloc[item['background_index']];assert row.split==split and row.base_window_index==item['base_background_index']
                    assert all(q['clip_id'] in known for q in item['references'])
                sampler,_=make_sampler(version,split,recipes['seed'],AugmentationConfig(**ready['augmentation_config']))
                replay=sampler.draw(min(recipes['batch_size'],len(desired)))
                for key,arr in zip(['db','share','presence','cell_share'],replay[:4]):assert np.array_equal(arr,data[key][:len(arr)],equal_nan=True)
                result['partitions'][split]={'n':len(desired),'max_energy_target_error':error,'positive_became_zero':0,'exact_replay':True}
        md=pd.read_parquet(path/'validation_metadata.parquet');assert len(md)==f['validation_examples']
        assert md.is_outer_oof.all() and md.fold.eq(f['fold']).all() and md.sample_id.is_unique
        assert ids.isdisjoint(set(md.sample_id));ids.update(md.sample_id)
        assert set(md.background_group)<=bs
        for q in md.reference_group:assert set(json.loads(q))<=rs
        assert not any((path/n).exists() for n in ['calibration.npz','domain_holdout.npz','sealed_holdout.npz'])
        rows.append(result)
    assert held_b==set(allb.background_group) and held_r==set(allr.reference_component)
    assert len(ids)==cv['outer_oof_examples']
    return {'passed':True,'source_hash':cv['source_hash'],'nested_early_stopping':True,'folds':rows,
        'outer_oof_examples':len(ids),'all_outer_source_units_assigned_once':True,
        'calibration_or_sealed_arrays_accessed':False,'challenge_test_accessed':False,'model_training_executed':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=ROOT/'data/synthetic/group_cv_nested5_pilot_v1');a=p.parse_args()
    report=verify(a.data_dir);(ROOT/'reports/group_cv_data_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
