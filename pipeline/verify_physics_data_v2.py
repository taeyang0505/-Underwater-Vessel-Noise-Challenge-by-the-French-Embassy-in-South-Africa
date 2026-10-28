"""Independent saved-train QA and byte-frozen heldout checks for physics v2."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from pipeline.build_quality_v2 import sha256,nearest_other_gap_seconds
from pipeline.physics_data_checks import audit_active_snr,audit_minute_snr
ROOT=Path(__file__).resolve().parents[1]

def verify(path):
    cv=json.loads((path/'cv_manifest.json').read_text());assert cv['ready_to_train']
    parent=ROOT/cv['parent_cv_dir'];assert sha256(parent/'cv_manifest.json')==cv['parent_cv_manifest_sha256']
    old=json.loads((parent/'cv_manifest.json').read_text());assert len(cv['folds'])==len(old['folds'])==5
    assert sha256(path/'reference_outer_assignment.parquet')==sha256(parent/'reference_outer_assignment.parquet')
    bmaster=pd.read_parquet(path/'background_outer_assignment.parquet');oldb=pd.read_parquet(parent/'background_outer_assignment.parquet')
    assert bmaster[oldb.columns].equals(oldb)
    result={'passed':True,'data_version':cv['version'],'source_hash':cv['source_hash'],
        'parent_cv_manifest_sha256':cv['parent_cv_manifest_sha256'],'all_source_assignments_preserved':True,
        'Test_calibration_sealed_or_outer_values_opened':False,'actual_training_executed':False,'folds':[]}
    total=0
    for f,previous in zip(cv['folds'],old['folds']):
        data=ROOT/f['data_dir'];prior=ROOT/previous['data_dir'];g=json.loads((data/'readiness_manifest.json').read_text())
        assert g['ready_to_train'] and g['version']=='quality_v2_physics_group_cv_prebuilt'
        assert g==json.loads((data/'manifest.json').read_text())
        assert g['file_sha256']==g['files_sha256']
        assert hashlib.sha256(json.dumps(g['file_sha256'],sort_keys=True).encode()).hexdigest()==g['data_package_hash']==f['data_package_hash']
        for name,h in g['file_sha256'].items():assert sha256(data/name)==h,name
        for name,h in g['generator_code_sha256'].items():assert sha256(ROOT/name)==h,name
        source=ROOT/g['source_version_dir'];assert sha256(source/'manifest.json')==g['source_version_manifest_sha256']
        s=json.loads((source/'manifest.json').read_text())
        for name,h in s['metadata_sha256'].items():assert sha256(source/name)==h,name
        b=pd.read_parquet(source/'background_manifest.parquet');r=pd.read_parquet(source/'reference_manifest.parquet')
        oldsource=ROOT/json.loads((prior/'readiness_manifest.json').read_text())['source_version_dir']
        oldframe=pd.read_parquet(oldsource/'background_manifest.parquet')
        assert b[oldframe.columns.drop('split')].equals(oldframe[oldframe.columns.drop('split')])
        changed=b.split!=oldframe.split
        assert (oldframe.loc[changed,'split']=='train').all()
        assert (b.loc[changed,'split']=='excluded_boundary_candidate').all() and b.loc[changed,'boundary_candidate_flag'].all()
        assert sha256(source/'reference_manifest.parquet')==sha256(oldsource/'reference_manifest.parquet')
        assert r.groupby('reference_component').split.nunique().max()==1 and r.groupby('sha256').split.nunique().max()==1
        gap=np.inf;active=b[b.split.isin(['train','validation','outer_validation'])]
        for _,group in active.groupby('sensor_group'):
            sec=pd.to_datetime(group.start_export_clock).to_numpy(dtype='datetime64[ns]').astype('int64')//10**9;labels=group.split.to_numpy()
            for split in set(labels):
                d=nearest_other_gap_seconds(sec[labels==split],sec[labels!=split])-900
                if len(d):gap=min(gap,float(d.min()))
        assert gap>=86400
        order=np.load(data/'training_background_order.npy');assert len(order)==len(set(order))
        assert set(order)==set(b.index[b.split.eq('train')]) and not b.loc[order,'boundary_candidate_flag'].any()
        recipe=json.loads((data/'train_recipes.json').read_text());items=[i for batch in recipe['batches'] for i in batch['items']]
        assert np.array_equal(order,[i['background_index'] for i in items])
        secondary=[int(i.get('secondary_background_index',-1)) for i in items if i.get('secondary_background_index',-1) is not None and int(i.get('secondary_background_index',-1))>=0]
        assert all(b.loc[i,'split']=='train' and not b.loc[i,'boundary_candidate_flag'] for i in secondary)
        known=set(r[r.split.eq('train')].clip_id)
        assert all(q['clip_id'] in known for item in items for q in item['references'])
        with np.load(data/'train.npz') as z:
            x=z['db'].astype('float64');valid=np.isfinite(x);e=np.where(valid,10**((x-np.nanmax(x,axis=(1,2),keepdims=True))/10),0)
            q=z['cell_share'];y=z['share'];recovered=(q*e).sum((1,2))/e.sum((1,2));error=float(np.max(np.abs(y-recovered)))
            assert len(y)==len(order) and error<3e-6 and (q[~valid]==0).all()
            assert (valid.all(2).sum(1)>=3).all() and ((y>=0)&(y<=1)).all()
            assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
            assert np.array_equal(z['presence'],(y>0).astype('float32'))
            requested=np.asarray([i['requested_share'] for i in items]);effective=np.asarray([i['effective_share'] for i in items])
            assert not ((requested>0)&(y==0)).any() and np.allclose(effective,y,atol=1e-6)
            active_ref=(q>0).any(2)&valid.all(2);coverage=active_ref.sum(1)/valid.all(2).sum(1)
            assert np.allclose(coverage,[i['support_coverage'] for i in items],atol=1e-6)
            checked=0
            for j,item in enumerate(items):
                for reference in item['references']:
                    eq=np.asarray(reference['reference_eq_db'],dtype=float);parameters=reference['reference_eq_parameters']
                    assert eq.shape==(25,) and np.isfinite(eq).all() and np.abs(eq).max()<=4+1e-9
                    if parameters:
                        axis=np.linspace(-1,1,25)
                        expected=(parameters['tilt_coefficient_db']*axis+parameters['quadratic_coefficient_db']*(axis**2-np.mean(axis**2)))*parameters['bound_scale']
                        assert np.allclose(eq,expected,atol=1e-10)
                    assert reference['reference_eq_applied_before_alpha'] is True
                if len(item['references'])==1 and item['references'][0]['kind']!='native_contiguous_15min_crop' and active_ref[j].sum()<=2:
                    assert y[j]<=min(.98,.25+.73*coverage[j])+2e-6;checked+=1
                if y[j]>0:
                    assert abs(10*np.log10(float(y[j])/(1-float(y[j])))-item['global_snr_db'])<2e-3
            snr_check=audit_active_snr(z['db'],q,[i['active_snr_db'] for i in items])
            minute_snr_check=audit_minute_snr(z['db'],q,[i['minute_snr_db'] for i in items],g['physics_config']['max_minute_snr_db'])
        frozen={}
        for split in ['validation','outer_validation']:
            for suffix in ['.npz','_recipes.json']:
                name=split+suffix;assert sha256(data/name)==sha256(prior/name);frozen[name]=sha256(data/name)
        for name in ['validation_metadata.parquet','inner_validation_metadata.parquet']:
            oldmd=pd.read_parquet(prior/name);md=pd.read_parquet(data/name)
            assert md[oldmd.columns].equals(oldmd)
            assert sha256(data/name.replace('.parquet','_original.parquet'))==sha256(prior/name)
            flags=b.set_index('base_window_index').loc[md.base_background_index]
            assert np.array_equal(md.boundary_artifact_candidate,flags.boundary_candidate_flag)
            assert np.array_equal(md.urn_8321_nominal_warning,flags.urn_nominal_band_warning)
        total+=len(order)
        result['folds'].append({'fold':f['fold'],'clean_primary_windows':len(order),'excluded_boundary_windows':int(changed.sum()),
            'primary_coverage_fraction':1.,'boundary_primary_or_secondary_used':0,'minimum_support_gap_hours':gap/3600,
            'max_energy_target_error':error,'single_short_support_cap_checks':checked,
            'active_snr_float32_interval_check':snr_check,
            'minute_snr_check':minute_snr_check,
            'frozen_heldout_npz_and_recipes_byte_identical':True,'frozen_heldout_sha256':frozen})
    assert total==cv['training_examples_total']
    result['training_examples_total']=total;result['all_examples_total']=total+30000
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=ROOT/'data/synthetic/group_cv_physics_v2_r1');a=p.parse_args()
    report=verify(a.data_dir);(ROOT/'reports/group_cv_physics_data_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
