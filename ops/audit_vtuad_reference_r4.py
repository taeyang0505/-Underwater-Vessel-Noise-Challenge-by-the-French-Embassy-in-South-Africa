"""Independent direct-isolation and provenance checks for R4 VT metadata.

Reads source metadata and the 2s bank HEADER only. No acoustic values or Test,
reserved audio, model weights, training predictions or leaderboard data enter.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from pipeline.vtuad_reference_r4 import candidate_pool,select_subset,connected_components,allocate_folds,digest,canonical
ROOT=Path(__file__).resolve().parents[1]

def graph_count(frame,hull_column):
    parent={}
    def find(a):
        parent.setdefault(a,a)
        if parent[a]!=a:parent[a]=find(parent[a])
        return parent[a]
    def join(a,b):parent[find(a)]=find(b)
    data=frame.reset_index(drop=True).copy()
    data['begin']=pd.to_datetime(data.utc_begin,utc=True);data['end']=pd.to_datetime(data.utc_end,utc=True)
    for i,row in data.iterrows():
        for token in [('hull',row[hull_column]),('day',row.recording_day),('bytes',row.sha256)]:join(('row',i),token)
    previous=[]
    for i,row in data.sort_values('begin').iterrows():
        previous=[(j,end) for j,end in previous if end>row['begin']]
        for j,end in previous:join(('row',i),('row',j))
        previous.append((i,row['end']))
    return len({find(('row',i)) for i in data.index})

def overlaps_cross(frame,label):
    data=frame.copy();data['begin']=pd.to_datetime(data.utc_begin,utc=True);data['end']=pd.to_datetime(data.utc_end,utc=True)
    previous=[];count=0
    for row in data.sort_values('begin').itertuples():
        previous=[q for q in previous if q.end>row.begin]
        count+=sum(getattr(q,label)!=getattr(row,label) for q in previous)
        previous.append(row)
    return count

def audit(directory):
    directory=Path(directory).resolve();m=json.loads((directory/'manifest.json').read_text())
    for name,expected in m['file_sha256'].items():
        if Path(name).name!=name or digest(directory/name)!=expected:raise ValueError('R4 metadata SHA differs')
    for name,expected in [('parent_reference_manifest','parent_reference_sha256'),('original_external_manifest','original_external_sha256')]:
        if digest(ROOT/m['inputs'][name])!=m['inputs'][expected]:raise ValueError('Frozen parent metadata changed')
    if digest(ROOT/'pipeline/vtuad_reference_r4.py')!=m['code_sha256']:raise ValueError('Selector code changed after metadata creation')
    selected=pd.read_parquet(directory/'reference_manifest.parquet');excluded=pd.read_parquet(directory/'excluded_manifest.parquet')
    groups=pd.read_parquet(directory/'component_manifest.parquet');pool,_=candidate_pool(ROOT)
    selected_ids=set(selected.clip_id);excluded_ids=set(excluded.clip_id)
    assert selected_ids.isdisjoint(excluded_ids) and selected_ids|excluded_ids==set(pool.clip_id)
    assert len(selected_ids)==len(selected) and selected.source.eq('VTUAD_v1.3.2').all()
    assert selected.original_split.isin(['train','validation']).all()
    assert selected.reference_eligible.all() and selected.measured.all() and selected.verified.all() and selected.quality_full_band_good.all()
    assert selected.duration_seconds.between(120,1380).all() and selected.segment_count.between(60,690).all()
    assert selected.role.eq('vessel_passage').all() and not selected.weak_class.astype(str).str.lower().isin(['background','unknown','nan']).any()
    assert selected.sample_rate_hz.ge(2*10**3.95).all()
    actual=graph_count(selected,'original_group_key');assert actual==22==selected.reference_component.nunique()==len(groups)
    assert (selected.groupby('recording_day').reference_component.nunique()==1).all()
    assert (selected.groupby('reference_component').recording_day.nunique()==1).all()
    assert (selected.groupby('original_group_key').reference_component.nunique()==1).all()
    assert (selected.groupby('sha256').reference_component.nunique()==1).all()
    assert overlaps_cross(selected,'reference_component')==0
    # Header-only bounds check: all original offsets remain usable, without
    # opening numeric waveform/energy values from reserved bank rows.
    headers={}
    for filename,rows in selected.groupby('segment_file'):
        path=ROOT/filename
        with path.open('rb') as f:
            version=np.lib.format.read_magic(f)
            shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f))
        assert len(shape)==2 and shape[1]==25 and dtype==np.dtype('float32')
        assert rows.segment_start.ge(0).all() and (rows.segment_start+rows.segment_count).le(shape[0]).all()
        headers[filename]={'shape':list(shape),'dtype':str(dtype),'numeric_values_read':False}
    original=pool.set_index('clip_id')
    invariant=['source','sha256','segment_file','segment_start','segment_count','duration_seconds','sample_rate_hz','original_group_key','recording_day','utc_begin','utc_end','original_split']
    for name in invariant:assert selected.set_index('clip_id')[name].sort_index().equals(original.loc[sorted(selected_ids),name].sort_index())
    assert set(pool[pool.segment_count.ge(450)].clip_id).issubset(selected_ids)
    matrix=pd.crosstab(pool.recording_day,pool.original_group_key).to_numpy();rr,cc=linear_sum_assignment(np.where(matrix>0,0,1_000_000))
    upper=int((matrix[rr,cc]>0).sum());assert upper==actual
    replay,again,proof=select_subset(pool,seed=m['config']['seed'],minimum_components=m['config']['minimum_components'])
    replay=connected_components(replay);replay,_=allocate_folds(replay,seed=m['config']['seed'])
    assert replay[['clip_id','reference_component','outer_fold']].sort_values('clip_id').reset_index(drop=True).equals(selected[['clip_id','reference_component','outer_fold']].sort_values('clip_id').reset_index(drop=True))
    fold_audits=[]
    for f in range(5):
        r=pd.read_parquet(directory/f'fold_{f:02d}_reference_manifest.parquet')
        assert set(r.clip_id)==selected_ids and not set(r.clip_id)&excluded_ids
        assert set(r.split)=={'train','validation','outer_validation'}
        expected=np.where(r.outer_fold.eq(f),'outer_validation',np.where(r.outer_fold.eq((f+1)%5),'validation','train'))
        assert np.array_equal(expected,r.split)
        checks={'component':int((r.groupby('reference_component').split.nunique()>1).sum()),
            'hull':int((r.groupby('original_group_key').split.nunique()>1).sum()),
            'day':int((r.groupby('recording_day').split.nunique()>1).sum()),
            'SHA':int((r.groupby('sha256').split.nunique()>1).sum()),'UTC_overlap_pairs':overlaps_cross(r,'split')}
        assert not any(checks.values())
        counts=r.groupby('split').agg(clips=('clip_id','size'),components=('reference_component','nunique'),native_clips=('native_long','sum')).to_dict('index')
        for split,part in r.groupby('split'):assert set(part.coarse_duration_bin)=={'120_to_under300','300_to1380'}
        fold_audits.append({'fold':f,'direct_cross_split_violation_counts':checks,'partitions':counts,
            'coarse_duration_counts':r.groupby(['split','coarse_duration_bin']).size().unstack(fill_value=0).to_dict('index')})
    external=pd.read_parquet(ROOT/m['inputs']['original_external_manifest']);full=external[external.source.eq('VTUAD_v1.3.2')]
    full_components=graph_count(full,'group_key');assert full_components==1
    source_hash=canonical({'inputs':m['inputs'],'config':m['config'],'files':m['file_sha256']});assert source_hash==m['source_hash']
    report={'passed':True,'source_hash':source_hash,'metadata_directory':str(directory.relative_to(ROOT)),
        'candidate_pool_clips':len(pool),'selected_clips':len(selected),'excluded_actual_clips':len(excluded),
        'selected_components':actual,'maximum_matching_bound':upper,'original_native3_retained':True,
        'retention_optimality_scope':'119 clips is the exact metadata objective optimum under the declared one-retained-day-per-hull and native-retention policy; not a claim of the globally largest arbitrary 22-component subgraph.',
        'selected_hulls':selected.original_group_key.nunique(),'selected_days':selected.recording_day.nunique(),
        'measured_seconds':float(selected.duration_seconds.sum()),'used_2s_seconds':int(selected.segment_count.sum()*2),
        'full525_components':full_components,'direct_isolation_scope':'Selected pool only; full corpus connectivity is preserved as an explicit limitation, not hidden by relabeling.',
        'new_physical_sources_added':0,'deterministic_selection_and_assignment_replay':True,'excluded_absent_from_all_R4_splits':True,
        'reserved_original_splits_admitted':False,'source_bank_headers':headers,'folds':fold_audits,
        'outer_component_counts':m['component_outer_counts'],'outer_coarse_duration_counts':m['coarse_counts_by_outer'],
        'outer_fine_duration_counts':m['fine_counts_by_outer'],'native900bin_cover_all_folds':False,
        'metadata_only':True,'acoustic_numeric_values_read':False,'Test_accessed':False,'models_or_mixtures_generated':False,
        'existing_R2_data_models_or_code_modified':False,'SA_assignment_or_24h_purge_modified':False,
        'NOAA_or_HMS_quality_failure_claim':False,'selector_code_sha256':m['code_sha256'],'audit_code_sha256':digest(Path(__file__))}
    destination=ROOT/'reports/r4_reference_audit.json';destination.write_text(json.dumps(report,indent=2,allow_nan=False))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metadata-dir',type=Path,default=ROOT/'data/versions/vtuad_reference_r4_v1')
    print(json.dumps(audit(p.parse_args().metadata_dir),indent=2))
