"""Build opt-in, versioned quality metadata without modifying the frozen v1 bank.

This module never opens challenge Test inputs or the sealed NOAA assessment.
The new split requires fresh task-model training; v1-fitted task weights and
preprocessors must not be treated as unfitted on this new validation split.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ACTIVE=['train','validation','calibration','domain_holdout']

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*2**20):h.update(chunk)
    return h.hexdigest()

def weekly_split(group,block,seed):
    bucket=int(hashlib.sha256(f'quality_v2|{seed}|{group}|{block}'.encode()).hexdigest()[:8],16)%10
    return 'validation' if bucket==0 else 'calibration' if bucket==1 else 'train'

def nearest_other_gap_seconds(times,other):
    """Nearest start-time distance; empty other partitions have infinite gap."""
    if len(other)==0:return np.full(len(times),np.inf)
    other=np.sort(other);pos=np.searchsorted(other,times)
    lo=other[np.clip(pos-1,0,len(other)-1)]
    hi=other[np.clip(pos,0,len(other)-1)]
    return np.minimum(np.abs(times-lo),np.abs(times-hi)).astype('float64')

def purged_split(frame,seed=42,purge_hours=24):
    w=frame.copy();t=pd.to_datetime(w.start_export_clock)
    w['base_window_index']=np.arange(len(w),dtype=np.int64)
    w['v1_split']=w.split
    w['recording_day']=t.dt.strftime('%Y-%m-%d')
    w['recording_week_start']=(t.dt.normalize()-pd.to_timedelta(t.dt.dayofweek,unit='D')).dt.strftime('%Y-%m-%d')
    w['split']=[('domain_holdout' if old=='domain_holdout' else weekly_split(g,b,seed))
                for old,g,b in zip(w.v1_split,w.sensor_group,w.recording_week_start)]
    w['proposed_split']=w.split
    # Work in explicit nanoseconds: parquet input clocks may use microseconds.
    seconds=t.to_numpy(dtype='datetime64[ns]').astype('int64')//10**9
    gap=np.full(len(w),np.inf)
    for _,ix in w.groupby('sensor_group',sort=False).groups.items():
        ids=np.asarray(ix,dtype=int);labels=w.proposed_split.iloc[ids].to_numpy()
        for split in np.unique(labels):
            chosen=ids[labels==split];other=ids[labels!=split]
            gap[chosen]=nearest_other_gap_seconds(seconds[chosen],seconds[other])
    # A clock window occupies 15 min, so account for support as well as starts.
    w['nearest_foreign_fold_support_gap_hours']=np.maximum(gap-15*60,0)/3600
    purge=(w.nearest_foreign_fold_support_gap_hours<purge_hours)&w.split.ne('domain_holdout')
    w.loc[purge,'split']='purged_temporal_boundary'
    w['quality_tier']=np.where(w.observed_minutes>=12,'high_coverage','sparse_observed')
    return w

def audit_source_duplicates(meta,segments):
    values=np.asarray(segments)
    _,inverse,counts=np.unique(values,axis=0,return_inverse=True,return_counts=True)
    duplicates=int((counts>1).sum());cross=[]
    if duplicates:
        owner=np.full(len(values),-1,dtype=np.int32)
        for i,r in enumerate(meta.itertuples()):owner[int(r.segment_start):int(r.segment_start+r.segment_count)]=i
        for code in np.flatnonzero(counts>1):
            rows=np.flatnonzero(inverse==code);clips=np.unique(owner[rows])
            folds=set(meta.iloc[clips].split)&set(ACTIVE)
            if len(folds)>1:cross.append({'clip_indices':clips.tolist(),'folds':sorted(folds)})
    return {'duplicate_band_vectors':duplicates,'cross_fold_band_vector_conflicts':cross,
            'limitation':'Power-vector equality is only an additional diagnostic, not proof against all re-encodings or overlapping recordings.'}

def main(a):
    base=ROOT/'data/processed';out=a.output
    if out.exists():raise FileExistsError(f'Use a fresh version path, never overwrite: {out}')
    out.mkdir(parents=True)
    bmeta=pd.read_parquet(base/'sa_window_manifest.parquet')
    emeta=pd.read_parquet(base/'external_clip_manifest.parquet')
    rows=pd.read_parquet(base/'sa_row_manifest.parquet')
    weak=pd.read_parquet(base/'sa_all_rows_weak_labels.parquet')
    assert len(rows)==len(weak)==1715779 and rows.row_id.equals(weak.row_id)
    assert not weak.is_verified_ground_truth.any()
    b=purged_split(bmeta,a.seed,a.purge_hours)
    active=b[b.split.isin(ACTIVE)]
    assert active.groupby(['sensor_group','recording_week_start']).split.nunique().max()==1
    assert active.loc[active.split.ne('domain_holdout'),'nearest_foreign_fold_support_gap_hours'].min()>=a.purge_hours
    b.to_parquet(out/'background_manifest.parquet',index=False,compression='zstd')
    e=emeta.copy();e['base_clip_index']=np.arange(len(e),dtype=np.int32)
    classes=e.weak_class.astype(str).str.lower()
    e['is_ambiguous_vessel_scenario']=e.archive_role.eq('vessel_passage')&classes.isin(['background','unknown','nan'])
    e['native_long_reference']=e.archive_role.eq('vessel_passage')&e.complete_minutes.ge(15)&~e.is_ambiguous_vessel_scenario
    e['measured_short_reference']=e.archive_role.eq('vessel_passage')&e.complete_minutes.lt(15)&~e.is_ambiguous_vessel_scenario
    e['primary_reference_eligible']=e.native_long_reference&e.split.isin(ACTIVE)
    e['secondary_reference_eligible']=e.measured_short_reference&e.split.isin(ACTIVE)
    e.to_parquet(out/'external_manifest.parquet',index=False,compression='zstd')
    effective=e[e.split.isin(ACTIVE)].groupby(['source','split']).agg(
        clips=('clip_id','size'),source_groups=('group_key','nunique'),days=('recording_day','nunique'),
        native_long_clips=('native_long_reference','sum'),measured_short_clips=('measured_short_reference','sum'))
    effective.reset_index().to_csv(out/'effective_reference_counts.csv',index=False)
    crossing=bmeta.assign(day=pd.to_datetime(bmeta.start_export_clock).dt.strftime('%Y-%m-%d')).groupby(['sensor_group','day']).split.nunique()
    native=e[e.primary_reference_eligible].groupby('split').size().to_dict()
    missing_native=[s for s in ['train','validation','calibration'] if native.get(s,0)==0]
    # Link weak scores as historical audit only; no confidence threshold turns them into truth.
    weak['pseudo_label_training_allowed']=False
    weak.to_parquet(out/'row_score_audit.parquet',index=False,compression='zstd')
    segments=np.load(base/'external_segments_power.npy',mmap_mode='r')
    audit={'version':'quality_v2','source_data_modified':False,'challenge_test_accessed':False,
        'sealed_NOAA_accessed':False,'source_sha256':{name:sha256(base/name) for name in [
            'sa_windows_db.npy','sa_window_manifest.parquet','external_segments_power.npy',
            'external_clip_manifest.parquet','sa_row_manifest.parquet','sa_all_rows_weak_labels.parquet']},
        'base_processed_relative_to_project':'data/processed','background_manifest':'background_manifest.parquet',
        'external_manifest':'external_manifest.parquet','all_rows_preserved':len(rows),
        'weak_scores_are_truth':False,'weak_scores_allowed_for_training':False,
        'v1_sensor_group_days_with_cross_fold_windows':int((crossing>1).sum()),
        'v1_background_coverage_counts':bmeta.observed_minutes.value_counts().sort_index().to_dict(),
        'background_split_policy':f'sensor_group × export-clock Monday weeks, metadata-hash seed {a.seed}; symmetric support-aware {a.purge_hours}h boundary purge',
        'background_split_counts':b.split.value_counts().to_dict(),
        'background_quality_counts':b.groupby(['split','quality_tier']).size().reset_index(name='windows').to_dict('records'),
        'do_not_reuse_v1_fitted_models_for_v2_validation':True,
        'domain_holdout_previously_inspected':True,
        'source_duplicate_audit':audit_source_duplicates(emeta,segments),
        'native_long_reference_counts':native,'missing_native_reference_partitions':missing_native,
        'native_first_full_training_ready':not missing_native,
        'readiness_gate':'Require verified additional native long references in every development partition before large-scale native-first materialization.',
        'augmentation_policy':'native crop/placement without repeated audio; optional common gain, time reverse, measured ambient addition and background blending; recompute observed-cell target from known components',
        'limitations':['SA export clock and level calibration remain unverified.',
            'HearMyShip hull recurrence is unknown, even with day separation.',
            'AIS and ship-event annotations do not prove isolated vessel sound.',
            'Synthetic sample count is not the number of independent source groups.']}
    (out/'manifest.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    report=ROOT/'reports/data_quality_v2_audit.json';report.write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k not in ['source_sha256','background_quality_counts']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'data/versions/quality_v2')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--purge-hours',type=float,default=24)
    main(p.parse_args())
