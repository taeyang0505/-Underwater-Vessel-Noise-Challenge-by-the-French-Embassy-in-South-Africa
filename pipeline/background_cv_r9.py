"""R9 background-only fivefold development CV, with four heldout stage roles.

Reuse exactly the original R4 allowed100965 SA source windows and existing
sensor-week buckets. No SA calibration/domain source is restored. Apply the
same actual-minute QC to every role; reserve900s support plus24h separation
within a sensor across all active roles. Reference509 pool is shared and is
NOT reference-independent CV. No model-performance or Test input is used.
"""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib,json
from pathlib import Path

import numpy as np
import pandas as pd
from pipeline.source_split_r6 import apply_minute_qc,minute_masks


ROOT=Path(__file__).resolve().parents[1]
VERSION='background_cv_r9_v1'
DEFAULT_OUTPUT='data/versions/background_cv_r9'
PARENT='data/synthetic/group_cv_passage_r4_v1/fold_00/sources'
BANK='data/processed/sa_windows_db.npy'
ROLES=('train','selection','ensemble_fit','ensemble_gate','report')
VALIDATION_ROLES=ROLES[1:]
FRACTIONS=(.5,.2,.15,.15)
PURGE_SECONDS=24*3600
WINDOW_SECONDS=900


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def write(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def heldout_group_roles(groups,fold,seed=20261026):
    groups=sorted(set(map(str,groups)),key=lambda v:hashlib.sha256(f'R9BG|roles|{seed}|{fold}|{v}'.encode()).hexdigest())
    exact=np.asarray(FRACTIONS)*len(groups);quota=np.floor(exact).astype(int)
    remaining=len(groups)-int(quota.sum())
    rank=sorted(range(4),key=lambda i:(-(exact[i]-quota[i]),i))
    for i in rank[:remaining]:quota[i]+=1
    if (quota<2).any():raise ValueError('Each heldout stage requires at least two whole source groups')
    result={};start=0
    for role,count in zip(VALIDATION_ROLES,quota):
        result.update({v:role for v in groups[start:start+count]});start+=count
    return result


def nearest_seconds(starts,other):
    if not len(other):return np.full(len(starts),np.inf)
    other=np.sort(other);positions=np.searchsorted(other,starts)
    lower=other[np.clip(positions-1,0,len(other)-1)]
    upper=other[np.clip(positions,0,len(other)-1)]
    return np.minimum(abs(starts-lower),abs(starts-upper)).astype(float)


def proposed_purge_gap(frame):
    """Preserve report>gate>fit>selection priority; TRAIN vs entire heldout."""
    seconds=pd.to_datetime(frame.start_export_clock).to_numpy(dtype='datetime64[ns]').astype('int64')//10**9
    gaps=np.full(len(frame),np.inf)
    higher={'train':VALIDATION_ROLES,'selection':('ensemble_fit','ensemble_gate','report'),
            'ensemble_fit':('ensemble_gate','report'),'ensemble_gate':('report',),'report':()}
    labels=frame.proposed_role.to_numpy()
    for ids in frame.groupby('sensor_group').indices.values():
        ids=np.asarray(ids)
        for role,other_roles in higher.items():
            chosen=ids[labels[ids]==role];other=ids[np.isin(labels[ids],other_roles)]
            if len(chosen) and len(other):gaps[chosen]=np.maximum(nearest_seconds(seconds[chosen],seconds[other])-WINDOW_SECONDS,0)
    return gaps


def final_cross_role_gaps(frame):
    seconds=pd.to_datetime(frame.start_export_clock).to_numpy(dtype='datetime64[ns]').astype('int64')//10**9
    gaps=np.full(len(frame),np.inf);labels=frame.split.to_numpy()
    for ids in frame.groupby('sensor_group').indices.values():
        ids=np.asarray(ids)
        for role in ROLES:
            chosen=ids[labels[ids]==role];other=ids[labels[ids]!=role]
            if len(chosen) and len(other):gaps[chosen]=np.maximum(nearest_seconds(seconds[chosen],seconds[other])-WINDOW_SECONDS,0)
    return gaps


def _role_stats(frame):
    return {role:{'windows':int((frame.split==role).sum()),
                  'source_groups':int(frame[frame.split==role].background_group.nunique()),
                  'site_counts':{str(k):int(v) for k,v in frame[frame.split==role].site.value_counts().items()}}
            for role in ROLES}


def build(output=DEFAULT_OUTPUT,seed=20261026):
    out=ROOT/output
    if out.exists():raise FileExistsError('Fresh backgroundCV directory required')
    source=ROOT/PARENT;parent=json.loads((source/'manifest.json').read_text())
    names=[PARENT+'/manifest.json',PARENT+'/background_manifest.parquet',
           'data/versions/quality_v2_native/manifest.json',
           'data/versions/quality_v2_native/background_manifest.parquet',
           'data/processed/sa_window_manifest.parquet','data/processed/sa_row_manifest.parquet',
           'reports/data_quality_boundary_flags.parquet','pipeline/source_split_r6.py',
           'pipeline/background_cv_r9.py','ops/build_background_cv_r9.py',
           'data/versions/vtuad_full_r9/manifest.json','data/versions/vtuad_full_r9/readiness.json']
    inputs={n:sha(ROOT/n) for n in names};bank_sha=sha(ROOT/BANK)
    quality_version=json.loads((ROOT/'data/versions/quality_v2_native/manifest.json').read_text())
    if bank_sha!=quality_version['frozen_v1_hashes']['sa_windows_db.npy']:
        raise ValueError('Original SA window bank changed')
    if sha(source/'background_manifest.parquet')!=parent['metadata_sha256']['background_manifest.parquet']:
        raise ValueError('Original R4 source pool metadata changed')
    frame=pd.read_parquet(source/'background_manifest.parquet').copy().sort_values('base_window_index').reset_index(drop=True)
    if len(frame)!=100965 or frame.base_window_index.duplicated().any() or frame.window_id.duplicated().any():
        raise ValueError('Exactly100965 unique original allowed windows required')
    original=pd.read_parquet(ROOT/'data/versions/quality_v2_native/background_manifest.parquet')
    allowed=set(original.loc[original.split.isin(['train','validation']),'base_window_index'])
    if set(frame.base_window_index)!=allowed:
        raise ValueError('SAcalibration/domain/originalpurged sources cannot be restored')
    if (frame.groupby('background_group').outer_fold.nunique().max()!=1
            or frame.groupby(['sensor_group','recording_week_start']).outer_fold.nunique().max()!=1
            or set(frame.outer_fold)!=set(range(5))):
        raise ValueError('Original sensor-week bucket identity failed')
    frame['r4_split']=frame.split
    frame['split']='allowed_background_pool'
    wm=pd.read_parquet(ROOT/'data/processed/sa_window_manifest.parquet')
    ids=frame.base_window_index.to_numpy(dtype='int64');native=wm.iloc[ids].reset_index(drop=True)
    for col in ['window_id','site','recorder','sensor_group','start_export_clock']:
        if not frame[col].astype(str).equals(native[col].astype(str)):
            raise ValueError('Original window identity differs: '+col)
    rows=pd.read_parquet(ROOT/'data/processed/sa_row_manifest.parquet',columns=
                         ['row_id','site','recorder','t','window_included','eligible_for_minute_window'])
    flags=pd.read_parquet(ROOT/'reports/data_quality_boundary_flags.parquet')
    bits,masked,counts=minute_masks(frame,rows,flags)
    frame['boundary_mask_bits']=bits
    frame['masked_minute_indices']=[[m for m in range(15) if int(b)&(1<<m)] for b in bits]
    frame['masked_candidate_minutes']=[int(b).bit_count() for b in bits]
    raw=np.load(ROOT/BANK,mmap_mode='r',allow_pickle=False)
    observed=[];raw_observed=[]
    for start in range(0,len(frame),512):
        ix=ids[start:start+512];a=np.asarray(raw[ix]);mask=np.isfinite(a)
        if np.isinf(a).any() or (mask.any(2)!=mask.all(2)).any():
            raise ValueError('Invalid partial-frequency original minute')
        if any(counts.get(int(i),0)!=int(v) for i,v in zip(ix,mask.any(2).sum(1))):
            raise ValueError('Original selected-minute provenance mismatch')
        b=apply_minute_qc(a,bits[start:start+len(ix)])
        raw_observed.extend(mask.any(2).sum(1).tolist());observed.extend(np.isfinite(b).all(2).sum(1).tolist())
    frame['raw_observed_minutes']=raw_observed;frame['qc_observed_minutes']=observed
    if not np.array_equal(frame.raw_observed_minutes.to_numpy(),frame.observed_minutes.to_numpy()):
        raise ValueError('Original coverage mismatch')
    frame['start']=pd.to_datetime(frame.start_export_clock)
    frame['end_export_clock']=frame['start']+pd.Timedelta(seconds=WINDOW_SECONDS)
    frame['end']=frame.end_export_clock
    frame['export_clock_not_verified_UTC']=True
    frame['group']=frame.background_group
    frame['sensor']=frame.sensor_group
    frame['qc_eligible']=frame.qc_observed_minutes.ge(3)
    out.mkdir(parents=True)
    frame.to_parquet(out/'background_pool_manifest.parquet',index=False,compression='zstd')
    masked.to_parquet(out/'masked_minute_rows.parquet',index=False,compression='zstd')
    assignments=[];folds=[];all_ready=True
    for fold in range(5):
        f=frame.copy();held=f.outer_fold.eq(fold)
        mapping=heldout_group_roles(f.loc[held,'background_group'],fold,seed)
        f['proposed_role']='train';f.loc[held,'proposed_role']=f.loc[held,'background_group'].map(mapping)
        f['split']=f.proposed_role
        f.loc[~f.qc_eligible,'split']='excluded_insufficient_observed_support'
        eligible=f[f.qc_eligible].reset_index(drop=True)
        gap=proposed_purge_gap(eligible)
        eligible['proposed_cross_role_support_gap_hours']=gap/3600
        eligible.loc[gap<PURGE_SECONDS,'split']='purged_temporal_boundary_r9'
        active=eligible[eligible.split.isin(ROLES)].copy().reset_index(drop=True)
        active['role']=active.split
        active['background_fold']=fold
        final_gap=final_cross_role_gaps(active)
        active['minimum_other_role_support_gap_hours']=final_gap/3600
        if (final_gap<PURGE_SECONDS).any() or active.groupby('background_group').split.nunique().max()!=1:
            raise RuntimeError('24h support/group isolation failed')
        stats=_role_stats(active)
        ready=all(v['windows']>0 and v['source_groups']>=2 for v in stats.values());all_ready &= ready
        directory=out/f'fold_{fold:02d}';directory.mkdir()
        role_paths={}
        for role in ROLES:
            part=active[active.split.eq(role)].sort_values('base_window_index').reset_index(drop=True)
            filename=role+'_manifest.parquet';part.to_parquet(directory/filename,index=False,compression='zstd');role_paths[role]=filename
        active[active.split.ne('train')].to_parquet(directory/'validation_manifest.parquet',index=False,compression='zstd')
        excluded=pd.concat([f[~f.qc_eligible],eligible[~eligible.split.isin(ROLES)]],ignore_index=True)
        excluded.to_parquet(directory/'excluded_manifest.parquet',index=False,compression='zstd')
        assigned=f[['base_window_index','window_id','background_group','outer_fold','proposed_role']].copy();assigned['background_fold']=fold;assignments.append(assigned)
        finite=final_gap[np.isfinite(final_gap)]
        held_ids=set(f.loc[held,'base_window_index'])
        if set(active.loc[active.split.eq('train'),'base_window_index'])&held_ids:
            raise RuntimeError('Heldout primary entered own-fold TRAIN')
        fold_summary={'fold':fold,'ready_to_materialize':ready,'roles':role_paths,'role_stats':stats,
                      'before_QC_and_purge':{'train_windows':int((~held).sum()),'heldout_windows':int(held.sum()),
                                             'heldout_group_quotas':{str(k):int(v) for k,v in pd.Series(mapping).value_counts().items()}},
                      'QC_excluded':int((~f.qc_eligible).sum()),
                      'purged_by_proposed_role':{str(k):int(v) for k,v in eligible[~eligible.split.isin(ROLES)].proposed_role.value_counts().items()},
                      'post_active_windows':len(active),'post_excluded_windows':len(excluded),
                      'minimum_cross_role_support_gap_hours':None if not len(finite) else float(finite.min()/3600),
                      'cross_role_support_gap_under24h_violations':int((final_gap<PURGE_SECONDS).sum()),
                      'background_group_overlap_violations':0,
                      'report_group_stage':'Report source groups not used bythisfold TRAIN/selection/ensemblefit/gate;otherCVfoldTRAINmayuse them',
                      'file_sha256':{p.name:sha(p) for p in directory.iterdir() if p.is_file()}}
        write(directory/'manifest.json',fold_summary);folds.append(fold_summary)
    pd.concat(assignments,ignore_index=True).to_parquet(out/'proposed_roles_allfolds.parquet',index=False,compression='zstd')
    for name,digest in inputs.items():
        if sha(ROOT/name)!=digest:raise ValueError('Input changed during backgroundCV build: '+name)
    if sha(ROOT/BANK)!=bank_sha:raise ValueError('SA bank changed during build')
    manifest={'version':VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),'ready_to_materialize':bool(all_ready),
              'model_training_ready':False,'seed':seed,'original_allowed_windows':100965,
              'background_bank':{'path':BANK,'shape':list(raw.shape),'dtype':str(raw.dtype),'sha256':bank_sha},
              'original_outer_bucket_counts':{str(k):int(v) for k,v in frame.outer_fold.value_counts().sort_index().items()},
              'source_groups':int(frame.background_group.nunique()),'sites':int(frame.site.nunique()),
              'source_week_and_existing_outer_assignment_reused':True,'TRAIN_fraction_before_purge':'four of fiveoriginalbuckets, approximately80%bywindowcount',
              'heldout_roles':list(VALIDATION_ROLES),'heldout_source_group_fractions':dict(zip(VALIDATION_ROLES,FRACTIONS)),
              'heldout_group_assignment':'SHA256 stableorder thenlargest-remainder quotas;no value/performancebasedseedretry',
              'purge':'Same sensor,900s nominalwindow support plus>=24h acrossallroles. Priorityreport>gate>fit>selection;TRAINpurgedagainstallheldoutcandidates',
              'QC':{'method':'same SourceSplitR6.apply_minute_qc/actual originalrow_id join for allroles','masked_selected_minutes':len(masked),
                    'windows_with_masked_minutes':int((bits>0).sum()),'eligible_windows_min3_observed':int(frame.qc_eligible.sum()),
                    'excluded_insufficient_support':int((~frame.qc_eligible).sum()),'numeric_source_indices_read':100965,
                    'outside_allowed_source_values_read':0,'physical_cause_of_boundary_flag_verified':False},
              'roles':list(ROLES),'folds':folds,'references':'shared509VTUADpool;notreference-independentCV',
              'report_policy':'No report role modelpredictionsbeforeallownfoldcheckpointandensemblepolicy frozen;not original unseenpubliccorpus',
              'development_CV':True,'source_only_metadata_not_synthetic_NPZ':True,
              'SA_calibration_domain_or_extra_originalpurged_sources_restored':False,
              'old_sources_models_arrays_modified':False,'model_performance_or_Test_used':False,
              'input_sha256':inputs,
              'file_sha256':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}}
    manifest['source_hash']=hashlib.sha256(canonical(manifest)).hexdigest();write(out/'manifest.json',manifest)
    readiness={'version':VERSION,'ready':bool(all_ready),'manifest_sha256':sha(out/'manifest.json'),'source_hash':manifest['source_hash'],
               'all_fivefolds_all_roles_nonempty_min2groups':bool(all_ready),'all_crossrole24h_purge_checks_passed':True,
               'all_source_group_crossrole_overlaps_zero':True,'new_synthetic_dataset_or_training_ready':False,
               'scope':'Backgroundsource metadata generationready;reference shared/no naturalshipGT/no newblind referenceOOF',
               'report_model_evaluation_authorized_before_freeze':False}
    write(out/'readiness.json',readiness)
    return manifest,readiness


def validate_background_cv(output=DEFAULT_OUTPUT,verify_bank=True):
    folder=ROOT/output;m=json.loads((folder/'manifest.json').read_text());g=json.loads((folder/'readiness.json').read_text())
    if m['version']!=VERSION or g['version']!=VERSION or g['ready'] is not True:
        raise ValueError('BackgroundCV metadata notready')
    if sha(folder/'manifest.json')!=g['manifest_sha256']:raise ValueError('Manifest changed')
    for name,digest in m['input_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen input changed: '+name)
    for name,digest in m['file_sha256'].items():
        if '..' in Path(name).parts or sha(folder/name)!=digest:raise ValueError('Artifact changed: '+name)
    if verify_bank and sha(ROOT/m['background_bank']['path'])!=m['background_bank']['sha256']:
        raise ValueError('Background bank changed')
    return {'status':'passed','source_hash':m['source_hash'],'manifest_sha256':g['manifest_sha256'],
            'folds':5,'roles':list(ROLES),'reference_independent':False,'model_fit_or_Test_accessed':False}
