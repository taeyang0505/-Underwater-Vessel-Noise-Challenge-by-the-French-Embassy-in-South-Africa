"""R9 full strict VTUAD recording pool, explicitly shared across background CV.

The user authorized restoring old VT reference reservations for this NEW
production version. Old data/models remain immutable. This is NOT reference-
independent CV: all509 vessel-archive recordings may enter every TRAIN fold.
Weak background classes inside that archive remain flagged; no measured
recording is asserted to contain isolated vessel-only energy. Sixteen
dedicated ambient recordings are separate controls, never auto-zero labels.
"""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
VERSION='vtuad_full_r9_v1'
DEFAULT_OUTPUT='data/versions/vtuad_full_r9'
BANK='data/processed/external_segments_power.npy'
BASE='data/processed/external_clip_manifest.parquet'
QUALITY='data/versions/quality_v2_native/reference_manifest.parquet'
REGISTRY='external/vtuad_v1.3.2/registry_anonymized.csv'
PARENT='data/versions/vtuad_reference_r4_v1/manifest.json'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def reference_indices_for_epoch(reference_count,n_examples,epoch,fold,seed=20261026):
    """Deterministic balanced cycling of recording INDICES, not waveform tiling.

    N examples per epoch must stay fixed within a fold. Epoch1 begins a fixed
    fold-specific permutation; following epochs continue that index stream.
    If N>=reference_count, every reference appears in every epoch. For a
    smaller N, ceil(reference_count/N) epochs cover the first full cycle.
    Recording bytes/timeline are never repeated inside one acoustic event.
    """
    values=[reference_count,n_examples,epoch,fold,seed]
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in values):
        raise ValueError('Integer reference count/N/epoch/fold/seed required')
    if reference_count<1 or n_examples<1 or epoch<1 or not 0<=fold<5 or seed<0:
        raise ValueError('Positive count/N/epoch, fold0..4 and nonnegative seed required')
    order=np.random.default_rng(np.random.SeedSequence([int(seed),int(fold),0x5239])).permutation(reference_count)
    offset=((int(epoch)-1)*int(n_examples))%int(reference_count)
    positions=(np.arange(n_examples,dtype=np.int64)+offset)%reference_count
    return order[positions].astype(np.int64)


def audit_slice(power):
    a=np.asarray(power)
    if a.ndim!=2 or a.shape[1]!=25 or a.dtype!=np.float32 or not len(a):
        raise ValueError('Native float32 nonempty2s band-power[K,25] required')
    finite=np.isfinite(a);nonnegative=(a>=0)&finite
    complete_row=nonnegative.all(1)&(np.where(finite,a,0.).sum(1)>0)
    minute_count=len(a)//30
    minutes=(complete_row[:minute_count*30].reshape(minute_count,30).all(1)
             if minute_count else np.empty(0,bool))
    raw=np.ascontiguousarray(a).tobytes()
    return {'valid_2s_segments':int(complete_row.sum()),'invalid_2s_segments':int((~complete_row).sum()),
            'complete_minute_count':minute_count,'valid_complete_minute_count':int(minutes.sum()),
            'valid_segment_counts_per_band':nonnegative.sum(0).astype(int).tolist(),
            'finite_nonnegative_all25':bool(nonnegative.all()),
            'strictly_positive_broadband_all_segments':bool(complete_row.all()),
            'has_measured_complete_minute':bool(minute_count and minutes.all()),
            'trailing_complete2s_bins_without_full_minute':len(a)%30,
            'expected_segment_slice_sha256':hashlib.sha256(raw).hexdigest(),
            'segment_array_identity_sha256':hashlib.sha256(canonical({'shape':list(a.shape),'dtype':str(a.dtype)})+raw).hexdigest()}


def _counts(frame):
    return {str(k):int(v) for k,v in frame.weak_class.value_counts().items()}


def build(output=DEFAULT_OUTPUT,seed=20261026):
    output=ROOT/output
    if output.exists():raise FileExistsError('New pool output must be fresh')
    paths=[BASE,QUALITY,REGISTRY,PARENT,'pipeline/reference_pool_r9.py','ops/build_reference_pool_r9.py']
    pins={n:sha(ROOT/n) for n in paths}
    parent=json.loads((ROOT/PARENT).read_text());bank_sha=sha(ROOT/BANK)
    if bank_sha!=parent['bank_sha256_from_frozen_R2_source_manifest']:
        raise ValueError('Original converted band bank changed')
    if pins[BASE]!=parent['inputs']['original_external_sha256']:
        raise ValueError('Original source metadata changed')
    base=pd.read_parquet(ROOT/BASE)
    base=base[base.source=='VTUAD_v1.3.2'].copy().sort_values(['archive_role','name','clip_id']).reset_index(drop=True)
    if base.archive_role.value_counts().to_dict()!={'vessel_passage':509,'ambient_control':16}:
        raise ValueError('Expected all509 vessel-archive and16 dedicated ambient records')
    quality=pd.read_parquet(ROOT/QUALITY);quality=quality[quality.source=='VTUAD_v1.3.2'].set_index('clip_id')
    registry=pd.read_csv(ROOT/REGISTRY);lookup={str(v['path']).removeprefix('./'):v for v in registry.to_dict('records')}
    bank=np.load(ROOT/BANK,mmap_mode='r',allow_pickle=False)
    if bank.ndim!=2 or bank.shape[1]!=25 or bank.dtype!=np.float32:
        raise ValueError('Unexpected original band-bank layout')
    rows=[];audit_rows=[]
    for record in base.to_dict('records'):
        first=int(record['segment_start']);count=int(record['segment_count']);fs=int(record['sample_rate_hz']);frames=int(record['frames'])
        if first<0 or count<1 or first+count>len(bank):raise ValueError('Bank slice out of bounds')
        if count!=frames//(2*fs) or int(record['minute_count'])!=count//30:
            raise ValueError('Native frame/segment/minute support mismatch')
        if record.get('provider_sha256_verified') is not True:
            raise ValueError('Provider integrity record is absent')
        if abs(float(record['duration_seconds'])-frames/fs)>1e-9:
            raise ValueError('Native header duration differs from frames/fs')
        if fs/2 < 10**3.95:raise ValueError('Native header cannot support25bands')
        audit=audit_slice(bank[first:first+count]);q=quality.loc[record['clip_id']]
        key='/'.join(record['name'].split('/')[-2:]);reg=lookup[key]
        warnings=list(q['quality_review_flags'])
        role=record['archive_role'];weak=str(record['weak_class'])
        row={**record,**audit,'role':role,'original_split':record['split'],'split':'shared_reference_pool',
             'power_cache':BANK,'segment_file':BANK,'power_cache_layout':'global float32[K,25]; slice bysegment_start/count',
             'header_sample_rate_hz':fs,
             'measured_used_seconds':2*count,'native_complete_minute_count':count//30,
             'hull_id':None if pd.isna(reg['hull_id']) else str(reg['hull_id']),
             'reference_component':'VTUAD_FULL_CORPUS_COMPONENT_0',
             'verified':True,'measured':True,'quality_full_band_good':audit['finite_nonnegative_all25'],
             'quality_review_flags':warnings,'original_quality_evidence':str(q['quality_evidence']),
             'is_weak_background_label_in_vessel_archive':bool(role=='vessel_passage' and weak.lower()=='background'),
             'reference_semantics':'whole measured recording added component;not verified isolated vessel energy',
             'automatic_presence_or_energy_label_from_original_class':False,
             'new_production_old_reference_holdouts_contaminated':True,
             'prior_reservations_restored_by_explicit_user_authorization':True,
             'reference_eligible':bool(role=='vessel_passage' and audit['has_measured_complete_minute']
                                      and audit['finite_nonnegative_all25'] and audit['strictly_positive_broadband_all_segments'])}
        rows.append(row)
        audit_rows.append({'clip_id':record['clip_id'],'sha256':record['sha256'],'archive_role':role,
                           'original_split':record['split'],'sample_rate_hz':fs,'frames':frames,
                           'header_duration_seconds':float(record['duration_seconds']),
                           'segment_start':first,'segment_count':count,**audit})
    frame=pd.DataFrame(rows)
    # Deduplicate only original WAV bytes, not coarse features, classes,hulls or days.
    frame=frame.sort_values(['archive_role','sha256','name']).reset_index(drop=True)
    dup=frame.duplicated('sha256',keep='first')
    duplicate_manifest=frame[dup].copy()
    unique=frame[~dup].copy()
    refs=unique[unique.archive_role=='vessel_passage'].sort_values(['clip_id','name']).reset_index(drop=True)
    ambient=unique[unique.archive_role=='ambient_control'].sort_values(['clip_id','name']).reset_index(drop=True)
    refs.insert(0,'reference_row_index',np.arange(len(refs),dtype=np.int64))
    ambient.insert(0,'ambient_row_index',np.arange(len(ambient),dtype=np.int64))
    for fold in range(5):refs[f'fold_{fold:02d}_TRAIN_eligible']=True
    ready=bool(len(refs)==509 and len(ambient)==16 and refs.reference_eligible.all()
               and ambient.finite_nonnegative_all25.all() and ambient.has_measured_complete_minute.all())
    order=np.stack([reference_indices_for_epoch(len(refs),len(refs),1,f,seed) for f in range(5)])
    schedule=[]
    for fold in range(5):
        for position,index in enumerate(order[fold]):
            row=refs.iloc[int(index)]
            schedule.append({'background_fold':fold,'position_in_reference_cycle':position,
                             'reference_row_index':int(index),'clip_id':str(row.clip_id),'source_WAV_sha256':str(row.sha256),
                             'usage':'TRAIN','reference_pool_shared_across_background_folds':True})
    if any(len(set(o))!=len(refs) for o in order):raise RuntimeError('Cycle coverage failed')
    # Only now create new output artifacts; never overwrite old versions or banks.
    output.mkdir(parents=True)
    tables={'reference_manifest.parquet':refs,'ambient_control_manifest.parquet':ambient,
            'clip_quality_manifest.parquet':pd.DataFrame(audit_rows),
            'duplicate_WAV_bytes_manifest.parquet':duplicate_manifest,
            'training_reference_schedule.parquet':pd.DataFrame(schedule)}
    for filename,table in tables.items():table.to_parquet(output/filename,index=False,compression='zstd')
    np.save(output/'reference_cycle_order.npy',order,allow_pickle=False)
    artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    for n,digest in pins.items():
        if sha(ROOT/n)!=digest:raise ValueError('Source metadata/code changed during audit')
    if sha(ROOT/BANK)!=bank_sha:raise ValueError('Original bank changed during audit')
    manifest={'version':VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),
              'output':str(output.relative_to(ROOT)),'ready_to_use_as_measured_reference_pool':ready,
              'not_a_trained_model_or_synthetic_dataset':True,
              'authority':'Explicit user authorization restores oldVTreference reservations forNEWproduction only',
              'counts':{'source_records':525,'vessel_archive_records':509,'ambient_records':16,
                        'duplicate_WAV_bytes_removed':int(dup.sum()),'admitted_reference_records':len(refs),
                        'separate_ambient_controls':len(ambient),'reference_weak_class_counts':_counts(refs),
                        'original_split_counts_restored':{str(k):int(v) for k,v in refs.original_split.value_counts().items()},
                        'source_hours_all525':float(frame.duration_seconds.sum()/3600),
                        'reference_header_seconds':float(refs.duration_seconds.sum()),
                        'reference_complete2s_seconds':int(refs.segment_count.sum()*2),
                        'ambient_header_seconds':float(ambient.duration_seconds.sum()),
                        'minimum_complete_minutes':int(refs.native_complete_minute_count.min()),
                        'no_measured_complete_minute_references':int((~refs.has_measured_complete_minute).sum()),
                        'reference_groups_full_corpus_graph':1,'original_hulls':int(refs.hull_id.nunique()),
                        'reference_recording_days':int(refs.recording_day.nunique())},
              'bank':{'path':BANK,'shape':list(bank.shape),'dtype':str(bank.dtype),'sha256':bank_sha,
                      'new_waveform_copy_or_upsampling':False},
              'schema':{'slice_hash':'SHA256 contiguous originalfloat32[K,25] bytes, no prefix',
                        'array_identity_hash':'SHA256 canonicalJSON(shape,dtype)+contiguousbytes',
                        'native_minute':'30complete2s bins, no pad/repeat/warp bythisbuilder'},
              'background_CV':{'folds':5,'reference_pool_shared_per_fold':True,
                               'reference_independence_claim':False,
                               'TRAIN_unique_references_per_cycle':[int(len(set(v))) for v in order],
                               'every_reference_TRAIN_eligible_in_all_five_folds':True,
                               'schedule_seed':seed,'epoch_function':'pipeline.reference_pool_r9.reference_indices_for_epoch',
                               'epoch_contract':'fixedn_examples per fold; contiguouscyclingofindexstream;N>=509coverseveryreferenceeach epoch',
                               'background_split_or_24h_purge_modified':False},
              'cautions':['Whole measured recording includes ambient;not verified isolated vessel-only energy.',
                          '22weak-background vessel-archive records intentionally retained under userall509 instruction;class isnotenergyGT.',
                          '16dedicatedambient controls areseparate;neverautomaticallyassignnaturalsourceabsenceorzeroenergy.',
                          'OldVTreferencecalibration/domain/outer splits no longerindependentfornewmodelsusingthispool.',
                          'ExistingR4/R5/R6 datasets,models,andpanels remain unchanged.',
                          'Per-WAVproviderSHA/header/waveformQCarehistoricalrecords;thisbuildrevalidatesconvertedbankonly.',
                          'LargeDCreviewflagsretained;constantdetrendingdoesnotpressure-calibratesensors.'],
              'input_sha256':pins,'file_sha256':artifacts}
    manifest['source_hash']=hashlib.sha256(canonical(manifest)).hexdigest()
    write(output/'manifest.json',manifest)
    readiness={'version':VERSION,'ready':ready,'manifest_sha256':sha(output/'manifest.json'),
               'source_hash':manifest['source_hash'],'all509_fullband_and_measured_minute':bool(refs.reference_eligible.all()),
               'ambient_controls_separate_not_automatic_zero_labels':True,
               'all_five_TRAIN_cycles_cover_every_reference':True,
               'old_reference_holdouts_independent_for_new_models':False,
               'scope':'Measuredreferencepool only;actualnewgenerator/backgroundCV/trainingreadinessrequiresseparategates',
               'old_bank_and_input_metadata_unchanged':True}
    write(output/'readiness.json',readiness)
    return manifest,readiness


def validate_reference_pool(output=DEFAULT_OUTPUT,verify_bank=True):
    p=ROOT/output;m=json.loads((p/'manifest.json').read_text());g=json.loads((p/'readiness.json').read_text())
    if m['version']!=VERSION or g['version']!=VERSION or g['ready'] is not True:
        raise ValueError('Reference pool is not ready')
    if sha(p/'manifest.json')!=g['manifest_sha256']:raise ValueError('Manifest SHA changed')
    for name,digest in m['file_sha256'].items():
        if Path(name).name!=name or sha(p/name)!=digest:raise ValueError('Pool artifact changed: '+name)
    for name,digest in m['input_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Original source/code changed: '+name)
    if verify_bank and sha(ROOT/m['bank']['path'])!=m['bank']['sha256']:raise ValueError('Original bank changed')
    refs=pd.read_parquet(p/'reference_manifest.parquet');ambient=pd.read_parquet(p/'ambient_control_manifest.parquet')
    order=np.load(p/'reference_cycle_order.npy',allow_pickle=False)
    if (len(refs)!=509 or len(ambient)!=16 or refs.sha256.duplicated().any()
            or not refs.reference_eligible.all() or order.shape!=(5,509)
            or any(set(v)!=set(range(509)) for v in order)):
        raise ValueError('Pool counts/quality/coverage changed')
    return {'status':'passed','source_hash':m['source_hash'],'manifest_sha256':g['manifest_sha256'],
            'references':len(refs),'ambient':len(ambient),'reference_independent_CV':False,
            'old_arrays_or_metadata_modified':False,'model_training_executed':False}
