"""Attach audited native recordings to an immutable, opt-in v2 package."""
from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np
import pandas as pd
import soundfile as sf
from pipeline.audio import CENTRES
from pipeline.build_quality_v2 import sha256,ACTIVE
ROOT=Path(__file__).resolve().parents[1]

def canonical_hash(records):
    return hashlib.sha256(json.dumps(records,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def main(a):
    if a.output.exists():raise FileExistsError('Use a fresh output path; never overwrite a data version')
    base_version=ROOT/'data/versions/quality_v2';manifest=json.loads((base_version/'manifest.json').read_text())
    base=ROOT/'data/processed'
    for name,h in manifest['source_sha256'].items():assert sha256(base/name)==h,'Frozen v1 source changed'
    adapter=pd.read_parquet(a.adapter)
    qc=pd.read_parquet(ROOT/'reports/data_quality_waveform_qc.parquet').set_index(['source','clip_id'])
    ext=pd.read_parquet(base_version/'external_manifest.parquet');records=[];excluded=[]
    blocking={'insufficient_header_nyquist_for_all_25_bands','digital_rail_fraction_above_0p1_percent_review',
              'digital_zero_run_at_least_1s_review','zero_waveform_energy'}
    for r in ext.itertuples():
        q=qc.loc[(r.source,r.clip_id)];flags=set(q.quality_flags)
        good=not bool(blocking&flags)
        eligible=bool((r.primary_reference_eligible or r.secondary_reference_eligible) and good)
        records.append({'source':str(r.source),'clip_id':str(r.clip_id),'group_key':str(r.group_key),
            'split':str(r.split),'role':str(r.archive_role),'weak_class':str(r.weak_class),
            'storage':'base_segments','segment_file':'data/processed/external_segments_power.npy',
            'segment_start':int(r.segment_start),'segment_count':int(r.segment_count),'minute_count':int(r.minute_count),
            'duration_seconds':float(r.duration_seconds),'sample_rate_hz':int(r.sample_rate_hz),
            'recording_day':str(r.recording_day),'verified':True,'measured':True,
            'quality_full_band_good':good,'quality_evidence':'25 finite bands/header Nyquist and waveform QC; sensor transfer calibration remains unverified',
            'quality_review_flags':sorted(flags),'reference_eligible':eligible,
            'native_long':bool(eligible and r.minute_count>=15),'sealed':False,
            'sha256':str(r.sha256),'checksum_limitation':'Frozen source WAV SHA-256 reverified',
            'provenance':'data/processed/external_clip_manifest.parquet'})
    seen_intervals=[]
    for r in adapter.itertuples():
        if not r.is_primary_native_reference:
            excluded.append({'clip_id':r.clip_id,'reason':'Not a verified primary positive; quarantined separately'})
            continue
        assert r.split in ['train','validation','calibration','sealed_holdout']
        assert 'oc02_02' not in str(r.deployment).lower(),'Existing sealed assessment must never be attached'
        assert str(r.weak_class).lower()=='ship'
        labels=[str(v).strip().lower() for v in r.annotation_labels_original]
        assert 'ship' in labels and 'noship' not in labels
        assert r.quality_full_band_good and r.native_duration_seconds==900 and r.minute_count==15
        assert r.header_fs/2>=float(CENTRES[-1]*10**.05)
        start=pd.to_datetime(r.utc_start,utc=True);end=pd.to_datetime(r.utc_end,utc=True)
        assert end-start==pd.Timedelta(minutes=15)
        assert pd.to_datetime(r.quality_start,utc=True)<=start and end<=pd.to_datetime(r.quality_end,utc=True)
        assert pd.to_datetime(r.annotation_start,utc=True)<=start and end<=pd.to_datetime(r.annotation_end,utc=True)
        assert r.digital_rail_fraction<=1e-3 and r.longest_zero_run_seconds<1
        for file_key,hash_key in [('excerpt_file','excerpt_sha256'),('segment_file','segment_file_sha256'),
                                  ('minute_file','minute_file_sha256'),('quality_metadata_file','quality_metadata_sha256')]:
            assert sha256(ROOT/getattr(r,file_key))==getattr(r,hash_key)
        info=sf.info(ROOT/r.excerpt_file)
        assert info.samplerate==r.header_fs and info.frames==r.native_crop_frames==900*r.header_fs
        segments=np.load(ROOT/r.segment_file,mmap_mode='r');minutes=np.load(ROOT/r.minute_file,mmap_mode='r')
        assert segments.shape==(450,25) and minutes.shape==(15,25)
        assert np.isfinite(segments).all() and (segments>=0).all() and (segments.sum(1)>0).all()
        assert np.allclose(segments.reshape(15,30,25).mean(1),minutes,rtol=2e-6,atol=1e-20)
        for group,old_start,old_end,old_fold in seen_intervals:
            if group==r.group_key:assert end<=old_start or start>=old_end,'Overlapping native references'
        seen_intervals.append((r.group_key,start,end,r.split))
        records.append({'source':str(r.source),'clip_id':str(r.clip_id),'group_key':str(r.group_key),
            'split':str(r.split),'role':'vessel_passage','weak_class':'ship','storage':'per_record_segments',
            'segment_file':str(r.segment_file),'segment_start':0,'segment_count':450,'minute_count':15,
            'duration_seconds':900.,'sample_rate_hz':int(r.header_fs),'recording_day':str(r.recording_day),
            'verified':True,'measured':True,'quality_full_band_good':True,
            'quality_evidence':'Explicit ship interval intersected with provider Good full-band interval; native header and waveform QC checked',
            'quality_review_flags':[],'reference_eligible':True,'native_long':True,
            'sealed':bool(r.sealed_holdout_locked),'sha256':str(r.excerpt_sha256),
            'checksum_limitation':str(r.checksum_limitation),'provenance':str(a.adapter.relative_to(ROOT))})
    frame=pd.DataFrame(records)
    active=frame[frame.reference_eligible & frame.split.isin(ACTIVE+['sealed_holdout'])]
    assert active.groupby(['source','group_key']).split.nunique().max()==1
    assert active.groupby('sha256').split.nunique().max()==1
    native=active[active.native_long].groupby('split').size().to_dict()
    for split in ['train','validation','calibration']:assert native.get(split,0)>0
    positive=adapter[adapter.is_primary_native_reference]
    assert positive.groupby('device_serial').split.nunique().max()==1
    assert positive.groupby('station').split.nunique().max()==1
    assert not positive.model_evaluated.any(),'New source must not have been selected from model performance'
    a.output.mkdir(parents=True)
    shutil.copy2(base_version/'background_manifest.parquet',a.output/'background_manifest.parquet')
    frame.to_parquet(a.output/'reference_manifest.parquet',index=False,compression='zstd')
    cfg={'version':'quality_v2_native','source_hash':canonical_hash(records),
        'background_version_manifest':'data/versions/quality_v2/manifest.json',
        'background_manifest':'background_manifest.parquet','reference_manifest':'reference_manifest.parquet',
        'base_background_array':'data/processed/sa_windows_db.npy',
        'frozen_v1_hashes':manifest['source_sha256'],'native_adapter_sha256':sha256(a.adapter),
        'native_long_counts':native,'excluded_adapter_records':excluded,
        'input_quality_gate_passed':True,'ready_to_materialize':True,'prebuilt_verification_required':True,
        'sealed_holdout_locked':True,'sealed_holdout_materialized':False,'model_evaluated_on_new_sealed':False,
        'requires_fresh_v2_task_training':True,'vendor_original_pretrained_weights_may_be_reused':True,
        'v1_fitted_context_scaler_KV_and_task_weights_must_not_be_reused':True,
        'challenge_test_accessed':False,'existing_sealed_NOAA_accessed':False,
        'target_policy':'Desired positive share must intersect measured reference energy under observed mask; no silent relabel-to-zero.',
        'sampler':'pipeline.native_sampler_v2.NativeFirstSampler','surrogate_envelopes_or_repeated_audio':False,
        'native_provider_whole_file_checksum_limitation':'Provider full-object checksums were not recomputed for pinned-range acquisitions; excerpt and derived-array SHA-256 verified.',
        'semantics':'Added-reference energy includes measured ambience; annotations and waveform QC do not prove isolated vessel energy or physical hull independence.'}
    (a.output/'manifest.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    print(json.dumps(cfg,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--adapter',type=Path,default=ROOT/'external/expansion_research/native_reference_adapter.parquet')
    p.add_argument('--output',type=Path,default=ROOT/'data/versions/quality_v2_native');main(p.parse_args())
