"""Audit acquired native data and export a locked adapter view without touching V1."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import soundfile as sf
from fetch_extension_plan import quality_intervals, HIGH
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'external/expansion_research'

def checksum(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    records=json.loads((BASE/'native_15min_manifest.json').read_text())
    converted=[];audits=[]
    for r in records:
        if r.get('failed'):continue
        dep=r['deployment'];md=BASE/dep/('SanctSound_'+dep.upper()+'.json')
        events=pd.read_csv(BASE/dep/'ship_events.csv')
        start=pd.Timestamp(r['start_utc']);end=pd.Timestamp(r['end_utc'])
        a=pd.to_datetime(events.ISOStartTime,utc=True);b=pd.to_datetime(events.ISOEndTime,utc=True)
        matching=events[(a<=start)&(b>=end)]
        original=sorted(set(matching.Labels.str.strip()))
        if r['annotation'].lower()=='ship':assert any(x.lower()=='ship' for x in original)
        else:
            assert r['annotation']=='NoShip' and r['prospective_split']=='ambient_control_quarantine'
            assert any(x.lower()=='noship' for x in original)
        meta=json.loads(md.read_text())
        assert any(g0<=start and g1>=end for g0,g1 in quality_intervals(meta)), 'Not wholly inside official full-band Good interval'
        if dep=='oc03_02':assert start>=pd.Timestamp('2019-11-08',tz='UTC'), 'First-week noise-floor exclusion'
        crop=BASE/r['crop_relative_path'];assert checksum(crop)==r['crop_sha256']
        with sf.SoundFile(crop) as f:
            fs=f.samplerate;frames=f.frames;channels=f.channels;subtype=f.subtype
            assert fs/2>=HIGH and channels==1 and frames==900*fs
            samples=f.read(dtype='int16' if subtype=='PCM_16' else 'int32',always_2d=True)[:,0]
        zero=np.flatnonzero(samples==0)
        breaks=np.r_[0,np.flatnonzero(np.diff(zero)>1)+1,len(zero)]
        longest=int(np.diff(breaks).max()) if len(zero) else 0
        maxrail=np.iinfo(samples.dtype).max if subtype=='PCM_16' else (2**23-1)<<8
        minrail=np.iinfo(samples.dtype).min
        rails=float(((samples==maxrail)|(samples==minrail)).mean())
        assert longest<fs,'Detected >=1s digital zero dropout'
        assert rails<=.001,'Excessive digital clipping'
        bp=BASE/r['band_power_relative_path'];z=np.load(bp);seg=z['segments'];minute=z['minutes']
        assert seg.shape==(450,25) and minute.shape==(15,25)
        assert np.isfinite(seg).all() and (seg>=0).all() and (seg.sum(1)>0).all()
        assert np.allclose(minute,seg.reshape(15,30,25).mean(1),rtol=1e-6)
        sfp=bp.with_name(bp.name.removesuffix('_band_power.npz')+'_segments_power.npy')
        mfp=bp.with_name(bp.name.removesuffix('_band_power.npz')+'_minutes_power.npy')
        np.save(sfp,seg);np.save(mfp,minute)
        split='sealed_holdout' if r['prospective_split']=='new_sealed_holdout' else r['prospective_split']
        row={'source':'NOAA_SanctSound_extension_v1','clip_id':r['clip_id'],'split':split,
            'archive_role':'vessel_passage' if r['annotation'].lower()=='ship' else 'ambient_control_weak_annotation',
            'weak_class':r['annotation'],'group_key':'NOAA_STATION_'+r['station'],
            'station':r['station'],'deployment':dep,'recording_day':str(start.date()),
            'device_serial':r['device_serial'],
            'utc_start':str(start),'utc_end':str(end),'utc_begin':str(start),
            'header_fs':fs,'sample_rate_hz':fs,'channels':channels,'native_subtype':subtype,
            'native_duration_seconds':frames/fs,'duration_seconds':frames/fs,
            'native_source_frames':r['native_source_frames'],'native_crop_start_frame':r['native_crop_start_frame'],
            'native_crop_frames':frames,'segment_file':str(sfp.relative_to(ROOT)),
            'minute_file':str(mfp.relative_to(ROOT)),'segment_count':450,'minute_count':15,
            'quality_full_band_good':True,'quality_start':r['quality_start'],'quality_end':r['quality_end'],
            'quality_metadata_file':str(md.relative_to(ROOT)),'quality_metadata_sha256':checksum(md),
            'additional_quality_exclusion':'2019-11-01 through 2019-11-07 high-frequency noise-floor warning excluded' if dep=='oc03_02' else '',
            'annotation_start':r['annotation_start'],'annotation_end':r['annotation_end'],
            'annotation_labels_original':original,'licence':'CC0-1.0',
            'source_url':'https://storage.googleapis.com/noaa-passive-bioacoustic/'+r['source_object'],
            'source_object':r['source_object'],'provider_generation':r['source_generation'],
            'provider_md5_base64':r['source_md5_base64'],'provider_full_checksum_verified':False,
            'excerpt_file':str(crop.relative_to(ROOT)),'excerpt_sha256':r['crop_sha256'],
            'segment_file_sha256':checksum(sfp),'minute_file_sha256':checksum(mfp),
            'range_ledger_available':True,'checksum_limitation':'Provider full-object MD5 not verified; exact pinned ranges and excerpt SHA-256 recorded.',
            'hull_isolation_verified':False,'is_primary_native_reference':r['annotation'].lower()=='ship',
            'sealed_holdout_locked':split=='sealed_holdout','model_evaluated':False,
            'digital_rail_fraction':rails,'longest_zero_run_seconds':longest/fs,
            'reference_kind':'measured_contiguous_native_15min_with_ambient_sound'}
        converted.append(row);audits.append({'clip_id':r['clip_id'],'checks':'native frames/fs, no >=1s zero gap, <=0.1% clipping, all 25 bands finite, linear minute aggregation, explicit annotation and full-band Good interval passed'})
    frame=pd.DataFrame(converted)
    pos=frame[frame.is_primary_native_reference]
    assert pos.groupby('group_key').split.nunique().max()==1
    assert pos.groupby('excerpt_sha256').split.nunique().max()==1
    for dep,g in pos.groupby('deployment'):
        g=g.sort_values('utc_start');previous=None
        for r in g.itertuples():
            start=pd.Timestamp(r.utc_start);end=pd.Timestamp(r.utc_end)
            assert previous is None or start>=previous,'Overlapping actual source intervals'
            previous=end
    (BASE/'native_reference_adapter.json').write_text(json.dumps(converted,indent=2))
    frame.to_parquet(BASE/'native_reference_adapter.parquet',index=False)
    counts=pos.groupby('split').agg(clips=('clip_id','size'),stations=('station','nunique'),days=('recording_day','nunique'),minutes=('minute_count','sum')).reset_index()
    # Count actual acquired bytes across initial, CI recovery and OC3 additions, including failed header probes.
    initial=json.loads((BASE/'native_15min_summary.json').read_text())['downloaded_range_bytes']
    acquired=initial+sum(r['acquired_range_bytes'] for r in records if not r.get('failed') and (r['deployment']=='oc03_02' or (r['deployment']=='ci01_01' and r['recording_day']=='20181204')))
    summary={'positive_native_clips':len(pos),'positive_native_minutes':int(pos.minute_count.sum()),
        'ambient_quarantine_clips':int((~frame.is_primary_native_reference).sum()),
        'unique_positive_stations':pos.station.nunique(),'unique_positive_deployments':pos.deployment.nunique(),
        'unique_positive_days':pos.recording_day.nunique(),'counts':counts.to_dict('records'),
        'network_range_bytes_including_initial_probes':acquired,'total_crop_bytes':sum((ROOT/x).stat().st_size for x in frame.excerpt_file),
        'test_accessed':False,'existing_OC02_02_accessed':False,'SB_sealed_model_evaluated':False,
        'checks_passed':audits,'adapter_manifest_sha256':checksum(BASE/'native_reference_adapter.json'),
        'limitations':['Few independent source groups: synthetic repetition is not new real-world information.',
            'Ship labels indicate events, not isolated ship-only energy or known independent hull identities.',
            'NoShip was quarantined, not treated as verified natural vessel absence.',
            'Only source ranges acquired: provider full-file MD5 is NOT verified.',
            'Digital energy is not pressure-calibrated; calibration PDFs retained but not applied.']}
    (BASE/'native_reference_final_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='checks_passed'},indent=2))
if __name__=='__main__':main()
