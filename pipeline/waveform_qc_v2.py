"""Read-only source waveform QC, separate from frozen band conversion outputs."""
from pathlib import Path
import io,hashlib,json,zipfile
import numpy as np
import pandas as pd
import soundfile as sf
from pipeline.audio import MultipartReader,CENTRES
ROOT=Path(__file__).resolve().parents[1]

def inspect(data):
    n=0;sum_x=0.;sum_x2=0.;peak=0.;zeros=0;rails=0;longest_zero=0;carry_zero=0
    with sf.SoundFile(io.BytesIO(data)) as f:
        fs=f.samplerate;frames=f.frames;channels=f.channels;subtype=f.subtype
        bits={'PCM_U8':8,'PCM_16':16,'PCM_24':24,'PCM_32':32}.get(subtype)
        threshold=1-2**(-(bits-1)) if bits else None
        while True:
            x=f.read(fs*60,dtype='float32',always_2d=True)[:,0]
            if not len(x):break
            if not np.isfinite(x).all():raise ValueError('Non-finite raw waveform samples')
            xd=x.astype('float64');n+=len(x);sum_x+=float(xd.sum());sum_x2+=float(np.dot(xd,xd))
            peak=max(peak,float(np.abs(x).max()));z=x==0;zeros+=int(z.sum())
            if threshold is not None:rails+=int(((x>=threshold)|(x<=-1)).sum())
            edges=np.diff(np.r_[False,z,False].astype('int8'))
            starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
            lengths=ends-starts
            if len(lengths):
                if starts[0]==0:lengths[0]+=carry_zero
                longest_zero=max(longest_zero,int(lengths.max()))
                carry_zero=int(ends[-1]-starts[-1]) if ends[-1]==len(z) else 0
                if len(starts)==1 and starts[0]==0 and ends[0]==len(z):carry_zero=int(lengths[0])
            else:carry_zero=0
    assert n==frames
    rms=float(np.sqrt(sum_x2/n));mean=sum_x/n;clip_fraction=rails/n if threshold is not None else None
    upper=float(CENTRES[-1]*10**.05)
    flags=[]
    if fs/2<upper:flags.append('insufficient_header_nyquist_for_all_25_bands')
    if clip_fraction is not None and clip_fraction>1e-3:flags.append('digital_rail_fraction_above_0p1_percent_review')
    if longest_zero/fs>=1:flags.append('digital_zero_run_at_least_1s_review')
    if rms==0:flags.append('zero_waveform_energy')
    elif abs(mean)/rms>.9:flags.append('large_DC_fraction_review')
    return {'header_sample_rate_hz':fs,'frames':frames,'channels':channels,'selected_channel':0,
        'sample_subtype':subtype,'actual_duration_seconds':frames/fs,'peak_abs_fullscale_units':peak,
        'rms_fullscale_units':rms,'mean_fullscale_units':mean,'digital_zero_fraction':zeros/n,
        'longest_digital_zero_seconds':longest_zero/fs,'digital_rail_fraction':clip_fraction,
        'quality_flags':flags,'no_automatic_semantic_or_pressure_calibration_claim':True}

def main():
    meta=pd.read_parquet(ROOT/'data/processed/external_clip_manifest.parquet');rows=[];fail=[]
    def add(r,data):
        try:
            assert hashlib.sha256(data).hexdigest()==r.sha256,'Frozen source SHA mismatch'
            q=inspect(data)
            assert q['header_sample_rate_hz']==r.sample_rate_hz
            assert abs(q['actual_duration_seconds']-r.duration_seconds)<1e-6
            rows.append({'source':r.source,'clip_id':r.clip_id,'name':r.name,'split':r.split,
                'group_key':r.group_key,'sha256':r.sha256,**q})
        except Exception as e:fail.append({'source':r.source,'clip_id':r.clip_id,'error':repr(e)})
        if (len(rows)+len(fail))%50==0:print('Waveform QC',len(rows),'failures',len(fail),flush=True)
    hroot=ROOT/'external/hearmyship_v1/wav'
    for r in meta[meta.source.eq('HearMyShip_V1')].itertuples():add(r,(hroot/(r.clip_id+'.wav')).read_bytes())
    vroot=ROOT/'external/vtuad_v1.3.2';parts=[vroot/f'vtuad_strict_wav_vessel.zip.part{i:02d}' for i in [1,2,3]]
    for reader,role in [(MultipartReader(parts),'vessel_passage'),(vroot/'vtuad_strict_wav_background.zip','ambient_control')]:
        with zipfile.ZipFile(reader) as z:
            for r in meta[meta.source.eq('VTUAD_v1.3.2')&meta.archive_role.eq(role)].itertuples():add(r,z.read(r.name))
        if isinstance(reader,MultipartReader):reader.close()
    frame=pd.DataFrame(rows)
    frame.to_parquet(ROOT/'reports/data_quality_waveform_qc.parquet',index=False,compression='zstd')
    flags=frame.explode('quality_flags').dropna(subset=['quality_flags'])
    report={'clips_inspected':len(frame),'all_source_checksums_reverified':not fail,'failures':fail,
        'flag_counts':flags.quality_flags.value_counts().to_dict(),
        'flagged_clips':flags[['source','clip_id','name','quality_flags']].to_dict('records'),
        'source_data_or_frozen_bank_modified':False,'challenge_test_accessed':False,'sealed_NOAA_accessed':False,
        'caveat':'A clean waveform QC result does not establish a vessel-only source, acoustic absence, physical hull independence or calibrated pressure. Review flags are not automatic deletion criteria.'}
    (ROOT/'reports/data_quality_waveform_qc.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
