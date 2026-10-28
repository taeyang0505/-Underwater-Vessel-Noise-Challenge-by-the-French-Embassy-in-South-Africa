"""Meaningful end-to-end data, energy and source-isolation checks."""
from pathlib import Path
import io,json
import numpy as np
import pandas as pd
import soundfile as sf
from pipeline.audio import audio_band_power
from pipeline.mixtures import MixtureSampler,cnn_channels
ROOT=Path(__file__).resolve().parents[1]

def main():
    checks={};p=ROOT/'data/processed'
    rows=pd.read_parquet(p/'sa_row_manifest.parquet');weak=pd.read_parquet(p/'sa_all_rows_weak_labels.parquet')
    assert len(rows)==len(weak)==1715779 and rows.row_id.equals(weak.row_id)
    assert weak.is_verified_ground_truth.eq(False).all()
    assert weak.teacher_score_raw_audit.notna().all()
    checks['all_original_rows_preserved_and_scored']=len(rows)
    # Known narrow-band sine: variance A^2/2 must land in its physical 1-kHz band.
    fs=32000;t=np.arange(fs*60)/fs;signal=.1*np.sin(2*np.pi*1000*t)
    buffer=io.BytesIO();sf.write(buffer,signal,fs,format='WAV',subtype='FLOAT')
    seg,minutes,_=audio_band_power(buffer.getvalue())
    assert np.allclose(minutes[0,15],.005,rtol=1e-5)
    assert minutes[0,np.arange(25)!=15].sum()<1e-10
    checks['known_sine_energy_band_and_units']='passed'
    # Check direct compatibility with the organizer's released conversion function.
    nb=json.loads((ROOT/'data/raw/maritime_starter_notebook.ipynb').read_text())
    source=next(''.join(c['source']) for c in nb['cells'] if 'def minute_band_power' in ''.join(c['source']))
    context={'np':np};exec(compile(source,'official_starter_reference','exec'),context)
    gold=context['minute_band_power'](signal,fs)
    assert np.allclose(gold,minutes,rtol=1e-5,atol=1e-12)
    checks['official_psd_reference_equivalence']='passed'
    # An explicitly removed minute must contribute neither target energy nor model pooling weight.
    for split in ['train','validation','calibration','domain_holdout']:
        db,y,z,q,recipe=MixtureSampler(split,seed=42).draw(32)
        c,w=cnn_channels(db)
        assert np.allclose((q*w).sum((1,2)),y,atol=2e-6)
        assert np.all(w[~np.isfinite(db)]==0)
        assert np.allclose(w.sum((1,2)),1)
        assert np.isfinite(c).all()
    checks['controlled_mixture_energy_targets_and_missing_mask']='passed'
    m=pd.read_parquet(p/'external_clip_manifest.parquet')
    active=m[m.split.isin(['train','validation','calibration','domain_holdout'])]
    assert active.groupby('sha256').split.nunique().max()==1
    v=active[active.source.eq('VTUAD_v1.3.2')]
    assert v.groupby('group_key').split.nunique().max()==1
    # Include every calendar day touched by each reference interval.
    days=[]
    for r in v.itertuples():
        for d in pd.date_range(pd.Timestamp(r.utc_begin).floor('D'),(pd.Timestamp(r.utc_end)-pd.Timedelta('1ns')).floor('D'),freq='D'):
            days.append((str(d.date()),r.split))
    assert pd.DataFrame(days,columns=['day','split']).groupby('day').split.nunique().max()==1
    h=active[active.source.eq('HearMyShip_V1')]
    assert h.groupby('recording_day').split.nunique().max()==1
    checks['external_recording_hull_day_and_byte_duplicate_isolation']='passed; HearMyShip hull identities unknown'
    checks['external_source_clips']=m.source.value_counts().to_dict()
    (ROOT/'verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
