"""Independent numeric golden cases and immutable-source quality-v2 checks."""
from pathlib import Path
import io,json
import numpy as np
import pandas as pd
import soundfile as sf
from pipeline.augmentation_v2 import (EnergyComponents,solve_reference_scale,
    UnobservedReferenceError,place_measured_segments,average_complete_minutes,
    blend_backgrounds,complete_audio_tail_audit)
from pipeline.audio import audio_band_power
from pipeline.build_quality_v2 import sha256,purged_split
from pipeline.native_sampler_v2 import NativeFirstSampler,ReferenceRecord,AugmentationConfig

ROOT=Path(__file__).resolve().parents[1]

def main():
    result={};b=np.array([[1.,9.]]);r=np.array([[9.,1.]]);o=np.ones((1,2),dtype=bool)
    m=EnergyComponents(b,r,o)
    assert m.share==.5
    assert np.isclose(m.common_gain(17).share,.5)
    assert np.isclose(m.frequency_response([10,0]).share,91/110)
    assert np.isclose(m.add_measured_ambient(np.ones((1,2))).share,10/22)
    assert m.with_mask([[True,False]]).share==.9
    assert m.reverse_time().share==m.share
    result['known_gain_response_noise_mask_target_cases']='passed; target copied only for provably invariant transforms'
    alpha,at=solve_reference_scale(b,r,o,.2)
    assert np.isclose(alpha,.25) and np.isclose(at.share,.2)
    assert np.allclose((at.cell_share*at.energy_weights).sum(),.2)
    try:solve_reference_scale(b,np.zeros_like(r),o,.2)
    except UnobservedReferenceError:pass
    else:raise AssertionError('Unobserved positive reference must be rejected, not relabelled silently')
    zero=solve_reference_scale(b,np.zeros_like(r),o,0)[1]
    assert zero.share==0
    for invalid in [-.1,1.,np.nan]:
        try:solve_reference_scale(b,r,o,invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid desired fraction accepted')
    result['boundary_component_solver_checks']='passed'
    # A 20-second clip crossing one minute boundary contributes 10 seconds to
    # each neighboring minute. The other 13 minutes are never replicated audio.
    s=place_measured_segments(np.ones((10,25)),25)
    mins=average_complete_minutes(s)
    assert np.allclose(mins[:2],1/6) and np.count_nonzero(mins[2:])==0
    assert np.allclose(average_complete_minutes(place_measured_segments(np.ones((10,25)),-5))[0],1/6)
    result['native_short_placement_no_repetition']='passed'
    mix,mask,g=blend_backgrounds(np.array([[2.,2.]]),np.array([[1.,np.nan]]),
        np.ones((1,2),bool),np.array([[True,False]]),.5)
    assert np.array_equal(mask,[[True,False]]) and g==2 and np.isclose(mix[0,0],2)
    result['background_blend_intersection_mask']='passed'
    # Explicitly distinguish the incomplete 2-second tail from the duration
    # excluded when only complete measured minute summaries are requested.
    fs=32000;t=np.arange(fs*63)/fs;signal=.1*np.sin(2*np.pi*1000*t)
    buf=io.BytesIO();sf.write(buf,signal,fs,format='WAV',subtype='FLOAT')
    seg,minutes,audit=audio_band_power(buf.getvalue())
    tails=complete_audio_tail_audit(audit['duration_seconds'],seg,minutes)
    assert len(seg)==31 and len(minutes)==1
    assert tails=={'trailing_seconds_below_two_second_segment':1.,
        'complete_segments_not_in_minute_array':1,'seconds_not_in_complete_minute_array':3.}
    result['audio_tail_units_and_minute_count']='passed'
    # A tiny synthetic clock fixture exposes a reused-recorder group at a
    # week boundary, checks nanosecond unit conversion and support-aware purge.
    dates=pd.date_range('2025-01-01',periods=730,freq='12h')
    fixture=pd.DataFrame({'site':['A']*len(dates),'recorder':['R']*len(dates),
        'start_export_clock':dates.astype(str),'sensor_group':[1]*len(dates),
        'split':['train']*len(dates),'observed_minutes':[15]*len(dates)})
    q=purged_split(fixture,42,24)
    active=q[q.split.isin(['train','validation','calibration'])]
    assert active.groupby(['sensor_group','recording_week_start']).split.nunique().max()==1
    assert active.nearest_foreign_fold_support_gap_hours.min()>=24
    result['weekly_clock_and_symmetric_purge_fixture']='passed'
    assert {'validation','calibration','purged_temporal_boundary'}.issubset(set(q.split))
    # Exercise the complete sampler on independent toy energy components,
    # including sparse coverage, blend/noise transforms and native support.
    rng=np.random.default_rng(9001);toy_db=rng.normal(90,3,(8,15,25))
    toy_db[::2,5:]=np.nan
    toy_meta=pd.DataFrame({'base_window_index':np.arange(8),'split':['train']*8,
        'site':['A']*4+['B']*4,'observed_minutes':[5,15]*4})
    refs=[ReferenceRecord('native','toy','group1','train',np.ones((500,25)),verified=True,quality_full_band_good=True),
          ReferenceRecord('short','toy','group2','train',np.ones((10,25)),verified=True,quality_full_band_good=True),
          ReferenceRecord('ambient','toy','group3','train',np.ones((480,25)),verified=True,
                          quality_full_band_good=True,role='ambient_control')]
    cfg=AugmentationConfig(background_blend_probability=1.,measured_ambient_probability=1.,common_response_range_db=3.)
    sampler=NativeFirstSampler(toy_db,toy_meta,refs,seed=42,config=cfg)
    db,y,presence,cell,recipes=sampler.draw(512)
    requested=np.array([r['desired_share'] for r in recipes['items']])
    assert np.allclose(y,requested,atol=1e-6) and np.array_equal(presence,(requested>0).astype('float32'))
    peak=np.nanmax(db,axis=(1,2),keepdims=True);energy=np.where(np.isfinite(db),10**((db-peak)/10),0)
    assert np.allclose((cell*energy).sum((1,2))/energy.sum((1,2)),y,atol=2e-6)
    assert all(i['observed_minutes']>=3 for i in recipes['items'])
    assert not recipes['reference_repetition_used'] and not recipes['surrogate_envelopes_used']
    result['native_first_full_sampler_known_component_fixture']='passed; all 512 requested fractions match observed targets'
    version=ROOT/'data/versions/quality_v2'
    manifest=json.loads((version/'manifest.json').read_text())
    for name,h in manifest['source_sha256'].items():assert sha256(ROOT/'data/processed'/name)==h
    result['all_v1_source_hashes_unchanged']='passed'
    result['challenge_test_accessed']=False;result['sealed_NOAA_accessed']=False
    (ROOT/'reports/data_quality_v2_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
