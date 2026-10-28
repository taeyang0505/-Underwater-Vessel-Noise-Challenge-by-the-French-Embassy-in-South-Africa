"""Read-only starter/helper versus current external-audio conversion audit.

Execute only the reviewed minute_band_power FunctionDef from notebook cell11.
No notebook imports/data/submission cells, models, Test, or report arrays run.
Synthetic WAV containers live in memory; one preselected TRAIN VTUAD WAV is
read and SHA-verified without changing source WAVs, archives, banks or caches.
"""
from __future__ import annotations
import ast
import csv
from datetime import datetime,timezone
import gc
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import scipy
from scipy.signal import periodogram,get_window
import soundfile as sf

from pipeline.audio import audio_band_power,MultipartReader,CENTRES
from pipeline.sampler_r9 import native_minutes

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r10_diagnostics'
ART=OUT/'starter_conversion_artifacts'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def bytes_sha(b):return hashlib.sha256(b).hexdigest()
def save(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def helper_from_reviewed_function():
    path=ROOT/'data/raw/maritime_starter_notebook.ipynb';book=json.loads(path.read_text())
    matches=[]
    for i,cell in enumerate(book['cells']):
        if cell.get('cell_type')!='code':continue
        source=''.join(cell.get('source',[]))
        if 'def minute_band_power(' in source:matches.append((i,source))
    if len(matches)!=1:raise ValueError('Exactly one reviewed helper required')
    index,source=matches[0];tree=ast.parse(source)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!='minute_band_power':
        raise ValueError('Helper cell contains unreviewed executable statements')
    blocked={'open','eval','exec','compile','globals','locals','getattr','setattr','__import__','os','sys','subprocess'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Name) and node.id in blocked:raise ValueError('Unreviewed helper operation')
        if isinstance(node,ast.Import):raise ValueError('Unreviewed helper import')
        if isinstance(node,ast.ImportFrom) and (node.module!='scipy.signal' or [v.name for v in node.names]!=['periodogram']):
            raise ValueError('Unexpected helper dependency')
    # No other notebook cell is evaluated; this statement only defines the
    # already inspected pure numeric helper, using our own NumPy import.
    ns={'np':np};exec(compile(tree,str(path)+':reviewed-helper-only','exec'),ns)
    (ART/'official_helper_reviewed.py').write_text(source)
    return ns['minute_band_power'],{'notebook_file':str(path.relative_to(ROOT)),
        'notebook_sha256':sha(path),'reviewed_cell_index':index,'helper_sha256':bytes_sha(source.encode()),
        'other_notebook_cells_executed':False,'global_dependency':'NumPy only; local scipy.signal.periodogram import'}


def compare(official,ours):
    a=np.asarray(official,dtype=float);b=np.asarray(ours,dtype=float)
    if a.shape!=b.shape:raise ValueError('Complete-minute output shape differs')
    valid=np.isfinite(a)&np.isfinite(b);err=np.abs(a-b)
    scale=np.maximum(np.nanmax(a,axis=1,keepdims=True),1e-30)
    significant=valid&(a>np.maximum(scale*1e-8,1e-25))
    ratios=np.divide(b,a,out=np.full_like(a,np.nan),where=significant)
    rel=np.divide(err,np.abs(a),out=np.zeros_like(a),where=significant)
    db=10*np.log10(ratios[significant]) if significant.any() else np.array([])
    return {'shape':list(a.shape),'finite_mask_equal':bool(np.array_equal(np.isfinite(a),np.isfinite(b))),
        'max_absolute_power_error':float(np.max(err[valid])) if valid.any() else None,
        'significance_rule':'official band power > max(1e-8*minute maximum,1e-25)',
        'significant_cells':int(significant.sum()),
        'significant_max_relative_error':float(rel[significant].max()) if significant.any() else None,
        'significant_ratio_min':float(ratios[significant].min()) if significant.any() else None,
        'significant_ratio_max':float(ratios[significant].max()) if significant.any() else None,
        'significant_max_abs_db_error':float(abs(db).max()) if len(db) else None,
        'all_cells_including_numerical_leakage_note':'Near-zero out-of-band FFT leakage is not a stable relative-error denominator.'}


def container(samples,fs,subtype='FLOAT'):
    b=io.BytesIO();sf.write(b,samples,fs,format='WAV',subtype=subtype);return b.getvalue()


def case(helper,name,data):
    with sf.SoundFile(io.BytesIO(data)) as f:
        samples=f.read(dtype='float64',always_2d=True);fs=f.samplerate
        info={'sample_rate_hz':fs,'frames':f.frames,'channels':f.channels,'subtype':f.subtype,'duration_seconds':f.frames/fs}
    row={'case':name,**info,'WAV_sha256':bytes_sha(data),'official_input_channel':0,
         'complete_minutes_expected':len(samples)//(60*fs),'complete_2s_expected':len(samples)//(2*fs)}
    a=b=None;arrays={}
    try:
        a=helper(samples[:,0],fs);row['official_status']='returned';row['official_shape']=list(a.shape)
        row['official_unsupported_band_indices']=np.flatnonzero(~np.isfinite(a).all(axis=0)).tolist()
        arrays['official_minute_power']=a
    except Exception as e:row.update(official_status='rejected',official_error=type(e).__name__+': '+str(e))
    try:
        segments,direct,audit=audio_band_power(data);row['external_status']='returned';row['external_audit']=audit
        arrays['local_2s_power']=segments;arrays['local_direct_minute_power']=direct
        try:
            b=native_minutes(segments);row['R9_native_status']='returned';arrays['R9_native_minute_power']=b
            row['direct_minutes_vs_R9_native_max_abs']=float(np.max(abs(direct.astype(float)-b)))
        except Exception as e:row.update(R9_native_status='rejected',R9_native_error=type(e).__name__+': '+str(e))
    except Exception as e:row.update(external_status='rejected',external_error=type(e).__name__+': '+str(e))
    if a is not None and b is not None:row['comparison']=compare(a,b)
    if len(samples):
        n=min(len(samples),2*fs);x=samples[:n,0]
        row['first_2s_mean_square']=float(np.mean(x*x))
    if info['channels']>1:
        try:helper(samples,fs)
        except Exception as e:row['official_direct_stereo_rejected']=type(e).__name__+': '+str(e)
        else:raise AssertionError('Reviewed official one-channel guard changed')
        other=helper(samples[:,1],fs)
        row['channel1_total_band_power']=float(np.nansum(other))
        row['channel0_total_band_power']=float(np.nansum(a))
    np.savez_compressed(ART/(name+'.npz'),**arrays)
    return row


def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    if ART.exists() or (OUT/'starter_conversion_audit.json').exists():raise FileExistsError('Fresh audit artifacts required')
    ART.mkdir()
    inputs=['data/raw/maritime_starter_notebook.ipynb','pipeline/audio.py','pipeline/prepare_external.py',
            'pipeline/sampler_r9.py','data/versions/vtuad_full_r9/reference_manifest.parquet',
            'data/processed/external_segments_power.npy']
    before={n:sha(ROOT/n) for n in inputs}
    def guard(event,args):
        if event=='socket.connect':raise PermissionError('Network prohibited in this audit')
        if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
            n=Path(args[0]).name.lower()
            if n in ['test.csv','report.npz','report_metadata.parquet','report_recipes.json']:
                raise PermissionError('Test/report reads prohibited')
    sys.addaudithook(guard)
    helper,provenance=helper_from_reviewed_function();rows=[]
    specifications=[('tone100_60s',32000,60.,'tone100'),('tone100_120s',32000,120.,'tone100'),
        ('tone100_125p5s',32000,125.5,'tone100'),('noise_119p9s',32000,119.9,'noise'),
        ('noise_61p25s',32000,61.25,'noise'),('noise_59p9s',32000,59.9,'noise'),
        ('noise_20s',32000,20.,'noise'),('noise_1p5s',32000,1.5,'noise'),
        ('white_noise_32k',32000,60.,'noise'),('white_noise_48k',48000,60.,'noise'),
        ('white_noise_96k',96000,60.,'noise'),('unsupported_16k',16000,60.,'noise'),
        ('nyquist_17825',17825,60.,'noise'),('nyquist_17826',17826,60.,'noise'),
        ('below_first_edge_28Hz',32000,60.,'tone28'),('inside_first_edge_28p5Hz',32000,60.,'tone28.5'),
        ('inside_first_high_35Hz',32000,60.,'tone35'),('above_first_high_35p5Hz',32000,60.,'tone35.5'),
        ('outside_all_bands_12k',32000,60.,'tone12000'),('DC_plus_noise',32000,60.,'DC'),
        ('linear_then_log',32000,60.,'alternating_gain'),('stereo_first_channel',32000,60.,'stereo'),
        ('silence_60s',32000,60.,'silence')]
    for index,(name,fs,duration,kind) in enumerate(specifications):
        frames=round(fs*duration);t=np.arange(frames,dtype=np.float64)/fs
        rng=np.random.default_rng(20261026+index)
        if kind=='noise':x=rng.normal(0,.1,frames)
        elif kind=='DC':x=.25+rng.normal(0,.05,frames)
        elif kind=='silence':x=np.zeros(frames)
        elif kind=='stereo':x=np.column_stack([.4*np.sin(2*np.pi*100*t),.1*np.sin(2*np.pi*1000*t)])
        elif kind=='alternating_gain':x=np.where(t<30,.1,.9)*np.sin(2*np.pi*100*t)
        else:x=.4*np.sin(2*np.pi*float(kind.removeprefix('tone'))*t)
        wav=container(x,fs);rows.append(case(helper,name,wav));del x,wav,t;gc.collect()
    # Independent one-sided FFT-density/normalization arithmetic on one block.
    fs=32000;n=2*fs;x=np.random.default_rng(811).normal(.2,.1,n);z=x-x.mean()
    win=.5-.5*np.cos(2*np.pi*np.arange(n)/n);hz=np.fft.rfftfreq(n,1/fs)
    manual=np.abs(np.fft.rfft(z*win))**2/(fs*np.sum(win*win));manual[1:-1]*=2
    freq,scipy_psd=periodogram(x,fs=fs,window='hann',detrend='constant',return_onesided=True,scaling='density')
    arithmetic={'segment_samples':n,'nfft':n,'df_hz':float(hz[1]-hz[0]),
        'rfftfreq_equal':bool(np.array_equal(freq,hz)),'Hann_sum_w2_over_N':float(np.sum(win*win)/n),
        'manual_Hann_vs_scipy_window_max_abs':float(np.max(abs(win-get_window('hann',n)))),
        'manual_density_vs_periodogram_max_abs':float(np.max(abs(manual-scipy_psd))),
        'integral_all_frequency':float(manual.sum()*(fs/n)),
        'window_weighted_detrended_mean_square':float(np.sum((z*win)**2)/np.sum(win*win)),
        'PSD_float32_input_dtype':str(periodogram(x.astype(np.float32),fs=fs,window='hann',detrend='constant',scaling='density')[1].dtype),
        'PSD_float64_input_dtype':str(scipy_psd.dtype)}
    assert arithmetic['rfftfreq_equal'] and abs(arithmetic['integral_all_frequency']-arithmetic['window_weighted_detrended_mean_square'])<1e-12
    bands=[]
    for j,c in enumerate(CENTRES):
        low,high=c/10**.05,c*10**.05;bins=(hz>=low)&(hz<high)
        bands.append({'band':j,'centre_hz':float(c),'low_inclusive_hz':float(low),'high_exclusive_hz':float(high),
            'bins_at_df0p5':int(bins.sum()),'sum_PSD_times_df_div_mean_PSD':float(bins.sum()*.5)})
    with (OUT/'starter_conversion_band_bounds.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(bands[0]));w.writeheader();w.writerows(bands)
    # Metadata-only preselection: first verified original-TRAIN non-background
    # class with120..300s, before looking at conversion outcomes. No reselection.
    m=pd.read_parquet(ROOT/'data/versions/vtuad_full_r9/reference_manifest.parquet')
    q=m[(m.original_split=='train')&(m.weak_class!='background')&(m.duration_seconds>=120)&(m.duration_seconds<=300)
        &m.verified&m.measured&m.quality_full_band_good].sort_values('clip_id');chosen=q.iloc[0]
    base=ROOT/'external/vtuad_v1.3.2';parts=[base/f'vtuad_strict_wav_vessel.zip.part{i:02d}' for i in range(1,4)]
    archive_stats={str(p.relative_to(ROOT)):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in parts}
    reader=MultipartReader(parts)
    try:
        with zipfile.ZipFile(reader) as archive:raw=archive.read(chosen['name'])
    finally:reader.close()
    assert bytes_sha(raw)==chosen.sha256
    actual=case(helper,'actual_VTUAD_original_TRAIN_120s',raw)
    assert actual['duration_seconds']==120 and actual['sample_rate_hz']==32000
    segments=np.load(ART/'actual_VTUAD_original_TRAIN_120s.npz',allow_pickle=False)['local_2s_power']
    bank=np.load(ROOT/chosen.power_cache,mmap_mode='r',allow_pickle=False)
    cached=np.asarray(bank[int(chosen.segment_start):int(chosen.segment_start)+int(chosen.segment_count)])
    assert bytes_sha(np.ascontiguousarray(cached).tobytes())==chosen.expected_segment_slice_sha256
    actual.update(source_clip_id=chosen.clip_id,source_WAV_sha256=chosen.sha256,archive_member=chosen['name'],
        original_split=chosen.original_split,weak_class=chosen.weak_class,
        preselection_candidates=len(q),current_reference_index=int(chosen.reference_row_index),
        existing_cache_segments_bitwise_equal=bool(np.array_equal(segments,cached)),
        existing_cache_segments_max_abs=float(np.max(abs(segments.astype(float)-cached.astype(float)))),
        official_vs_existing_R9_cache=compare(np.load(ART/'actual_VTUAD_original_TRAIN_120s.npz')['official_minute_power'],native_minutes(cached)))
    rows.append(actual)
    assert before=={n:sha(ROOT/n) for n in inputs}
    assert archive_stats=={str(p.relative_to(ROOT)):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in parts}
    result={'format':'r10_starter_conversion_audit_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'completed_no_original_implementation_modified','notebook_helper_provenance':provenance,
        'source_sha256_before_and_after_identical':before,'code_sha256':sha(__file__),
        'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'soundfile':sf.__version__},
        'synthetic_case_count':len(specifications),'actual_source_case_count':1,'cases':rows,
        'independent_fft_arithmetic':arithmetic,'band_bounds':bands,
        'conversion_equations':{'PSD':'abs(rfft((x-mean(x))*periodic_Hann))^2/(fs*sum(Hann^2)); double interior one-sided bins',
            'band_energy':'sum(PSD bins with low<=f<high)*df','minute':'mean of30 complete2s band-energy rows in linear units',
            'band_grid':'10^(1.5+.1*j), edges centre*10^(+/- .05)','nfft':'2*actual integer fs; no zero padding or resampling',
            'log':'not taken inside either helper; linear averaging precedes downstream logarithm'},
        'expected_policy_differences':['Official helper accepts one channel only; our WAV reader explicitly chooses channel0.',
            'Official helper emits NaN for unsupported bands; ours rejects a WAV lacking any of25 bands.',
            'Official helper rejects <60s; external extraction retains complete2s rows but R9 native_minutes rejects <30rows.',
            'Official helper returns zero for silence; ours rejects zero-power segments/minutes.',
            'Official helper computes fromfloat64 samples; ours decodesfloat32, FFT PSD isfloat32 and storedbandpower isfloat32.'],
        'no_model_fit_Test_report_network_or_external_transmission':True,
        'source_audio_archives_numeric_bank_and_existing_cache_unchanged':True,
        'scope':'Parity with optional starter conversion does not verify SA export calibration or the hidden synthesis generator.',
        'elapsed_seconds':time.perf_counter()-start}
    save(OUT/'starter_conversion_audit.json',result)
    fields=['case','sample_rate_hz','duration_seconds','channels','official_status','external_status','R9_native_status',
            'significant_max_relative_error','significant_max_abs_db_error','max_absolute_power_error']
    with (OUT/'starter_conversion_cases.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:row.get(k,row.get('comparison',{}).get(k)) for k in fields})
    print(json.dumps({'status':result['status'],'cases':len(rows),'actual':actual,
        'elapsed_seconds':result['elapsed_seconds'],'report_sha256':sha(OUT/'starter_conversion_audit.json')},indent=2))


if __name__=='__main__':main()
