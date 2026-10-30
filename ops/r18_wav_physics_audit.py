"""Read-only R18 WAV/PSD/source-unit audit. No Test reads, fitting or matching."""
from pathlib import Path
import ast, hashlib, io, json, os, sys, time, zipfile, warnings
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'2')
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import periodogram
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.sampler_r9 import native_minutes
CENTRES=10.**(1.5+.1*np.arange(25))
OUT=ROOT/'reports/r18_fundamental_audit_r1/wav'

def multipart_reader(paths):
    # Load only the reviewed read-only ZIP adapter class; the audio module's
    # optional libsndfile binding is not installed in this isolated EDA env.
    p=ROOT/'pipeline/audio.py';tree=ast.parse(p.read_text())
    cls=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='MultipartReader']
    assert len(cls)==1
    ns={'np':np,'io':io,'Path':Path}
    exec(compile(ast.Module(body=cls,type_ignores=[]),str(p)+':MultipartReader-only','exec'),ns)
    return ns['MultipartReader'](paths)

def read_wav(raw):
    fs,x=wavfile.read(io.BytesIO(raw))
    channels=1 if x.ndim==1 else x.shape[1]
    frames=len(x);kind=str(x.dtype)
    if np.issubdtype(x.dtype,np.signedinteger):
        sample=x.astype(np.float64)/(2.**(np.iinfo(x.dtype).bits-1))
    elif x.dtype==np.dtype('uint8'):sample=(x.astype(np.float64)-128.)/128.
    elif np.issubdtype(x.dtype,np.floating):sample=x.astype(np.float64)
    else:raise ValueError('Unexpected WAV sample encoding '+kind)
    if sample.ndim==2:sample=sample[:,0]
    return {'actual_fs':int(fs),'actual_channels':int(channels),'actual_frames':frames,'actual_subtype':'scipy_dtype_'+kind},sample

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def save(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n')

def compare(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    valid=np.isfinite(a)&np.isfinite(b)
    sig=valid&(a>np.maximum(np.nanmax(a,axis=-1,keepdims=True)*1e-8,1e-25))
    r=b[sig]/a[sig]
    return {'max_abs_power_error':float(np.max(abs(a[valid]-b[valid]))),
            'significant_max_relative_error':float(np.max(abs(r-1))) if len(r) else 0.,
            'significant_max_abs_db_error':float(np.max(abs(10*np.log10(r)))) if len(r) else 0.,
            'finite_mask_equal':bool(np.array_equal(np.isfinite(a),np.isfinite(b))),
            'compared_cells':int(valid.sum()),'significant_cells':int(sig.sum())}

def helper():
    p=ROOT/'data/raw/maritime_starter_notebook.ipynb'
    b=json.loads(p.read_text()); found=[]
    for i,c in enumerate(b['cells']):
        s=''.join(c.get('source',[]))
        if c.get('cell_type')=='code' and 'def minute_band_power(' in s:found.append((i,s))
    assert len(found)==1
    i,s=found[0]; tree=ast.parse(s)
    assert len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef)
    assert tree.body[0].name=='minute_band_power'
    for n in ast.walk(tree):
        if isinstance(n,ast.ImportFrom):assert n.module=='scipy.signal' and [v.name for v in n.names]==['periodogram']
        assert not (isinstance(n,ast.Import))
        if isinstance(n,ast.Name):assert n.id not in {'open','eval','exec','compile','__import__','os','sys','subprocess'}
    ns={'np':np};exec(compile(tree,str(p)+':reviewednumericfunction','exec'),ns)
    (OUT/'official_helper_reviewed.py').write_text(s)
    return ns['minute_band_power'],b,i

def fractions(p):
    p=np.asarray(p,float);s=p.sum(-1)
    return p[...,:9].sum(-1)/s,p[...,20:25].sum(-1)/s

def generator_statistics():
    rows=[];paths=[]
    for role in ['TRAIN_D0','TRAIN_H0','local_gate','local_report','domain_holdout']:
        p=ROOT/'reports/r17_eda_audit_20261030/eda'/f'{role}.parquet';paths.append(p)
        x=pd.read_parquet(p)
        def summary(z,grouping,value):
            row={'dataset':role,'grouping':grouping,'value':value,'n':len(z),'share_mean':float(z.share.mean()),
                 'LF_energy_p50':float(z.LF_energy_fraction.median()),'HF_bump_p50':float(z.bump_high_db.median())}
            for col in ['active_minutes','q05_contaminated_band_fraction','q20_contaminated_band_fraction']:
                if col in z:row[col+'_p50']=float(z[col].median())
            rows.append(row)
        for scope,key in [('all',np.ones(len(x),bool)),('positive',x.share.gt(0)),('zero',x.share.eq(0))]:summary(x[key],'presence',scope)
        if role.startswith('TRAIN'):
            for col in ['operator','family']:
                for key,z in x[x.share>0].groupby(col):summary(z,col,key)
    pd.DataFrame(rows).to_csv(OUT/'declared_generator_observed_structure.csv',index=False)
    save(OUT/'generator_statistics_sources.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    a=pd.read_parquet(paths[0]);b=pd.read_parquet(paths[1]);assert np.array_equal(a.sample_id,b.sample_id)
    p=a.share.gt(0).to_numpy()
    save(OUT/'paired_D0_H0_geometry_effect.json',{
        'same_background_reference_operator_pairing':True,'positive_rows':int(p.sum()),
        'H0_minus_D0_share_mean':float((b.share-a.share).mean()),
        'H0_HF_bump_less_than_D0_positive_fraction':float((b.bump_high_db[p]<a.bump_high_db[p]).mean()),
        'H0_minus_D0_HF_bump_positive_p50_db':float(np.median(b.bump_high_db[p]-a.bump_high_db[p])),
        'H0_minus_D0_LF_fraction_positive_p50':float(np.median(b.LF_energy_fraction[p]-a.LF_energy_fraction[p]))})
    c=pd.read_csv(OUT/'all509_cache_units_audit.csv')
    path=ROOT/'data/synthetic/r16_multisupport_v1/fold_00/train_references.parquet';r=pd.read_parquet(path)
    values=[]
    for name,z in [('all509',c),('used60',c[c.reference_row_index.isin(r.reference_row_index)])]:
        values.append({'pool':name,'clips':len(z),'native_minutes':int(z.native_complete_minutes.sum()),
            'duration_hours':float(z.duration_seconds.sum()/3600),
            'LF_fraction_clip_median':float(z.LF_energy_fraction_integrated.median()),
            'HF_fraction_clip_median':float(z.HF_energy_fraction_integrated.median())})
    pd.DataFrame(values).to_csv(OUT/'reference_pool_spectral_summary.csv',index=False)
    save(OUT/'reference_selection_source_SHA.json',{str(path.relative_to(ROOT)):sha(path)})

def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    sources=['data/raw/maritime_starter_notebook.ipynb','pipeline/audio.py','pipeline/sampler_r9.py',
             'pipeline/prepare_external.py','ops/r16_forward_data.py','pipeline/r16_multisupport_data.py',
             'data/versions/vtuad_full_r9/reference_manifest.parquet','data/processed/external_segments_power.npy',
             'data/processed/external_minutes_power.npy','scratch/discussions/thread_bodies.json']
    before={n:sha(ROOT/n) for n in sources}
    def guard(event,args):
        if event=='socket.connect':raise PermissionError('No network in numerical audit')
        if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
            n=Path(args[0]).name.lower()
            if n in ('test.csv','samplesubmission.csv') or 'query_predictions' in n:
                raise PermissionError('No Test/query access in numerical audit')
    sys.addaudithook(guard)
    official,book,cell=helper()
    m=pd.read_parquet(ROOT/sources[6]);assert len(m)==509 and m.sha256.nunique()==509
    bank=np.load(ROOT/'data/processed/external_segments_power.npy',mmap_mode='r')
    minute_bank=np.load(ROOT/'data/processed/external_minutes_power.npy',mmap_mode='r')
    # Exact numeric band widths at the official df=.5Hz; PSD-mean is a different representation.
    hz=np.fft.rfftfreq(64000,1/32000);width=[]
    for j,c in enumerate(CENTRES):
        ix=(hz>=c/10**.05)&(hz<c*10**.05)
        width.append({'band':j,'centre_hz':float(c),'low_hz':float(c/10**.05),'high_hz':float(c*10**.05),
                      'n_fft_bins':int(ix.sum()),'effective_integrated_width_hz':float(ix.sum()*.5),
                      'integration_minus_PSDmean_db':float(10*np.log10(ix.sum()*.5))})
    pd.DataFrame(width).to_csv(OUT/'band_filterbank_units.csv',index=False)
    widths=np.array([r['effective_integrated_width_hz'] for r in width])
    cached=[]
    for r in m.itertuples():
        a=np.asarray(bank[r.segment_start:r.segment_start+r.segment_count])
        h=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        assert h==r.expected_segment_slice_sha256
        native=native_minutes(a)
        mb=np.asarray(minute_bank[r.minute_start:r.minute_start+r.minute_count])
        cmp=compare(native,mb)
        p=native.sum(0);lf,hf=fractions(p);wronglf,wronghf=fractions(p/widths)
        cached.append({'reference_row_index':int(r.reference_row_index),'clip_id':r.clip_id,
            'duration_seconds':float(r.duration_seconds),'sample_rate_hz':int(r.sample_rate_hz),
            'native_complete_minutes':len(native),'trailing_seconds_lost_at_minute_stage':float(r.duration_seconds-60*len(native)),
            'segment_sha_matches':True,'LF_energy_fraction_integrated':float(lf),'HF_energy_fraction_integrated':float(hf),
            'LF_energy_fraction_if_wrong_PSDmean':float(wronglf),'HF_energy_fraction_if_wrong_PSDmean':float(wronghf),
            'weak_class':r.weak_class,**{'native_vs_minute_bank_'+k:v for k,v in cmp.items()}})
    pd.DataFrame(cached).to_csv(OUT/'all509_cache_units_audit.csv',index=False)
    # Entire509 WAV corpus: every complete minute is recomputed by official
    # float64 helper and compared to cached native minute matrices.
    full_ids=set(m.clip_id)
    parts=[ROOT/f'external/vtuad_v1.3.2/vtuad_strict_wav_vessel.zip.part{i:02d}' for i in range(1,4)]
    astats={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in parts}
    reader=multipart_reader(parts);rows=[]
    try:
        with zipfile.ZipFile(reader) as z:
            for i,r in enumerate(m.sort_values('clip_id').itertuples()):
                raw=z.read(r.name)
                assert hashlib.sha256(raw).hexdigest()==r.sha256
                header,sample=read_wav(raw)
                assert header['actual_fs']==r.sample_rate_hz and header['actual_frames']==r.frames and header['actual_channels']==r.channels
                if r.clip_id not in full_ids:sample=sample[:60*header['actual_fs']]
                out=official(sample,header['actual_fs'])
                cache=native_minutes(bank[r.segment_start:r.segment_start+r.segment_count])[:len(out)]
                cmp=compare(out,cache)
                row={'clip_id':r.clip_id,'reference_row_index':int(r.reference_row_index),'raw_WAV_sha_matches':True,
                     **header,'recomputed_minutes':len(out),'full_duration_recomputed':r.clip_id in full_ids,**cmp}
                rows.append(row)
                if i%25==0 or i==len(m)-1:
                    print(json.dumps({'completed_WAVs':i+1,'total':len(m),'elapsed_seconds':time.perf_counter()-started}),flush=True)
                    save(OUT/'progress.json',{'completed':i+1,'total':len(m)})
    finally:reader.close()
    pd.DataFrame(rows).to_csv(OUT/'all509_fresh_WAV_helper_vs_cache.csv',index=False)
    # Independent FFT arithmetic and white-noise band-width fixture.
    fs=32000;n=2*fs;rng=np.random.default_rng(2026103001);x=rng.normal(.2,.1,n)
    win=.5-.5*np.cos(2*np.pi*np.arange(n)/n);a=x-x.mean()
    manual=abs(np.fft.rfft(a*win))**2/(fs*(win*win).sum());manual[1:-1]*=2
    freq,psd=periodogram(x,fs=fs,window='hann',detrend='constant',scaling='density')
    parseval=float(np.sum((a*win)**2)/(win*win).sum())
    fft={'df_hz':fs/n,'manual_vs_scipy_max_abs':float(np.max(abs(manual-psd))),
         'integrated_psd':float(manual.sum()*(fs/n)),'window_weighted_mean_square':parseval,
         'parseval_error':abs(float(manual.sum()*(fs/n))-parseval),
         'highest_band_upper_hz':float(CENTRES[-1]*10**.05),
         'minimum_integer_sample_rate_all25':int(np.ceil(2*CENTRES[-1]*10**.05)),
         'white_PSD_integrated_last_over_first':float(widths[-1]/widths[0]),
         'white_PSD_last_vs_first_db':float(10*np.log10(widths[-1]/widths[0]))}
    save(OUT/'independent_PSD_arithmetic.json',fft)
    # A constant gain is unidentifiable for uncalibrated WAV and intentionally discarded.
    t=np.arange(60*fs)/fs;signal=.04*np.sin(2*np.pi*100*t)+.01*np.sin(2*np.pi*3000*t)
    gains=[];base=official(signal,fs)
    for gain in [.01,.1,1.,10.]:
        out=official(signal*gain,fs);norm=out/out.sum()
        gains.append({'amplitude_gain':gain,'expected_power_factor':gain**2,
                      'power_scale_max_error':float(np.max(abs(out-base*gain**2))),
                      'normalized_power_gain_invariance_max_error':float(np.max(abs(norm-base/base.sum())))})
    pd.DataFrame(gains).to_csv(OUT/'digital_gain_invariance.csv',index=False)
    # Read documentation as source material; never execute notebook loading/submission cells.
    notices=[]
    for i,c in enumerate(book['cells']):
        s=''.join(c.get('source',[]))
        if c.get('cell_type')=='markdown' and i in [2,5,9,10]:notices.append({'cell':i,'text':s})
    save(OUT/'official_notebook_notices.json',notices)
    discussions=json.loads((ROOT/'scratch/discussions/thread_bodies.json').read_text())
    rules=[{'id':d['id'],'url':d['url'],'complete':d['complete'],'body':d['body']} for d in discussions if d['id'] in [35368,35370]]
    save(OUT/'official_rules_archived.json',rules)
    generator_statistics()
    assert before=={n:sha(ROOT/n) for n in sources}
    assert astats=={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in parts}
    frame=pd.DataFrame(rows);cache=pd.DataFrame(cached)
    result={'status':'complete','source_code_sha256':sha(__file__),'source_SHA256_unchanged':before,
        'raw_WAVs_SHA_and_header_verified':len(frame),'all509_first_complete_minute_recomputed':True,
        'full_duration_recomputed_clip_count':len(full_ids),'recomputed_complete_minutes':int(frame.recomputed_minutes.sum()),
        'maximum_significant_relative_helper_vs_cache_error':float(frame.significant_max_relative_error.max()),
        'maximum_significant_abs_dB_helper_vs_cache_error':float(frame.significant_max_abs_db_error.max()),
        'all509_segment_slice_SHA_verified':bool(cache.segment_sha_matches.all()),
        'full25band_header_supported':bool((frame.actual_fs/2>=CENTRES[-1]*10**.05).all()),
        'all509_native_vs_minute_cache_max_relative':float(cache.native_vs_minute_bank_significant_max_relative_error.max()),
        'LF_fraction_p50_integrated':float(cache.LF_energy_fraction_integrated.median()),
        'LF_fraction_p50_if_wrong_PSDmean':float(cache.LF_energy_fraction_if_wrong_PSDmean.median()),
        'HF_fraction_p50_integrated':float(cache.HF_energy_fraction_integrated.median()),
        'HF_fraction_p50_if_wrong_PSDmean':float(cache.HF_energy_fraction_if_wrong_PSDmean.median()),
        'all509_trailing_seconds_at_complete_minute_stage':float(cache.trailing_seconds_lost_at_minute_stage.sum()),
        'helper_notebook_cell':cell,'independent_FFT':fft,
        'Test_accessed':False,'model_fitting':False,'new_generator_parameters':False,
        'raw_decoder':'scipy.io.wavfile; signed PCM divided by 2^(container_bits-1), uint8 centered128, float kept; channel0',
        'official_live_browse':'35368/35370/rules open failed using web tool; timestamped complete local archives used, not claimed freshly reverified',
        'elapsed_seconds':time.perf_counter()-started}
    save(OUT/'RESULTS.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
