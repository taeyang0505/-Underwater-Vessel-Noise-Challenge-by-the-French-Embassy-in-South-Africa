"""R20 CPU-only, exhaustive physical descriptions; no model or Test fitting.

The saved Test input is described in its own quarantined CSV directory. Native
VTUAD clips keep their measured duration; no stretching, looping or gap fill.
"""
from pathlib import Path
import gc, hashlib, json, os, sys, time, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r20_integrated_cpu_diagnostics_r1/physical_profiles'
FREQ=np.array([31.6,39.8,50.1,63.1,79.4,100,125.9,158.5,199.5,251.2,316.2,398.1,501.2,631,794.3,1000,1258.9,1584.9,1995.3,2511.9,3162.3,3981.1,5011.9,6309.6,7943.3])
GROUPS={'broadband':slice(0,25),'LF_31to200':slice(0,9),'MF_251to2512':slice(9,20),'HF_3162to7943':slice(20,25),'safe_1000to5012':slice(15,23),'low_31to501':slice(0,13)}
QS=[.1,.5,.9]
HASHES={}
COUNTS=[]
START=time.monotonic()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def register(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    key=str(p.relative_to(ROOT))
    if key not in HASHES:HASHES[key]={'sha256':sha(p),'bytes':p.stat().st_size}
    return p

def jwrite(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n')

def csv(path,rows):
    pd.DataFrame(rows).to_csv(path,index=False,float_format='%.9g')

def progress(**v):
    v.update(elapsed_seconds=round(time.monotonic()-START,2))
    print(json.dumps(v,ensure_ascii=False),flush=True);jwrite('progress.json',v)

def summaries(v):
    a=np.asarray(v,float);a=a[np.isfinite(a)]
    if not len(a):return {'n_valid':0,'mean':np.nan,'std':np.nan,'p10':np.nan,'p50':np.nan,'p90':np.nan}
    q=np.quantile(a,QS)
    return {'n_valid':len(a),'mean':a.mean(),'std':a.std(),'p10':q[0],'p50':q[1],'p90':q[2]}

def row_summaries(a,base,axis_label=None):
    a=np.asarray(a)
    if a.ndim==1:return [{**base,**summaries(a)}]
    return [{**base,axis_label:i,**summaries(a[:,i])} for i in range(a.shape[1])]

def corr_pair(x,y):
    m=np.isfinite(x)&np.isfinite(y);n=m.sum(1)
    xx=np.where(m,x,0.);yy=np.where(m,y,0.)
    sx=xx.sum(1);sy=yy.sum(1)
    cov=(xx*yy).sum(1)-sx*sy/np.maximum(n,1)
    vx=(xx*xx).sum(1)-sx*sx/np.maximum(n,1);vy=(yy*yy).sum(1)-sy*sy/np.maximum(n,1)
    den=np.sqrt(np.maximum(vx,0)*np.maximum(vy,0))
    return np.divide(cov,den,out=np.full(len(x),np.nan),where=(n>=3)&(den>1e-12)).clip(-1,1)

def band_correlations(a):
    """Pairwise-complete Pearson, >=3 shared minutes, nonconstant bands."""
    finite=np.isfinite(a)
    if np.all(finite==finite[:,:,:1]):
        n=finite[:,:,0].sum(1)
        mu=np.nansum(a,1)/np.maximum(n[:,None],1)
        c=np.where(finite,a-mu[:,None,:],0.)
        den=np.sqrt((c*c).sum(1))
        normalized=np.divide(c,den[:,None,:],out=np.zeros_like(c),where=den[:,None,:]>1e-12)
        r=np.einsum('ntf,ntg->nfg',normalized,normalized,optimize=True)
        valid=(n[:,None,None]>=3)&(den[:,:,None]>1e-12)&(den[:,None,:]>1e-12)
    else:
        m=finite.astype(float);x=np.where(finite,a,0.)
        n=np.einsum('ntf,ntg->nfg',m,m,optimize=True)
        sx=np.einsum('ntf,ntg->nfg',x,m,optimize=True);sy=sx.swapaxes(1,2)
        sxx=np.einsum('ntf,ntg->nfg',x*x,m,optimize=True)
        cov=np.einsum('ntf,ntg->nfg',x,x,optimize=True)-sx*sy/np.maximum(n,1)
        vx=sxx-sx*sx/np.maximum(n,1);vy=vx.swapaxes(1,2)
        den=np.sqrt(np.maximum(vx,0)*np.maximum(vy,0))
        valid=(n>=3)&(den>1e-12)
        r=np.divide(cov,den,out=np.zeros_like(cov),where=valid)
    return np.where(valid,np.clip(r,-1,1),np.nan).astype(np.float32)

def calculate(a):
    a=np.asarray(a,dtype=np.float64);finite=np.isfinite(a)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        med=np.nanmedian(a,axis=1);lo=np.nanmin(a,axis=1);hi=np.nanmax(a,axis=1)
        quant=np.nanquantile(a,[.1,.9],axis=1)
        diff=np.abs(np.diff(a,axis=1));adj=np.nanmean(diff,axis=1)
        offset=np.nanmax(a,axis=(1,2),keepdims=True)
        e=np.where(finite,10.**((a-offset)/10.),0.)
        total=e.sum((1,2));wf=e.sum(1)/total[:,None]
        band={'energy_fraction_wf':wf,'max_minus_median_db':hi-med,'median_minus_min_db':med-lo,
              'q90_minus_q10_db':quant[1]-quant[0],'temporal_std_db':np.nanstd(a,axis=1),
              'adjacent_valid_abs_delta_db_mean':adj}
        globalf={'n_obs':finite.any(2).sum(1),'n_complete25':finite.all(2).sum(1),
                 'finite_cell_fraction':finite.mean((1,2))}
        minutes={};curves={};persistence={}
        nobs=finite.any(2).sum(1)
        for name,sl in GROUPS.items():
            valid=finite[:,:,sl].any(2);p=e[:,:,sl].sum(2)
            curve=np.where(valid,10*np.log10(np.maximum(p,1e-300)),np.nan)
            curves[name]=curve
            frac=np.where(valid,p/total[:,None],np.nan)
            cmed=np.nanmedian(curve,axis=1)
            rel=curve-cmed[:,None]
            counts=valid.sum(1)
            minutes[name]={'energy_fraction':frac[:,:15],'energy_vs_observed_mean':(frac/np.maximum(p.sum(1)/total,1e-300)[:,None]*counts[:,None])[:,:15],
                           'curve_minus_temporal_median_db':rel[:,:15], 'observed':valid[:,:15]}
            globalf[name+'_energy_fraction']=p.sum(1)/total
            globalf[name+'_bump_db']=np.nanmax(curve,1)-cmed
            globalf[name+'_dynamic_range_db']=np.nanmax(curve,1)-np.nanmin(curve,1)
            globalf[name+'_temporal_std_db']=np.nanstd(curve,1)
            globalf[name+'_lag1_corr']=corr_pair(curve[:,:-1],curve[:,1:])
            persistence[name]=np.sum((rel>3)&valid,axis=1).astype(np.int16)
        for x,y in [('LF_31to200','MF_251to2512'),('LF_31to200','HF_3162to7943'),('MF_251to2512','HF_3162to7943'),('low_31to501','HF_3162to7943'),('LF_31to200','safe_1000to5012')]:
            globalf[x+'__'+y+'_corr']=corr_pair(curves[x],curves[y])
        for x,y in [('HF_3162to7943','LF_31to200'),('MF_251to2512','LF_31to200')]:
            globalf[x+'__'+y+'_energy_ratio']=globalf[x+'_energy_fraction']/np.maximum(globalf[y+'_energy_fraction'],1e-300)
        # Orthogonal first/second polynomial contrasts over log10 frequency.
        t=np.log10(FREQ);t-=t.mean();t2=t*t-(t*t).mean();t2-=t*(t2@t)/(t@t)
        relative=med-np.nanmedian(med,axis=1,keepdims=True)
        if np.isfinite(relative).all():
            globalf['relative_median_spectrum_slope_db_per_decade']=relative@t/(t@t)
            globalf['relative_median_spectrum_quadratic_db_per_decade2']=relative@t2/(t2@t2)
        else:
            slope=[];curv=[]
            for r in relative:
                m=np.isfinite(r)
                p=np.polyfit(t[m],r[m],2) if m.sum()>=3 else [np.nan]*3
                curv.append(p[0]);slope.append(p[1])
            globalf['relative_median_spectrum_slope_db_per_decade']=np.array(slope)
            globalf['relative_median_spectrum_quadratic_db_per_decade2']=np.array(curv)
    return band,globalf,minutes,persistence,band_correlations(a)

def profile(name,batches,description,expected=None,labels=None,quarantine=False):
    dest=OUT/('test_descriptive_only' if quarantine else 'public_and_synthetic')
    dest.mkdir(parents=True,exist_ok=True)
    bands={};globals_={};mins={};pers={};cor=[];total=0
    for a in batches:
        b,g,m,p,c=calculate(a);total+=len(a)
        for k,v in b.items():bands.setdefault(k,[]).append(v.astype(np.float32))
        for k,v in g.items():globals_.setdefault(k,[]).append(v.astype(np.float32))
        for gn,items in m.items():
            for metric,v in items.items():
                # Native clips shorter than 15 are explicitly missing, never 0.
                z=np.full((len(a),15),False if metric=='observed' else np.nan,dtype=bool if metric=='observed' else np.float32)
                z[:,:v.shape[1]]=v
                mins.setdefault((gn,metric),[]).append(z)
        for k,v in p.items():pers.setdefault(k,[]).append(v)
        cor.append(c)
        if total%32768<len(a):progress(dataset=name,processed=total,expected=expected)
    if expected is not None:assert total==expected,(name,total,expected)
    bandrows=[]
    for metric,parts in bands.items():
        values=np.concatenate(parts)
        for f in range(25):bandrows.append({'dataset':name,'metric':metric,'band':f,'frequency_hz':FREQ[f],'n_samples':total,**summaries(values[:,f])})
    csv(dest/(name+'_band_profiles.csv'),bandrows)
    globalv={k:np.concatenate(v) for k,v in globals_.items()}
    grows=[]
    slices={'all':np.ones(total,bool)}
    if labels is not None:
        assert len(labels)==total
        slices.update({'y_eq0':labels==0,'weak_0lt_y_le010':(labels>0)&(labels<=.1),'y_gt010':labels>.1})
    for sl,mask in slices.items():
        for metric,v in globalv.items():grows.append({'dataset':name,'slice':sl,'metric':metric,'n_samples':int(mask.sum()),**summaries(v[mask])})
    csv(dest/(name+'_global_profiles.csv'),grows)
    mrows=[]
    for (group,metric),parts in mins.items():
        values=np.concatenate(parts)
        for minute in range(15):
            stats=summaries(values[:,minute])
            if metric=='observed':stats['missing_rate']=1-values[:,minute].mean()
            mrows.append({'dataset':name,'group':group,'metric':metric,'minute':minute+1,'n_samples':total,**stats})
    csv(dest/(name+'_minute_profiles.csv'),mrows)
    prows=[]
    for group,parts in pers.items():
        values=np.concatenate(parts)
        for i in list(range(16))+['gt15']:
            n=int((values>15).sum()) if i=='gt15' else int((values==i).sum())
            prows.append({'dataset':name,'group':group,'minutes_above_own_median_plus3db':i,'count':n,'fraction':n/total})
    csv(dest/(name+'_persistence.csv'),prows)
    # Exact all-sample quantiles. 256,029 * 25 * 25 * FP32 = 640MB, freed
    # before the next source. No representative/reservoir sampling is used.
    cc=np.concatenate(cor)
    crows=[]
    for f in range(25):
        for h in range(25):crows.append({'dataset':name,'band_i':f,'frequency_i_hz':FREQ[f],'band_j':h,'frequency_j_hz':FREQ[h],**summaries(cc[:,f,h])})
    csv(dest/(name+'_band_correlations.csv'),crows)
    COUNT={'dataset':name,'samples':total,'description':description,'quarantine_Test_descriptives':quarantine,
           'samples_at_least3_observed':int((globalv['n_obs']>=3).sum()),'n_obs_p10_p50_p90':np.quantile(globalv['n_obs'],QS).tolist(),
           'all_25band_observed_minutes':int(globalv['n_complete25'].sum()),'finite_cell_fraction_mean':float(globalv['finite_cell_fraction'].mean())}
    COUNTS.append(COUNT)
    jwrite('dataset_counts.json',COUNTS)
    progress(dataset=name,processed=total,status='complete')
    del cc,cor,bands,globals_,mins,pers;gc.collect()
    return grows

def batches(a,size=1024):
    for st in range(0,len(a),size):yield a[st:st+size]

def sa_eligible_batches(meta,bank):
    flagtable=pd.read_parquet(register('reports/data_quality_boundary_flags.parquet'),columns=['row_id'])
    flagset=set(flagtable.row_id)
    counts=[]
    for key,h in meta.groupby(['site','recorder'],sort=False,observed=True):
        h=h.copy();h['minute']=pd.to_datetime(h.t).to_numpy(dtype='datetime64[m]').astype(np.int64)
        h['duration_distance']=abs(h.dur-60)
        h=h.sort_values(['duration_distance','t','row_id']).drop_duplicates('minute').sort_values('minute')
        minute=h.minute.to_numpy(np.int64);ids=h.row_id.to_numpy(np.int64)
        eligible=h.eligible_for_minute_window.to_numpy(bool);flagged=np.array([i in flagset for i in ids])
        starts=np.unique(np.concatenate([(minute//5)*5-z for z in [0,5,10]]))
        stat={'site':key[0],'recorder':key[1],'original_rows':len(meta[(meta.site==key[0])&(meta.recorder==key[1])]),
              'deduplicated_clock_rows':len(h),'nonempty_15min_stride5_candidates':len(starts),'full_clock15':0,'full_finite25_clock15':0,
              'all15_original_row_eligible':0,'any_boundary_flagged_candidate':0,'any_ineligible_present_candidate':0,
              'partial_clock_candidate':0,'eligible15_with_boundary_flags':0,'cross_source_file_eligible15':0}
        sf=pd.Categorical(h.src_file).codes
        for st in range(0,len(starts),1024):
            grid=starts[st:st+1024,None]+np.arange(15)
            ix=np.searchsorted(minute,grid);safe=np.minimum(ix,len(minute)-1)
            present=(ix<len(minute))&(minute[safe]==grid)
            selected=bank[ids[safe]]
            allpresent=present.all(1);allfinite=np.isfinite(selected).all((1,2))&allpresent
            good=allpresent&eligible[safe].all(1)
            hasflag=(flagged[safe]&present).any(1)
            stat['full_clock15']+=int(allpresent.sum());stat['full_finite25_clock15']+=int(allfinite.sum())
            stat['all15_original_row_eligible']+=int(good.sum());stat['any_boundary_flagged_candidate']+=int(hasflag.sum())
            stat['any_ineligible_present_candidate']+=int((~eligible[safe]&present).any(1).sum())
            stat['partial_clock_candidate']+=int((~allpresent).sum())
            stat['eligible15_with_boundary_flags']+=int((good&hasflag).sum())
            stat['cross_source_file_eligible15']+=int((good&((sf[safe].max(1)-sf[safe].min(1))>0)).sum())
            if good.any():yield selected[good]
        counts.append(stat)
    csv(OUT/'sa_all_window_eligibility_flags_by_recorder.csv',counts)
    agg=pd.DataFrame(counts).select_dtypes('number').sum().to_dict()
    assert int(agg['original_rows'])==1715779
    assert int(agg['all15_original_row_eligible'])==256029,agg
    jwrite('sa_all_window_counts.json',agg)

def sa_fixed_batches(bank,manifest):
    for st in range(0,len(manifest),1024):
        h=manifest.iloc[st:st+1024];a=np.array(bank[h.base_window_index.to_numpy(int)],copy=True)
        bits=h.boundary_mask_bits.to_numpy(int)
        mask=((bits[:,None]>>np.arange(15))&1).astype(bool)
        a[mask]=np.nan
        yield a

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'PROTOCOL.md').exists():raise RuntimeError('Write protocol before data reads')
    if (OUT/'COMPLETED.json').exists():raise FileExistsError('Completed profile output is immutable')
    register(__file__);register(OUT/'PROTOCOL.md')
    # Numeric invariance and masked-pair correlation tests on fixtures only.
    rng=np.random.default_rng(20261030);fixture=rng.normal(size=(4,15,25));fixture[1,0]=np.nan;fixture[2,2,4]=np.nan
    b1,g1,m1,p1,c1=calculate(fixture);b2,g2,m2,p2,c2=calculate(fixture+20)
    assert all(np.allclose(b1[k],b2[k],equal_nan=True) for k in b1)
    assert np.allclose(c1,c2,equal_nan=True)
    assert np.isclose(c1[2,3,4],np.corrcoef(fixture[2,np.isfinite(fixture[2,:,4]),3],fixture[2,np.isfinite(fixture[2,:,4]),4])[0,1])
    summary=[]
    # SA only: exact original eligibility, no Test-dependent selection.
    meta=pd.read_parquet(register('data/processed/sa_row_manifest.parquet'))
    for c in ['site','recorder','src_file']:meta[c]=meta[c].astype('category')
    raw=np.load(register('data/processed/sa_rows_db.npy'),mmap_mode='r',allow_pickle=False)
    assert len(meta)==len(raw)==1715779
    summary+=profile('SA_eligible15_stride5',sa_eligible_batches(meta,raw),'All 15 clock minutes satisfy original row eligibility; 5-minute stride, recorder clock join; overlapping, not independent',256029)
    del meta,raw;gc.collect()
    bank=np.load(register('data/processed/sa_windows_db.npy'),mmap_mode='r',allow_pickle=False)
    train=pd.read_parquet(register('data/versions/background_cv_r9/fold_00/train_manifest.parquet'))
    assert len(train)==78869
    summary+=profile('SA_primary78869_QC',sa_fixed_batches(bank,train),'All R9 Fold0 primary backgrounds; original bank plus boundary-minute masking, no synthesis',78869)
    # Whole native clips: complete 60s blocks only, variable measured length.
    refbank=np.load(register('data/processed/external_segments_power.npy'),mmap_mode='r',allow_pickle=False)
    sys.path.insert(0,str(ROOT));from pipeline.sampler_r9 import native_minutes
    native_metadata=[]
    for name,path,n in [('VTUAD509_nativeclip','data/versions/vtuad_full_r9/reference_manifest.parquet',509),('VTUAD16ambient_nativeclip','data/versions/vtuad_full_r9/ambient_control_manifest.parquet',16)]:
        frame=pd.read_parquet(register(path));assert len(frame)==n
        clips=[]
        for row in frame.itertuples():
            part=np.asarray(refbank[int(row.segment_start):int(row.segment_start)+int(row.segment_count)])
            if hasattr(row,'expected_segment_slice_sha256'):assert hashlib.sha256(np.ascontiguousarray(part).tobytes()).hexdigest()==row.expected_segment_slice_sha256
            m=native_minutes(part)
            with np.errstate(divide='ignore'):db=np.where(m>0,10*np.log10(m),np.nan)
            clips.append(db)
            native_metadata.append({'dataset':name,'clip_id':row.clip_id,'name':row.name,'weak_class':row.weak_class,'complete_native_minutes':len(m),'header_duration_seconds':row.duration_seconds,'first15_present_minutes':min(15,len(m)),'sample_rate_hz':row.sample_rate_hz,'sha256':row.sha256})
        aa=np.full((n,max(map(len,clips)),25),np.nan)
        for i,a in enumerate(clips):aa[i,:len(a)]=a
        summary+=profile(name,batches(aa),'One native measured clip per sample, full variable length; minute-position tables show only first 15, denominator uses full native duration; uncalibrated digital power',n)
        del aa,clips;gc.collect()
    csv(OUT/'native_clip_counts_lengths.csv',native_metadata)
    del refbank;gc.collect()
    for name,path,n in [('R16_D0_TRAIN','data/synthetic/r16_multisupport_v1/fold_00/train_D0.npz',61446),('R16_H0_TRAIN','data/synthetic/r16_multisupport_v1/fold_00/train_H0.npz',61446),('R18_D0_TRAIN','data/synthetic/r18_production_v1/train_D0.npz',60000),('R18_H0_TRAIN','data/synthetic/r18_production_v1/train_H0.npz',60000),('R19_vanilla2000','reports/r19_multislice_law_transfer_r1/law_panels/vanilla_addition.npz',2000)]:
        with np.load(register(path),allow_pickle=False) as z:
            a=z['db'];y=z['share']
            summary+=profile(name,batches(a),'All saved synthetic rows; no regeneration, normalization or feature fitting',n,labels=y)
        del a,y;gc.collect()
    # Last and quarantined: saved query input, not labels/predictions/raw CSV.
    with np.load(register('reports/r18_production_r1/finalization/inputs/query.npz'),allow_pickle=False) as z:
        assert set(z.files)=={'db','sample_id'}
        a=z['db']
        assert len(a)==1500
        profile('Test_saved_query1500',batches(a),'All 1500 saved input matrices; descriptive only, no labels, fitting, source matching, inference or candidate selection',1500,quarantine=True)
    csv(OUT/'public_and_synthetic'/'global_all_datasets.csv',summary)
    jwrite('input_sha256.json',HASHES)
    jwrite('checks.json',{'gain20db_invariance':True,'pairwise_missing_correlation_fixture':True,'all_datasets_exhaustive_no_sampling':True,'Test_descriptive_only_separate_outputs':True,'no_fitting_or_inference':True,'frequency_group_definition':{k:[int(v.start),int(v.stop)-1] for k,v in GROUPS.items()},'sample_unit_warning':'Native clips variable length; SA stride5 windows overlap; synthetic rows reuse recordings; none equals independent sample count.'})
    make_report()
    files={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.rglob('*') if p.is_file() and p.name not in {'COMPLETED.json','progress.json'}}
    jwrite('COMPLETED.json',{'status':'complete','elapsed_seconds':time.monotonic()-START,'files':files,'output_bytes':sum(x['bytes'] for x in files.values())})
    progress(status='complete',datasets=len(COUNTS),output_bytes=sum(x['bytes'] for x in files.values()))

def make_report():
    rows=pd.read_csv(OUT/'public_and_synthetic/global_all_datasets.csv')
    wanted=['LF_31to200_energy_fraction','MF_251to2512_energy_fraction','HF_3162to7943_energy_fraction','HF_3162to7943_bump_db','low_31to501_bump_db','LF_31to200__HF_3162to7943_corr']
    p=rows[(rows['slice']=='all')&rows.metric.isin(wanted)].pivot(index='dataset',columns='metric',values='p50')
    count=pd.DataFrame(COUNTS)[['dataset','samples','samples_at_least3_observed','all_25band_observed_minutes','quarantine_Test_descriptives']]
    text='# R20 대역·시간·상관구조 전수 물리 프로파일\n\n'
    text+='원본과 저장 배열의 기술 통계를 계산했다. 학습·추론·Test 목표값 역산 및 원음 매칭은 수행하지 않았다. Test 결과는 `test_descriptive_only/`에 따로 보존했다. 이 비교는 숨겨진 Test 음성/양성의 조건부 분포를 알려주지 않는다.\n\n'
    text+='## 실제 사용량과 표본 단위\n\n'+count.to_markdown(index=False)+'\n\n'
    text+='SA 256,029창은 원본 1,715,779행에 대한 녹음기별 5분 stride 15분 clock 창 중 15분 모두 기존 적격 행인 전수다. 창끼리 중첩되며 256,029개의 독립 녹음을 뜻하지 않는다. `SA_primary78869_QC`는 실제 R9 주 배경 풀의 경계 분 NaN 처리까지 반영한 별도 전수다.\n\n'
    text+='VTUAD는 509개 전체와 전용 ambient 16개를 모두 포함했다. 각 clip의 원래 길이를 보존했으며 반복/신축하지 않았다. 따라서 native clip 전체 지속 시간 통계와 15분 합성 창의 지속 시간 통계를 직접 동등한 표본으로 취급하면 안 된다. 첫 15분 위치별 표에는 길이 부족에 따른 결측을 구분해 보고한다.\n\n'
    text+='## 공개·합성 데이터의 창별 지표 중앙값\n\n'+p.round(6).to_markdown()+'\n\n'
    text+='LF=31.6–199.5Hz, MF=251.2–2511.9Hz, HF=3162.3–7943.3Hz이다. 기존 `bump_low_db`와 같은 정의인 `low_31to501_bump_db`는 31.6–501.2Hz를 사용한다. 모든 bump는 대역 합산 선형 파워의 분별 dB 곡선에서 max−시간 중앙값이다.\n\n'
    text+='## 세부 산출물 읽기\n\n- `*_band_profiles.csv`: 25대역별 정규화 에너지 분율, 시간 상승폭/하락폭/폭/표준편차/인접 분 차이의 창 간 p10/p50/p90.\n- `*_minute_profiles.csv`: 1–15분의 에너지 비중, 관측 분 평균 대비 에너지 비, 창 중앙값 대비 상대 dB 및 결측률.\n- `*_persistence.csv`: broadband/LF/MF/HF/안전 대역의 창 중앙값+3dB 초과 관측 분 수. native clip은 15분 초과 별도 표시.\n- `*_band_correlations.csv`: 창 내 유효 공통 관측 분 ≥3인 25×25 대역 Pearson 상관의 평균/표준편차/p10/p50/p90 및 유효 창 수.\n- `*_global_profiles.csv`: 대역군 에너지 분율·bump·시간 변화·lag1·대역군 상관·상대 스펙트럼 기울기/곡률. 합성은 전체/무추가/약한 양성/나머지 양성 기술 통계를 별도로 제공한다.\n- `sa_all_window_eligibility_flags_by_recorder.csv`, `sa_all_window_counts.json`: 전체 비어 있지 않은 clock 창 수와 부적격/경계/결측 조건별 개수.\n- `native_clip_counts_lengths.csv`: 525개 clip 길이·클래스·채널 sample rate·원본 SHA.\n- `input_sha256.json`, `COMPLETED.json`: 입력/출력 파일 SHA-256.\n\n'
    text+='## 해석 한계\n\nSA export의 dB와 VTUAD 비보정 디지털 mean-square 파워는 절대 물리 단위가 서로 검증되지 않았다. 본 표는 각 샘플 내부 정규화·차분·상관만 비교한다. 에너지 비중/시간 대비가 닮더라도 정답 전달성과 Public 성능이 보장되지는 않는다. 기술 통계로 Test의 원래 배경 B와 추가 성분 A를 분해하거나 Test 음성의 특징을 확정할 수 없다.\n'
    (OUT/'R20_PHYSICAL_PROFILES.md').write_text(text)

if __name__=='__main__':
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    main()
