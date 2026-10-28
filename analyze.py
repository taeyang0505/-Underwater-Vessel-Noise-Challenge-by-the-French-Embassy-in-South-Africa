"""Reproducible descriptive EDA. Does not fit/calibrate a model on Test.csv."""
from pathlib import Path
import os, sys, json, hashlib, platform, warnings
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/underwater-mpl')
if Path('/private/tmp/underwater-eda-deps').exists():
    sys.path.insert(0, '/private/tmp/underwater-eda-deps')
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RAW, EXT = ROOT/'data/raw', ROOT/'data/extracted'
OUT, FIG = ROOT/'eda/tables', ROOT/'eda/figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'figure.dpi': 140, 'savefig.dpi': 180, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})

def save_table(rows, name):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(OUT/f'{name}.csv', index=False)
    return df

def quant(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return dict(zip(['min','p01','p10','p50','p90','p99','max'],
                    map(float, np.quantile(x, [0,.01,.1,.5,.9,.99,1])))) if x.size else {}

def figsave(fig, name):
    fig.savefig(FIG/f'{name}.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)

def candidate_windows(t, good):
    """Overlapping 15-row windows, exactly 60 s apart within a recorder."""
    n=len(t)
    if n<15:return np.empty(0,dtype=int)
    times=pd.DatetimeIndex(t).as_unit('ns').asi8
    gaps=np.diff(times)!=60_000_000_000
    cg=np.r_[0,np.cumsum(~np.asarray(good,dtype=bool))]
    cb=np.r_[0,np.cumsum(gaps)]
    return np.flatnonzero((cg[15:]-cg[:-15]==0)&(cb[14:]-cb[:-14]==0))

def main():
    test=pd.read_csv(RAW/'Test.csv',dtype={'ID':str})
    sample=pd.read_csv(RAW/'sample_submission.csv',dtype={'ID':str})
    common=[c.split('_',1)[1] for c in test.columns[1:26]]
    freqs=np.array([float(c[4:]) for c in common])
    expected=[f'm{m:02d}_{c}' for m in range(1,16) for c in common]
    assert test.columns.tolist()==['ID']+expected
    assert sample.columns.tolist()==['ID','share_added_vessel','score_added_vessel']
    assert test.ID.is_unique and sample.ID.is_unique and set(test.ID)==set(sample.ID)
    checksum=[]
    for line in (EXT/'SHA256SUMS.txt').read_text().splitlines():
        h,rel=line.split(maxsplit=1)
        actual=hashlib.sha256((EXT/rel).read_bytes()).hexdigest()
        checksum.append({'file':rel,'expected':h,'actual':actual,'match':h==actual})
    assert all(x['match'] for x in checksum)
    save_table(checksum,'checksum_verification')
    inventory=[]
    for folder in [RAW,EXT]:
        for p in sorted(folder.rglob('*')):
            if p.is_file():
                size=p.stat().st_size
                inventory.append({'file':str(p.relative_to(ROOT)), 'bytes':size,
                                  'MB_decimal':size/1e6,'MiB':size/2**20,
                                  'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    inv=save_table(inventory,'file_inventory')
    schemas=[]; sites=[]; cols=[]; recorders=[]; recorder_bands=[]; bands=[]
    daily=[]; duration_counts=[]; sentinel=[]; samples=[]; allcommon=[]; rowmetrics=[]
    for path in sorted((EXT/'acoustics').glob('*.parquet')):
        df=pd.read_parquet(path)
        df['t']=pd.to_datetime(df['t'])
        site=df.site.iloc[0]; n=len(df)
        print(f'Processing {site}: {n:,}',flush=True)
        tol=[c for c in df if c.startswith('TOL_')]
        a=df[common].to_numpy(dtype=np.float32)
        finite=np.isfinite(a)
        spread=np.max(a,axis=1)-np.min(a,axis=1)
        sent=(a==-156.5).any(1)
        allsent=(a==-156.5).all(1)
        dur=df.dur.to_numpy()
        base=finite.all(1)&(dur>=55)&~sent
        screening=base&(spread>=6)
        allcommon.append(a.copy())
        rowmetrics.append(pd.DataFrame({'site':site,'dur':dur,'spread':spread,
                                       'sentinel':sent,'negative':(a<0).any(1)}))
        checks={
            'qc_duration_mismatch':int((df.qc_duration_ge55.to_numpy()!=(dur>=55)).sum()),
            'qc_finite_mismatch':int((df.qc_25band_finite.to_numpy()!=finite.all(1)).sum()),
            'qc_spread_mismatch':int((df.qc_25band_spread_ge6.to_numpy()!=(spread>=6)).sum())}
        s={'site':site,'rows':n,'columns':len(df.columns),'tol_columns':len(tol),
           'recorders':df.recorder.nunique(),'source_files':df.src_file.nunique(),
           'first_t':str(df.t.min()),'last_t':str(df.t.max()),
           'duplicate_site_t':int(df.t.duplicated().sum()),
           'missing_t':int(df.t.isna().sum()),
           'tol_missing_cells':int(df[tol].isna().sum().sum()),
           'tol_total_cells':n*len(tol),'common_missing_cells':int((~finite).sum()),
           'common_cells':n*25,'common_infinite_cells':int(np.isinf(a).sum()),
           'all_common_missing_rows':int((~finite).all(1).sum()),
           'partial_common_missing_rows':int(((~finite).any(1)&finite.any(1)).sum()),
           'duration_not60':int((dur!=60).sum()),'duration_lt55':int((dur<55).sum()),
           'duration_gt60':int((dur>60).sum()),'spread_lt6':int((spread<6).sum()),
           'exact_flat_rows':int((spread==0).sum()),'sentinel_rows':int(sent.sum()),
           'all_sentinel_rows':int(allsent.sum()),'negative_rows':int((a<0).any(1).sum()),
           'base_screen_rows':int(base.sum()),'stricter_screen_rows':int(screening.sum()),
           'raw_candidate_windows15':0,'base_candidate_windows15':0,
           'stricter_candidate_windows15':0, **checks}
        s.update({f'duration_{k}':v for k,v in quant(dur).items()})
        s.update({f'spread_{k}':v for k,v in quant(spread).items()})
        for field in pq.ParquetFile(path).schema_arrow:
            schemas.append({'site':site,'column':field.name,'dtype':str(field.type),'nullable':field.nullable})
        for c in df:
            x=df[c]
            cols.append({'site':site,'column':c,'rows':n,'nulls':int(x.isna().sum()),
                         'null_fraction':float(x.isna().mean()),'dtype':str(x.dtype),
                         'infinite':int(np.isinf(x).sum()) if pd.api.types.is_numeric_dtype(x) else 0})
        for c in common:
            bands.append({'site':site,'band':c,'frequency_hz':float(c[4:]),**quant(df[c])})
        counts=df.groupby(df.t.dt.floor('D')).size()
        daily.extend({'site':site,'date':str(k.date()),'rows':int(v)} for k,v in counts.items())
        duration_counts.extend({'site':site,'dur':float(k),'rows':int(v)} for k,v in df.dur.value_counts().sort_index().items())
        for rec,g in df.groupby('recorder',sort=False):
            g=g.sort_values('t',kind='stable')
            idx=g.index.to_numpy()
            dsec=g.t.diff().dt.total_seconds().dropna()
            rawstarts=candidate_windows(g.t,np.ones(len(g),dtype=bool))
            basestarts=candidate_windows(g.t,base[idx])
            strictstarts=candidate_windows(g.t,screening[idx])
            s['raw_candidate_windows15']+=len(rawstarts)
            s['base_candidate_windows15']+=len(basestarts)
            s['stricter_candidate_windows15']+=len(strictstarts)
            r={'site':site,'recorder':rec,'rows':len(g),'source_files':g.src_file.nunique(),
               'nominal_sample_rate_hz':int(g.sample_rate_hz.iloc[0]),
               'first_t':str(g.t.min()),'last_t':str(g.t.max()),
               'diff_eq60':int((dsec==60).sum()),'diff_lt60':int((dsec<60).sum()),
               'diff_gt60':int((dsec>60).sum()),'diff_gt3600':int((dsec>3600).sum()),
               'max_gap_seconds':float(dsec.max()) if len(dsec) else None,
               'duplicate_t':int(g.t.duplicated().sum()),'off_minute_rows':int((g.t.dt.second!=0).sum()),
               'duration_not60':int((g.dur!=60).sum()),'duration_lt55':int((g.dur<55).sum()),
               'spread_lt6':int((spread[idx]<6).sum()),'sentinel_rows':int(sent[idx].sum()),
               'negative_rows':int((a[idx]<0).any(1).sum()),
               'filename_error_rows':int(g.src_file.str.contains('ERROR',case=False).sum()),
               'finite_6309_7943_rows':int(np.isfinite(g[['TOL_6309.6','TOL_7943.3']]).all(axis=1).sum()),
               'raw_windows15':len(rawstarts),'base_windows15':len(basestarts),'stricter_windows15':len(strictstarts)}
            recorders.append(r)
            for c in tol:
                v=g[c].to_numpy();miss=~np.isfinite(v)
                recorder_bands.append({'site':site,'recorder':rec,'band':c,'frequency_hz':float(c[4:]),
                                       'rows':len(g),'missing':int(miss.sum()),'missing_fraction':float(miss.mean()),
                                       'all_missing':bool(miss.all()),'intermittent_missing':bool(miss.any() and not miss.all())})
            if len(basestarts):
                start=basestarts[len(basestarts)//2]
                samples.append({'site':site,'recorder':rec,'start':str(g.t.iloc[start]),
                                'values':a[idx[start:start+15]].tolist()})
        if sent.any() or (spread==0).any():
            chosen=df.loc[sent|(spread==0),['site','recorder','src_file','t','dur']+common].copy()
            chosen['common_spread']=spread[chosen.index]
            sentinel.append(chosen)
        sites.append(s)
        del df
    ss=save_table(sites,'site_summary')
    ss['row_fraction']=ss.rows/ss.rows.sum()
    ss.to_csv(OUT/'site_summary.csv',index=False)
    save_table(schemas,'schema');save_table(cols,'column_missingness')
    rr=save_table(recorders,'recorder_summary')
    rb=save_table(recorder_bands,'recorder_band_missingness')
    bs=save_table(bands,'common_band_site_stats')
    dd=save_table(daily,'daily_coverage')
    save_table(duration_counts,'duration_counts')
    if sentinel:save_table(pd.concat(sentinel,ignore_index=True),'flat_and_sentinel_rows')
    vals=np.concatenate(allcommon); del allcommon
    metrics=pd.concat(rowmetrics,ignore_index=True)
    global_bands=[]
    for j,c in enumerate(common):
        global_bands.append({'dataset':'training_all','band':c,'frequency_hz':freqs[j],**quant(vals[:,j])})
    testarr=test[expected].to_numpy(dtype=np.float32).reshape(-1,15,25)
    missing=~np.isfinite(testarr)
    for j,c in enumerate(common):
        global_bands.append({'dataset':'test_descriptive_only','band':c,'frequency_hz':freqs[j],**quant(testarr[:,:,j])})
    gb=save_table(global_bands,'common_band_global_stats')
    testwindow=pd.DataFrame({'ID':test.ID,'missing_cells':missing.sum((1,2)),
                             'missing_minutes':missing.all(2).sum(1),
                             'partial_missing_minutes':(missing.any(2)&~missing.all(2)).sum(1),
                             'observed_minutes':(~missing.all(2)).sum(1),
                             'min_db':np.nanmin(testarr,axis=(1,2)),
                             'max_db':np.nanmax(testarr,axis=(1,2))})
    save_table(testwindow,'test_window_quality')
    save_table([{'minute':m+1,'band':common[j],'frequency_hz':freqs[j],
                 'missing_windows':int(missing[:,m,j].sum()),'missing_fraction':float(missing[:,m,j].mean())}
                for m in range(15) for j in range(25)],'test_minute_band_missingness')
    patterns=[]
    for pattern,count in zip(*np.unique(missing.all(2),axis=0,return_counts=True)):
        patterns.append({'missing_minute_positions':','.join(str(i+1) for i in np.flatnonzero(pattern)) or 'none',
                         'windows':int(count)})
    save_table(patterns,'test_missing_patterns')
    totals={key:int(ss[key].sum()) for key in ['rows','tol_missing_cells','tol_total_cells','common_missing_cells',
                'common_cells','duration_not60','duration_lt55','duration_gt60','spread_lt6','exact_flat_rows',
                'sentinel_rows','negative_rows','raw_candidate_windows15','base_candidate_windows15',
                'stricter_candidate_windows15','qc_duration_mismatch','qc_finite_mismatch','qc_spread_mismatch']}
    summary={'analysis_date':'2026-10-24','training':totals,
             'training_shape':[int(ss.rows.sum()),int(ss['columns'].iloc[0])],
             'training_tol_columns':62,'training_common_shape':list(vals.shape),
             'site_count':len(ss),'recorder_groups':len(rr),'unique_recorder_ids':int(rr.recorder.nunique()),
             'global_first_t':ss.first_t.min(),'global_last_t':ss.last_t.max(),
             'test':{'shape':list(test.shape),'array_shape':list(testarr.shape),'missing_cells':int(missing.sum()),
                     'total_cells':missing.size,'missing_fraction':float(missing.mean()),
                     'windows_with_missing':int(missing.any((1,2)).sum()),'complete_windows':int((~missing.any((1,2))).sum()),
                     'whole_missing_minutes':int(missing.all(2).sum()),
                     'partial_missing_minutes':int((missing.any(2)&~missing.all(2)).sum()),
                     'all_missing_windows':int(missing.all((1,2)).sum()),
                     'duplicate_IDs':int(test.ID.duplicated().sum()),
                     'duplicate_acoustic_rows':int(test[expected].duplicated().sum()),
                     'infinite_cells':int(np.isinf(testarr).sum()),'min_db':float(np.nanmin(testarr)),
                     'max_db':float(np.nanmax(testarr))},
             'sample':{'shape':list(sample.shape),'ID_set_matches':True,'ID_order_matches':bool(test.ID.equals(sample.ID)),
                       'placeholder_values':{c:sample[c].unique().tolist() for c in sample.columns[1:]}},
             'sizes':{'raw_bytes':int(inv[inv.file.str.startswith('data/raw/')].bytes.sum()),
                      'extracted_bytes':int(inv[inv.file.str.startswith('data/extracted/')].bytes.sum()),
                      'common_float32_bytes':int(vals.nbytes),'test_float32_bytes':int(testarr.nbytes)},
             'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                            'pyarrow':pyarrow.__version__,'matplotlib':matplotlib.__version__},
             'scope':'Descriptive EDA only. No model fitting, calibration, pseudo-labels, or training examples from test inputs.'}
    (ROOT/'eda/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (ROOT/'eda/training_examples.json').write_text(json.dumps(samples,indent=2))
    # 1. Representation-wide missingness, with shared frequency ordering.
    missdf=pd.DataFrame(cols);tolcols=[c for c in missdf.column.unique() if c.startswith('TOL_')]
    p=missdf[missdf.column.isin(tolcols)].pivot(index='site',columns='column',values='null_fraction').reindex(index=ss.site,columns=tolcols)
    fig,ax=plt.subplots(figsize=(16,5.5))
    im=ax.imshow(p.to_numpy()*100,aspect='auto',vmin=0,vmax=100,cmap='viridis')
    ax.set_yticks(range(len(ss)),ss.site);ax.set_xticks(range(0,len(tolcols),3),[x[4:] for x in tolcols[::3]],rotation=60)
    ax.set_xlabel('Band centre (Hz; all 62 exported bands)');ax.set_title('Training missing values by site and frequency band')
    fig.colorbar(im,ax=ax,label='Missing cells (%)',pad=.01);figsave(fig,'01_training_missingness')
    # 2. Site imbalance and quality screening. Counts are observations, not minutes.
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    axes[0].barh(ss.site,ss.rows,color='#247b91');axes[0].invert_yaxis()
    axes[0].set_xlabel('Observation records');axes[0].set_title('Site sampling is highly uneven')
    for i,row in ss.iterrows():axes[0].text(row.rows+7000,i,f'{row.rows:,}',va='center',fontsize=8)
    axes[0].set_xlim(0,ss.rows.max()*1.22)
    for col,label,color in [('duration_lt55','Duration <55s','#c1692d'),('spread_lt6','25-band spread <6 dB','#704fa0'),('sentinel_rows','Contains -156.5 dB','#bd3d58')]:
        axes[1].plot(ss.site,100*ss[col]/ss.rows,'o-',label=label,color=color)
    axes[1].tick_params(axis='x',labelrotation=65);axes[1].set_ylabel('Records (%)');axes[1].set_title('Screening indicators overlap; none is a vessel label');axes[1].legend(fontsize=8)
    figsave(fig,'02_sites_and_quality')
    # 3. Test missingness structure.
    fig,axes=plt.subplots(1,3,figsize=(16,4),layout='constrained')
    im=axes[0].imshow(missing.mean(0).T*100,aspect='auto',cmap='magma',vmin=0)
    axes[0].set_xticks(range(0,15,2),range(1,16,2));axes[0].set_yticks(range(0,25,4),[f'{f:g}' for f in freqs[::4]])
    axes[0].set_xlabel('Relative minute');axes[0].set_ylabel('Band centre (Hz)');axes[0].set_title('Identical missingness across all 25 bands')
    fig.colorbar(im,ax=axes[0],label='Missing windows (%)')
    vc=testwindow.missing_minutes.value_counts().sort_index()
    axes[1].bar(vc.index,vc.values,color='#247b91');axes[1].set_xticks(vc.index)
    for k,v in vc.items():axes[1].text(k,v+10,str(v),ha='center')
    axes[1].set_ylim(0,vc.max()*1.18);axes[1].set_xlabel('Missing complete minutes / window');axes[1].set_ylabel('Windows');axes[1].set_title('775 complete; 725 with missing minutes')
    axes[2].bar(range(1,16),missing.all(2).sum(0),color='#c1692d');axes[2].set_xticks(range(1,16,2));axes[2].set_xlabel('Relative minute');axes[2].set_ylabel('Missing minute observations');axes[2].set_title('Missingness depends on minute position')
    figsave(fig,'03_test_missingness')
    # 4. Source-export dB distributions. Descriptive comparison, no calibration.
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    med=bs.pivot(index='site',columns='frequency_hz',values='p50').reindex(index=ss.site,columns=freqs)
    im=axes[0].imshow(med,aspect='auto',cmap='cividis');axes[0].set_yticks(range(len(ss)),ss.site)
    axes[0].set_xticks(range(0,25,4),[f'{f:g}' for f in freqs[::4]],rotation=45)
    axes[0].set_xlabel('Band centre (Hz)');axes[0].set_title('Site median levels vary strongly');fig.colorbar(im,ax=axes[0],label='Source-export dB')
    for dataset,label,color in [('training_all','Training (all observed rows)','#247b91'),('test_descriptive_only','Test (observed cells only)','#c1692d')]:
        g=gb[gb.dataset==dataset];axes[1].plot(g.frequency_hz,g.p50,label=label,color=color)
        axes[1].fill_between(g.frequency_hz,g.p10,g.p90,alpha=.15,color=color)
    axes[1].set_xscale('log');axes[1].set_xlabel('Band centre (Hz)');axes[1].set_ylabel('Source-export dB')
    axes[1].set_title('Median and 10th–90th percentiles; descriptive only');axes[1].legend(fontsize=8)
    figsave(fig,'04_spectral_distributions')
    # 5. Daily coverage: no interpolation across missing days.
    dd['date']=pd.to_datetime(dd.date)
    calendar=pd.date_range(dd.date.min(),dd.date.max(),freq='D')
    grid=dd.pivot(index='site',columns='date',values='rows').reindex(index=ss.site,columns=calendar).fillna(0)
    fig,ax=plt.subplots(figsize=(15,4.5),layout='constrained')
    im=ax.imshow(np.log1p(grid),aspect='auto',cmap='viridis',vmin=0)
    ax.set_yticks(range(len(ss)),ss.site)
    ticks=[i for i,t in enumerate(calendar) if t.month==1 and t.day==1]
    ax.set_xticks(ticks,[str(calendar[i].year) for i in ticks]);ax.set_xlabel('Original export clock; timezone unverified')
    ax.set_title('Discontinuous observation coverage across 2015–2025')
    fig.colorbar(im,ax=ax,label='log(1 + records/day)');figsave(fig,'05_daily_coverage')
    # 6. Independent training examples, one per recorder group.
    chosen=samples[:6]
    fig,axes=plt.subplots(2,3,figsize=(15,7),layout='constrained')
    for ax,item in zip(axes.flat,chosen):
        im=ax.imshow(np.array(item['values']).T,aspect='auto',origin='lower',cmap='cividis',vmin=35,vmax=130)
        ax.set_title(item['site']+'\n'+item['start'],fontsize=9)
        ax.set_xticks([0,7,14],[1,8,15]);ax.set_yticks([0,8,16,24],[f'{f:g}' for f in freqs[[0,8,16,24]]])
        ax.set_xlabel('Relative minute');ax.set_ylabel('Band centre (Hz)')
    fig.colorbar(im,ax=list(axes.flat),label='Source-export dB (display clipped to 35–130)')
    figsave(fig,'06_training_windows')
    # 7. Duration distribution, plus exact-flat ERROR-file rows.
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    bins=[0,10,30,55,59.99,60.01,61,120,361]
    hist,_=np.histogram(metrics.dur,bins=bins)
    labels=['<10','10–<30','30–<55','55–<60','60','>60–<61','61–<120','120–360']
    axes[0].bar(range(len(hist)),hist,color='#247b91');axes[0].set_yscale('log');axes[0].set_xticks(range(len(hist)),labels,rotation=45)
    axes[0].set_ylabel('Records (log scale)');axes[0].set_xlabel('Observed duration (seconds)');axes[0].set_title('A row is not always a complete minute')
    counts=metrics.groupby('site').sentinel.sum().reindex(ss.site)
    axes[1].barh(counts.index,counts.values,color='#bd3d58');axes[1].invert_yaxis()
    axes[1].set_xlabel('Rows containing -156.5 dB');axes[1].set_title('Finite-value QC does not catch sentinel-like values')
    figsave(fig,'07_duration_and_sentinel')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
