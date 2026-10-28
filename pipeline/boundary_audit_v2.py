"""Recheck training-only recorder boundaries; flags are not exclusion rules."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pipeline.build_quality_v2 import sha256
ROOT=Path(__file__).resolve().parents[1]

def main():
    md=pd.read_parquet(ROOT/'data/processed/sa_row_manifest.parquet')
    db=np.load(ROOT/'data/processed/sa_rows_db.npy',mmap_mode='r')
    order=md.sort_values(['site','recorder','t','row_id'],kind='stable').index.to_numpy()
    frame=md.iloc[order].copy();values=np.asarray(db[order]);t=pd.to_datetime(frame.t)
    second=t.to_numpy(dtype='datetime64[s]').astype('int64');minute=second//60
    eligible=(second%60==0)&np.isfinite(values).all(1)&~(values<0).any(1)
    peak=values.max(1);broadband=peak+10*np.log10((10**((values-peak[:,None])/10)).sum(1))
    flags=np.zeros(len(frame),bool);reports=[]
    for recorder,ix in frame.reset_index(drop=True).groupby('recorder',sort=True).groups.items():
        idx=np.asarray(ix);idx=idx[eligible[idx]]
        if not len(idx):continue
        start=np.r_[True,np.diff(minute[idx])!=1];end=np.r_[start[1:],True]
        run=np.cumsum(start);length=np.bincount(run)[run];multi=length>=3
        baseline=pd.Series(broadband[idx]).groupby(run).transform('median').to_numpy()
        excess=broadband[idx]-baseline
        first=float(np.median(excess[start&multi])) if (start&multi).any() else 0.
        last=float(np.median(excess[end&multi])) if (end&multi).any() else 0.
        if first>3:flags[idx[start]]=True
        if last>3:flags[idx[end]]=True
        edge=flags[idx];candidate=idx[edge]
        reports.append({'recorder':recorder,'runs':int(run.max()),'median_first_minute_excess_db':first,
            'median_last_minute_excess_db':last,'flagged_rows':int(edge.sum()),
            'flagged_rows_window_included':int(frame.iloc[candidate].window_included.sum()),
            'flagged_duration_quantiles':np.quantile(frame.iloc[candidate].dur,[0,.5,1]).tolist() if len(candidate) else [],
            'verified_instrument_transient':False})
    flagged=frame.loc[flags,['row_id','site','recorder','t','dur','src_file','window_included']].copy()
    flagged['candidate_reason']='Recorder-level median run-edge broadband excess >3dB; physical cause unverified'
    flagged['exclusion_authorized']=False;flagged.to_parquet(ROOT/'reports/data_quality_boundary_flags.parquet',index=False)
    prior=pd.read_csv(ROOT/'eda_deep/tables/run_edge_artifacts.csv')
    expected=dict(zip(prior.recorder,prior.masked_rows))
    assert all(q['flagged_rows']==expected[q['recorder']] for q in reports)
    assert len(flagged)==12643
    urn=frame.recorder.eq('URN_8321_8321.1.12000').to_numpy();uv=values[urn]
    # Exact band edges, not rounded display centres.
    centres=10**(1.5+.1*np.arange(25));upper=centres*10**.05;lower=centres/10**.05
    nominal_fs=frame.loc[urn,'sample_rate_hz'].unique().tolist()
    u={'rows':int(urn.sum()),'metadata_sample_rate_hz':nominal_fs,
        'nominal_nyquist_hz':6000,'bands_crossing_or_above_nominal_nyquist':[
            {'centre_hz':float(centres[i]),'lower_hz':float(lower[i]),'upper_hz':float(upper[i]),
             'level_min_median_max_db':np.quantile(uv[:,i],[0,.5,1]).tolist()} for i in np.flatnonzero(upper>6000)],
        'waveform_sampling_rate_verified':False,'correction_applied':False,
        'interpretation':'Nominal 12k metadata cannot support complete 6.31/7.94k bands; nonzero exported levels require provider provenance clarification, not inferred upsampling or blanket band removal.'}
    b=pd.read_parquet(ROOT/'data/versions/quality_v2_native/background_manifest.parquet')
    b['start_export_clock']=pd.to_datetime(b.start_export_clock)
    keys=flagged[flagged.window_included].copy();keys['start_export_clock']=pd.to_datetime(keys.t).dt.floor('15min')
    affected=b.merge(keys[['recorder','start_export_clock']].drop_duplicates(),on=['recorder','start_export_clock'],how='inner')
    report={'training_only':True,'challenge_test_accessed':False,'candidate_rows':len(flagged),
        'candidate_rows_already_in_frozen_windows':int(flagged.window_included.sum()),
        'affected_windows_by_current_source_split':affected.split.value_counts().to_dict(),
        'reproduces_existing_candidate_counts_exactly':True,'rows_excluded_by_this_audit':0,
        'instrument_cause_verified':False,'recorder_results':reports,'urn_8321_nominal_rate_warning':u,
        'source_sha256':{n:sha256(ROOT/'data/processed'/n) for n in ['sa_rows_db.npy','sa_row_manifest.parquet']},
        'recommendation':'Preserve flags and all frozen inputs; compare predeclared edge-retention ablations on inner folds only after obtaining provider acquisition details.'}
    (ROOT/'reports/data_quality_boundary_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['recorder_results','source_sha256']},indent=2))

if __name__=='__main__':main()
