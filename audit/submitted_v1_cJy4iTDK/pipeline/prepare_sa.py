"""Preserve every SA row; construct clock-aligned masked windows without gap filling."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed';OUT.mkdir(parents=True,exist_ok=True)
FREQS=10**(1.5+.1*np.arange(25))
COMMON=['TOL_'+x for x in ['31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5','251.2','316.2','398.1','501.2','631','794.3','1000','1258.9','1584.9','1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3']]

def connected_sensor_groups(recorders):
    """Keep sites linked by a reused recorder identifier in one evaluation group."""
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for row in recorders.itertuples():
        a,b=find('site:'+row.site),find('rec:'+row.recorder)
        parent[a]=b
    roots=sorted({find('site:'+x) for x in recorders.site},
                 key=lambda y:min(int(x.split()[0]) for x in recorders.site if find('site:'+x)==y))
    return {x:roots.index(find('site:'+x)) for x in recorders.site}

def main():
    rec=pd.read_csv(ROOT/'data/extracted/recorders.csv')
    groups=connected_sensor_groups(rec)
    paths=sorted((ROOT/'data/extracted/acoustics').glob('*.parquet'))
    total=int(rec.rows.sum())
    db=np.lib.format.open_memmap(OUT/'sa_rows_db.npy',mode='w+',dtype='float32',shape=(total,25))
    metadata=[];window_arrays=[];window_meta=[];offset=0;input_hashes={}
    for path in paths:
        print('Preparing',path.name,flush=True)
        df=pd.read_parquet(path,columns=['site','recorder','src_file','t','dur','sample_rate_hz',
                                          'qc_duration_ge55','qc_25band_finite','qc_25band_spread_ge6']+COMMON)
        n=len(df);a=df[COMMON].to_numpy(np.float32);db[offset:offset+n]=a
        t=pd.to_datetime(df.t);finite=np.isfinite(a).all(1)
        sentinel=(a==-156.5).any(1);negative=(a<0).any(1)
        duration=df.dur.to_numpy()
        # Screening is explicitly separate from a semantic vessel label.
        usable=finite&~sentinel&(duration>=55)&(duration<=65)&(t.dt.second.to_numpy()<=5)
        md=df.drop(columns=COMMON).copy()
        md['row_id']=np.arange(offset,offset+n,dtype=np.int64)
        md['source_row_index']=np.arange(n,dtype=np.int32)
        md['source_parquet']=path.name
        md['sensor_group']=md.site.map(groups).astype('int8')
        md['sentinel_like']=sentinel;md['negative_level']=negative
        md['eligible_for_minute_window']=usable
        md['label_type']='unlabelled_observation'
        md['window_included']=False
        md['pseudo_label_status']='pending_external_teacher'
        md.loc[~finite|sentinel,'pseudo_label_status']='invalid_measurement_keep_row'
        for recorder,g in df.groupby('recorder',sort=False):
            ix=g.index.to_numpy();ix=ix[usable[ix]]
            if len(ix)==0:continue
            work=pd.DataFrame({'local_row':ix,'t':t.iloc[ix].to_numpy(),
                               'duration_distance':np.abs(duration[ix]-60)})
            work['minute']=work.t.dt.floor('min')
            # Do not double-count split boundary exports as independent full minutes.
            work=work.sort_values(['duration_distance','t','local_row']).drop_duplicates('minute')
            work=work.sort_values('minute')
            work['start']=work.minute.dt.floor('15min')
            keys,unique=pd.factorize(work.start,sort=True)
            minutes=((work.minute-work.start).dt.total_seconds()/60).to_numpy(int)
            arr=np.full((len(unique),15,25),np.nan,dtype=np.float32)
            arr[keys,minutes]=a[work.local_row.to_numpy()]
            observed=np.isfinite(arr).all(2).sum(1)
            keep=observed>=3
            ids=np.flatnonzero(keep);kept_lookup={x:i for i,x in enumerate(ids)}
            rows_in_kept=work.local_row.to_numpy()[keep[keys]]
            md.loc[rows_in_kept,'window_included']=True
            site=g.site.iloc[0];group=groups[site]
            for key in ids:
                start=pd.Timestamp(unique[key])
                # Whole export-clock days are isolated across train/validation/calibration.
                token=f'{site}|{recorder}|{start.date()}'
                bucket=int(hashlib.sha256(token.encode()).hexdigest()[:8],16)%10
                split='domain_holdout' if group==5 else 'validation' if bucket==0 else 'calibration' if bucket==1 else 'train'
                window_meta.append({'window_id':f'SA_{offset}_{len(window_meta):08d}',
                                    'site':site,'recorder':recorder,'start_export_clock':str(start),
                                    'sensor_group':group,'split':split,'observed_minutes':int(observed[key])})
            window_arrays.append(arr[keep])
        metadata.append(md);offset+=n
        input_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert offset==total==1715779
    db.flush()
    md=pd.concat(metadata,ignore_index=True)
    assert md.row_id.is_unique and md.row_id.tolist()==list(range(total))
    md.to_parquet(OUT/'sa_row_manifest.parquet',index=False,compression='zstd')
    windows=np.concatenate(window_arrays)
    wm=pd.DataFrame(window_meta)
    assert len(windows)==len(wm)
    np.save(OUT/'sa_windows_db.npy',windows)
    wm.to_parquet(OUT/'sa_window_manifest.parquet',index=False,compression='zstd')
    # Test is deliberately not read by this preparation path.
    info={'rows_preserved':total,'row_array_shape':list(db.shape),
          'eligible_rows':int(md.eligible_for_minute_window.sum()),
          'rows_in_windows':int(md.window_included.sum()),'window_shape':list(windows.shape),
          'window_split_counts':wm.split.value_counts().to_dict(),'sensor_groups':groups,
          'input_sha256':input_hashes,'clock':'original export clock; timezone unverified',
          'window_rule':'15-minute fixed bins; >=3 observed full-ish minutes; gaps remain NaN; no interpolation',
          'semantic_labels':'none; model outputs are weak pseudo-scores, never verified natural-vessel labels',
          'test_accessed':False}
    (OUT/'sa_preparation_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
