"""Exhaustive read-only public SA lineage / unfiltered clock-window audit.

No fitting, label inference, source matching to Test, or source mutation.
Outputs are compact CSV(.gz), JSON and Markdown; window tensors exist only
for the current <=1024-window batch. All stored Test features are reused.
"""
from pathlib import Path
import gc, hashlib, json, os, sys, warnings
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.prepare_sa import COMMON
from ops.r17_audit_common import physical_features

OUT = ROOT / 'reports/r18_public_gap_audit_r1/background'
Q = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
FEATURES = ['n_obs', 'n_full25', 'LF_energy_fraction', 'bump_high_db', 'bump_low_db']
INPUTS = {}
CHECKS = {}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def save(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda v: v.item() if isinstance(v, np.generic) else str(v))+'\n')

def csv(name, value):
    pd.DataFrame(value).to_csv(OUT/name, index=False,
        compression={'method':'gzip', 'compresslevel':3, 'mtime':0} if name.endswith('.gz') else None)

def read(name):
    p = ROOT/name
    INPUTS[name] = sha(p)
    return pd.read_parquet(p)

def guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        name = Path(os.fsdecode(args[0])).name.lower()
        if name in {'test.csv','sample_submission.csv','query_predictions.npz'}:
            raise PermissionError('Use saved Test input features only; no raw Test, submissions or query predictions')

def note(**v):
    print(json.dumps(v, ensure_ascii=False), flush=True)
    save('progress.json', v)

def feature_batch(a):
    """Exactly the R17 power-sum/max-minus-temporal-median definitions.

    xarray carries the three physical axes for finite masks and energy sums;
    NumPy handles tiny temporal medians, matching the existing EDA formula.
    """
    a = np.asarray(a, dtype=np.float64)
    x = xr.DataArray(a, dims=('window','minute','band'))
    finite = np.isfinite(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        off = np.nanmax(a, axis=(1,2), keepdims=True)
        e = xr.DataArray(np.where(np.isfinite(a), 10**((a-off)/10), 0), dims=x.dims)
        den = e.sum(('minute','band')).values
        lf = e.isel(band=slice(0,9)).sum(('minute','band')).values
        result = {'n_obs':finite.any('band').sum('minute').values.astype('int8'),
                  'n_full25':finite.all('band').sum('minute').values.astype('int8'),
                  'LF_energy_fraction':np.divide(lf, den, out=np.full(len(a),np.nan), where=den>0)}
        for name, lo, hi in [('bump_low_db',0,13), ('bump_high_db',20,25)]:
            b=a[:,:,lo:hi]; o=np.isfinite(b); base=np.nanmax(b, axis=(1,2), keepdims=True)
            s=np.where(o,10**((b-base)/10),0).sum(2)
            curve=np.where(o.any(2),10*np.log10(np.maximum(s,1e-300))+base[:,0,0,None],np.nan)
            result[name]=np.nanmax(curve,axis=1)-np.nanmedian(curve,axis=1)
    return pd.DataFrame(result)

def features(a):
    return pd.concat([feature_batch(a[i:i+1024]) for i in range(0,len(a),1024)],ignore_index=True)

def add_conditions(f):
    f['HF_ge16'] = f.bump_high_db.ge(16)
    f['LF_le058'] = f.LF_energy_fraction.le(.58)
    f['HF_ge16_LF_le058'] = f.HF_ge16 & f.LF_le058
    f['HF12to20_LF048to068'] = f.bump_high_db.between(12,20) & f.LF_energy_fraction.between(.48,.68)
    f['LF_bump_ge068'] = f.bump_low_db.ge(6.8)
    return f

def summary(f, label, groupcols=()):
    rows=[]
    iterator=f.groupby(list(groupcols), observed=True, dropna=False, sort=True) if groupcols else [((),f)]
    for key,g in iterator:
        if not isinstance(key,tuple):key=(key,)
        common={'dataset':label, **dict(zip(groupcols,key)), 'windows':len(g)}
        for col in FEATURES:
            vals=g[col].to_numpy(float);vals=vals[np.isfinite(vals)]
            row={**common,'feature':col,'n_valid':len(vals),'missing':len(g)-len(vals)}
            if len(vals):
                row.update({'mean':vals.mean(),'std':vals.std(),**{f'p{int(q*100):02d}':v for q,v in zip(Q,np.quantile(vals,Q))}})
            rows.append(row)
    return rows

def conditions(f,label,groupcols=()):
    rows=[]
    iterator=f.groupby(list(groupcols), observed=True, dropna=False, sort=True) if groupcols else [((),f)]
    cols=['HF_ge16','LF_le058','HF_ge16_LF_le058','HF12to20_LF048to068','LF_bump_ge068']
    for key,g in iterator:
        if not isinstance(key,tuple):key=(key,)
        for c in cols:
            rows.append({'dataset':label,**dict(zip(groupcols,key)),'condition':c,'windows':len(g),
                         'count':int(g[c].sum()),'fraction':float(g[c].mean())})
    return rows

def ledger():
    wm=read('data/processed/sa_window_manifest.parquet')
    wm['base_window_index']=np.arange(len(wm))
    wm['start_minute']=pd.to_datetime(wm.start_export_clock).to_numpy(dtype='datetime64[m]').astype('int64')
    wm['prepare_split']=wm['split']
    quality=read('data/versions/quality_v2_native/background_manifest.parquet')
    from pipeline.build_quality_v2 import purged_split
    rebuilt=purged_split(wm.drop(columns=['base_window_index','start_minute','prepare_split']),42,24)
    CHECKS['quality_v2_split_rebuilt_exact'] = rebuilt.split.equals(quality.split)
    assert CHECKS['quality_v2_split_rebuilt_exact']
    wm['quality_v2_split']=quality.split.values
    parent=read('data/synthetic/group_cv_passage_r4_v1/fold_00/sources/background_manifest.parquet')
    pool=read('data/versions/background_cv_r9/background_pool_manifest.parquet')
    assert len(pool)==100965 and not pool.base_window_index.duplicated().any()
    assert set(parent.base_window_index)==set(pool.base_window_index)==set(quality.loc[quality.split.isin(['train','validation']),'base_window_index'])
    wm['R9_allowed']=wm.base_window_index.isin(pool.base_window_index)
    wm['R4_historical_split']=wm.base_window_index.map(parent.set_index('base_window_index').split).fillna('outside_R4_allowed')
    roleparts=[]
    for role in ['train','selection','ensemble_fit','ensemble_gate','report','excluded']:
        r=read(f'data/versions/background_cv_r9/fold_00/{role}_manifest.parquet')
        roleparts.append(r)
    roles=pd.concat(roleparts,ignore_index=True)
    assert len(roles)==len(pool) and roles.base_window_index.is_unique
    wm['R9_fold0_role']=wm.base_window_index.map(roles.set_index('base_window_index').split).fillna('outside_R9_allowed')
    wm['boundary_mask_bits']=wm.base_window_index.map(pool.set_index('base_window_index').boundary_mask_bits).fillna(0).astype('uint16')
    wm['qc_observed_minutes']=wm.base_window_index.map(pool.set_index('base_window_index').qc_observed_minutes)
    wm['in_primary78869']=wm.R9_fold0_role.eq('train')
    wm['selection_reason']=np.select([
        wm.in_primary78869,~wm.R9_allowed,wm.R9_fold0_role.eq('purged_temporal_boundary_r9'),
        wm.R9_fold0_role.eq('excluded_insufficient_observed_support')],
        ['in_primary78869','outside_allowed_'+wm.quality_v2_split.astype(str),'R9_24h_purge','R9_QC_observed_lt3'],
        default='R9_heldout_'+wm.R9_fold0_role.astype(str))
    for name,path in [('in_R16_fold0_train','data/synthetic/r16_multisupport_v1/fold_00/train_backgrounds.parquet'),
                      ('in_R18_train12000','data/synthetic/r18_production_v1/train_backgrounds.parquet')]:
        f=read(path);wm[name]=wm.base_window_index.isin(f.base_window_index)
    assert wm.in_primary78869.sum()==78869
    csv('original133641_window_lineage.csv.gz',wm.drop(columns='start_minute'))
    for dims in [['site'],['site','recorder']]:
        for stage in ['prepare_split','quality_v2_split','R4_historical_split','R9_fold0_role','selection_reason']:
            csv('window_lineage_'+('_'.join(dims))+'_'+stage+'.csv',
                wm.groupby(dims+[stage],observed=True).agg(windows=('window_id','size'),observed_row_occurrences=('observed_minutes','sum')).reset_index())
    return wm,pool,roles

def candidate_windows(g, a, policy, wm):
    """All nonempty 15-clock-minute supports at a 5-minute clock stride.

    Candidate starts are the union of floor5(observed minute)-{0,5,10}.
    This covers every nonempty grid-aligned window without enumerating huge
    empty inter-deployment intervals. Leading/trailing partial windows stay.
    No duration, level, sentinel, coverage or label filter is used.
    """
    parts=[]
    groupcols=['recorder']+(['src_file'] if policy=='within_file' else [])
    for key,h in g.groupby(groupcols,sort=False,observed=True):
        h=h.sort_values(['duration_distance','t','source_row_index']).drop_duplicates('minute').sort_values('minute')
        minute=h.minute.to_numpy(np.int64); local=h.source_row_index.to_numpy(int)
        starts=np.unique(np.concatenate([(minute//5)*5-z for z in [0,5,10]]))
        lookup=wm[wm.recorder.eq(h.recorder.iloc[0]) & wm.site.eq(h.site.iloc[0])].set_index('start_minute')
        sf=pd.Categorical(h.src_file).codes
        t_ns=h.t.to_numpy(dtype='datetime64[ns]').astype('int64')
        for st in range(0,len(starts),1024):
            s=starts[st:st+1024];grid=s[:,None]+np.arange(15)
            ix=np.searchsorted(minute,grid);safe=np.minimum(ix,len(minute)-1)
            present=(ix<len(minute)) & (minute[safe]==grid)
            value=a[local[safe]].copy();value[~present]=np.nan
            f=feature_batch(value)
            f['site']=h.site.iloc[0];f['recorder']=h.recorder.iloc[0]
            f['source_file']=h.src_file.iloc[0] if policy=='within_file' else '<recorder_clock_join>'
            f['start_minute']=s
            f['n_raw_rows_selected']=present.sum(1).astype('int8')
            f['n_eligible_rows']=np.where(present,h.eligible.to_numpy(bool)[safe],False).sum(1).astype('int8')
            f['n_sentinel_rows']=np.where(present,h.sentinel.to_numpy(bool)[safe],False).sum(1).astype('int8')
            f['n_negative_rows']=np.where(present,h.negative.to_numpy(bool)[safe],False).sum(1).astype('int8')
            f['n_boundary_flagged_rows']=np.where(present,h.boundary_flagged.to_numpy(bool)[safe],False).sum(1).astype('int8')
            # Number of distinct observed source files, independently of NaN bands.
            fileids=np.where(present,sf[safe],-1)
            ordered=np.sort(fileids,axis=1)
            nf=(ordered[:,0]>=0).astype(int)+( (ordered[:,1:]!=ordered[:,:-1]) & (ordered[:,1:]>=0)).sum(1)
            f['n_source_files']=nf.astype('int8');f['crosses_source_file']=nf>1
            delta=np.diff(t_ns[safe],axis=1)/1e9
            full=present.all(1)
            f['full_clock15']=full
            f['full15_exact60s_timestamps']=full & np.equal(delta,60).all(1)
            f['full15_55to65s_timestamp_steps']=full & ((delta>=55)&(delta<=65)).all(1)
            # Exact clock identity is distinguished from equal source content.
            match=lookup.reindex(s)
            f['base_window_index']=match.base_window_index.fillna(-1).to_numpy(int)
            for c in ['R9_allowed','in_primary78869','in_R16_fold0_train','in_R18_train12000']:
                f[c]=match[c].fillna(False).to_numpy(bool)
            f['selection_reason']=match.selection_reason.fillna('not_original_fixed15_min3_window').to_numpy()
            # Existing raw/QC tensors contain exactly eligible chosen rows.
            same=present.sum(1)==f.n_eligible_rows.to_numpy()
            original_n=match.observed_minutes.fillna(-1).to_numpy(int)
            f['same_raw_bank_content']=same & (present.sum(1)==original_n) & f.base_window_index.ge(0).to_numpy()
            maskbits=match.boundary_mask_bits.fillna(0).to_numpy(int)
            f['same_primary_QC_content']=f.in_primary78869 & f.same_raw_bank_content & (maskbits==0)
            parts.append(add_conditions(f))
    return pd.concat(parts,ignore_index=True)

def main():
    sys.addaudithook(guard)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'COMPLETED.json').exists():raise FileExistsError('Completed audit output is immutable')
    for n in ['pipeline/prepare_sa.py','pipeline/build_quality_v2.py','pipeline/source_split_r6.py',
              'pipeline/background_cv_r9.py','pipeline/r16_production_data.py','pipeline/r18_production_data.py',
              'ops/r17_audit_common.py','ops/r17_eda_execute.py']:
        INPUTS[n]=sha(ROOT/n)
    save('PROTOCOL.json',{'scope':'public SA selection-bias/descriptive forensics only','SA_rows_expected':1715779,
        'no_model_fitting':True,'no_Test_target_inference':True,'no_Test_source_retrieval_or_matching':True,
        'no_source_mutation':True,'unfiltered_window_stride_minutes':5,'original_window_stride_minutes':15,
        'policies':['within_file','recorder_clock_join'],'grid':'15 clock minutes; observed source clocks sorted; no gap fill/interpolation',
        'duplicates':'choose smallest abs(duration-60), then timestamp, then source_row_index; retained-row counts disclosed',
        'raw_invalid_values':'retain sentinel/negative/short-duration values, count flags; only native NaN/Inf excluded from finite energy sums',
        'boundary_edge_windows':'all nonempty supports retained including <15 observed slots, first/last export edges',
        'timezone':'original export clock; UTC/local timezone unverified','features':'same LF0..8 power fraction, HF20..24 and LF0..12 temporal max-minus-median as R17',
        'absolute_conditions':'HF>=16; LF<=.58; joint; HF[12,20] & LF[.48,.68]; LF bump>=6.8. Descriptive boxes, no Test label decision.',
        'code_sha256':sha(__file__)})
    wm,pool,roles=ledger()
    meta=read('data/processed/sa_row_manifest.parquet')
    for c in meta.select_dtypes(include='object'):meta[c]=meta[c].astype('category')
    rawpath=ROOT/'data/processed/sa_rows_db.npy';bankpath=ROOT/'data/processed/sa_windows_db.npy'
    INPUTS['data/processed/sa_rows_db.npy']=sha(rawpath);INPUTS['data/processed/sa_windows_db.npy']=sha(bankpath)
    rowsbank=np.load(rawpath,mmap_mode='r',allow_pickle=False);bank=np.load(bankpath,mmap_mode='r',allow_pickle=False)
    prep=json.loads((ROOT/'data/processed/sa_preparation_manifest.json').read_text())
    INPUTS['data/processed/sa_preparation_manifest.json']=sha(ROOT/'data/processed/sa_preparation_manifest.json')
    flags=read('reports/data_quality_boundary_flags.parquet');flagids=set(flags.row_id)
    rowtables=[];rowchecks=[];source_checks=[];allwindows={'within_file':[],'recorder_clock_join':[]}
    selected_original=np.zeros(len(meta),bool);offset=0
    origkey=wm.set_index(['site','recorder','start_minute']).base_window_index
    for path in sorted((ROOT/'data/extracted/acoustics').glob('*.parquet')):
        digest=sha(path);INPUTS[str(path.relative_to(ROOT))]=digest
        assert digest==prep['input_sha256'][path.name]
        g=pd.read_parquet(path,columns=['site','recorder','src_file','t','dur','sample_rate_hz']+COMMON)
        n=len(g);a=g[COMMON].to_numpy(np.float32);g=g.drop(columns=COMMON)
        assert np.array_equal(a,np.asarray(rowsbank[offset:offset+n]),equal_nan=True)
        native=meta.iloc[offset:offset+n]
        for c in ['site','recorder','src_file','t','dur','sample_rate_hz']:
            assert np.array_equal(g[c].astype(str).to_numpy(),native[c].astype(str).to_numpy())
        g['source_row_index']=np.arange(n);g['row_id']=np.arange(offset,offset+n)
        g['minute']=pd.to_datetime(g.t).to_numpy(dtype='datetime64[m]').astype('int64')
        g['duration_distance']=abs(g.dur-60)
        g['finite']=np.isfinite(a).all(1);g['sentinel']=(a==-156.5).any(1);g['negative']=(a<0).any(1)
        g['dur_ge55']=g.dur.ge(55);g['dur_le65']=g.dur.le(65);g['second_le5']=pd.to_datetime(g.t).dt.second.le(5)
        g['eligible']=g.finite & ~g.sentinel & g.dur_ge55 & g.dur_le65 & g.second_le5
        g['boundary_flagged']=g.row_id.isin(flagids)
        assert np.array_equal(g.eligible.to_numpy(),native.eligible_for_minute_window.to_numpy())
        eligible=g[g.eligible].sort_values(['duration_distance','t','source_row_index']).drop_duplicates(['recorder','minute']).sort_values(['recorder','minute'])
        original_clock=(eligible.minute.to_numpy()//15)*15
        idx=pd.MultiIndex.from_arrays([eligible.site,eligible.recorder,original_clock])
        wi=origkey.reindex(idx).fillna(-1).to_numpy(int)
        selected_original[eligible.row_id.to_numpy()[wi>=0]]=True
        count=pd.Series(wi[wi>=0]).value_counts()
        assert np.array_equal(count.sort_index().to_numpy(),wm.loc[count.sort_index().index,'observed_minutes'].to_numpy())
        maxdiff=0.0;compared=0
        for st in range(0,len(eligible),12000):
            eh=eligible.iloc[st:st+12000];ew=wi[st:st+12000];valid=ew>=0
            e=eh[valid];ids=ew[valid];mi=e.minute.to_numpy()-(e.minute.to_numpy()//15)*15
            selected=a[e.source_row_index.to_numpy()]
            assert np.array_equal(selected,np.asarray(bank[ids,mi]),equal_nan=True)
            compared+=len(ids)
        dup_all=g.duplicated(['recorder','minute']).sum();dup_eligible=g[g.eligible].duplicated(['recorder','minute']).sum()
        source_checks.append({'source_parquet':path.name,'rows':n,'raw_sha_verified':True,'row_bank_exact':True,
            'eligible_mask_exact':True,'raw_to_window_selected_vectors_exact':compared,'duplicate_minute_rows_all':int(dup_all),
            'duplicate_minute_rows_eligible':int(dup_eligible),'duplicate_exact_timestamps':int(g.duplicated(['recorder','t']).sum()),
            'NaT_rows':int(g.t.isna().sum()),'partial_finite_band_rows':int((np.isfinite(a).any(1)&~np.isfinite(a).all(1)).sum())})
        # Sequential attrition is disjoint. Nonexclusive flag counts are separate.
        remaining=np.ones(n,bool);g['row_exclusion_reason']='eligible_selected_or_sparse_window'
        for reason,bad in [('nonfinite_25bands',~g.finite),('sentinel_minus156_5',g.sentinel),
            ('duration_lt55',~g.dur_ge55),('duration_gt65',~g.dur_le65),('timestamp_second_gt5',~g.second_le5)]:
            take=remaining & bad.to_numpy();g.loc[take,'row_exclusion_reason']=reason;remaining[take]=False
        eligible_ids=set(eligible.row_id)
        g.loc[g.eligible & ~g.row_id.isin(eligible_ids),'row_exclusion_reason']='duplicate_minute_not_selected'
        g.loc[g.eligible & g.row_id.isin(eligible_ids) & ~g.row_id.isin(eligible.row_id.to_numpy()[wi>=0]),'row_exclusion_reason']='fixed15_window_observed_lt3'
        g.loc[g.row_id.isin(eligible.row_id.to_numpy()[wi>=0]),'row_exclusion_reason']='included_in_original133641'
        for dims in [['site'],['site','recorder'],['site','recorder','sample_rate_hz']]:
            rr=g.groupby(dims+['row_exclusion_reason'],observed=True).size().reset_index(name='rows');rr['grouping']='|'.join(dims);rowtables.extend(rr.to_dict('records'))
        for keys,h in g.groupby(['site','recorder'],observed=True):
            rowchecks.append({'site':keys[0],'recorder':keys[1],'rows':len(h),
                **{c:int((~h[c]).sum()) for c in ['finite','dur_ge55','dur_le65','second_le5']},
                'sentinel':int(h.sentinel.sum()),'negative':int(h.negative.sum()),'eligible':int(h.eligible.sum()),
                'boundary_flagged':int(h.boundary_flagged.sum())})
        for policy in allwindows:
            f=candidate_windows(g,a,policy,wm)
            for c in ['site','recorder','source_file','selection_reason']:f[c]=f[c].astype('category')
            allwindows[policy].append(f)
            note(phase='unfiltered_source_complete',source=path.name,policy=policy,windows=len(f),rows=n)
        offset+=n;del g,a,eligible;gc.collect()
    assert offset==len(meta)==1715779
    assert np.array_equal(selected_original,meta.window_included.to_numpy())
    CHECKS.update({'raw_source_rows_exact':offset,'original_selected_rows_exact':int(selected_original.sum()),
        'eligible_rows_exact':int(meta.eligible_for_minute_window.sum()),'original_window_mask_exact':True,
        'all_source_SHA_match_original_preparation':True})
    csv('row_disjoint_filter_attrition.csv',rowtables);csv('row_nonexclusive_flags_by_recorder.csv',rowchecks);csv('raw_source_reconstruction_checks.csv',source_checks)
    del meta,rowsbank;gc.collect()
    qrows=[];crows=[];comparisons=[]
    for policy,parts in allwindows.items():
        f=pd.concat(parts,ignore_index=True);allwindows[policy]=None;del parts;gc.collect()
        for c in ['site','recorder','source_file','selection_reason']:f[c]=f[c].astype('category')
        dt=pd.to_datetime(f.start_minute,unit='m');f['export_month']=dt.dt.month.astype('int8');f['export_hour']=dt.dt.hour.astype('int8')
        f['start_export_clock']=dt.astype(str)
        csv('all_unfiltered_stride5_'+policy+'_windows.csv.gz',f.drop(columns='start_minute'))
        for dims in [[],['site'],['site','recorder'],['n_obs'],['site','n_obs'],['site','recorder','n_obs'],
                     ['selection_reason'],['site','selection_reason'],['n_obs','in_primary78869'],['site','export_month'],['site','export_hour']]:
            qrows.extend(summary(f,policy,dims));crows.extend(conditions(f,policy,dims))
        for selector,mask in [('all_nonempty',np.ones(len(f),bool)),('all_finite15',f.n_full25.eq(15)),
             ('all_eligible15',f.n_eligible_rows.eq(15)),('all_finite15_within_one_file',f.n_full25.eq(15)&~f.crosses_source_file),
             ('clock_in_primary78869',f.in_primary78869),('same_primary_QC_content',f.same_primary_QC_content),
             ('not_clock_in_primary78869',~f.in_primary78869)]:
            sub=f[mask];qrows.extend(summary(sub,policy+'__'+selector));crows.extend(conditions(sub,policy+'__'+selector))
        comparisons.append({'policy':policy,'windows':len(f),'unique_clock_keys':len(f.drop_duplicates(['site','recorder','start_minute'])),
            'clock_in_primary78869':int(f.in_primary78869.sum()),'same_primary_QC_content':int(f.same_primary_QC_content.sum()),
            'same_original_raw_content':int(f.same_raw_bank_content.sum()),'crosses_source_file':int(f.crosses_source_file.sum()),
            'full_clock15':int(f.full_clock15.sum()),'full25_bands15':int(f.n_full25.eq(15).sum()),
            'full15_exact60s':int(f.full15_exact60s_timestamps.sum()),'full15_55to65s':int(f.full15_55to65s_timestamp_steps.sum()),
            'full15_all_eligible':int(f.n_eligible_rows.eq(15).sum()),'HF_ge16_LF_le058':int(f.HF_ge16_LF_le058.sum()),
            'HF_ge16_LF_le058_full25_15':int((f.HF_ge16_LF_le058&f.n_full25.eq(15)).sum())})
        note(phase='policy_summary_complete',**comparisons[-1]);del f;gc.collect()
    # All original windows, before and after actual admitted boundary QC.
    fp=features(bank);parity=physical_features(np.asarray(bank[:128]));small=feature_batch(bank[:128])
    for c in ['n_obs','LF_energy_fraction','bump_low_db','bump_high_db']:
        diff=np.max(abs(small[c].to_numpy()-parity[c].to_numpy()))
        CHECKS['R17_feature_parity_maxdiff_'+c]=float(diff);assert diff<1e-10
    for c in ['site','recorder','base_window_index','selection_reason','in_primary78869','in_R16_fold0_train','in_R18_train12000']:
        fp[c]=wm[c].values
    fp=add_conditions(fp);csv('original133641_raw_features.csv.gz',fp)
    for dims in [[],['site'],['site','recorder'],['n_obs'],['selection_reason'],['site','selection_reason']]:
        qrows.extend(summary(fp,'original_fixed15_raw',dims));crows.extend(conditions(fp,'original_fixed15_raw',dims))
    ids=pool.base_window_index.to_numpy(int);bparts=[]
    for st in range(0,len(ids),1024):
        ix=ids[st:st+1024];a=np.asarray(bank[ix]).copy();bits=pool.boundary_mask_bits.to_numpy(int)[st:st+len(ix)]
        a[((bits[:,None]>>np.arange(15))&1).astype(bool)]=np.nan;bparts.append(feature_batch(a))
    fq=pd.concat(bparts,ignore_index=True)
    assert np.array_equal(fq.n_obs,pool.qc_observed_minutes)
    for c in ['site','recorder','base_window_index','selection_reason','in_primary78869','in_R16_fold0_train','in_R18_train12000']:
        fq[c]=wm.iloc[ids][c].values
    fq=add_conditions(fq);csv('R9_allowed100965_QC_features.csv.gz',fq)
    for ds,sub in [('R9_allowed100965_QC',fq),('primary78869_QC',fq[fq.in_primary78869]),
                   ('R16_fold0_train_QC',fq[fq.in_R16_fold0_train]),('R18_train12000_QC',fq[fq.in_R18_train12000])]:
        for dims in [[],['site'],['site','recorder'],['n_obs'],['site','n_obs']]:
            qrows.extend(summary(sub,ds,dims));crows.extend(conditions(sub,ds,dims))
    test=read('reports/r17_eda_audit_20261030/eda/Test_descriptive_only.parquet')
    # This stored table contains every parsed Test row (no sampling in its code).
    # Reused saved-input completeness proof, without opening query tensors here.
    proof_path=ROOT/'reports/r18_public_gap_audit_r1/query_observation_completeness.json'
    if proof_path.exists():
        proof=json.loads(proof_path.read_text())
        assert proof['shape']==[1500,15,25] and proof['partial_band_minutes']==0 and proof['n_obs_any_equals_all'] is True
        INPUTS[str(proof_path.relative_to(ROOT))]=sha(proof_path)
        test['n_full25']=test.n_obs
    else:test['n_full25']=np.nan
    test=add_conditions(test)
    for dims in [[],['n_obs']]:qrows.extend(summary(test,'Test_input_descriptive_saved_all1500',dims));crows.extend(conditions(test,'Test_input_descriptive_saved_all1500',dims))
    csv('feature_quantiles_all_groups.csv',qrows);csv('absolute_shape_conditions_all_groups.csv',crows);csv('unfiltered_window_policy_counts.csv',comparisons)
    stages=[{'unit':'raw acoustic rows','stage':'raw_source','count':offset},
        {'unit':'raw acoustic rows','stage':'eligible_minute_rows','count':CHECKS['eligible_rows_exact']},
        {'unit':'raw acoustic rows','stage':'included_in_original_windows','count':CHECKS['original_selected_rows_exact']},
        {'unit':'15-minute fixed clock windows','stage':'original133641','count':len(wm)},
        {'unit':'15-minute fixed clock windows','stage':'R9_allowed100965','count':len(pool)}]
    for r,n in roles.split.value_counts().items():stages.append({'unit':'15-minute fixed clock windows','stage':'R9_fold0_'+r,'count':int(n)})
    for c in ['in_R16_fold0_train','in_R18_train12000']:stages.append({'unit':'15-minute fixed clock windows','stage':c,'count':int(wm[c].sum())})
    csv('stage_counts_units.csv',stages)
    save('VERIFICATION.json',CHECKS)
    save('INPUT_SHA256.json',INPUTS)
    mainq=pd.DataFrame(qrows);mainc=pd.DataFrame(crows)
    keepq=mainq[mainq.dataset.isin(['original_fixed15_raw','primary78869_QC','R16_fold0_train_QC','R18_train12000_QC',
        'recorder_clock_join__all_nonempty','recorder_clock_join__all_finite15','within_file__all_finite15','Test_input_descriptive_saved_all1500'])]
    # Dataset-only rows have no grouping keys; retain only rows from [] grouping.
    groupkeys=['site','recorder','n_obs','selection_reason','in_primary78869','export_month','export_hour']
    keepq=keepq[keepq[[c for c in groupkeys if c in keepq]].isna().all(1)]
    keepc=mainc[mainc.dataset.isin(keepq.dataset.unique())]
    keepc=keepc[keepc[[c for c in groupkeys if c in keepc]].isna().all(1)]
    csv('headline_feature_distributions.csv',keepq);csv('headline_shape_conditions.csv',keepc)
    raw_ret=CHECKS['original_selected_rows_exact']/offset
    text='# SA 전수 배경 선택편향 감사\n\n'
    text+='원본 1,715,779는 **행**, 78,869는 **15분 창(R9 Fold0 TRAIN)**이다. 두 수를 나누어 95.4%를 버렸다고 할 수 없다. 원본 행은 100% 저장되어 있고, 그중 기존 창에 들어간 행 비율은 '+f'{100*raw_ret:.4f}%'+'이다. 같은 창 단위의 Fold0 TRAIN/전체 원본 창 비율은 '+f'{100*78869/len(wm):.4f}%'+'이다. 후자는 calibration/domain/heldout/시간분리 목적의 창을 포함한다.\n\n'
    text+=pd.DataFrame(stages).to_markdown(index=False)+'\n\n'
    text+='기존 prepare_sa.py는 **stride15 고정 clock bin**이다. 새 전수 감사는 stride5로 모든 비어 있지 않은 clock support를 구성한다. recorder_clock_join은 파일을 넘어서도 같은 recorder의 실제 시각을 정렬하되 시간 gap을 NaN으로 남긴다. within_file은 src_file을 넘지 않는다. 어느 쪽도 행을 붙여 연속 15분처럼 만들지 않는다. 모든 nonempty edge/sparse 창을 보존하고 n_obs별 전체 분포를 별도로 제공한다.\n\n'
    text+=pd.DataFrame(comparisons).to_markdown(index=False)+'\n\n'
    text+='원본에서 NaN/Inf band는 유한 관측수와 에너지합에서 빠진다. sentinel=-156.5, 음수, 55초 미만/65초 초과, second>5 행은 unfiltered 창에서 삭제하지 않고 flag로 표시한다. 이러한 창과 관측 15분/25band/eligible15 창을 구분해 해석해야 한다. 중복 minute은 duration이 60초에 가장 가까운 행, 이른 timestamp, 이른 source_row_index 순으로 한 행을 선택한다.\n\n'
    text+='`clock_in_primary78869`는 동일한 site/recorder/start clock key가 기존 풀에 있다는 뜻이다. 원시 내용까지 같은지(`same_raw_bank_content`), 기존 QC 내용까지 같은지(`same_primary_QC_content`)를 별도로 계산했다. stride5의 시작이 기존 stride15와 다르면 기존 창 ID를 억지로 배정하지 않는다.\n\n'
    text+='HF bump=밴드20..24(3162..7943Hz) 선형 power합을 dB로 변환한 분별 곡선의 max−median, LF bump=밴드0..12(31.6..501.2Hz)의 같은 정의다. LF energy fraction은 전체 관측 power 대비 밴드0..8(31.6..199.5Hz) power 비율이다. 이 LF fraction과 LF bump는 서로 다른 주파수 범위를 쓴다. 기존 R17 정의와 수치 parity를 검증했다.\n\n'
    text+='HF>=16dB 및 LF fraction<=.58, 주변 box HF[12,20]/LF[.48,.68]는 입력 형태를 설명하는 절대 조건이다. top10%만 선별하지 않고 min/p01/p05/p10/p25/p50/p75/p90/p95/p99/max 전체 분포, 모든 site/recorder/n_obs 및 기존 포함/제외 이유별 집계를 제공한다. 이 조건은 선박/무선박 라벨이 아니다. SA도 검증된 음성이 아니라 미라벨 자연 관측이다.\n\n'
    text+='Test는 저장된 R17 입력 특징 1,500개 전체를 재사용한다(생성 코드에 sampling 없음). HF16.25dB/LF.5825는 전체 Test 혼합입력 분포의 중앙값이며, 음성 조건부분포가 아니다. Test의 라벨, 예측, 제출물은 열지 않았고 원본 SA 유사창을 Test 생성 원본으로 역추정하거나 매칭하지 않았다. 관측 형태가 비슷하다는 사실은 배경 성분 또는 실제 생성 원천을 확인하지 않는다. Test 통계로 threshold/model/policy를 적합하지 않았다.\n\n'
    text+='5분 stride의 이웃 15분 창은 최대 10분의 원천 행을 공유한다. 파일 경계에서 나뉜 창도 같은 clock key를 가질 수 있다. 따라서 창 수나 조건 hit 수를 독립 녹음/독립 사건 수로 해석하거나 일반적인 독립표본 신뢰구간에 바로 넣을 수 없다.\n\n'
    text+='세부 근거: row_disjoint_filter_attrition.csv, window_lineage_* CSV, all_unfiltered_stride5_*_windows.csv.gz, feature_quantiles_all_groups.csv, absolute_shape_conditions_all_groups.csv, VERIFICATION.json, INPUT_SHA256.json. 소스코드 근거: prepare_sa.py:38-81, build_quality_v2.py:33-57, background_cv_r9.py:117-177, source_split_r6.py:92-111, r16_production_data.py:121-140, r18_production_data.py:53-67, r17_audit_common.py:34-50.\n'
    (OUT/'SA_PUBLIC_BACKGROUND_FORENSICS.md').write_text(text)
    enrich_outputs()
    pins={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['COMPLETED.json','OUTPUT_SHA256.json','progress.json','run.log','enrich_run.log']}
    pins[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    save('OUTPUT_SHA256.json',pins)
    save('COMPLETED.json',{'status':'completed','rows_verified':offset,'models_trained':0,'Test_targets_or_predictions_accessed':False,
        'source_mutations':False,'window_policies':comparisons,'all_verifications_passed':True,'output_files_hashed':len(pins)})
    note(phase='completed',files=len(pins))

def enrich_outputs():
    """Derive validity/overlap detail only from this audit's saved tables."""
    wm=pd.read_csv(OUT/'original133641_window_lineage.csv.gz')
    m=read('data/versions/background_cv_r9/masked_minute_rows.parquet')
    for dims in [['site'],['site','recorder'],['site','recorder','candidate_reason']]:
        csv('minute_QC_masked_'+('_'.join(dims))+'.csv',m.groupby(dims).size().reset_index(name='masked_original_rows'))
    reason=wm.groupby('selection_reason').agg(windows=('window_id','size'),original_selected_rows=('observed_minutes','sum')).reset_index()
    csv('selection_reason_same_unit_counts.csv',reason)
    units={'raw_rows_stored':1715779,'raw_rows_deleted':0,'rows_in_original_windows':1698184,
           'row_admission_rate_to_original_windows':1698184/1715779,
           'rows_not_in_original_windows':1715779-1698184,
           'original_window_count':len(wm),'primary_fold0_TRAIN_window_count':int(wm.in_primary78869.sum()),
           'same_unit_primary_window_fraction':float(wm.in_primary78869.mean()),
           'primary_original_rows_before_minute_QC':int(wm.loc[wm.in_primary78869,'observed_minutes'].sum()),
           'primary_original_rows_after_minute_QC':int(wm.loc[wm.in_primary78869,'qc_observed_minutes'].sum()),
           'all_R9_allowed_masked_minutes':len(m),
           'primary_masked_minutes':int((wm.loc[wm.in_primary78869,'observed_minutes']-wm.loc[wm.in_primary78869,'qc_observed_minutes']).sum()),
           'invalid_row_divided_by_window_claim':1-78869/1715779}
    save('selection_unit_correction.json',units)
    qrows=[];crows=[];overlap=[]
    for policy in ['recorder_clock_join','within_file']:
        f=pd.read_csv(OUT/f'all_unfiltered_stride5_{policy}_windows.csv.gz')
        f['validity_class']=np.select([
            f.n_eligible_rows.eq(15),f.n_eligible_rows.eq(f.n_raw_rows_selected),
            f.n_sentinel_rows.gt(0),f.n_negative_rows.gt(0)],
            ['eligible15','eligible_sparse_edge','contains_sentinel','contains_other_negative'],
            default='contains_duration_or_second_exclusion')
        for dims in [['validity_class'],['site','validity_class'],['site','recorder','validity_class'],
                     ['validity_class','selection_reason'],['site','validity_class','selection_reason']]:
            qrows.extend(summary(f,policy,dims));crows.extend(conditions(f,policy,dims))
        # Fixed-grid start membership and overlapping sliding source rows are different.
        originalgrid=pd.to_datetime(f.start_export_clock).dt.minute.mod(15).eq(0)
        c=f.groupby(['site','recorder','start_export_clock'],sort=False).size()
        overlap.append({'policy':policy,'windows':len(f),'duplicate_clock_keys_extra_windows':int(len(f)-len(c)),
            'clock_keys_with_multiple_file_windows':int((c>1).sum()),
            'all_unfiltered_original_stride15_clock_starts':int(originalgrid.sum()),
            'original_stride15_clock_keys_in_primary78869':int((originalgrid&f.in_primary78869).sum()),
            'striding_overlap':'neighboring stride5 windows share up to10/15 clock-minute slots; not independent source recordings',
            'whole_corpus_interpretation':'all public SA observed inputs; no verified vessel absence labels'})
    csv('unfiltered_cleanliness_feature_quantiles.csv',qrows)
    csv('unfiltered_cleanliness_shape_conditions.csv',crows)
    save('source_overlap_diagnostics.json',overlap)
    q=pd.read_csv(OUT/'feature_quantiles_all_groups.csv')
    missing_full25=q.dataset.eq('Test_input_descriptive_saved_all1500') & q.feature.eq('n_full25')
    statcols=['n_valid','missing','mean','std']+[f'p{int(z*100):02d}' for z in Q]
    proof_path=ROOT/'reports/r18_public_gap_audit_r1/query_observation_completeness.json'
    proof=None
    if proof_path.exists():
        proof=json.loads(proof_path.read_text())
        assert proof['shape']==[1500,15,25] and proof['partial_band_minutes']==0 and proof['n_obs_any_equals_all'] is True
        INPUTS[str(proof_path.relative_to(ROOT))]=sha(proof_path)
        ref=q.dataset.eq('Test_input_descriptive_saved_all1500') & q.feature.eq('n_obs')
        assert np.array_equal(q.loc[ref,'windows'],q.loc[missing_full25,'windows'])
        q.loc[missing_full25,statcols]=q.loc[ref,statcols].to_numpy()
    else:
        q.loc[missing_full25,'n_valid']=0;q.loc[missing_full25,'missing']=q.loc[missing_full25,'windows']
        q.loc[missing_full25,statcols[2:]]=np.nan
    csv('feature_quantiles_all_groups.csv',q)
    c=pd.read_csv(OUT/'absolute_shape_conditions_all_groups.csv')
    cols=['site','recorder','n_obs','selection_reason','in_primary78869','export_month','export_hour']
    q=q[q[[x for x in cols if x in q]].isna().all(1)]
    c=c[c[[x for x in cols if x in c]].isna().all(1)]
    names=['original_fixed15_raw','primary78869_QC','R16_fold0_train_QC','R18_train12000_QC',
           'recorder_clock_join__all_nonempty','recorder_clock_join__all_finite15',
           'recorder_clock_join__all_eligible15','within_file__all_eligible15','Test_input_descriptive_saved_all1500']
    q=q[q.dataset.isin(names)&q.feature.isin(['bump_high_db','bump_low_db','LF_energy_fraction'])]
    c=c[c.dataset.isin(names)&c.condition.eq('HF_ge16_LF_le058')]
    csv('headline_feature_distributions.csv',q);csv('headline_shape_conditions.csv',c)
    text='\n## 유효 행 조건과 입력 형태의 실제 차이\n\n'
    text+='원본 창에 들어가지 않은 행은 17,595개(1.0255%)다. disjoint 사유는 sentinel 653, duration<55초 16,880, duration>65초 22, 3분 미만 clock bin 40이다. 비유한 25band/second>5의 추가 순차 제외는 0이다(겹치는 원인은 별도 비배타 표 참조). 원본을 삭제한 행은 없다.\n\n'
    text+='R9 전체 허용100,965창에서 run-edge 후보 6,254분을 mask했고 그중 한 창만 3분 미만으로 탈락했다. Fold0 TRAIN78,869창에는 원본 선택1,017,474행이 있었고, 4,453분 mask 후1,013,021분이 관측된다. 이 run-edge 후보는 원인 미확인 QC 플래그이며 확인된 비선박/선박 라벨이 아니다.\n\n'
    text+=q[['dataset','feature','windows','p50','p90','p99']].to_markdown(index=False,floatfmt='.6f')+'\n\n'
    text+=c[['dataset','windows','count','fraction']].to_markdown(index=False,floatfmt='.6f')+'\n\n'
    text+='unfiltered 완전25band15분 창296,798개 중 joint조건은2,061개지만, duration/second도 모두 기존 eligibility를 충족하는 15분256,029창에서는379개(0.1480%)다. 그러므로 unfiltered의 큰 bump 증가를 모두 유효한 자연배경 다양성이라고 해석할 수 없다. 반대로 유효한 유사 입력이 전혀 없다는 주장도 이 집계와 맞지 않는다. 기존 원본fixed15 joint154/133,641(0.1152%), primary QC110/78,869(0.1395%), 실제R18 배경18/12,000(0.1500%)다.\n\n'
    text+='Test423/1,500=28.2%는 전체 혼합입력의 형태 조건 빈도다. SA 미라벨 자연관측의 빈도와의 차이는 입력분포 차이를 보여주며, 이를 Test 음성 분포, 음성 배경 손실, 실제 Test 생성 원천, 특정 원인에 의한 실패 비율로 바꾸어 말할 근거는 없다. 새 조건 적합이나 모델훈련은 없다.\n'
    report=OUT/'SA_PUBLIC_BACKGROUND_FORENSICS.md'
    prior=report.read_text().split('\n## 유효 행 조건과 입력 형태의 실제 차이\n')[0]
    prior=prior.replace('원본 행 보존율은 98.9745%이고,','원본 행은 100% 저장되어 있고, 그중 기존 창에 들어간 행 비율은 98.9745%이다. ')
    if '5분 stride의 이웃' not in prior:
        prior+='\n5분 stride의 이웃 15분 창은 최대10분의 원천 행을 공유한다. 파일 경계에서 나뉜 창도 같은 clock key를 가질 수 있다. 창 수/조건hit를 독립 녹음이나 독립 사건 수로 볼 수 없다.\n'
    if proof is not None:
        text+='\n저장된 입력 completeness 검산(query_observation_completeness.json, query.npz SHA '+proof['sha256']+')은 전체1500×15×25에서 partial-band minute=0과 any-band n_obs=all25 n_obs를 확인한다. 이 감사는 원본Test나 query.npz를 다시 열지 않고 그 검증 JSON을 인용하여 Test의 n_full25를 확정했다.\n'
    report.write_text(prior+text)
    paired_minute_qc()
    if (OUT/'INPUT_SHA256.json').exists():
        pins=json.loads((OUT/'INPUT_SHA256.json').read_text());pins.update(INPUTS);save('INPUT_SHA256.json',pins)
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    protocol['final_code_sha256']=sha(__file__)
    protocol['enrichment_scope']='validity/overlap summaries from saved exhaustive tables; no source values recomputed or changed'
    protocol['saved_Test_n_full25']='verified equal to n_obs by separate saved-input partial-band=0 completeness proof' if proof is not None else 'unknown from reused feature table; never inferred from any-band n_obs'
    save('PROTOCOL.json',protocol)

def paired_minute_qc():
    """Hold exact base-window identities fixed, isolating only minute QC."""
    before=pd.read_csv(OUT/'original133641_raw_features.csv.gz').set_index('base_window_index')
    after=pd.read_csv(OUT/'R9_allowed100965_QC_features.csv.gz').set_index('base_window_index')
    meta=pd.read_csv(OUT/'original133641_window_lineage.csv.gz').set_index('base_window_index')
    assert before.index.is_unique and after.index.is_unique and len(after)==100965
    before=before.loc[after.index]
    for c in ['site','recorder','in_primary78869']:
        assert np.array_equal(before[c],after[c])
    assert int(after.in_primary78869.sum())==78869
    pair=after[['site','recorder','in_primary78869','in_R16_fold0_train','in_R18_train12000']].copy()
    pair['boundary_mask_bits']=meta.loc[pair.index,'boundary_mask_bits']
    pair['masked_minutes']=[int(z).bit_count() for z in pair.boundary_mask_bits]
    pair['had_minute_QC']=pair.masked_minutes.gt(0)
    for c in ['n_obs','LF_energy_fraction','bump_high_db','bump_low_db']:
        pair[c+'_before']=before[c];pair[c+'_after']=after[c]
        pair[c+'_delta']=after[c]-before[c]
        pair[c+'_changed']=~np.isclose(before[c],after[c],atol=1e-10,rtol=0,equal_nan=True)
    pair['joint_before']=before.HF_ge16_LF_le058
    pair['joint_after']=after.HF_ge16_LF_le058
    pair['joint_transition']=np.select([pair.joint_before & pair.joint_after,pair.joint_before & ~pair.joint_after,
       ~pair.joint_before & pair.joint_after],['stays_true','true_to_false','false_to_true'],default='stays_false')
    pair['drops_below_min3']=pair.n_obs_before.ge(3)&pair.n_obs_after.lt(3)
    csv('minute_QC_paired_windows.csv.gz',pair.reset_index())
    rows=[];totals=[]
    for name,sub in [('R9_allowed100965',pair),('primary78869',pair[pair.in_primary78869]),
                     ('R16_fold0_train61446',pair[pair.in_R16_fold0_train]),('R18_train12000',pair[pair.in_R18_train12000])]:
        total={'pool':name,'windows':len(sub),'masked_minutes':int(sub.masked_minutes.sum()),
               'windows_with_masked_minutes':int(sub.had_minute_QC.sum()),
               'windows_with_observed_count_change':int(sub.n_obs_changed.sum()),
               'drops_below_min3':int(sub.drops_below_min3.sum()),
               'joint_before':int(sub.joint_before.sum()),'joint_after':int(sub.joint_after.sum()),
               **{z:int(sub.joint_transition.eq(z).sum()) for z in ['stays_true','true_to_false','false_to_true','stays_false']}}
        totals.append(total)
        for dims in [[],['site'],['site','recorder']]:
            iterator=sub.groupby(dims,observed=True,dropna=False) if dims else [((),sub)]
            for key,g in iterator:
                if not isinstance(key,tuple):key=(key,)
                common={'pool':name,**dict(zip(dims,key)),'windows':len(g),'masked_minutes':int(g.masked_minutes.sum())}
                for feature in ['n_obs','LF_energy_fraction','bump_high_db','bump_low_db']:
                    b=g[feature+'_before'].to_numpy(float);a=g[feature+'_after'].to_numpy(float);d=a-b
                    rows.append({**common,'feature':feature,'before_p50':float(np.nanmedian(b)),
                        'before_p90':float(np.nanquantile(b,.9)),'after_p50':float(np.nanmedian(a)),
                        'after_p90':float(np.nanquantile(a,.9)),'paired_delta_p50':float(np.nanmedian(d)),
                        'paired_delta_p90':float(np.nanquantile(d,.9)),
                        'changed_windows':int(g[feature+'_changed'].sum()),
                        'joint_before':int(g.joint_before.sum()),'joint_after':int(g.joint_after.sum())})
    csv('minute_QC_paired_summary.csv',rows);csv('minute_QC_paired_count_transitions.csv',totals)
    headline=pd.DataFrame(rows);headline=headline[headline.site.isna()&headline.recorder.isna()]
    text='# 동일 SA 창의 minute QC 전후 비교\n\n'
    text+='site/recorder/start/base_window_index를 완전히 고정한 paired 비교다. 전체133,641창과78,869 subset의 site/구성 차이를 섞지 않았다. before는 기존 prepare를 거친 원시 창, after는 정확히 같은 창에 run-edge minute mask만 적용한 값이다. 추가 행 eligibility 변경이나 Test 참조는 없다.\n\n'
    text+=pd.DataFrame(totals).to_markdown(index=False)+'\n\n'
    text+=headline[['pool','feature','windows','before_p50','before_p90','after_p50','after_p90','changed_windows','paired_delta_p50','paired_delta_p90']].to_markdown(index=False,floatfmt='.8f')+'\n\n'
    text+='joint는 HF bump>=16dB 및 LF energy fraction<=.58이다. 특징 변화 건수는 절대차>1e-10 기준이다. 창을 고정하고 mask 후3분 미만의 한 창도 paired 표에서 버리지 않고 기록했다. 이 비교는 미라벨 관측 형태의 차이이며 선박 라벨이나 Test 음성을 복원하지 않는다.\n'
    (OUT/'MINUTE_QC_PAIRED_EFFECTS.md').write_text(text)
    report=OUT/'SA_PUBLIC_BACKGROUND_FORENSICS.md';t=report.read_text()
    t+='\n## 동일 창에 고정한 minute QC 효과\n\n'+pd.DataFrame(totals).to_markdown(index=False)+'\n\n'
    t+='minute_QC_paired_summary.csv에 동일 창 before/after의 HF/LF fraction/LF bump p50/p90 및 변화 창 수를 기록했다. 전체133,641창과부분78869창 차이의 site선택 효과를 minute QC 효과로 혼동하지 않는다. 세부 paired 모든 창은 minute_QC_paired_windows.csv.gz, 보고서는 MINUTE_QC_PAIRED_EFFECTS.md다.\n'
    report.write_text(t)

if __name__=='__main__':
    # A fresh destination permits a full independent rerun without overwriting
    # the completed audit. Example: --output reports/audit_recheck/background.
    if '--output' in sys.argv:
        target=Path(sys.argv[sys.argv.index('--output')+1])
        OUT=target if target.is_absolute() else ROOT/target
    if '--enrich' in sys.argv:
        sys.addaudithook(guard);enrich_outputs()
        pins={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['COMPLETED.json','OUTPUT_SHA256.json','progress.json','run.log','enrich_run.log']}
        pins[str(Path(__file__).relative_to(ROOT))]=sha(__file__);save('OUTPUT_SHA256.json',pins)
        done=json.loads((OUT/'COMPLETED.json').read_text());done['output_files_hashed']=len(pins);done['validity_and_overlap_enrichment']=True;save('COMPLETED.json',done)
    else:main()
