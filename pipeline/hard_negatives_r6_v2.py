"""TRAIN-only natural LF-transient background candidates for R6 v2.

These are no-addition controls with known artificial A=0, not verified
ship-absent recordings. No natural-vessel labels are inferred. Region powers
are sums of linear band powers; bump=max-minute dB minus temporal median dB.
All thresholds are fixed before model fitting. No non-TRAIN values are read.
"""
from __future__ import annotations
from pathlib import Path
import json
import math
import warnings
import numpy as np
import pandas as pd
from pipeline.source_split_r6 import SourceSplitR6, ROOT, sha, canonical_sha, write_new, relative

FORMAT='r6_v2_train_lf_hard_negative_candidates_v1'
FREQUENCIES=10.**(1.5+.1*np.arange(25))
REGIONS={'low':np.arange(9),'safe_mid_high':np.arange(15,23),'high':np.arange(20,25)}
POLICY={'lf_quantile':.90,'lf_excess_over_safe_high_db':3.,'minimum_observed_minutes_per_region':3,
        'safe_high_definition':'max(safe_mid_high_bump_db,high_bump_db)',
        'bump_definition':'max-minute regional broadband power dB minus temporal median regional broadband power dB',
        'broadband_definition':'sum linear powers of every band in the declared region; complete regional minutes only',
        'quantile_scope':'all eligible TRAIN windows, pooled; no site/Test population matching',
        'input_db_basis':'admitted raw SA window dB after actual-minute QC; no extra rounding for candidate selection',
        'quantile_method':'linear','threshold_ties':'LF >= TRAIN 90th quantile; ties included and counted',
        'source_role':'train','natural_ship_ground_truth':False,'known_added_component_A':0}


def transient_features(db):
    """db[N,15,25] only. Return scalar regional bump diagnostics per window."""
    if isinstance(db,dict):raise TypeError('Pass db only, no labelled mappings')
    x=np.asarray(db)
    if x.ndim!=3 or x.shape[1:]!=(15,25) or not len(x) or not np.issubdtype(x.dtype,np.floating) or np.isinf(x).any():raise ValueError('Expected nonempty finite/NaN float db[N,15,25]')
    out={};eligible=np.ones(len(x),bool)
    for name,ix in REGIONS.items():
        v=x[:,:,ix].astype(np.float64);ok=np.isfinite(v).all(2);count=ok.sum(1)
        # logsumexp is stable even when arbitrary common dB gain is present.
        natural=v*np.log(10.)/10.
        power_db=np.logaddexp.reduce(np.where(np.isfinite(natural),natural,-np.inf),axis=2)*10./np.log(10.)
        power_db=np.where(ok,power_db,np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            baseline=np.nanmedian(power_db,axis=1);peak=np.nanmax(power_db,axis=1)
        out[name+'_bump_db']=peak-baseline;out[name+'_observed_minutes']=count
        eligible&=(count>=POLICY['minimum_observed_minutes_per_region'])&np.isfinite(peak)&np.isfinite(baseline)
    out['safe_high_bump_db']=np.maximum(out['safe_mid_high_bump_db'],out['high_bump_db'])
    out['eligible_for_train_quantile']=eligible
    return out


def candidate_mask(features):
    valid=np.asarray(features['eligible_for_train_quantile'],bool)
    low=np.asarray(features['low_bump_db'],float);high=np.asarray(features['safe_high_bump_db'],float)
    if not valid.any():raise ValueError('No eligible TRAIN windows for fixed LF quantile')
    threshold=float(np.quantile(low[valid],POLICY['lf_quantile'],method='linear'))
    selected=valid&(low>=threshold)&(low>high+POLICY['lf_excess_over_safe_high_db'])
    return selected,threshold


def candidates(source_version,output,*,root=ROOT,chunk_size=512):
    """Create immutable TRAIN-only candidate metadata; return JSON manifest.

    Source indices are original global SA raw-bank coordinates. No reference
    numeric slice, INNER, cross panel, old OUTER or challenge query is loaded.
    """
    root=Path(root).resolve();output=Path(output);output=output if output.is_absolute() else root/output
    if output.exists():raise FileExistsError('Fresh hard-negative candidate output required')
    if type(chunk_size) is not int or chunk_size<1:raise ValueError('Positive chunk_size required')
    loader=SourceSplitR6(source_version,root=root);source_file=loader.folder/'manifest.json';source_sha=sha(source_file)
    frame=loader.background_manifest[loader.background_manifest.split.eq('train')].copy().reset_index(drop=True)
    if not len(frame) or not frame.r6_role.eq('train').all():raise ValueError('Nonempty strict TRAIN role required')
    parts=[]
    for i in range(0,len(frame),chunk_size):
        chunk=frame.iloc[i:i+chunk_size];db,md=loader.load_backgrounds('train',chunk.base_window_index.to_numpy())
        if not np.array_equal(md.base_window_index,chunk.base_window_index):raise RuntimeError('TRAIN window order changed')
        parts.append(pd.DataFrame(transient_features(db)))
    features=pd.concat(parts,ignore_index=True);mask,threshold=candidate_mask(features)
    meta=frame[['base_window_index','window_id','site','recorder','sensor_group','start_export_clock','recording_week_start','qc_observed_minutes','boundary_mask_bits','r6_role']].copy()
    meta=pd.concat([meta,features],axis=1);meta['lf_train_q90_threshold_db']=threshold;meta['hard_negative_candidate']=mask
    subset=meta[mask].copy().reset_index(drop=True)
    if subset.empty:raise ValueError('Fixed pre-fit hard-negative definition has no TRAIN candidates; no threshold retry')
    all_sites=sorted(frame.site.unique());site_counts={s:int(subset.site.eq(s).sum()) for s in all_sites}
    audit={'train_windows_examined':len(frame),'eligible_train_windows':int(features.eligible_for_train_quantile.sum()),
        'candidate_count':len(subset),'unique_candidate_windows':int(subset.base_window_index.nunique()),
        'lf_train_q90_threshold_db':threshold,'lf_exact_threshold_tie_count':int((features.low_bump_db==threshold).sum()),
        'candidate_site_counts':site_counts,'sites_without_candidates':[s for s in all_sites if not site_counts[s]],
        'train_site_counts':{s:int(frame.site.eq(s).sum()) for s in all_sites},
        'references_or_nontrain_values_read':False,'model_fitting_executed':False,
        'known_A_zero_is_exact_artificial_target':True,'natural_ship_absence_claimed':False,
        'historical_EDA_5725_count_reused':False,'challenge_Test_statistics_used':False}
    source_relative=relative(root,loader.folder)
    code_names=['pipeline/hard_negatives_r6_v2.py','ops/build_hard_negatives_r6_v2.py']
    code_hashes={n:sha(root/n) for n in code_names}
    output.mkdir(parents=True)
    meta.to_parquet(output/'train_window_features.parquet',index=False,compression='zstd')
    subset.to_parquet(output/'hard_negative_candidates.parquet',index=False,compression='zstd')
    np.save(output/'candidate_indices.npy',subset.base_window_index.to_numpy(np.int64),allow_pickle=False)
    m={'format':FORMAT,'source_version':source_relative,'source_manifest_sha256':source_sha,
       'source_hash':loader.manifest['source_hash'],'policy':POLICY,'regions':{name:{'indices':ix.tolist(),'centres_hz':FREQUENCIES[ix].tolist()} for name,ix in REGIONS.items()},
       'audit':audit,'file_sha256':{n:sha(output/n) for n in ['train_window_features.parquet','hard_negative_candidates.parquet','candidate_indices.npy']},
       'code_sha256':code_hashes,'role_scope':'train_only','ready_for_train_source_draws':True}
    m['candidate_version_hash']=canonical_sha(m)
    if sha(source_file)!=source_sha or any(sha(root/n)!=h for n,h in code_hashes.items()):raise ValueError('Source/code changed during candidate construction')
    write_new(output/'manifest.json',m)
    return m


def _load(version,*,root=ROOT):
    root=Path(root).resolve();folder=Path(version);folder=folder if folder.is_absolute() else root/folder
    m=json.loads((folder/'manifest.json').read_text())
    if m.get('format')!=FORMAT or m.get('role_scope')!='train_only' or m.get('ready_for_train_source_draws') is not True:raise ValueError('Verified TRAIN-only candidate version required')
    if canonical_sha({k:v for k,v in m.items() if k!='candidate_version_hash'})!=m['candidate_version_hash'] or m['policy']!=POLICY:raise ValueError('Candidate policy/manifest changed')
    for n,h in m['file_sha256'].items():
        if sha(folder/n)!=h:raise ValueError('Candidate artifact changed: '+n)
    for n,h in m['code_sha256'].items():
        if sha(root/n)!=h:raise ValueError('Candidate code changed: '+n)
    loader=SourceSplitR6(m['source_version'],root=root)
    if sha(loader.folder/'manifest.json')!=m['source_manifest_sha256'] or loader.manifest['source_hash']!=m['source_hash']:raise ValueError('Source version changed')
    indices=np.load(folder/'candidate_indices.npy',allow_pickle=False)
    frame=pd.read_parquet(folder/'hard_negative_candidates.parquet')
    train=loader.background_manifest[loader.background_manifest.split.eq('train')]
    if indices.ndim!=1 or indices.dtype!=np.dtype('int64') or not len(indices) or len(np.unique(indices))!=len(indices) or not set(indices)<=set(train.base_window_index):raise ValueError('Candidate indices escape unique TRAIN source scope')
    if not np.array_equal(indices,frame.base_window_index) or not frame.r6_role.eq('train').all():raise ValueError('Candidate index/metadata mismatch')
    return loader,frame,indices,m


def load_candidate_indices(version,*,root=ROOT):
    """Owned int64 array of original global TRAIN indices, in manifest order."""
    return _load(version,root=root)[2].copy()


def _balanced(frame,n,rng):
    if n==0:return np.empty(0,np.int64)
    if n<0 or frame.empty:raise ValueError('Nonempty eligible TRAIN pool required')
    sites=sorted(frame.site.unique());out=[]
    for i,site in enumerate(sites):
        count=n//len(sites)+int(i<n%len(sites));pool=frame.loc[frame.site.eq(site),'base_window_index'].to_numpy(np.int64)
        if count:out.extend(np.concatenate([rng.permutation(pool) for _ in range(math.ceil(count/len(pool)))])[:count])
    return np.asarray(out,np.int64)[rng.permutation(n)]


def draw_training_indices(version,n,seed,*,root=ROOT,zero_fraction=.20,hard_negative_fraction=.10):
    """Return indices/explicit_zero/hard_negative and an audit, all TRAIN.

    Exactly half of round(.20*N) explicit no-source examples are hard-negative
    draws (floor if odd). This is ~10% of all examples. The remaining examples
    draw baseline TRAIN sources with equal-site allocation. No model fit.
    """
    if type(n) is not int or n<1 or zero_fraction!=.20 or hard_negative_fraction!=.10:raise ValueError('Fixed pre-fit 20% zero / half hard-negative policy required')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:raise ValueError('Explicit nonnegative integer seed required')
    loader,frame,_,m=_load(version,root=root);train=loader.background_manifest[loader.background_manifest.split.eq('train')]
    rng=np.random.default_rng(seed);nzero=round(.20*n);nhard=nzero//2;nbaseline=n-nhard
    hard=_balanced(frame,nhard,rng);baseline=_balanced(train,nbaseline,rng)
    indices=np.r_[hard,baseline];is_hard=np.arange(n)<nhard;zero=is_hard.copy()
    if nzero>nhard:zero[nhard+rng.permutation(nbaseline)[:nzero-nhard]]=True
    order=rng.permutation(n);indices=indices[order];is_hard=is_hard[order];zero=zero[order]
    if not is_hard[zero].sum()==nhard or not zero.sum()==nzero or (is_hard&~zero).any():raise AssertionError('No-addition hard-negative schedule failed')
    sites=frame.set_index('base_window_index').site;hs=sites.loc[indices[is_hard]].to_numpy() if nhard else np.empty(0,str)
    audit={'candidate_version_hash':m['candidate_version_hash'],'seed':int(seed),'n':n,'explicit_zero_count':int(nzero),'hard_negative_count':int(nhard),
        'baseline_source_draw_count':int(nbaseline),'hard_negative_unique_windows':len(np.unique(indices[is_hard])),
        'hard_negative_site_draw_counts':{s:int((hs==s).sum()) for s in sorted(frame.site.unique())},
        'sites_without_candidates':m['audit']['sites_without_candidates'],'sampling':'equal-site candidates among eligible sites; baseline equal-site all TRAIN; cycle only when site pool exhausted',
        'index_order_sha256':__import__('hashlib').sha256(np.ascontiguousarray(indices).tobytes()).hexdigest(),
        'source_role':'train','artificial_no_addition_hard_negative_target':0,'natural_ship_absence_claimed':False}
    return {'indices':indices,'explicit_zero':zero,'hard_negative':is_hard,'draw_audit':audit}


def fixture():
    x=np.full((20,15,25),60.,dtype=np.float64);x[:2,7,:9]+=20;x[2,7,:]+=20
    x[3,7,20:]+=30;x[4,0]=np.nan;x[5,:13]=np.nan
    f=transient_features(x);selected,threshold=candidate_mask(f)
    assert selected[:2].all() and not selected[2] and not selected[3] and not selected[5]
    assert f['low_bump_db'][0]>19.999999 and f['high_bump_db'][0]==0
    gained=transient_features(x+37.)
    for n in ['low_bump_db','safe_mid_high_bump_db','high_bump_db']:
        assert np.allclose(f[n],gained[n],atol=1e-12,equal_nan=True)
    chunk={n:np.r_[transient_features(x[:7])[n],transient_features(x[7:])[n]] for n in f}
    assert all(np.allclose(f[n],chunk[n],equal_nan=True) for n in f)
    assert np.isnan(x[4,0]).all() and np.all(x[:2,7,:9]==80)
    frame=pd.DataFrame({'site':['a','a','b'],'base_window_index':[1,2,3]})
    a=_balanced(frame,9,np.random.default_rng(42));b=_balanced(frame,9,np.random.default_rng(42))
    assert np.array_equal(a,b) and sum(np.isin(a,[1,2]))==5 and sum(a==3)==4
    return {'status':'PASS','synthetic_only':True,'LF_transient_selected_broadband_HF_rejected':True,
        'missing_constant_gain_chunk_safe':True,'deterministic_equal_site_draw':True,'label_or_model_fit':False,'fixture_threshold_db':threshold}
