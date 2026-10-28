"""TRAIN-only aggregate background priors; query transforms use db alone.

High-band anchors are a declared counterfactual assumption, not certified
vessel/background separation or a calibrated acoustic propagation model.
"""
import json
import warnings
import numpy as np

FREQUENCIES=10.**(1.5+.1*np.arange(25))
QUANTILES=(.1,.2,.5)
REGIMES=('all','mid_high','high')
FIELDS=('background_fraction_raw','estimated_added_fraction','anchor_log_rmse_db',
        'infeasible_energy_fraction','top2_score_gap_db','anchor_scale_time_cv')
FEATURE_NAMES=[f'q{int(q*100):02d}_{r}_{field}' for q in QUANTILES for r in REGIMES for field in FIELDS]

def _input(db):
    x=np.asarray(db)
    if x.ndim==2:x=x[None]
    if x.ndim!=3 or x.shape[2]!=25 or not len(x):raise ValueError('Expected nonempty [N,T,25] db')
    o=np.isfinite(x)
    if np.isinf(x).any() or not o.any((1,2)).all():raise ValueError('Finite nonempty observations are required')
    x=x.astype(np.float64);peak=np.nanmax(x,(1,2),keepdims=True)
    e=np.where(o,10**((x-peak)/10),0)
    if not (e[o]>0).all():raise ValueError('Unsupported energy underflow')
    return e,o

def _nanquantile(x,q,axis):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanquantile(x,q,axis=axis)

def _shape(power):
    available=np.isfinite(power)&(power>0)
    fill=_nanquantile(np.where(available,power,np.nan),.5,1)
    if not np.isfinite(fill).all():raise ValueError('Known background must have positive measured energy')
    p=np.where(available,power,fill[:,None]*1e-12)
    p=np.maximum(p,np.maximum(p.sum(1,keepdims=True)*1e-12,1e-30))
    log=np.log(p);return log-log.mean(1,keepdims=True)

def _distances(x,c):
    return np.maximum((x*x).sum(1)[:,None]+(c*c).sum(1)[None,:]-2*x@c.T,0)

def fit_background_prior(db,cell_share,seed=42,clusters=32):
    """Learn aggregate log-spectral centres from known TRAIN B=(1-q)E.

    Returns a JSON-safe dict; no individual rows, retrieval IDs or source tags.
    Empty/tiny clusters are merged rather than retaining singleton templates.
    """
    e,o=_input(db);q=np.asarray(cell_share,dtype=float)
    if q.ndim==2:q=q[None]
    if q.shape!=e.shape or not np.isfinite(q).all() or ((q<0)|(q>1)).any():raise ValueError('Invalid TRAIN cell_share')
    if type(clusters) is not int or clusters<1:raise ValueError('clusters must be a positive integer')
    b=np.where(o,(1-q)*e,np.nan)
    shape=_shape(_nanquantile(b,.5,1));n=len(shape)
    if n<2:raise ValueError('At least two training backgrounds are required for aggregate priors')
    rng=np.random.default_rng(seed);minimum=8 if n>=128 else 2;k=min(clusters,max(1,n//minimum))
    centres=[shape[int(rng.integers(n))]];nearest=_distances(shape,np.asarray(centres))[:,0]
    for _ in range(1,k):
        if nearest.sum()<=1e-18:break
        index=int(rng.choice(n,p=nearest/nearest.sum()));centres.append(shape[index])
        nearest=np.minimum(nearest,_distances(shape,shape[index:index+1])[:,0])
    centres=np.asarray(centres)
    for _ in range(16):
        labels=_distances(shape,centres).argmin(1);counts=np.bincount(labels,minlength=len(centres));new=centres.copy()
        for j in range(len(centres)):
            if counts[j]:new[j]=shape[labels==j].mean(0)
        if np.max(np.abs(new-centres))<1e-7:centres=new;break
        centres=new
    for _ in range(4):
        labels=_distances(shape,centres).argmin(1);counts=np.bincount(labels,minlength=len(centres));keep=counts>=minimum
        if keep.all():break
        centres=centres[keep] if keep.any() else shape.mean(0,keepdims=True)
    labels=_distances(shape,centres).argmin(1);counts=np.bincount(labels,minlength=len(centres))
    centres=np.stack([shape[labels==j].mean(0) for j in range(len(centres))])
    power=np.exp(centres-centres.max(1,keepdims=True));power/=power.sum(1,keepdims=True)
    return {'version':'aggregate_background_prior_v1','seed':int(seed),'training_rows':n,
        'requested_clusters':clusters,'actual_clusters':len(centres),'minimum_cluster_members':minimum,
        'cluster_counts':counts.tolist(),'centres_log_shape':centres.tolist(),'centres_power_shape':power.tolist(),
        'band_centres_hz':FREQUENCIES.tolist(),'feature_names':FEATURE_NAMES,
        'background_aggregation':'Per-band time median of known TRAIN B=(1-q)E; centred log-shape kmeans.',
        'anchor_policy':{'all':'all measured bands','mid_high':'centres >=500 Hz','high':'centres >=2000 Hz'},
        'query_quantiles':list(QUANTILES),'map_policy':'q20/high-anchor template; per-minute high-anchor common amplitude',
        'individual_training_examples_saved':False,'query_requires_labels_or_source_tags':False,
        'physical_interpretation':'Counterfactual background-manifold features, not verified source separation.'}

def _prior(prior):
    if prior.get('version')!='aggregate_background_prior_v1':raise ValueError('Unsupported prior')
    p=np.asarray(prior['centres_power_shape'],dtype=float)
    if p.ndim!=2 or p.shape[1]!=25 or not np.isfinite(p).all() or (p<=0).any():raise ValueError('Invalid prior centres')
    return p/p.sum(1,keepdims=True)

def _anchor(valid,regime):
    selected=valid&({'all':np.ones(25,bool),'mid_high':FREQUENCIES>=500,'high':FREQUENCIES>=2000}[regime])[None,:]
    # Missing anchors fall back to measured bands only, never invented cells.
    insufficient=selected.sum(1)<2;selected[insufficient]=valid[insufficient]
    return selected

def _fit_templates(level,valid,power,regime):
    n=len(level);k=len(power);anchors=_anchor(valid,regime)
    logq=np.log(np.maximum(level,1e-30));logp=np.log(power)
    difference=logq[:,None,:]-logp[None,:,:]
    logscale=_nanquantile(np.where(anchors[:,None,:],difference,np.nan),.5,2)
    logscale=np.clip(logscale,-60,60)
    residual=(difference-logscale[:,:,None])*(10/np.log(10))
    squared=np.where(anchors[:,None,:],residual**2,np.inf);ordered=np.sort(squared,axis=2)
    keep=np.maximum(1,np.ceil(.8*anchors.sum(1)).astype(int));use=np.arange(25)[None,None,:]<keep[:,None,None]
    rmse=np.sqrt(np.where(use,ordered,0).sum(2)/keep[:,None])
    # Only predicted BG above observed quantile violates nonnegative addition.
    over=np.where(valid[:,None,:],np.maximum(-residual,0),0)
    feasibility=over.sum(2)/valid.sum(1)[:,None]
    score=rmse+.5*feasibility;best=score.argmin(1);rows=np.arange(n)
    second=np.partition(score,1,axis=1)[:,1] if k>1 else score[:,0]
    chosen=power[best];scale=np.exp(logscale[rows,best])
    return chosen,scale,rmse[rows,best],np.maximum(second-score[rows,best],0),best,anchors

def _minute_scale(e,o,chosen,anchors):
    ratio=np.log(np.maximum(e,1e-30))-np.log(chosen[:,None,:])
    mask=o&anchors[:,None,:]
    # A minute with missing high bands uses its actually observed bands.
    missing=~mask.any(2);mask=np.where(missing[:,:,None],o,mask)
    logscale=_nanquantile(np.where(mask,ratio,np.nan),.5,2)
    return np.where(o.any(2),np.exp(np.clip(np.nan_to_num(logscale,nan=0),-60,60)),0)

def transform_background_prior(db,prior):
    """Return finite [N,54] features using each query window independently."""
    e,o=_input(db);p=_prior(prior);valid=o.any(1);total=e.sum((1,2));features=[]
    for quantile in QUANTILES:
        level=_nanquantile(np.where(o,e,np.nan),quantile,1);level=np.nan_to_num(level,nan=1e-30)
        for regime in REGIMES:
            chosen,scale,error,gap,_,anchors=_fit_templates(level,valid,p,regime)
            background=chosen[:,None,:]*scale[:,None,None]
            raw=np.where(o,background,0).sum((1,2))/total
            excess=np.where(o,np.maximum(e-background,0),0).sum((1,2))/total
            infeasible=np.where(o,np.maximum(background-e,0),0).sum((1,2))/total
            minute=_minute_scale(e,o,chosen,anchors);m=o.any(2);mean=minute.sum(1)/m.sum(1)
            cv=np.sqrt(np.where(m,(minute-mean[:,None])**2,0).sum(1)/m.sum(1))/np.maximum(mean,1e-30)
            features.extend([np.clip(raw,0,4),np.clip(excess,0,1),np.clip(error,0,120),
                np.clip(infeasible,0,4),np.clip(gap,0,120),np.clip(cv,0,10)])
    result=np.stack(features,1).astype(np.float32)
    if result.shape[1]!=54 or not np.isfinite(result).all():raise RuntimeError('Invalid background-prior features')
    return result

def background_prior_maps(db,prior):
    """Return q0 and raw Bhat/E maps [N,T,25]; query db is the ONLY input.

    Template identity uses the window's q20 HIGH-band shape. The intensity of
    that template is then fit separately in each minute from high bands.
    Bhat/E is clipped to [0,10] to retain background overshoot/confidence.
    q0=clip(1-Bhat/E,0,1); both maps are zero outside the observed mask.
    """
    e,o=_input(db);p=_prior(prior);valid=o.any(1)
    level=_nanquantile(np.where(o,e,np.nan),.2,1);level=np.nan_to_num(level,nan=1e-30)
    chosen,_,error,gap,index,anchors=_fit_templates(level,valid,p,'high')
    scale=_minute_scale(e,o,chosen,anchors);background=chosen[:,None,:]*scale[:,:,None]
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        raw=np.divide(background,e,out=np.zeros_like(e),where=o&(e>0))
    bf=np.where(o,np.clip(np.nan_to_num(raw,nan=0,posinf=10),0,10),0)
    q0=np.where(o,np.clip(1-bf,0,1),0);weighted=(q0*e).sum((1,2))/e.sum((1,2))
    audit={'examples':len(e),'clusters':len(p),'anchor':'high >=2000 Hz; measured-band fallback only',
        'template_quantile':.2,'per_minute_common_scale':True,'query_labels_or_source_tags_used':False,
        'query_population_fit':False,'finite_maps':bool(np.isfinite(q0).all() and np.isfinite(bf).all()),
        'unobserved_maps_zero':True,'q0_semantics':'clipped counterfactual added fraction per cell',
        'B_fraction_semantics':'Raw Bhat/E clipped to [0,10], not globally normalized energy weights; retains overshoot beyond one',
        'background_overshoot_cells':int(((raw>1)&o).sum()),'raw_B_fraction_clip10_cells':int(((raw>10)&o).sum()),
        'raw_B_fraction_max_finite':float(raw[o&np.isfinite(raw)].max()) if (o&np.isfinite(raw)).any() else None,
        'raw_B_fraction_nonfinite_cells':int((o&~np.isfinite(raw)).sum()),
        'estimated_added_share_mean':float(weighted.mean()),'anchor_log_rmse_db_mean':float(error.mean()),
        'template_counts':np.bincount(index,minlength=len(p)).tolist()}
    return q0.astype(np.float32),bf.astype(np.float32),audit

def numerical_fixture():
    """Known persistent LF addition, zero-addition and inference-isolation cases."""
    rng=np.random.default_rng(13);base=np.exp(np.linspace(-.7,.7,25))
    b=base[None,None,:]*rng.uniform(.7,1.4,(32,15,1));x=(10*np.log10(b)+83).astype(np.float32)
    prior=fit_background_prior(x,np.zeros_like(x),seed=42,clusters=1)
    known=base[None,:]*np.linspace(.7,1.4,15)[:,None];added=np.zeros_like(known);added[:,:13]=9*known[:,:13]
    db=(10*np.log10(known+added)+83).astype(np.float32);truth=float(added.sum()/(known+added).sum())
    q,bf,audit=background_prior_maps(db,prior);e,o=_input(db);estimate=float((q*e).sum()/e.sum())
    floor=_nanquantile(e,.2,1);naive=float(np.maximum(e-floor[:,None,:],0).sum()/e.sum())
    assert abs(estimate-truth)<1e-5 and abs(estimate-truth)<abs(naive-truth)
    clean=(10*np.log10(known)+83).astype(np.float32);zero,_,_=background_prior_maps(clean,prior)
    assert zero.max()<1e-5
    restored=json.loads(json.dumps(prior));assert np.array_equal(transform_background_prior(db,prior),transform_background_prior(db,restored))
    assert np.allclose(transform_background_prior(db+27,prior),transform_background_prior(db,prior),atol=2e-5)
    shifted=background_prior_maps(db+27,prior)[0];assert np.allclose(shifted,q,atol=2e-5)
    masked=db.copy();masked[2:4]=np.nan;masked[5,24]=np.nan
    mq,mb,_=background_prior_maps(masked,prior);assert np.isfinite(mq).all() and np.isfinite(mb).all()
    assert (mq[0][~np.isfinite(masked)]==0).all() and (mb[0][~np.isfinite(masked)]==0).all()
    one=transform_background_prior(db,prior)[0];batch=transform_background_prior(np.stack([db,db+27]),prior)[0]
    assert np.array_equal(one,batch) and len(prior['feature_names'])==54 and prior['cluster_counts']==[32]
    return {'passed':True,'known_persistent_LF_share':truth,'counterfactual_estimate':estimate,
        'within_window_q20_estimate':naive,'zero_addition_max_q0':float(zero.max()),
        'features':54,'gain_invariant':True,'query_batch_independent':True,'JSON_reload_exact':True,
        'finite_masked_maps':True,'no_individual_examples_saved':True,'actual_training_executed':False}

if __name__=='__main__':print(json.dumps(numerical_fixture(),indent=2))
