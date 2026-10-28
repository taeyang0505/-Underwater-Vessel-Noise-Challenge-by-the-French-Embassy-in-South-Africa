"""Training-only known-component domain randomization, implemented in NumPy.

These smooth log-frequency responses are augmentation assumptions, not an exact
underwater propagation model. No data files, model state or global RNG are read.
"""
import numpy as np


def _ratio(a, b):
    result=np.zeros_like(a,dtype=np.float64)
    np.divide(a,b,out=result,where=b>0)
    result[(b<=0)&(a>0)]=np.inf
    return result


def _max_db(ratio):
    positive=ratio[ratio>0]
    return float(10*np.log10(positive.max())) if len(positive) else None


def _rounding_minute_limits(db, q, energy, observed, selected):
    """Allow only an original float32 excess consistent with half-ULP rounding."""
    v=q.astype(np.float64)
    previous=np.nextafter(q,np.asarray(-np.inf,dtype=q.dtype)).astype(np.float64)
    loq=np.clip((v+previous)/2,0,1)
    delta=np.where(observed,np.abs(np.spacing(db)).astype(np.float64)/2,0)
    elo=energy*10**(-delta/10);ehi=energy*10**(delta/10)
    lower=_ratio((loq*elo).sum(2),((1-loq)*ehi).sum(2))
    actual=_ratio((v*energy).sum(2),((1-v)*energy).sum(2))
    if np.any((lower>100*(1+3e-7))&selected[:,None]):
        raise ValueError('common_response cannot preserve q and repair an input genuinely above the 20 dB minute limit')
    limits=np.maximum(100.,actual)*(1+2e-12)
    return limits,int(np.sum(selected&np.any(actual>100.,axis=1)))


def _rounding_support_limits(db,q,energy,observed):
    v=q.astype(float);a=v*energy;b=(1-v)*energy
    active=a.sum(2)>0;eligible=(active.sum(1)>0)&(active.sum(1)<=2)
    cap=np.minimum(.98,.25+.73*active.sum(1)/observed.any(2).sum(1))
    actual=a.sum((1,2))/energy.sum((1,2))
    previous=np.nextafter(q,np.asarray(-np.inf,dtype=q.dtype)).astype(float)
    loq=np.clip((v+previous)/2,0,1)
    delta=np.where(observed,np.abs(np.spacing(db)).astype(float)/2,0)
    alo=(loq*energy*10**(-delta/10)).sum((1,2))
    bhi=((1-loq)*energy*10**(delta/10)).sum((1,2))
    lower=alo/(alo+bhi)
    genuine=eligible&(lower>cap+1e-9)
    limit=np.where(eligible,np.maximum(cap,actual),1.)+2e-12
    return limit,genuine


def augment_components(db, cell_share, rng, mode='reference_smooth', probability=.75, strength_db=12.):
    """Return ({db, share, presence, cell_share}, numeric audit).

    Inputs are floating [N,T,F] arrays (or a single [T,F]); NaN db cells are
    unobserved and must have zero q. ``rng`` is a Generator or explicit seed.
    strength_db bounds MAX ABSOLUTE response at every band, not the endpoint
    difference: peak-to-peak response can be up to 2*strength_db. The curve is
    a tilt plus mean-centered quadratic over log-frequency band index.

    reference_smooth filters A=qE, restores the original global added share,
    then enforces <=2-active-minute support and 20dB minute limits by scaling A.
    common_response filters A and B alike, preserves q exactly, and recomputes
    share. If needed its curve is weakened/removed, rather than rescaling A.
    off (and probability-zero rows) preserves input db/q exactly. Its returned
    share is reconstructed from the stored pair; original NPZ scalar rounding
    cannot be recovered because that scalar is not an input to this function.
    """
    if mode not in {'reference_smooth','common_response','off'}:raise ValueError('Unknown augmentation mode')
    if not np.isfinite(probability) or not 0<=probability<=1:raise ValueError('probability must be in [0,1]')
    if not np.isfinite(strength_db) or not 0<=strength_db<=12:raise ValueError('strength_db must be in [0,12]')
    source=np.asarray(db);truth=np.asarray(cell_share)
    if source.shape!=truth.shape or source.ndim not in [2,3]:raise ValueError('db/q must share [T,F] or [N,T,F] dimensions')
    if source.dtype.kind!='f' or truth.dtype.kind!='f':raise ValueError('Floating db and q arrays are required')
    single=source.ndim==2;x=source[None] if single else source;q=truth[None] if single else truth
    if x.shape[0]<1 or x.shape[1]<1 or x.shape[2]<2:raise ValueError('Nonempty time/band dimensions required')
    observed=np.isfinite(x)
    if np.isinf(x).any() or not observed.any((1,2)).all():raise ValueError('Each example needs finite observations and no infinite db')
    if not np.isfinite(q).all() or ((q<0)|(q>1)).any() or (q[~observed]!=0).any():raise ValueError('q must be finite in [0,1], and zero outside the observed mask')
    gen=rng if isinstance(rng,np.random.Generator) else np.random.default_rng(rng)
    n,t,f=x.shape;peak=np.nanmax(x.astype(np.float64),axis=(1,2),keepdims=True)
    energy=np.where(observed,10**((x.astype(np.float64)-peak)/10),0)
    if not (energy[observed]>0).all():raise ValueError('Observed energy underflowed; input db range is unsupported')
    a=q.astype(np.float64)*energy;b=energy-a
    amass=a.sum((1,2));bmass=b.sum((1,2));original=amass/(amass+bmass);positive=amass>0
    before=_ratio(a.sum(2),b.sum(2));outx=x.copy();outq=q.copy()
    selected=np.zeros(n,bool) if mode=='off' else gen.random(n)<probability
    applied=selected&positive if mode=='reference_smooth' else selected.copy()
    if mode=='off':applied[:]=False
    axis=np.linspace(-1,1,f);basis=np.stack([axis,axis**2-np.mean(axis**2)])
    coefficients=np.zeros((n,2)) if mode=='off' else gen.uniform(-strength_db,strength_db,(n,2))
    curve=coefficients@basis;extent=np.max(np.abs(curve),axis=1)
    curve*=np.minimum(1,strength_db/np.maximum(extent,1e-30))[:,None];curve[~applied]=0
    scale=np.ones(n);backtracked=np.zeros(n,bool);identity=np.zeros(n,bool)
    support_capped=np.zeros(n,bool);minute_capped=np.zeros(n,bool);quantization_rows=0;restore_error=0.
    preexisting_support=np.zeros(n,bool);preexisting_skipped=np.zeros(n,bool)
    newa=a.copy();newb=b.copy()
    if mode=='common_response' and applied.any():
        support_limits,preexisting_support=_rounding_support_limits(x,q,energy,observed)
        limits,quantization_rows=_rounding_minute_limits(x,q,energy,observed,applied)
        preexisting_skipped=applied&preexisting_support
        applied[preexisting_support]=False;curve[~applied]=0
        identity|=preexisting_skipped
        for _ in range(24):
            response=10**((curve*scale[:,None])/10)
            ratios=_ratio((a*response[:,None,:]).sum(2),(b*response[:,None,:]).sum(2))
            mass_a=(a*response[:,None,:]).sum((1,2));mass_b=(b*response[:,None,:]).sum((1,2))
            bad=applied&(np.any(ratios>limits,axis=1)|(mass_a/(mass_a+mass_b)>support_limits))
            if not bad.any():break
            backtracked|=bad;scale[bad]*=.5
        else:
            identity|=bad;scale[bad]=0
        curve*=scale[:,None];response=10**(curve/10)
        newa=a*response[:,None,:];newb=b*response[:,None,:]
        outx[applied]=np.where(observed[applied],x[applied].astype(float)+curve[applied,None,:],np.nan).astype(x.dtype)
        # q is copied, not re-estimated or changed by an A-only correction.
        assert np.array_equal(outq,q)
    elif mode=='reference_smooth' and applied.any():
        if np.any((bmass<=0)&applied):raise ValueError('Reference share restoration needs positive background mass')
        candidate=a*10**(curve[:,None,:]/10)
        factor=np.ones(n);factor[applied]=amass[applied]/candidate.sum((1,2))[applied]
        candidate*=factor[:,None,None]
        restored=candidate.sum((1,2))/(candidate.sum((1,2))+bmass)
        restore_error=float(np.max(np.abs(restored[applied]-original[applied])))
        active=(candidate.sum(2)>0)&observed.any(2);coverage=active.sum(1)/observed.any(2).sum(1)
        limit=np.minimum(.98,.25+.73*coverage);eligible=applied&(active.sum(1)<=2)
        current=candidate.sum((1,2))/(candidate.sum((1,2))+bmass)
        support_capped=eligible&(current>limit)
        multiplier=np.ones(n)
        multiplier[support_capped]=limit[support_capped]*bmass[support_capped]/((1-limit[support_capped])*candidate.sum((1,2))[support_capped])
        candidate*=multiplier[:,None,None]
        minutea=candidate.sum(2);minuteb=b.sum(2)
        if np.any((minutea>0)&(minuteb<=0)&applied[:,None]):
            raise ValueError('A positive component on a zero-background minute cannot satisfy the SNR cap with a positive global scale')
        upper=np.ones_like(minutea)
        np.divide(100*minuteb,minutea,out=upper,where=minutea>0)
        multiplier=np.minimum(1,upper.min(1));multiplier[~applied]=1
        minute_capped=applied&(multiplier<1-1e-12)
        candidate*=multiplier[:,None,None];newa[applied]=candidate[applied]
        e=newa+newb;fraction=np.divide(newa,e,out=np.zeros_like(e),where=observed&(e>0))
        outx[applied]=np.where(observed[applied],10*np.log10(np.maximum(e[applied],1e-300))+peak[applied],np.nan).astype(x.dtype)
        outq[applied]=fraction[applied].astype(q.dtype)
    total=newa+newb;share=(newa.sum((1,2))/total.sum((1,2))).astype(np.float32)
    if np.any(positive&(share==0)):raise ValueError('Positive added energy underflowed to a zero target')
    if not np.isfinite(share).all() or ((share<0)|(share>1)).any():raise ValueError('Invalid augmented share')
    if not np.array_equal(np.isfinite(outx),observed):raise RuntimeError('Observation mask changed')
    if not np.isfinite(outq).all() or ((outq<0)|(outq>1)).any():raise RuntimeError('Invalid augmented cell targets')
    sx=outx.astype(float);se=np.where(observed,10**((sx-np.nanmax(sx,axis=(1,2),keepdims=True))/10),0)
    recovered=(outq.astype(float)*se).sum((1,2))/se.sum((1,2))
    error=float(np.max(np.abs(recovered-share)))
    if error>3e-6:raise RuntimeError('Saved augmented cell/window energy targets disagree')
    after=_ratio(newa.sum(2),newb.sum(2))
    if mode=='reference_smooth' and np.any((after>100*(1+1e-10))&applied[:,None]):raise RuntimeError('Reference minute SNR cap failed')
    if mode=='common_response' and applied.any() and np.any((after>limits*(1+1e-10))&applied[:,None]):raise RuntimeError('Common-response minute constraint failed')
    if mode=='common_response' and applied.any() and np.any(applied&(share.astype(float)>support_limits+1e-7)):raise RuntimeError('Common-response introduced a support-cap violation')
    data={'db':outx[0] if single else outx,'cell_share':outq[0] if single else outq,
        'share':share[0] if single else share,'presence':(share>0).astype(np.float32)[0] if single else (share>0).astype(np.float32)}
    audit={'mode':mode,'examples':n,'probability':float(probability),'strength_db':float(strength_db),
        'strength_definition':'Maximum absolute per-band smooth response; endpoint-to-endpoint span <= 2*strength_db.',
        'selected_count':int(selected.sum()),'augmented_count':int(np.sum(applied&~identity)),
        'positive_examples':int(positive.sum()),'zero_examples':int((~positive).sum()),
        'cap_count':int((support_capped|minute_capped|backtracked).sum()),
        'support_cap_count':int(support_capped.sum()),'minute_snr_cap_count':int(minute_capped.sum()),
        'common_backtrack_count':int(backtracked.sum()),'common_identity_fallback_count':int(identity.sum()),
        'preexisting_support_cap_violation_count':int(preexisting_support.sum()),
        'preexisting_support_cap_identity_count':int(preexisting_skipped.sum()),
        'new_support_cap_violation_count':0,
        'input_float_rounding_excess_rows':quantization_rows,'curve_abs_max_db':float(np.abs(curve).max()),
        'curve_peak_to_peak_max_db':float(np.ptp(curve,axis=1).max()),
        'reference_share_restored_count':int(applied.sum()) if mode=='reference_smooth' else 0,
        'max_reference_share_restore_error':restore_error,'max_reconstruction_error':error,
        'max_minute_snr_db_before':_max_db(before),'max_minute_snr_db_after':_max_db(after),
        'mask_preserved':True,'q_preserved':bool(np.array_equal(outq,q)),
        'positive_became_zero_count':0,'input_data_modified':False,
        'interpretation':'Known-component smooth domain randomization; not calibrated or exact propagation physics.'}
    return data,audit
