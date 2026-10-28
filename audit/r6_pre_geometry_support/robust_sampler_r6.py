"""R6 declared peak-SNR band-power simulator, independent of frozen R4/R5.

No requested-share inversion, waveform repetition or Test-informed matching.
Native source trajectories remain single-pass. Channel/CPA operators here
are controlled band-power approximations, not calibrated propagation or a
claim about the challenge's hidden generator. The caller fixes zero counts.
"""
from __future__ import annotations
import hashlib,json
import numpy as np

BAND_CENTRES=10.**(1.5+.1*np.arange(25))
FAMILIES=('minute_first','short_passage','edge_straddling','two_second_first_no_warp')

def _serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):_serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_serial(v) for v in value]
    return value

def _rng(rng):
    if isinstance(rng,np.random.Generator):return rng,None
    if isinstance(rng,(bool,np.bool_)) or not isinstance(rng,(int,np.integer)) or rng<0:raise TypeError('Pass a NumPy Generator or a nonnegative integer seed')
    return np.random.default_rng(int(rng)),int(rng)

def energy_weights(db):
    x=np.asarray(db,dtype='float64');o=np.isfinite(x)
    if x.shape!=(15,25) or np.isinf(x).any() or o.any(1).sum()<3:raise ValueError('Expected [15,25], finite/NaN and at least3 observed minutes')
    peak=float(np.nanmax(x));e=np.where(o,10**((x-peak)/10),0)
    if not (e[o]>0).all() or not np.isfinite(e).all() or e.sum()<=0:raise ValueError('Unsupported energy dynamic range')
    return e/e.sum()

def rounding_share_gap_bound(max_abs_db_error):
    """Tight distribution-free |E_w(q)-E_wrounded(q)| bound for q∈[0,1].

    Energy multipliers lie in [a,b]=[10^(-δ/10),10^(δ/10)]. Their normalized
    reweighting changes any bounded mean by at most
    (sqrt(b)-sqrt(a))/(sqrt(b)+sqrt(a))=tanh(ln(10)*δ/20).
    The nominal δ=.05dB gives about .0057564; float32 storage adds half-ULP.
    """
    d=float(max_abs_db_error)
    if not np.isfinite(d) or d<0:raise ValueError('Finite nonnegative dB error bound required')
    return float(np.tanh(np.log(10.)*d/20.))

def _reference(power):
    s=np.array(power,dtype='float64',copy=True)
    if s.ndim!=2 or s.shape[1]!=25 or not len(s) or not np.isfinite(s).all() or (s<0).any() or not (s.sum(1)>0).all():raise ValueError('Finite nonnegative native 2s×25 reference powers required, with positive row energy')
    return s

def _channel(rng,eq_max_db,lowpass_probability):
    maximum=float(eq_max_db);probability=float(lowpass_probability)
    if not np.isfinite(maximum) or not 0<=maximum<=12 or not np.isfinite(probability) or not 0<=probability<=1:raise ValueError('Declared EQ0..12dB and lowpass probability0..1 required')
    if maximum==0:return np.zeros(25),{'enabled':False,'combined_max_abs_db':0.,'lowpass_applied':False}
    x=np.linspace(-1,1,25);tilt,quadratic=rng.uniform(-maximum,maximum,2);curve=tilt*x+quadratic*(x*x-np.mean(x*x));lowpass=bool(rng.random()<probability);cutoff=None
    if lowpass:
        cutoff=float(10**rng.uniform(np.log10(200.),np.log10(2000.)))
        curve+=-10*np.log10(1+(BAND_CENTRES/cutoff)**2)
    scale=min(1.,maximum/max(float(np.max(abs(curve))),1e-300));curve*=scale
    return curve,{'enabled':True,'tilt_coefficient_db':float(tilt),'quadratic_coefficient_db':float(quadratic),'combined_bound_scale':float(scale),'combined_max_abs_db':float(np.max(abs(curve))),'lowpass_applied':lowpass,'lowpass_probability':probability,'lowpass_cutoff_hz':cutoff,'lowpass_shape':'first-order power response before joint EQ bounding','response_semantics':'reference-only band-power multiplier; not exact waveform filtering'}

def _supported_starts(series,observed,starts,steps):
    """Admissible placement depends on measured support, never target/share."""
    starts=np.asarray(starts,dtype='int64');o=np.repeat(observed,steps,axis=0) if steps>1 else observed
    n=len(o);k=len(series);good=[];strictly_positive=bool((series>0).all());prefix=np.r_[0,np.cumsum(o.any(1))]
    for start in starts:
        lo=max(0,int(start));hi=min(n,int(start)+k)
        if lo>=hi:continue
        if strictly_positive:valid=prefix[hi]>prefix[lo]
        else:valid=bool(((series[lo-int(start):hi-int(start)]>0)&o[lo:hi]).any())
        if valid:good.append(int(start))
    return good

def _place_2s(series,start):
    out=np.zeros((450,25),dtype='float64');lo=max(0,start);hi=min(450,start+len(series));out[lo:hi]=series[lo-start:hi-start]
    return out.reshape(15,30,25).mean(1),{'window_start_2s':int(start),'source_visible_begin_2s':int(lo-start),'source_visible_end_2s':int(hi-start),'left_window_edge_reference_power':float(out[0].sum()),'right_window_edge_reference_power':float(out[-1].sum())}

def _native_minute_peak(series,minimum_complete_minutes=1):
    count=len(series)//30
    if count<minimum_complete_minutes:raise ValueError('Full native trajectory needs the declared number of complete60s bins for peak SNR')
    peak=float(series[:count*30].reshape(count,30,25).mean(1).sum(1).max())
    if not np.isfinite(peak) or peak<=0:raise ValueError('Positive full-trajectory native-minute peak required')
    return peak

def _trajectory(s,observed,rng,family):
    k=len(s);base={'native_input_segments':k,'native_input_seconds':2*k,'waveform_repeated':False,'timewarp_applied':False,'window_boundary_fade_forced':False}
    if family=='minute_first':
        complete=k//30
        if not complete:raise ValueError('minute_first requires a complete native60s, never pad/repeat')
        count=min(15,complete);first=int(rng.integers(complete-count+1));minute=s[first*30:(first+count)*30].reshape(count,30,25).mean(1)
        starts=_supported_starts(minute,observed,np.arange(16-count),1)
        if not starts:raise ValueError('No observed overlap for minute_first; no resampling/retry')
        start=int(rng.choice(starts));out=np.zeros((15,25));out[start:start+count]=minute
        return out,{**base,'aggregation_order':'native complete60s mean then minute-aligned placement','native_crop_begin_2s':first*30,'native_crop_end_2s':(first+count)*30,'complete_minutes_used':count,'native_incomplete_tail_segments_excluded':k%30,'window_start_minute':start,'support_seconds':60*count,'full_trajectory_minute_broadband_peak':float(minute.sum(1).max()),'left_window_edge_reference_power':float(out[0].sum()),'right_window_edge_reference_power':float(out[-1].sum())}
    if family=='short_passage':
        eligible=[n for n in [30,60,90] if n<=k]
        if not eligible:raise ValueError('short_passage requires native>=60s; never repeat shorter clips')
        count=int(rng.choice(eligible));first=int(rng.integers(k-count+1));series=s[first:first+count].copy();duration=2.*count;cpa=float(rng.uniform(.25,.75)*duration);width=float(rng.uniform(.15,.5)*duration)
        edges=2*np.arange(count+1);power_envelope=width/2*np.diff(np.arctan((edges-cpa)/width));series*=power_envelope[:,None]
        starts=_supported_starts(series,observed,np.arange(451-count),30)
        if not starts:raise ValueError('No observed overlap for short_passage; no retry')
        out,placement=_place_2s(series,int(rng.choice(starts)))
        return out,{**base,**placement,'aggregation_order':'native contiguous2s crop × smooth CPA power then placement/minute mean','native_crop_begin_2s':first,'native_crop_end_2s':first+count,'support_seconds':int(duration),'full_trajectory_minute_broadband_peak':_native_minute_peak(series),'cpa_time_seconds_in_crop':cpa,'cpa_width_seconds':width,'cpa_envelope':'1/(1+((t-cpa)/width)^2), exact average per2s bin','envelope_min':float(power_envelope.min()),'envelope_max':float(power_envelope.max()),'envelope_interpretation':'controlled smooth band-power modulation, not calibrated propagation'}
    if family=='two_second_first_no_warp':
        _native_minute_peak(s,2);starts=_supported_starts(s,observed,np.arange(0,451-k) if k<=450 else np.arange(450-k,1),30)
        if not starts:raise ValueError('No observed overlap for native2s placement; no retry')
        out,placement=_place_2s(s,int(rng.choice(starts)))
        return out,{**base,**placement,'aggregation_order':'whole native2s trajectory then maximal-overlap placement/minute mean','native_crop_begin_2s':0,'native_crop_end_2s':k,'support_seconds':2*k,'full_trajectory_minute_broadband_peak':_native_minute_peak(s,2)}
    if family.startswith('edge_straddling'):
        full_peak=_native_minute_peak(s,2)
        side=family.removeprefix('edge_straddling').lstrip('_')
        if side not in {'','left','right','both'}:raise ValueError('Unknown edge family')
        candidates=np.arange(-k+1,450);ends=candidates+k
        mask=(candidates<0)|(ends>450) if not side else (candidates<0)&(ends>0) if side=='left' else (candidates<450)&(ends>450) if side=='right' else (candidates<0)&(ends>450)
        starts=_supported_starts(s,observed,candidates[mask],30)
        if not starts:raise ValueError('No native edge-straddling placement with observed overlap; no repeat/retry')
        start=int(rng.choice(starts));out,placement=_place_2s(s,start)
        return out,{**base,**placement,'aggregation_order':'whole native trajectory shifted continuously across window boundary then minute mean','native_crop_begin_2s':0,'native_crop_end_2s':k,'support_seconds':2*k,'full_trajectory_minute_broadband_peak':full_peak,'left_straddles':bool(start<0),'right_straddles':bool(start+k>450)}
    raise ValueError('Unknown declared R6 family')

def synthesize(background_db,reference_power_2s,rng,family='minute_first',eq_max_db=12.,snr_db=None,zero=False,*,lowpass_probability=.25):
    """Return db(float32,.1dB), unrounded_db/q(float64), share/presence/recipe.

    alpha = 10^(DeltaL/10) * median_observed_minute(sum_f B) /
                                max_full_native_minute(sum_f R).
    B uses a common peak-relative scale for numerical stability; alpha is in
    those documented units. R includes native ambient content, not pure ship.
    Zero probability/count is decided by the caller, never inside this API.
    """
    if type(zero) is not bool:raise TypeError('Explicit zero bool required')
    if not np.isfinite(eq_max_db) or not 0<=eq_max_db<=12 or not np.isfinite(lowpass_probability) or not 0<=lowpass_probability<=1:raise ValueError('Declared EQ/lowpass range violated')
    aliases={'native_minute_first':'minute_first','2s_first_no_warp':'two_second_first_no_warp'};family=aliases.get(family,family)
    if family not in FAMILIES and family not in {'edge_straddling_left','edge_straddling_right','edge_straddling_both'}:raise ValueError('Unapproved family')
    generator,seed=_rng(rng);state=_serial(generator.bit_generator.state);state_hash=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    x=np.array(background_db,dtype='float64',copy=True);o=np.isfinite(x);energy_weights(x);peak=float(np.nanmax(x));b=np.where(o,10**((x-peak)/10),0);minute=o.any(1);bmedian=float(np.median(b.sum(1)[minute]))
    if not bmedian>0:raise ValueError('Observed broadband background median must be positive')
    r=np.zeros((15,25));trajectory={'native_input_segments':0,'waveform_repeated':False,'timewarp_applied':False,'window_boundary_fade_forced':False};curve=np.zeros(25);channel={'enabled':False,'lowpass_applied':False};delta=None;alpha=0.;rpeak=0.
    if not zero:
        s=_reference(reference_power_2s);curve,channel=_channel(generator,eq_max_db,lowpass_probability);s=s*10**(curve[None,:]/10);r,trajectory=_trajectory(s,o,generator,family);r=np.where(o,r,0);rpeak=float(trajectory['full_trajectory_minute_broadband_peak'])
        if not r.sum()>0 or not rpeak>0 or not np.isfinite(rpeak):raise ValueError('No positive observed reference/full native peak; no target-conditioned retry')
        delta=float(generator.uniform(-10,10)) if snr_db is None else float(snr_db)
        if not np.isfinite(delta):raise ValueError('Finite optional SNR required')
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):alpha=float(10**(np.float64(delta)/10)*bmedian/rpeak)
        if not np.isfinite(alpha) or alpha<=0:raise ValueError('SNR scale unsupported numerically; never silently clip share')
    added=alpha*r;e=b+added
    if not np.isfinite(e).all() or not (e[o]>0).all():raise ValueError('Finite positive mixture energy required')
    q=np.divide(added,e,out=np.zeros_like(e),where=o);share=float(added.sum()/e.sum());unrounded=np.full((15,25),np.nan);unrounded[o]=10*np.log10(e[o])+peak
    rounded=np.round(unrounded,1).astype('float32')
    if np.isinf(rounded).any():raise ValueError('Rounded float32 storage unsupported')
    observed_error=float(np.max(abs(rounded[o].astype(float)-unrounded[o])));half_ulp=float(np.max(abs(np.spacing(rounded[o])))/2);delta_bound=.05+half_ulp+1e-12;bound=rounding_share_gap_bound(delta_bound)
    raw_identity=float(np.sum(energy_weights(unrounded)*q));rounded_identity=float(np.sum(energy_weights(rounded)*q));gap=abs(rounded_identity-share)
    if abs(raw_identity-share)>1e-11 or observed_error>delta_bound+1e-12 or gap>bound+1e-11:raise RuntimeError('Raw/rounded energy identity or proven rounding bound failed')
    if not np.array_equal(np.isnan(rounded),~o) or not np.array_equal(np.isnan(unrounded),~o) or not ((q>=0)&(q<=1)).all() or (q[~o]!=0).any():raise RuntimeError('Mask/component fraction invariant failed')
    actual_snr=None if zero else float(10*np.log10(alpha*rpeak/bmedian));observed_snr=None if zero else float(10*np.log10(added.sum(1)[minute].max()/bmedian))
    if not zero and not np.isclose(actual_snr,delta,atol=1e-10,rtol=0):raise RuntimeError('Observed peak SNR identity failed')
    recipe={'version':'robust_sampler_r6_v1','family':family,'seed_if_integer':seed,'rng_bit_generator':generator.bit_generator.__class__.__name__,'rng_state_before_sha256':state_hash,'zero':zero,'snr_db_requested':delta,'snr_db_actual_full_reference_peak':actual_snr,'snr_db_actual_observed_window_peak':observed_snr,'default_snr_sampling':'Uniform[-10,+10]dB; declared sensitivity range, not Test-representative','background_peak_db':peak,'background_observed_broadband_median_peak_relative':bmedian,'reference_full_trajectory_minute_broadband_peak':rpeak,'reference_observed_window_broadband_peak':float(r.sum(1)[minute].max()),'reference_peak_scope':'after EQ/CPA shaping, full continuous selected reference trajectory before window crop/observation mask; complete native60s bins, native-start alignment','alpha_peak_relative_background_units':alpha,'alpha_definition':'10^(snr_db/10)*Bmedian/Rpeak_full; outside-window CPA never renormalized to visible tail','share_derived_from_components':share,'requested_share':None,'share_cap_or_high_share_deletion':False,'global_added_to_background_snr_db':None if zero else float(10*np.log10(added.sum()/b.sum())),'reference_eq_db':curve.tolist(),'channel':channel,'trajectory':trajectory,'observed_minutes':int(minute.sum()),'raw_energy_identity_error':abs(raw_identity-share),'rounded_weight_q_pool':rounded_identity,'rounded_weight_vs_raw_target_gap':gap,'rounding_nominal_step_db':.1,'rounding_storage_half_ulp_db':half_ulp,'rounding_max_abs_db_error':observed_error,'rounding_error_bound_db':delta_bound,'rounding_share_gap_theory_bound':bound,'rounding_nominal_float64_tight_bound':rounding_share_gap_bound(.05),'rounding_theory':'For q in[0,1], bounded energy reweighting TV <= tanh(ln(10)*delta_db/20)','label_energy_basis':'unrounded mixture energy; rounded model weights need not pool to raw share exactly','reference_semantics':'whole measured recording content, not verified isolated vessel-only energy','official_hidden_generator_reproduced':False}
    return {'db':rounded,'unrounded_db':unrounded,'share':share,'presence':int(share>0),'cell_share':q,'recipe':recipe}
