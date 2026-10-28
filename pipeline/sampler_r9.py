"""Minute-first controlled additions. No Test fitting or source matching.

The reference is first reduced from complete native 2-second band powers to
complete 60-second means. All EQ, crop and placement below use that matrix.
No time warp, waveform repetition, interpolation or forced boundary fade.
"""
from __future__ import annotations
import numpy as np
from pipeline.relative_grid_r6 import quantization_pooling_bound


def native_minutes(power_2s):
    x = np.asarray(power_2s, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 25 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Native finite nonnegative [segments,25] band power required')
    count = len(x)//30
    if count < 1:
        raise ValueError('No complete native minute; never pad/repeat short audio')
    result = x[:count*30].reshape(count,30,25).mean(axis=1)
    if not (result.sum(axis=1)>0).all():
        raise ValueError('Zero-power native minute unsupported')
    return result


def energy_weights(db):
    x = np.asarray(db, dtype=np.float64)
    o = np.isfinite(x)
    if x.shape != (15,25) or not o.any() or np.isinf(x).any():
        raise ValueError('Observed/NaN [15,25] required')
    p = np.where(o,10.**((x-np.nanmax(x))/10.),0.)
    return p/p.sum()


def smooth_eq(rng, maximum=6.):
    if not 0 <= maximum <= 6:
        raise ValueError('Declared smooth EQ bound is <=6dB')
    axis = np.linspace(-1,1,25)
    curve = rng.uniform(-maximum,maximum)*axis + rng.uniform(-maximum/2,maximum/2)*(axis*axis-(axis*axis).mean())
    top = float(np.max(np.abs(curve)))
    if top>maximum and top>0: curve *= maximum/top
    return curve


def place_minutes(native, rng, family):
    m = len(native)
    if family == 'native_short':
        length = int(rng.integers(1,min(3,m)+1))
        begin = int(rng.integers(m-length+1))
        part = native[begin:begin+length]
        start = int(rng.integers(16-length))
    elif family == 'native_inside':
        length = min(15,m)
        begin = int(rng.integers(m-length+1))
        part = native[begin:begin+length]
        start = int(rng.integers(16-length))
    elif family == 'native_edge':
        part = native
        begin = 0
        length = m
        starts = np.arange(-m+1,15)
        starts = starts[(starts<0)|(starts+m>15)]
        if len(starts):
            start = int(rng.choice(starts))
        else:
            # A one-minute source cannot straddle an integer minute boundary.
            # Preserve its real minute and document the geometric fallback.
            start = int(rng.integers(15))
    else:
        raise ValueError('Unknown minute-first family')
    result = np.zeros((15,25),dtype=np.float64)
    left,right = max(0,start),min(15,start+length)
    if right<=left: raise RuntimeError('No geometric overlap')
    result[left:right] = part[left-start:right-start]
    return result, {'family':family,'native_minutes':m,'crop_begin_minute':begin,
                    'selected_minutes':length,'window_start_minute':start,
                    'geometric_overlap_minutes':right-left,
                    'edge_one_minute_fallback':family=='native_edge' and m==1}


def synthesize(background_db, reference_power_2s, rng, *, family='native_inside',
               snr_range=(-10.,10.), eq_max_db=6., no_source=False,
               reference_minute_power=None):
    """Return a known-component mixture, rounded db, raw q/share and receipt."""
    if type(no_source) is not bool: raise TypeError('Explicit no_source bool required')
    x = np.array(background_db,dtype=np.float64,copy=True)
    o = np.isfinite(x)
    energy_weights(x)
    peak = float(np.nanmax(x))
    b = np.where(o,10.**((x-peak)/10.),0.)
    bmedian = float(np.median(b.sum(axis=1)[o.any(axis=1)]))
    added = np.zeros((15,25),dtype=np.float64)
    trajectory = {'family':'no_source','native_minutes':0}
    eq = np.zeros(25)
    alpha = 0.
    snr = None
    full_peak = None
    if not no_source:
        # Conversion is deliberately before EQ and every temporal operation.
        if reference_minute_power is None:
            minute = native_minutes(reference_power_2s)
        else:
            if reference_power_2s is not None:
                raise ValueError('Pass native 2-second powers OR the verified minute cache')
            minute = np.asarray(reference_minute_power, dtype=np.float64)
            if (minute.ndim != 2 or minute.shape[1] != 25 or not len(minute)
                    or not np.isfinite(minute).all() or (minute < 0).any()
                    or not (minute.sum(axis=1) > 0).all()):
                raise ValueError('Finite nonnegative native minute cache required')
        eq = smooth_eq(rng,eq_max_db)
        transformed = minute*10.**(eq[None,:]/10.)
        full_peak = float(transformed.sum(axis=1).max())
        placed,trajectory = place_minutes(transformed,rng,family)
        snr = float(rng.uniform(*snr_range))
        alpha = float(10.**(snr/10.)*bmedian/full_peak)
        added = np.where(o,alpha*placed,0.)
    mixture = b+added
    q = np.divide(added,mixture,out=np.zeros_like(added),where=o)
    share = float(added.sum()/mixture.sum())
    raw = np.full((15,25),np.nan)
    raw[o] = 10.*np.log10(mixture[o])+peak
    rounded = np.round(raw,1).astype(np.float32)
    w = energy_weights(rounded)
    residual = abs(float((energy_weights(raw)*q).sum())-share)
    half_ulp = float(np.max(np.abs(np.spacing(rounded[o])))/2.)
    bound = quantization_pooling_bound(.05+half_ulp+1e-12)
    gap = abs(float((w*q).sum())-share)
    if residual>1e-10 or gap>bound+2e-7 or not np.array_equal(np.isfinite(rounded),o):
        raise RuntimeError('Physical mixture/mask/rounding bound failed')
    if not np.isfinite(q).all() or (q<0).any() or (q>1).any():
        raise RuntimeError('Invalid cell fraction')
    recipe = {'version':'minute_first_r9_v1','minute_first':True,'time_warp':False,
              'forced_boundary_fade':False,'source_repetition':False,
              'no_source_request':no_source,'observed_added_present':bool(share>0),
              'observed_minutes':int(o.any(axis=1).sum()),'requested_full_peak_snr_db':snr,
              'full_native_minute_peak_after_EQ':full_peak,'background_median_minute_power':bmedian,
              'alpha':alpha,'reference_eq_db':eq.tolist(),'trajectory':trajectory,
              'raw_label_identity_error':residual,'rounded_pool_gap':gap,'rounding_bound':bound,
              'source_interpretation':'whole measured recording, not verified isolated vessel-only sound'}
    return {'db':rounded,'unrounded_db':raw,'cell_share':q.astype(np.float32),
            'share':np.float32(share),'presence':np.int64(share>0),'recipe':recipe}
