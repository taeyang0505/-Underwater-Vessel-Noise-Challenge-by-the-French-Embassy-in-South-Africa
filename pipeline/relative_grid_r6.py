"""R6 pure NumPy db-only relative grids. No Torch/OpenMP engine import.

Inputs are 15 minutes x 25 fixed third-octave bands (31.6--7943.3 Hz).
Missing cells are NaN, never zero-valued measurements. All features AND the
pooling weights use Q(db)=round(db, 1), computed in float64. Supervision is
external: this module neither accepts nor changes known q/y/presence labels.

Channels, with med_f=observed temporal median, p20_f=temporal 20% quantile,
med_win=median over all observed window cells, and adjacent-pair mask P:
  0 (Q-med_f)/10; 1 (Q-p20_f)/15; 2 P*(Q[t]-Q[t-1])/10;
  3 1[f>=316.2Hz]*relu(channel2)*co_rise[t];
  4 (med_f-med_win)/20; 5 observed mask.
co_rise[t] = tanh(mean_low(max(delta_db, 0))/3.0), where mean_low averages
the observed adjacent pairs among the first 9 bands (31.6--199.5 Hz,
nominal <=200 Hz). Its denominator excludes missing pairs; no pairs gives
zero. The 3-dB divisor is a pre-fit numerical continuity scale, selected
before any dataset fit or held-out model performance was viewed. This
replaces the initial hard positive-sign fraction, which jumped at zero.
All non-mask channels are zero in missing cells.
Channel 3 is zero in indices 0--9; indices 10--24 (316.2--7943.3 Hz,
nominal >=300 Hz) measure their own rise gated by simultaneous low-band rise.

Common additive dB gain leaves these features and normalized weights
unchanged AFTER rounding. For gain applied BEFORE rounding, this is exact
only when rounding commutes (e.g. 0.1-dB-grid gains away from floating-point
ties). Arbitrary gains can cross quantization boundaries. Channels 0--3 are
also invariant to static per-band offsets under that same rounding caveat;
channel 4 and the energy weights are deliberately NOT band-offset invariant.

The weights are proportional to 10**(Q/10), not to the dB numbers. Keeping
the original raw-precision labels is intentional and leaves a bounded
pooling mismatch; see quantization_pooling_bound(). This is an ablation
representation, not a claim that quantization preserves exact raw physics.
"""
from __future__ import annotations

import math
import warnings
import numpy as np


FEATURE_NAMES = (
    'band_temporal_median_relative_div10',
    'band_temporal_q20_relative_div15',
    'adjacent_observed_delta_div10',
    'positive_mid_high_delta_times_smooth_low_band_co_rise',
    'band_median_minus_window_median_div20',
    'observed_mask',
)
LOW_BAND_COUNT = 9
MID_HIGH_BAND_START = 10
QUANTIZATION_DECIMALS = 1


def quantization_pooling_bound(max_abs_db_error: float = .05) -> float:
    """Sharp general bound on |sum(q*w_raw)-sum(q*w_rounded)| for 0<=q<=1.

    If each energy is reweighted by r in [exp(-h),exp(h)], h=ln(10)*delta/10,
    the largest total-variation distance is tanh(h/2). Therefore the bound
    is tanh(ln(10)*delta/20), not twice this value. Mask must be unchanged.
    Floating-point error in saved weights is additional to this real-valued
    quantization bound. This function does not inspect or redefine labels.
    """
    if not np.isfinite(max_abs_db_error) or max_abs_db_error < 0:
        raise ValueError('A finite nonnegative dB error is required')
    return math.tanh(math.log(10.) * float(max_abs_db_error) / 20.)


def quantized_db(db) -> np.ndarray:
    """Return an owned float64 [N,15,25] array; db is the ONLY query input."""
    if isinstance(db, dict):
        raise TypeError('Pass db only; labelled dictionaries are forbidden')
    x = np.asarray(db)
    if (x.ndim != 3 or x.shape[1:] != (15, 25) or not len(x)
            or not np.issubdtype(x.dtype, np.floating)):
        raise ValueError('Expected nonempty floating db[N,15,25]')
    if np.isinf(x).any() or not np.isfinite(x).any(axis=(1, 2)).all():
        raise ValueError('Infinite values and wholly unobserved windows are invalid')
    return np.round(x.astype(np.float64), QUANTIZATION_DECIMALS)


def relative_grid(db) -> tuple[np.ndarray, np.ndarray]:
    """Return x6[N,6,15,25], w[N,15,25], both finite float32 owned arrays.

    Windows are transformed independently. No fitted parameters, labels,
    source IDs, background priors, or cross-window statistics are accepted.
    Absent entire bands are supported and receive zero features/weights.
    """
    x = quantized_db(db)
    observed = np.isfinite(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(x, axis=1, keepdims=True)
        p20 = np.nanquantile(x, .2, axis=1, keepdims=True, method='linear')
        window_med = np.nanmedian(x, axis=(1, 2), keepdims=True)
    delta = np.zeros_like(x)
    pairs = observed[:, 1:] & observed[:, :-1]
    delta[:, 1:] = np.where(pairs, x[:, 1:] - x[:, :-1], 0.) / 10.
    low_pairs = pairs[:, :, :LOW_BAND_COUNT]
    low_positive_db = np.where(low_pairs,
                               np.maximum(delta[:, 1:, :LOW_BAND_COUNT]*10., 0.), 0.)
    co_rise = np.zeros(x.shape[:2] + (1,), dtype=np.float64)
    count = low_pairs.sum(axis=2, keepdims=True)
    low_mean_db = np.divide(low_positive_db.sum(axis=2, keepdims=True), count,
                            out=np.zeros_like(count, dtype=np.float64), where=count > 0)
    co_rise[:, 1:] = np.tanh(low_mean_db / 3.)
    coherence = np.maximum(delta, 0.) * co_rise
    coherence[:, :, :MID_HIGH_BAND_START] = 0.
    channels = np.stack([
        np.where(observed, (x-med)/10., 0.),
        np.where(observed, (x-p20)/15., 0.),
        delta,
        coherence,
        np.where(observed, np.broadcast_to((med-window_med)/20., x.shape), 0.),
        observed.astype(np.float64),
    ], axis=1)
    peak = np.nanmax(x, axis=(1, 2), keepdims=True)
    energy = np.where(observed, np.power(10., (x-peak)/10.), 0.)
    if not np.isfinite(energy).all() or not (energy[observed] > 0).all():
        raise ValueError('Observed energy underflow/overflow is unsupported')
    weights = energy / energy.sum(axis=(1, 2), keepdims=True)
    if not np.isfinite(channels).all():
        raise ValueError('Non-finite relative features')
    return channels.astype(np.float32), weights.astype(np.float32)
