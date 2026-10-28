"""Stateless R6-v2 band-power summaries from db only (pure NumPy).

Input is [N,15,25], quantized here as float64 round(db,1). A band-group
minute is usable only when every constituent frequency is observed. Group
levels are 10*log10(sum_f 10**(db_f/10)); summing dB is never used.

Nine values are followed by nine defined masks and nine valid-minute counts.
Undefined temporal summaries are filled with zero, with mask/count retained.
All-NaN windows and single-minute support are safe diagnostic inputs; this
does not replace a caller's independent production-input eligibility gate.

The summaries are invariant to a common additive dB offset AFTER rounding.
For offsets applied BEFORE rounding, exact invariance additionally requires
rounding to commute (e.g. decimal-grid gains away from floating-point ties).
Arbitrary pre-round gains can cross quantization boundaries, especially for
correlations of nearly constant series. Static per-band offset invariance is
NOT guaranteed: summed-power weights, temporal shapes and tilt can change.

No fit API, learned state, source metadata, labels, query cohort statistics,
file I/O, Torch, background-prior retrieval or Test access are used.
"""
from __future__ import annotations

import numpy as np


SCHEMA = 'r6_physics_bandpower_v2_27'
QUANTIZATION_DECIMALS = 1
GROUP_BANDS = {
    'low': tuple(range(0, 9)),       # nominal <=200Hz: 31.6--199.5
    'mid': tuple(range(10, 20)),     # nominal 300--3000Hz: 316.2--2511.9
    'high': tuple(range(20, 25)),    # nominal 3--8kHz: 3162.3--7943.3
    'safe': tuple(range(15, 23)),    # nominal 1--5kHz: 1000--5011.9
}
SIGNAL_NAMES = (
    'bump_low_db', 'bump_mid_db', 'bump_high_db', 'bump_safe_db',
    'corr_low_mid', 'corr_low_high', 'corr_low_safe',
    'std_broadband_db', 'tilt_median_high_minus_low_db',
)
FEATURE_NAMES = (SIGNAL_NAMES
                 + tuple(n + '_defined' for n in SIGNAL_NAMES)
                 + tuple(n + '_valid_minutes' for n in SIGNAL_NAMES))
FEATURE_COUNT = len(FEATURE_NAMES)


def feature_specification() -> dict:
    """Return an independent JSON-compatible definition/name/count receipt."""
    return {
        'schema': SCHEMA,
        'feature_count': FEATURE_COUNT,
        'feature_names': list(FEATURE_NAMES),
        'signal_names': list(SIGNAL_NAMES),
        'groups_zero_based': {k: list(v) for k, v in GROUP_BANDS.items()},
        'safe_band_upper_actual_centre_hz': 5011.9,
        'quantization_decimals': QUANTIZATION_DECIMALS,
        'query_inputs': ['db'],
        'learned_state': False,
        'definitions': {
            'group_level_db': '10log10(sum linear powers); all group frequencies observed in the minute',
            'bump': 'max(group_level_db)-median(group_level_db), >=2 complete minutes',
            'correlation': 'Pearson of paired complete group dB levels; >=3 pairs and both nonconstant',
            'broadband_std': 'population std(ddof=0) of all25-band summed-power dB; >=2 complete minutes',
            'tilt': 'median(high_level_db-low_level_db) over >=1 paired complete minute',
            'missing': 'undefined value=0 plus explicit defined mask and integer valid-minute count',
            'constant_tolerance': 'centered sum of squares >1e-20 for each Pearson series',
        },
        'common_offset_invariance': 'exact mathematically on quantized input; pre-round arbitrary gain can change quantization',
        'per_band_offset_invariance_guaranteed': False,
    }


def _quantized(db) -> np.ndarray:
    if isinstance(db, dict):
        raise TypeError('Pass db only; labels, masks, recipes and source tags are not query inputs')
    x = np.asarray(db)
    if x.ndim != 3 or x.shape[1:] != (15, 25) or x.dtype.kind != 'f':
        raise ValueError('Expected floating db[N,15,25], with NaN for missing cells')
    if np.isinf(x).any():
        raise ValueError('Infinite db is invalid; missing cells must be NaN')
    # Owned copy: never mutate the caller, including a read-only view.
    x = np.round(x.astype(np.float64), QUANTIZATION_DECIMALS)
    if np.isinf(x).any():
        raise ValueError('Quantization overflow')
    return x


def _levels(x: np.ndarray, bands) -> tuple[np.ndarray, np.ndarray]:
    """Stable group sum-power levels; never sum an incomplete group."""
    a = x[:, :, bands]
    valid = np.isfinite(a).all(axis=2)
    # Replace every incomplete group by a harmless constant before arithmetic.
    a = np.where(valid[:, :, None], a, 0.)
    peak = a.max(axis=2)
    energy = np.power(10., (a - peak[:, :, None]) / 10.)
    level = peak + 10. * np.log10(energy.sum(axis=2))
    return np.where(valid, level, 0.), valid


def _median(v, valid):
    """Small fixed15-minute axis; avoid all-NaN NumPy warning/fill ambiguity."""
    count = valid.sum(axis=1)
    ordered = np.sort(np.where(valid, v, np.inf), axis=1)
    lo = np.maximum((count - 1) // 2, 0)
    hi = np.maximum(count // 2, 0)
    rows = np.arange(len(v))
    a = np.where(count > 0, ordered[rows, lo], 0.)
    b = np.where(count > 0, ordered[rows, hi], 0.)
    return a / 2. + b / 2.


def _correlation(a, b, valid):
    count = valid.sum(axis=1)
    # Own-window centering also avoids large common gain in Pearson products.
    am = np.divide(np.where(valid, a, 0.).sum(1), count,
                   out=np.zeros(len(a)), where=count > 0)
    bm = np.divide(np.where(valid, b, 0.).sum(1), count,
                   out=np.zeros(len(a)), where=count > 0)
    ac = np.where(valid, a - am[:, None], 0.)
    bc = np.where(valid, b - bm[:, None], 0.)
    aa = (ac * ac).sum(1)
    bb = (bc * bc).sum(1)
    defined = (count >= 3) & (aa > 1e-20) & (bb > 1e-20)
    den = np.sqrt(aa) * np.sqrt(bb)
    corr = np.divide((ac * bc).sum(1), den,
                     out=np.zeros(len(a)), where=defined)
    return np.clip(corr, -1., 1.), defined, count


def physics_features_r6_v2(db, *, return_diagnostics=False):
    """Return finite float32[N,27], optionally (features, diagnostics).

    Counts/defined masks are features as well as diagnostics. No feature
    depends on other windows, so chunking and query order cannot alter it.
    Broad std uses all25 complete frequencies; low/high tilt uses paired
    group minutes. A zero feature is never represented as measured validity
    when its summary is undefined. Empty batches return shape[0,27].
    """
    x = _quantized(db)
    levels = {name: _levels(x, bands) for name, bands in GROUP_BANDS.items()}
    value, defined, counts = [], [], []
    for name in ['low', 'mid', 'high', 'safe']:
        level, valid = levels[name]
        count = valid.sum(axis=1)
        ok = count >= 2
        maximum = np.max(np.where(valid, level, -np.inf), axis=1)
        maximum = np.where(count > 0, maximum, 0.)
        bump = maximum - _median(level, valid)
        value.append(np.where(ok, np.maximum(bump, 0.), 0.))
        defined.append(ok)
        counts.append(count)
    for name in ['mid', 'high', 'safe']:
        low, low_valid = levels['low']
        level, valid = levels[name]
        corr, ok, count = _correlation(low, level, low_valid & valid)
        value.append(corr)
        defined.append(ok)
        counts.append(count)
    broad, broad_valid = _levels(x, tuple(range(25)))
    count = broad_valid.sum(axis=1)
    mean = np.divide(np.where(broad_valid, broad, 0.).sum(1), count,
                     out=np.zeros(len(x)), where=count > 0)
    centred = np.where(broad_valid, broad - mean[:, None], 0.)
    var = np.divide((centred * centred).sum(1), count,
                    out=np.zeros(len(x)), where=count > 0)
    value.append(np.where(count >= 2, np.sqrt(np.maximum(var, 0.)), 0.))
    defined.append(count >= 2)
    counts.append(count)
    high, high_valid = levels['high']
    low, low_valid = levels['low']
    paired = high_valid & low_valid
    count = paired.sum(axis=1)
    value.append(_median(high - low, paired))
    defined.append(count >= 1)
    counts.append(count)
    values = np.stack(value, axis=1)
    definition = np.stack(defined, axis=1)
    count_array = np.stack(counts, axis=1)
    result = np.concatenate([values, definition, count_array], axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite physics summary; unsupported numeric dynamic range')
    if return_diagnostics:
        return result, {
            'schema': SCHEMA, 'feature_count': FEATURE_COUNT,
            'feature_names': list(FEATURE_NAMES), 'signal_names': list(SIGNAL_NAMES),
            'defined': definition, 'valid_minute_counts': count_array,
            'group_complete_minutes': {k: v[1].sum(1) for k, v in levels.items()},
            'quantization_decimals': QUANTIZATION_DECIMALS,
        }
    return result
