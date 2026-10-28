"""Thirteen db-only R5 features, including one explicitly TRAIN-fitted PCA.

Feature families are 1 high-frequency bump, 1 low/high temporal correlation,
10 peak-minute mid-band excess fractions, and 1 PCA3 spectral-shape residual.
They are new definitions, not a reproduction of historical EDA KS numbers.
KS separation of train versus Test is not supervised target validation.

The caller supplies db after its declared preprocessing/quantization policy.
This module never rounds, reads a dataset, fits queries, consumes labels or
recipes, or infers source identities. All PCA state is bound to TRAIN and the
caller's preprocessing policy; the first twelve features are per-window only.
"""
from pathlib import Path
import hashlib
import json
import re
import warnings

import numpy as np
from pipeline.mixtures import per_band_baseline_db


SCHEMA = 'physics_features13_r5_v1'
PCA_VERSION = 'train_window_logmean_shape_pca3_r5_v1'
FREQUENCIES_HZ = np.array([31.6,39.8,50.1,63.1,79.4,100.,125.9,158.5,199.5,251.2,
    316.2,398.1,501.2,631.,794.3,1000.,1258.9,1584.9,1995.3,2511.9,3162.3,3981.1,5011.9,6309.6,7943.3])
LOW_BANDS = np.flatnonzero(FREQUENCIES_HZ <= 200)
HIGH_BANDS = np.flatnonzero((FREQUENCIES_HZ >= 3000) & (FREQUENCIES_HZ <= 8000))
MID_BANDS = np.flatnonzero((FREQUENCIES_HZ >= 300) & (FREQUENCIES_HZ <= 3000))
FIXED_FEATURE_NAMES = ['bump_high_db', 'corr_low_high'] + [
    'peak_mid_excess_fraction_' + ('%g' % FREQUENCIES_HZ[i]).replace('.', 'p') + 'hz' for i in MID_BANDS]
FEATURE_NAMES = FIXED_FEATURE_NAMES + ['pca3_shape_residual_rms_db']
DEFINITIONS = {
    'high': 'sum linear power across 3000<=centre<=8000Hz; only complete high-band minutes; max dB minus median dB',
    'correlation': 'Pearson correlation of low<=200Hz and high>=3000Hz summed-power dB; complete-band paired minutes; >=3 pairs and nonconstant signals; otherwise zero',
    'mid_peak': 'earliest maximum summed 300<=centre<=3000Hz linear-power minute among complete-mid-band minutes',
    'mid_excess': 'relu(peak-minute linear power - power at per-band observed-time 20th percentile dB), divided by total observed window linear energy',
    'pca_input': '25-dimensional log10 observed-time MEAN LINEAR POWER per band, converted to dB and centered by that window observed-band dB mean',
    'pca_fit': 'TRAIN observed-band means impute missing bands; center by TRAIN means; covariance eigendecomposition; top3 orthonormal components; no per-band variance scaling',
    'pca_output': 'RMS reconstruction residual in centered spectral-shape dB, evaluated over query observed bands only; no observed bands -> zero',
}
EXISTING_MAPPING = {
    'bump_high_db': 'Not the existing upper-two-band temporal ranges or adjacent-frequency jumps.',
    'corr_low_high': 'Not present in existing physics45/fixed features; historical EDA used a different <=520Hz low band.',
    'peak_mid_excess_fraction': 'Reuses per_band_baseline_db(.2). Existing baseline_excess_spectrum sums excess over all minutes/all25bands; retrieval applies TRAIN prototype similarities to that sum. This family selects one MID-power peak minute and emits10 explicit normalized values.',
    'pca3_shape_residual_rms_db': 'New fold-TRAIN window-spectrum fit. Historical EDA PCA on observation rows is neither loaded nor reused.',
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def _binding(binding):
    if not isinstance(binding, dict):
        raise ValueError('An explicit TRAIN and preprocessing binding is required')
    result = dict(binding)
    allowed = {'train_npz_sha256', 'data_package_hash', 'source_hash', 'preprocessing_policy_sha256',
               'seed', 'fold', 'train_examples', 'quantization_decimals'}
    if set(result) - allowed:
        raise ValueError('Unexpected labels, recipe/source tags or undeclared preprocessing fields in binding')
    for key in ['train_npz_sha256', 'data_package_hash', 'source_hash', 'preprocessing_policy_sha256']:
        if not isinstance(result.get(key), str) or re.fullmatch('[0-9a-f]{64}', result[key]) is None:
            raise ValueError('Missing or invalid binding hash: ' + key)
    for key in ['seed', 'fold', 'train_examples']:
        value = result.get(key)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < (1 if key == 'train_examples' else 0):
            raise ValueError('Invalid integer TRAIN binding: ' + key)
        result[key] = int(value)
    if result['fold'] > 4:
        raise ValueError('A five-fold source package is required')
    if 'quantization_decimals' not in result:
        raise ValueError('Raw/quantized input policy must be explicit')
    value = result['quantization_decimals']
    if value is not None:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value <= 15:
            raise ValueError('quantization_decimals must be None for raw, or an integer in [0,15]')
        result['quantization_decimals'] = int(value)
    return json.loads(_canonical(result))


def _db(db):
    if isinstance(db, dict):
        raise TypeError('Pass db only; query labels, recipes and source metadata are not feature inputs')
    original = np.asarray(db)
    if original.ndim != 3 or original.shape[1:] != (15, 25) or original.dtype.kind not in 'fiu':
        raise ValueError('Expected real db [N,15,25]')
    if np.isinf(original).any():
        raise ValueError('Infinite db is invalid; unobserved cells use NaN')
    return original.astype(np.float64, copy=False)


def _power(x):
    observed = np.isfinite(x)
    peak = np.max(np.where(observed, x, -np.inf), axis=(1, 2), keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.)
    energy = np.where(observed, np.power(10., np.where(observed, (x - peak) / 10, 0)), 0.)
    return observed, peak, energy


def _group_levels(x, bands):
    values = x[:, :, bands]
    complete = np.isfinite(values).all(axis=2)
    maximum = np.max(np.where(np.isfinite(values), values, -np.inf), axis=2)
    safe_maximum = np.where(np.isfinite(maximum), maximum, 0.)
    relative = np.where(np.isfinite(values), values - safe_maximum[:, :, None], -np.inf)
    summed = np.power(10., relative / 10).sum(axis=2)
    levels = safe_maximum + 10 * np.log10(np.maximum(summed, 1e-300))
    return np.where(complete, levels, np.nan), complete


def _fixed(x):
    n = len(x)
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return np.empty((0, 12), dtype=np.float32), {'high_complete_minutes': empty, 'corr_paired_minutes': empty,
            'corr_defined': empty.astype(bool), 'mid_peak_minute_zero_based': empty}
    high, high_valid = _group_levels(x, HIGH_BANDS)
    low, low_valid = _group_levels(x, LOW_BANDS)
    high_count = high_valid.sum(axis=1)
    high_maximum = np.max(np.where(high_valid, high, -np.inf), axis=1)
    high_maximum = np.where(high_count > 0, high_maximum, 0.)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        high_median = np.nanmedian(high, axis=1)
    high_median = np.where(high_count > 0, high_median, 0.)
    bump = high_maximum - high_median
    paired = high_valid & low_valid
    count = paired.sum(axis=1)
    a = np.where(paired, low, 0.); b = np.where(paired, high, 0.)
    a = np.where(paired, a - np.divide(a.sum(1), count, out=np.zeros(n), where=count > 0)[:, None], 0.)
    b = np.where(paired, b - np.divide(b.sum(1), count, out=np.zeros(n), where=count > 0)[:, None], 0.)
    denominator = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    corr_defined = (count >= 3) & (denominator > 1e-12)
    correlation = np.divide(np.sum(a * b, axis=1), denominator, out=np.zeros(n), where=corr_defined)
    correlation = np.clip(correlation, -1., 1.)
    mid, mid_valid = _group_levels(x, MID_BANDS)
    peak_minute = np.argmax(np.where(mid_valid, mid, -np.inf), axis=1)
    has_peak = mid_valid.any(axis=1)
    observed, window_peak, energy = _power(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        floor = per_band_baseline_db(x, .2)
    floor_energy = np.where(np.isfinite(floor), np.power(10., np.where(np.isfinite(floor), (floor - window_peak) / 10., 0.)), 0.)
    excess = np.where(observed, np.maximum(energy - floor_energy, 0.), 0.)
    peak_excess = excess[np.arange(n), peak_minute][:, MID_BANDS]
    total = energy.sum(axis=(1, 2))
    peak_excess = np.divide(peak_excess, total[:, None], out=np.zeros_like(peak_excess), where=total[:, None] > 0)
    peak_excess[~has_peak] = 0.
    result = np.column_stack([bump, correlation, peak_excess]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite fixed physics features')
    diagnostics = {'high_complete_minutes': high_count, 'corr_paired_minutes': count,
        'corr_defined': corr_defined, 'mid_peak_minute_zero_based': np.where(has_peak, peak_minute, -1)}
    return result, diagnostics


def fixed_physics_features_r5(db, *, return_diagnostics=False):
    """Return [N,12]; no fitted state, target, cohort statistic or quantization."""
    result, diagnostics = _fixed(_db(db))
    return (result, diagnostics) if return_diagnostics else result


def _spectral_shapes(x):
    """Log of observed-time mean linear power; then own-window gain centering."""
    observed, _, energy = _power(x)
    count = observed.sum(axis=1)
    band_observed = count > 0
    mean_power = np.divide(energy.sum(axis=1), count, out=np.zeros((len(x), 25)), where=band_observed)
    logmean = 10 * np.log10(np.maximum(mean_power, 1e-300))
    mean_db = np.divide(np.where(band_observed, logmean, 0.).sum(axis=1), band_observed.sum(axis=1),
        out=np.zeros(len(x)), where=band_observed.sum(axis=1) > 0)
    return np.where(band_observed, logmean - mean_db[:, None], np.nan), band_observed


def physics_pca_payload_sha256(state):
    return hashlib.sha256(_canonical({k: v for k, v in state.items() if k != 'payload_sha256'})).hexdigest()


def fit_physics_pca(train_db, binding, split='train'):
    """Fit PCA3 to processed TRAIN mixture db only, without q/y or old PCA.

    The caller has verified the raw TRAIN NPZ SHA. Its binding also declares
    the preprocessing policy and quantization applied BEFORE this function.
    The digest here additionally identifies the exact processed db array.
    """
    if split != 'train':
        raise ValueError('Physics PCA may only fit the verified TRAIN split')
    identity = _binding(binding)
    x = _db(train_db)
    if len(x) != identity['train_examples']:
        raise ValueError('TRAIN row count differs from its binding')
    shapes, observed = _spectral_shapes(x)
    usable = observed.any(axis=1)
    if not usable.any():
        raise ValueError('No observed TRAIN spectral shapes available for PCA')
    count = observed[usable].sum(axis=0)
    mean = np.divide(np.where(observed[usable], shapes[usable], 0.).sum(axis=0), count,
        out=np.zeros(25), where=count > 0)
    centered = np.where(observed[usable], shapes[usable], mean) - mean
    covariance = centered.T @ centered / max(int(usable.sum()) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    components = vectors[:, order[:3]].T.copy()
    for component in components:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    values = np.maximum(values[order], 0.)
    original = np.asarray(train_db)
    h = hashlib.sha256(_canonical({'shape': list(original.shape), 'dtype': str(original.dtype)}))
    h.update(np.ascontiguousarray(original).tobytes())
    state = {'version': PCA_VERSION, 'feature_schema': SCHEMA, 'feature_names': list(FEATURE_NAMES),
        'definitions': dict(DEFINITIONS), 'fit_split': 'train', 'training_binding': identity,
        'training_db_sha256': h.hexdigest(), 'training_rows': len(x),
        'fit_rows_with_observations': int(usable.sum()), 'all_missing_training_rows_excluded': int((~usable).sum()),
        'pca_components': 3, 'shape_mean_db': mean.tolist(), 'components': components.tolist(),
        'eigenvalues': values[:3].tolist(), 'effective_rank': int((values > max(float(values.max()) * 1e-12, 1e-12)).sum()),
        'observed_band_training_counts': count.tolist(), 'training_targets_used': False,
        'query_requires_labels_or_source_tags': False, 'historical_EDA_PCA_reused': False}
    state['payload_sha256'] = physics_pca_payload_sha256(state)
    validate_physics_pca(state, expected_binding=identity)
    return state


def validate_physics_pca(state, *, expected_binding, expected_payload_sha256=None):
    if not isinstance(state, dict) or state.get('version') != PCA_VERSION or state.get('feature_schema') != SCHEMA:
        raise ValueError('Unsupported physics PCA state')
    if _binding(state.get('training_binding')) != _binding(expected_binding):
        raise ValueError('Physics PCA TRAIN or preprocessing binding differs')
    if state.get('fit_split') != 'train' or state.get('pca_components') != 3 or state.get('feature_names') != FEATURE_NAMES or state.get('definitions') != DEFINITIONS:
        raise ValueError('Physics feature definition or fit scope changed')
    if any(state.get(k) is not False for k in ['training_targets_used', 'query_requires_labels_or_source_tags', 'historical_EDA_PCA_reused']):
        raise ValueError('Physics PCA provenance is unsafe or unspecified')
    mean = np.asarray(state.get('shape_mean_db'), dtype=float)
    components = np.asarray(state.get('components'), dtype=float)
    values = np.asarray(state.get('eigenvalues'), dtype=float)
    if mean.shape != (25,) or components.shape != (3, 25) or values.shape != (3,) or not all(np.isfinite(a).all() for a in [mean, components, values]):
        raise ValueError('Invalid PCA dimensions or values')
    if (values < 0).any() or not np.allclose(components @ components.T, np.eye(3), atol=1e-8, rtol=0):
        raise ValueError('Invalid PCA eigenvalues or orthonormal components')
    if state.get('training_rows') != state['training_binding']['train_examples']:
        raise ValueError('Physics PCA row binding differs')
    actual = physics_pca_payload_sha256(state)
    if actual != state.get('payload_sha256') or expected_payload_sha256 is not None and actual != expected_payload_sha256:
        raise ValueError('Physics PCA payload SHA mismatch')
    return actual


def physics_features_r5(db, state, *, expected_binding, expected_payload_sha256=None, return_diagnostics=False):
    """Transform db-only queries with the frozen TRAIN PCA; return [N,13]."""
    validate_physics_pca(state, expected_binding=expected_binding, expected_payload_sha256=expected_payload_sha256)
    x = _db(db)
    fixed, diagnostics = _fixed(x)
    if len(x):
        shapes, observed = _spectral_shapes(x)
        mean = np.asarray(state['shape_mean_db'], dtype=float)
        components = np.asarray(state['components'], dtype=float)
        filled = np.where(observed, shapes, mean)
        reconstruction = (filled - mean) @ components.T @ components + mean
        difference = np.where(observed, shapes - reconstruction, 0.)
        count = observed.sum(axis=1)
        residual = np.sqrt(np.divide((difference ** 2).sum(axis=1), count, out=np.zeros(len(x)), where=count > 0))
    else:
        residual = np.empty(0); count = np.empty(0, dtype=np.int64)
    features = np.column_stack([fixed, residual]).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite PCA physics feature')
    diagnostics['pca_observed_bands'] = count
    return (features, diagnostics) if return_diagnostics else features


def save_physics_pca(path, state, *, expected_binding):
    payload = validate_physics_pca(state, expected_binding=expected_binding)
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError('A fitted physics PCA file is immutable')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(state, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return {'filename': path.name, 'file_sha256': _sha_file(path), 'payload_sha256': payload,
            'training_binding': state['training_binding'], 'feature_schema': SCHEMA}


def load_physics_pca(path, *, expected_binding, expected_file_sha256, expected_payload_sha256=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or _sha_file(path) != expected_file_sha256:
        raise ValueError('Frozen physics PCA file SHA mismatch')
    state = json.loads(path.read_text())
    validate_physics_pca(state, expected_binding=expected_binding, expected_payload_sha256=expected_payload_sha256)
    return state
