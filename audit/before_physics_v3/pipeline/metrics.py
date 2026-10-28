"""Development metrics for labelled synthetic examples, never challenge Test.

RMSE is the sole primary selection metric. Other metrics expose systematic
failures; they do not silently change the competition objective. Group bootstrap
intervals describe uncertainty conditional on the available source collection.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score


def _vectors(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Expected nonempty, equally sized one-dimensional vectors')
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError('Non-finite values cannot be silently dropped from metrics')
    return y, p


def _basic(y, p):
    e = p - y
    return {'n': int(len(y)), 'rmse': float(np.sqrt(np.mean(e * e))),
            'mae': float(np.abs(e).mean()), 'bias_prediction_minus_truth': float(e.mean())}


def block_bootstrap_rmse(y, p, groups, n_resamples=500, seed=20261024):
    """Resample complete source groups, not correlated synthetic rows."""
    y, p = _vectors(y, p)
    g = np.asarray(groups).astype(str)
    if g.shape != y.shape:
        raise ValueError('Groups must align exactly with labelled examples')
    keys, idx = np.unique(g, return_inverse=True)
    counts = np.bincount(idx)
    sse = np.bincount(idx, weights=(p - y) ** 2)
    if len(keys) < 2:
        return {'groups': len(keys), 'interval_95': None,
                'reason': 'At least two independent groups are required'}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_resamples):
        draw = rng.integers(0, len(keys), len(keys))
        vals.append(np.sqrt(sse[draw].sum() / counts[draw].sum()))
    return {'groups': len(keys), 'resamples': n_resamples,
            'interval_95': np.quantile(vals, [.025, .975]).tolist(),
            'few_groups_warning': bool(len(keys) < 10),
            'scope': 'One-way group bootstrap; not a joint two-way source uncertainty bound.'}


def evaluate_regression(y, p, presence=None, scores=None, db=None, groups=None):
    y, p = _vectors(y, p)
    if ((y < 0) | (y > 1)).any():
        raise ValueError('Share truth must be bounded in [0, 1]')
    result = _basic(y, p)
    result['score_1_minus_rmse'] = 1 - result['rmse']
    result['primary_selection_metric'] = 'rmse'
    denom = np.sum((y - y.mean()) ** 2)
    result['r2'] = float(1 - np.sum((p - y) ** 2) / denom) if denom > 0 else None
    result['spearman'] = float(spearmanr(y, p).statistic) if np.ptp(y) > 0 and np.ptp(p) > 0 else None
    ae = np.abs(p - y)
    result['absolute_error_quantiles'] = dict(zip(['p50', 'p90', 'p95', 'p99'], np.quantile(ae, [.5, .9, .95, .99]).tolist()))
    tail = np.argsort(ae)[-max(1, int(np.ceil(.1 * len(y)))):]
    result['worst_error_decile_rmse'] = _basic(y[tail], p[tail])['rmse']
    result['outside_unit_interval_count'] = int(((p < 0) | (p > 1)).sum())
    result['saturation_fraction'] = float(((p <= .001) | (p >= .999)).mean())
    slices = {}
    labels = ['zero', '(0,.1]', '(.1,.3]', '(.3,.6]', '(.6,.9]', '(.9,1]']
    bins = [(y == 0), (y > 0) & (y <= .1), (y > .1) & (y <= .3),
            (y > .3) & (y <= .6), (y > .6) & (y <= .9), y > .9]
    for name, mask in zip(labels, bins):
        slices[name] = _basic(y[mask], p[mask]) if mask.any() else {'n': 0}
    result['target_slices'] = slices
    calibration = []
    for i in range(10):
        mask = (p >= i / 10) & ((p < (i + 1) / 10) if i < 9 else (p <= 1))
        if mask.any():
            calibration.append({'prediction_bin': [i / 10, (i + 1) / 10],
                'n': int(mask.sum()), 'mean_prediction': float(p[mask].mean()),
                'mean_truth': float(y[mask].mean()), **_basic(y[mask], p[mask])})
    result['regression_calibration_bins'] = calibration
    if db is not None:
        db = np.asarray(db)
        if db.shape != (len(y), 15, 25):
            raise ValueError('Expected aligned 15 x 25 band-energy windows')
        minutes = np.isfinite(db).any(2).sum(1)
        result['coverage_slices'] = {}
        for lo, hi in [(0, 5), (6, 12), (13, 15)]:
            mask = (minutes >= lo) & (minutes <= hi)
            result['coverage_slices'][f'{lo}-{hi}_observed_minutes'] = _basic(y[mask], p[mask]) if mask.any() else {'n': 0}
    if presence is not None or scores is not None:
        if presence is None or scores is None:
            raise ValueError('Both presence and scores are required together')
        z, s = _vectors(presence, scores)
        if z.shape != y.shape or not np.isin(z, [0, 1]).all() or ((s < 0) | (s > 1)).any():
            raise ValueError('Expected aligned binary addition labels and bounded diagnostic scores')
        two_classes = len(np.unique(z)) == 2
        ece = 0.
        for i in range(10):
            mask = (s >= i / 10) & ((s < (i + 1) / 10) if i < 9 else (s <= 1))
            if mask.any():
                ece += mask.mean() * abs(s[mask].mean() - z[mask].mean())
        result['controlled_addition_detection'] = {
            'roc_auc': float(roc_auc_score(z, s)) if two_classes else None,
            'average_precision': float(average_precision_score(z, s)) if two_classes else None,
            'brier': float(np.mean((s - z) ** 2)), 'ece_10_equal_width_bins': float(ece),
            'positive_prevalence': float(z.mean()),
            'semantics': 'Controlled added-reference presence, not natural vessel truth; share proxies are not calibrated presence probabilities.'}
    if groups is not None:
        mapping = groups if isinstance(groups, dict) else {'source_group': groups}
        result['group_metrics'] = {}
        for axis, raw in mapping.items():
            g = np.asarray(raw).astype(str)
            if g.shape != y.shape:
                raise ValueError(f'Misaligned groups on axis {axis}')
            rows = [{'group': key, **_basic(y[g == key], p[g == key])} for key in np.unique(g)]
            bootstrap = block_bootstrap_rmse(y, p, g) if 'reference' not in axis.lower() else {
                'interval_95': None,
                'reason': 'Reference combinations can share underlying sources; combination labels are not independent bootstrap blocks.',
                'scope': 'Descriptive slices only. Use source-isolated outer-fold variation for uncertainty.'}
            result['group_metrics'][axis] = {
                'n_groups': len(rows), 'macro_group_rmse': float(np.mean([v['rmse'] for v in rows])),
                'worst_group_rmse': float(max(v['rmse'] for v in rows)),
                'groups': rows, 'source_group_rmse_bootstrap': bootstrap}
    return result
