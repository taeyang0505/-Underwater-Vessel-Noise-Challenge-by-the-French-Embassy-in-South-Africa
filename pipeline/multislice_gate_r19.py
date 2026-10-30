"""Fail-closed R19 adoption gate for already-produced local predictions.

This module neither trains nor calibrates. ``panels`` maps the four required
names to share/weights/active_minutes arrays. ``bases`` maps panel names to
baseline share predictions. ``candidates`` maps candidate names to the same
prediction mapping. Missing, empty, malformed, or nonfinite required data fail.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class GatePolicy:
    version: str = 'r19_multislice_v1'
    baseline: str = 'old_B1'
    panels: tuple[str, ...] = (
        'local_gate', 'domain_holdout', 'native12_VS_hf_high_negative',
        'native12_VS_st_francis_dense_negative',
    )
    ordinary_axes: tuple[tuple[str, str], ...] = (
        ('all', 'rmse'), ('y=0', 'rmse'), ('y=0', 'positive_bias'),
        ('0<y<=0.10', 'rmse'), ('0<y<=0.10', 'positive_bias'),
    )
    long_panels: tuple[str, ...] = (
        'native12_VS_hf_high_negative', 'native12_VS_st_francis_dense_negative',
    )
    long_axis: tuple[str, str] = ('positive_active>=12', 'rmse')
    absolute_tolerance: float = 1e-6
    positive_bias: str = 'max(weighted_mean(prediction-target),0)'
    missing_data: str = 'reject'
    compensation_across_axes: bool = False


DEFAULT_POLICY = GatePolicy()


def policy_dict(policy: GatePolicy = DEFAULT_POLICY) -> dict:
    """Return a detached JSON-safe description, never the mutable default."""
    return json.loads(json.dumps(asdict(policy)))


def policy_sha256(policy: GatePolicy = DEFAULT_POLICY) -> str:
    return hashlib.sha256(json.dumps(policy_dict(policy), sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def _array(value, n=None, *, bounded=False):
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or (n is not None and len(a) != n) or not len(a):
        raise ValueError('Expected nonempty aligned one-dimensional array')
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite required data')
    if bounded and ((a < 0).any() or (a > 1).any()):
        raise ValueError('Target/prediction outside [0,1]')
    return a


def _prepare(panel, base, prediction, needs_long):
    if not isinstance(panel, Mapping):
        raise ValueError('Missing panel')
    y = _array(panel['share'], bounded=True)
    b = _array(base, len(y), bounded=True)
    s = _array(prediction, len(y), bounded=True)
    w = _array(panel.get('weights', np.ones(len(y))), len(y))
    if (w < 0).any() or not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError('Invalid panel weights')
    if needs_long:
        active = _array(panel['active_minutes'], len(y))
        if ((active < 0) | (active > 15) | (active != np.floor(active))).any():
            raise ValueError('Invalid active minute count')
    else:
        active = None
    return y, b, s, w, active


def _mask(y, active, name):
    if name == 'all': return np.ones(len(y), dtype=bool)
    if name == 'y=0': return y == 0
    if name == '0<y<=0.10': return (y > 0) & (y <= .10)
    if name == 'positive_active>=12': return (y > 0) & (active >= 12)
    raise ValueError('Unknown slice: ' + name)


def _metric(prediction, y, w, name):
    w = w / w.sum()
    error = prediction - y
    if name == 'rmse': return float(np.sqrt(w @ (error * error)))
    if name == 'positive_bias': return max(float(w @ error), 0.)
    raise ValueError('Unknown metric: ' + name)


def evaluate_candidate(panels, bases, candidates, policy: GatePolicy = DEFAULT_POLICY):
    """Evaluate every candidate on every required axis; no averaging of failures.

    Returns a JSON-safe dictionary. A candidate passes iff all 22 checks pass.
    This gate does not choose calibration parameters or use report/Test data.
    """
    if policy.absolute_tolerance < 0 or not np.isfinite(policy.absolute_tolerance):
        raise ValueError('Invalid tolerance')
    results = {}
    for name, predictions in candidates.items():
        checks = []
        for panel_name in policy.panels:
            axes = list(policy.ordinary_axes)
            needs_long = panel_name in policy.long_panels
            if needs_long: axes.append(policy.long_axis)
            try:
                d = _prepare(panels.get(panel_name), bases.get(panel_name),
                             predictions.get(panel_name) if isinstance(predictions, Mapping) else None,
                             needs_long)
                error = None
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                d = None; error = str(exc)
            for slice_name, metric_name in axes:
                row = {'candidate': name, 'panel': panel_name, 'slice': slice_name,
                       'metric': metric_name, 'n': 0, 'baseline': None,
                       'candidate_value': None, 'delta': None,
                       'tolerance': policy.absolute_tolerance, 'passed': False,
                       'available': False, 'error': error}
                if d is not None:
                    y, b, s, w, active = d
                    ix = _mask(y, active, slice_name)
                    row['n'] = int(ix.sum())
                    if not ix.any() or w[ix].sum() <= 0:
                        row['error'] = 'Empty or zero-weight required slice'
                    else:
                        before = _metric(b[ix], y[ix], w[ix], metric_name)
                        after = _metric(s[ix], y[ix], w[ix], metric_name)
                        delta = after - before
                        row.update(baseline=before, candidate_value=after, delta=delta,
                                   passed=delta <= policy.absolute_tolerance,
                                   available=True, error=None)
                checks.append(row)
        results[name] = {'passed': all(c['passed'] for c in checks),
                         'failed_axes': sum(not c['passed'] for c in checks),
                         'axes': len(checks), 'checks': checks}
    return {'policy': policy_dict(policy), 'policy_sha256': policy_sha256(policy),
            'results': results}
