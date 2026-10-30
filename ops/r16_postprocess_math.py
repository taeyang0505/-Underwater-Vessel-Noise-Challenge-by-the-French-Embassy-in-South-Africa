"""Pure, fit-fixed postprocessing candidates; no file I/O or training side effects.

Only permitted local fit data may be passed to ``fit_*``/``select_*``. Their
JSON-serializable states are then frozen before application to any evaluation
query. Application never computes query-cohort means, standard deviations or
ranks. These candidates do not modify the immutable production inference code.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np


METHODS = ("probability", "standardized_logit", "ecdf")
DEFAULT_EPSILONS = (0.0, 0.01, 0.02, 0.03, 0.04)


def _finite(a, name):
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must contain only finite values")
    return a


def _unit(a, name):
    a = _finite(a, name)
    if ((a < 0) | (a > 1)).any():
        raise ValueError(f"{name} must lie in [0,1]")
    return a


def _probabilities(a):
    a = _unit(a, "probabilities")
    if a.ndim != 2 or not a.shape[1]:
        raise ValueError("probabilities must have shape [N, members]")
    return a


def validate_convex_weights(weights, n_members):
    """Validate, without silently normalizing, nonnegative weights summing to 1."""
    w = _finite(weights, "weights")
    if w.shape != (n_members,) or (w < 0).any():
        raise ValueError("weights must be nonnegative with one per member")
    if not np.isclose(w.sum(), 1.0, rtol=0, atol=1e-9):
        raise ValueError("weights must sum to one")
    return w


def convex_pool(values, weights):
    """Pool a finite [N, members] matrix with fixed convex weights."""
    x = _finite(values, "values")
    if x.ndim != 2:
        raise ValueError("values must have shape [N, members]")
    return x @ validate_convex_weights(weights, x.shape[1])


def _logit(p, epsilon):
    p = np.clip(p, epsilon, 1 - epsilon)
    return np.log(p) - np.log1p(-p)


def _sigmoid(x):
    # Prevent overflow; this is a fixed scalar transformation, not calibration.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


def fit_score_transform(fit_probabilities, method="probability", *, clip_epsilon=1e-6):
    """Fit a JSON-safe score transform on local fit probabilities only.

    ECDF uses a per-member sorted reference and mid-ties:
    (count(reference < query) + count(reference <= query)) / (2*n_fit).
    Constant-logit fit columns use scale 1, rather than divide by zero.
    """
    p = _probabilities(fit_probabilities)
    if not len(p):
        raise ValueError("fit data cannot be empty")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    epsilon = float(clip_epsilon)
    if not np.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("clip_epsilon must be in (0,0.5)")
    state = {"format": "r16_fit_fixed_score_transform_v1", "method": method,
             "n_members": p.shape[1], "n_fit": len(p), "clip_epsilon": epsilon}
    if method == "standardized_logit":
        z = _logit(p, epsilon)
        std = z.std(axis=0)
        constant = std <= 1e-12
        state.update(center=z.mean(axis=0).tolist(),
                     scale=np.where(constant, 1.0, std).tolist(),
                     constant_fit_columns=constant.tolist())
    elif method == "ecdf":
        state["sorted_reference"] = np.sort(p, axis=0).T.tolist()
    return state


def apply_score_transform(probabilities, state):
    """Apply frozen fit state; results are independent of query batch membership."""
    p = _probabilities(probabilities)
    if state.get("format") != "r16_fit_fixed_score_transform_v1":
        raise ValueError("unknown fitted transform format")
    if p.shape[1] != state.get("n_members") or state.get("n_fit", 0) < 1:
        raise ValueError("query/member count or fit state is invalid")
    method = state.get("method")
    if method == "probability":
        return p.copy()
    if method == "standardized_logit":
        epsilon = float(state["clip_epsilon"])
        if not np.isfinite(epsilon) or not 0 < epsilon < 0.5:
            raise ValueError("invalid frozen logit epsilon")
        center = _finite(state["center"], "fit center")
        scale = _finite(state["scale"], "fit scale")
        if center.shape != (p.shape[1],) or scale.shape != center.shape or (scale <= 0).any():
            raise ValueError("invalid frozen logit statistics")
        return (_logit(p, epsilon) - center) / scale
    if method == "ecdf":
        ref = _unit(state["sorted_reference"], "fit reference")
        if ref.shape != (p.shape[1], state["n_fit"]) or (np.diff(ref, axis=1) < 0).any():
            raise ValueError("invalid frozen sorted ECDF reference")
        out = np.empty_like(p)
        for j in range(p.shape[1]):
            left = np.searchsorted(ref[j], p[:, j], side="left")
            right = np.searchsorted(ref[j], p[:, j], side="right")
            out[:, j] = (left + right) / (2.0 * ref.shape[1])
        return out
    raise ValueError("unknown fitted transform method")


def apply_score_policy(probabilities, weights, transform):
    """Fixed-member transform -> fixed convex pooling -> finite [0,1] score.

    A final sigmoid maps standardized-logit pooled scores into [0,1] while
    preserving their ordering. No true share is used in this classifier path.
    """
    score = convex_pool(apply_score_transform(probabilities, transform), weights)
    if transform["method"] == "standardized_logit":
        score = _sigmoid(score)
    return np.clip(score, 0.0, 1.0)


def classifier_weight_candidates(member_names: Sequence[str]):
    """The production classifier's finite grid: 6 candidates when Cat is present."""
    names = list(member_names)
    if not names or names[0] != "lightgbm5" or len(names) != len(set(names)):
        raise ValueError("unique names with lightgbm5 first are required")
    neural = [j for j, n in enumerate(names) if n not in ("lightgbm5", "catboost5")]
    if not neural:
        raise ValueError("at least one neural member is required for this grid")
    tree = np.zeros(len(names)); tree[0] = 1.0
    candidates = {"lightgbm_only": tree.tolist()}
    if "catboost5" in names:
        for mass in (0.25, 0.5):
            v = tree.copy(); v[0] -= mass; v[names.index("catboost5")] = mass
            candidates[f"tree_cat_{mass}"] = v.tolist()
    for mass in (0.75, 0.6, 0.5):
        v = np.zeros(len(names)); v[0] = mass; v[neural] = (1 - mass) / len(neural)
        candidates[f"lightgbm{mass}_neural_uniform"] = v.tolist()
    return candidates


def _sample_weights(sample_weight, n):
    w = np.ones(n) if sample_weight is None else _finite(sample_weight, "sample_weight")
    if w.shape != (n,) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("sample weights must be nonnegative with positive total")
    return w


def weighted_auc(labels, scores, sample_weight=None):
    """Exact pair-weighted ROC-AUC with half credit for ties; None for one class."""
    labels = _finite(labels, "labels")
    scores = _finite(scores, "scores")
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("labels/scores must be matching nonempty vectors")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("AUC labels must be binary")
    w = _sample_weights(sample_weight, len(labels))
    positive = labels == 1
    wp, wn = w[positive].sum(), w[~positive].sum()
    if wp <= 0 or wn <= 0:
        return None
    order = np.argsort(scores, kind="stable")
    s, positive, w = scores[order], positive[order], w[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    group_p = np.add.reduceat(w * positive, starts)
    group_n = np.add.reduceat(w * ~positive, starts)
    neg_before = np.cumsum(group_n) - group_n
    return float(np.sum(group_p * (neg_before + 0.5 * group_n)) / (wp * wn))


def select_classifier_policy(fit_probabilities, fit_y, member_names, *,
                             method="probability", candidates=None, sample_weight=None):
    """Select one predeclared fixed candidate on local fit AUC only.

    AUC ties prefer higher LightGBM mass, matching the immutable baseline.
    Returns JSON-safe state; evaluation queries never enter this function.
    """
    p = _probabilities(fit_probabilities)
    y = _unit(fit_y, "fit share")
    if y.shape != (len(p),) or p.shape[1] != len(member_names):
        raise ValueError("fit labels/members must align")
    transform = fit_score_transform(p, method)
    candidates = classifier_weight_candidates(member_names) if candidates is None else candidates
    if not candidates:
        raise ValueError("at least one predeclared candidate is required")
    aucs = {}
    for name, weights in candidates.items():
        score = apply_score_policy(p, weights, transform)
        auc = weighted_auc(y > 0, score, sample_weight)
        if auc is None:
            raise ValueError("fit must include weighted positive and zero targets")
        aucs[name] = auc
    chosen = max(candidates, key=lambda k: (aucs[k], candidates[k][0]))
    return {"format": "r16_fit_fixed_classifier_policy_v1", "member_names": list(member_names),
            "transform": transform, "candidate": chosen, "weights": list(candidates[chosen]),
            "fit_auc": aucs[chosen], "candidate_fit_aucs": aucs,
            "score_share_weight": 0.0}


def stretched_mask(q, epsilon):
    """Candidate mask shrinkage, not a physical upper bound or guaranteed denoiser."""
    q = _unit(q, "q")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or not 0 <= epsilon <= 0.04:
        raise ValueError("epsilon must lie in the predeclared [0,0.04] range")
    return np.clip((q - epsilon) / (1 - epsilon), 0.0, 1.0)


def pool_mask_shares(q, w, *, epsilon=0.0):
    """Apply candidate epsilon to q[N,...], then pool with unchanged w[N,...]."""
    q = _unit(q, "q")
    w = _finite(w, "physical weights")
    if q.ndim < 2 or w.shape != q.shape or (w < 0).any():
        raise ValueError("q and nonnegative w must have matching [N,...] shapes")
    flat_w = w.reshape(len(w), -1)
    if not np.allclose(flat_w.sum(axis=1), 1.0, rtol=0, atol=2e-6):
        raise ValueError("physical w must already sum to one per window")
    return np.sum(flat_w * stretched_mask(q, epsilon).reshape(len(q), -1), axis=1)


def stretched_share_grid(q, w, epsilons=DEFAULT_EPSILONS):
    """Return [N,len(epsilons)] shares, preserving the caller's epsilon order."""
    epsilons = tuple(epsilons)
    if not epsilons:
        raise ValueError("epsilon grid cannot be empty")
    return np.column_stack([pool_mask_shares(q, w, epsilon=e) for e in epsilons])


def summary_metrics(y, share, score=None, sample_weight=None):
    """Weighted regression/AUC summaries; no unobserved target inference."""
    y = _unit(y, "true share"); share = _unit(share, "predicted share")
    if y.ndim != 1 or share.shape != y.shape or not len(y):
        raise ValueError("true/predicted share must be matching nonempty vectors")
    w = _sample_weights(sample_weight, len(y)); w = w / w.sum()
    error = share - y
    result = {"n": len(y), "rmse": float(np.sqrt(np.sum(w * error ** 2))),
              "bias": float(w @ error), "prediction_mean": float(w @ share),
              "prediction_rms": float(np.sqrt(w @ (share ** 2))),
              "share_auc": weighted_auc(y > 0, share, w)}
    if score is not None:
        score = _unit(score, "classifier score")
        if score.shape != y.shape:
            raise ValueError("classifier score must align with labels")
        result["score_auc"] = weighted_auc(y > 0, score, w)
    return result
