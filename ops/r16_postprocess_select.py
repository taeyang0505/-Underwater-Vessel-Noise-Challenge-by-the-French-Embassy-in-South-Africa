"""Local-only finite postprocessing selection, separate from frozen production.

There is no file I/O, model inference, Test/report reader, or training here.
Gate labels are policy-selection labels; these panels are NOT untouched final
tests after this procedure. Caller must pin all arrays/source identities before
running. A tree fallback is explicit and its trade-offs are reported.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

try:
    from .r16_postprocess_math import (
        METHODS, apply_score_policy, classifier_weight_candidates, convex_pool,
        fit_score_transform, summary_metrics, validate_convex_weights,
    )
except ImportError:  # Direct fixture execution from ops/.
    from r16_postprocess_math import (
        METHODS, apply_score_policy, classifier_weight_candidates, convex_pool,
        fit_score_transform, summary_metrics, validate_convex_weights,
    )


REGIMES = ("D0", "H0")
EPSILONS = (0.0, 0.005, 0.01, 0.02)
FULL_AXES = ("local_gate", "domain_holdout", "native12_VS_hf_high_negative",
             "native12_VS_st_francis_dense_negative")
REQUIRED = {"fit", *FULL_AXES}
TOL = 1e-6
MIN_SLICE_N = 20


def _finite_unit(a, name):
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise ValueError(f"{name} must be finite and within [0,1]")
    return a


def _validate_panel(panel, members, *, require_target=True):
    y = None
    if require_target or "target_share" in panel:
        y = _finite_unit(panel["target_share"], "target_share")
        if y.ndim != 1 or not len(y):
            raise ValueError("nonempty target_share vector required")
    # Frozen inference is target-free; labels are mandatory only for selection.
    n, m = len(panel["share_D0"]), len(members)
    if not n or (y is not None and len(y) != n):
        raise ValueError("nonempty aligned predictions are required")
    for regime in REGIMES:
        share = _finite_unit(panel[f"share_{regime}"], f"share_{regime}")
        prob = _finite_unit(panel[f"classifier_{regime}"], f"classifier_{regime}")
        stretch = _finite_unit(panel[f"stretch_{regime}"], f"stretch_{regime}")
        if share.shape != (n, m) or prob.shape != share.shape or stretch.shape != (n, len(EPSILONS), m):
            raise ValueError("panel shapes must be [N,5] shares/probabilities and [N,4,5] stretch")
        if not np.array_equal(stretch[:, 0], share):
            raise ValueError("epsilon=0 must exactly preserve saved baseline shares")
        for j, name in enumerate(members):
            if name in ("lightgbm5", "catboost5"):
                if not np.array_equal(stretch[:, :, j], np.repeat(share[:, j:j+1], len(EPSILONS), axis=1)):
                    raise ValueError("mask stretch must not change tree predictions")
    weight = _weight(panel)
    if weight is not None:
        if weight.shape != (n,) or not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
            raise ValueError("invalid panel sample weights")
    if "support_minutes" in panel:
        support = np.asarray(panel["support_minutes"], float)
        if support.shape != (n,) or not np.isfinite(support).all() or (support < 0).any():
            raise ValueError("invalid support_minutes")
    return y


def _weight(panel):
    if "sample_weight" in panel and "weights" in panel:
        if not np.array_equal(panel["sample_weight"], panel["weights"]):
            raise ValueError("conflicting sample weight aliases")
    raw = panel.get("sample_weight", panel.get("weights"))
    return None if raw is None else np.asarray(raw, dtype=np.float64)


def _flatten_prob(panel):
    return np.column_stack([panel[f"classifier_{r}"] for r in REGIMES])


def _subpanel(panel, mask):
    n = len(panel["target_share"])
    keys = ["target_share", "support_minutes", "weights", "sample_weight"]
    keys += [f"{kind}_{r}" for kind in ("share", "classifier", "stretch") for r in REGIMES]
    return {k: np.asarray(panel[k])[mask] for k in keys if k in panel and len(panel[k]) == n}


def _regression_spec(member_weights, *, d0_weight=0.5, epsilon=0.0, source="identity"):
    return {"source": source, "D0_H0_weights": [float(d0_weight), float(1-d0_weight)],
            "epsilon": float(epsilon),
            "member_weights": {r: np.asarray(member_weights[r], float).tolist() for r in REGIMES}}


def _policy(members, reg, cls, name):
    return {"format": "r16_local_postprocess_candidate_v1", "name": name,
            "member_order": list(members), "regression": deepcopy(reg),
            "classification": deepcopy(cls), "score_share_weight": 0.0}


def apply_policy(panel, policy):
    """Apply a frozen policy to labeled/unlabeled arrays; no query-cohort fitting."""
    if policy.get("format") != "r16_local_postprocess_candidate_v1" or policy.get("score_share_weight") != 0:
        raise ValueError("unknown or coupled classifier policy")
    members = policy["member_order"]
    _validate_panel(panel, members, require_target=False)
    reg = policy["regression"]
    dh = validate_convex_weights(reg["D0_H0_weights"], 2)
    matches = np.flatnonzero(np.isclose(EPSILONS, reg["epsilon"], rtol=0, atol=1e-12))
    if len(matches) != 1:
        raise ValueError("epsilon is outside the fixed candidate grid")
    index = int(matches[0])
    prediction = np.zeros(len(panel["share_D0"]))
    for mass, regime in zip(dh, REGIMES):
        values = np.asarray(panel[f"stretch_{regime}"], float)[:, index]
        prediction += mass * convex_pool(values, reg["member_weights"][regime])
    cls = policy["classification"]
    score = apply_score_policy(_flatten_prob(panel), cls["weights"], cls["transform"])
    return {"share": np.clip(prediction, 0, 1), "score": score}


def _metrics(panel, policy):
    values = apply_policy(panel, policy)
    y = np.asarray(panel["target_share"], float)
    weight = _weight(panel)
    result = summary_metrics(y, values["share"], values["score"], weight)
    w = np.ones(len(y)) if weight is None else weight
    result["underprediction_mean"] = float(np.average(np.maximum(y-values["share"], 0), weights=w))
    return result


def _metrics_all(axes, policy):
    return {name: _metrics(panel, policy) for name, panel in axes.items()}


def _best_tree(fit, members, original):
    """Choose one of three common D/H cat masses on fit only, separately by task."""
    m = len(members); cat = members.index("catboost5")
    reg_choices, cls_choices, audit = [], [], []
    for mass in (0.0, 0.25, 0.5):
        weight = np.zeros(m); weight[0] = 1-mass; weight[cat] = mass
        reg = _regression_spec({r: weight for r in REGIMES}, source=f"tree_cat_{mass}")
        cls = {"source": f"tree_cat_{mass}", "method": "probability",
               "weights": np.concatenate([.5*weight, .5*weight]).tolist(),
               "transform": original["classification"]["transform"]}
        policy = _policy(members, reg, cls, f"tree_cat_{mass}")
        metrics = _metrics(fit, policy)
        audit.append({"cat_mass": mass, "fit_metrics": metrics})
        reg_choices.append((metrics["rmse"], mass, reg))
        if metrics["score_auc"] is None:
            raise ValueError("tree fit requires both classes")
        cls_choices.append((metrics["score_auc"], -mass, cls))
    reg = min(reg_choices, key=lambda x: (x[0], x[1]))[2]
    cls = max(cls_choices, key=lambda x: (x[0], x[1]))[2]
    return _policy(members, reg, cls, "fit_selected_tree_baseline"), audit


def _reg_gate(candidate, baseline, *, epsilon):
    reasons = []
    improvement = baseline["local_gate"]["rmse"] - candidate["local_gate"]["rmse"]
    if improvement <= TOL:
        reasons.append("local_gate_RMSE_improvement_not_strictly_above_1e-6")
    for axis in baseline:
        if candidate[axis]["rmse"] > baseline[axis]["rmse"] + TOL:
            reasons.append(axis + ":RMSE_worsened")
        if abs(candidate[axis]["bias"]) > abs(baseline[axis]["bias"]) + TOL:
            reasons.append(axis + ":absolute_bias_worsened")
        if epsilon and axis.endswith(("weak_positive", "long_positive")):
            if candidate[axis]["underprediction_mean"] > baseline[axis]["underprediction_mean"] + TOL:
                reasons.append(axis + ":underprediction_mean_worsened")
    return reasons


def _cls_gate(candidate, baseline):
    reasons = []
    for axis in FULL_AXES:
        a, b = candidate[axis]["score_auc"], baseline[axis]["score_auc"]
        if a is None or b is None:
            reasons.append(axis + ":AUC_unavailable")
        elif a < b-TOL:
            reasons.append(axis + ":AUC_worsened")
    a, b = candidate["local_gate"]["score_auc"], baseline["local_gate"]["score_auc"]
    if a is None or b is None or a-b <= TOL:
        reasons.append("local_gate_AUC_improvement_not_strictly_above_1e-6")
    return reasons


def _worse_than_tree(metrics, tree):
    return [axis for axis in FULL_AXES if metrics[axis]["rmse"] > tree[axis]["rmse"] + TOL]


def select(bundle, baselinepolicy):
    """Select finite independent candidates on fit + declared gates only.

    Allowed keys are exactly fit plus the four declared gates. This API cannot
    accept report/Test data. All comparisons are against original identity;
    regression and classifier choices never form a joint search. A final
    tree fallback affects regression only. Returned metrics disclose trade-offs.
    """
    if set(bundle) != REQUIRED:
        raise ValueError("bundle must contain exactly fit and the four declared local gates; no Test/report")
    members = list(baselinepolicy["member_order"])
    if len(members) != 5 or members[0] != "lightgbm5" or members[1] != "catboost5" or len(set(members)) != 5:
        raise ValueError("expected LGB, Cat and three unique neural families")
    if baselinepolicy.get("D0_H0_weights") != [.5, .5] or baselinepolicy.get("score_share_weight") != 0:
        raise ValueError("baseline D/H or independent score contract differs")
    for panel in bundle.values():
        _validate_panel(panel, members)
    fit = bundle["fit"]
    transform = fit_score_transform(_flatten_prob(fit), "probability")
    reg_weights = {r: baselinepolicy["experts"][r]["regression_weights"] for r in REGIMES}
    for regime in REGIMES:
        w = validate_convex_weights(reg_weights[regime], len(members))
        if w[0] < .5-TOL or w[1] > .5+TOL or (w[2:] > .25+TOL).any():
            raise ValueError("baseline regression caps differ")
    cls_weights = np.concatenate([.5 * np.asarray(baselinepolicy["experts"][r]["independent_classifier_weights"])
                                  for r in REGIMES])
    baseline = _policy(members, _regression_spec(reg_weights),
                       {"source": "baseline", "method": "probability", "weights": cls_weights.tolist(),
                        "transform": transform}, "immutable_identity")
    axes = {key: bundle[key] for key in FULL_AXES}
    gate = bundle["local_gate"]; y = np.asarray(gate["target_share"], float)
    subgroup_masks = {"local_gate_weak_positive": (y > 0) & (y <= .03)}
    if "support_minutes" in gate:
        support = np.asarray(gate["support_minutes"], float)
        subgroup_masks.update(local_gate_short_positive=(y > 0) & (support <= 3),
                              local_gate_long_positive=(y > 0) & (support >= 12))
    slice_audit = {}
    for name, mask in subgroup_masks.items():
        n = int(mask.sum()); available = n >= MIN_SLICE_N
        if available:
            axes[name] = _subpanel(gate, mask)
        slice_audit[name] = {"n": n, "used_as_gate": available,
                             "reason": "n>=20" if available else "n<20_not_a_selection_axis"}
    if "support_minutes" not in gate:
        slice_audit["support_subgroups"] = {"used_as_gate": False, "reason": "support_minutes_not_supplied"}
    native = bundle["native12_VS_hf_high_negative"]
    if "support_minutes" in native:
        native_y = np.asarray(native["target_share"], float)
        native_support = np.asarray(native["support_minutes"], float)
        mask = (native_y > 0) & (native_support >= 12)
        n = int(mask.sum()); available = n >= MIN_SLICE_N
        if available:
            axes["native12_long_positive"] = _subpanel(native, mask)
        slice_audit["native12_long_positive"] = {
            "n": n, "used_as_gate": available,
            "source": "native12_VS_hf_high_negative",
            "reason": "n>=20" if available else "n<20_not_a_selection_axis"}
    else:
        slice_audit["native12_long_positive"] = {
            "used_as_gate": False, "reason": "support_minutes_not_supplied",
            "source": "native12_VS_hf_high_negative"}
    baseline_metrics = _metrics_all(axes, baseline)
    tree, tree_fit_audit = _best_tree(fit, members, baseline)
    tree_metrics = _metrics_all(axes, tree)
    candidates = [{"id": "identity", "task": "regression", "eligible": True,
                   "reasons": ["unchanged_baseline_default"], "fit_metrics": _metrics(fit, baseline),
                   "gate_metrics": baseline_metrics}]
    reg_candidates = []
    specs = [(_regression_spec(reg_weights, d0_weight=d, source=f"DH_{d}"), f"DH_{d}") for d in (.6, .4)]
    specs += [(_regression_spec(reg_weights, epsilon=e, source=f"epsilon_{e}"), f"epsilon_{e}")
              for e in EPSILONS[1:]]
    for reg, name in specs:
        policy = _policy(members, reg, baseline["classification"], name)
        metrics = _metrics_all(axes, policy)
        fit_metrics = _metrics(fit, policy)
        reasons = _reg_gate(metrics, baseline_metrics, epsilon=reg["epsilon"])
        candidates.append({"id": name, "task": "regression", "eligible": not reasons,
                           "reasons": reasons, "fit_metrics": fit_metrics, "gate_metrics": metrics,
                           "regression_spec": reg})
        if not reasons:
            reg_candidates.append((fit_metrics["rmse"], name, policy))
    winner = min(reg_candidates, key=lambda x: (x[0], x[1]))[2] if reg_candidates else baseline
    proposed_reg_metrics = _metrics_all(axes, winner)
    fallback_axes = _worse_than_tree(proposed_reg_metrics, tree_metrics)
    final_reg = tree["regression"] if fallback_axes else winner["regression"]

    grid = {"original_baseline": cls_weights.tolist()}
    for name, weight in classifier_weight_candidates(members).items():
        grid[name] = np.concatenate([.5 * np.asarray(weight), .5 * np.asarray(weight)]).tolist()
    classifier_candidates = []
    for method in METHODS:
        state = fit_score_transform(_flatten_prob(fit), method)
        method_fit = []
        for name, weight in grid.items():
            cls = {"source": name, "method": method, "weights": weight, "transform": state}
            policy = _policy(members, baseline["regression"], cls, method+":"+name)
            fit_metrics = _metrics(fit, policy)
            if fit_metrics["score_auc"] is None:
                raise ValueError("classifier fit requires both classes")
            method_fit.append((fit_metrics["score_auc"], weight[0]+weight[5], name, policy, fit_metrics))
        best = max(method_fit, key=lambda x: (x[0], x[1], x[2] == "original_baseline"))
        _, _, name, policy, fit_metrics = best
        metrics = _metrics_all({k: bundle[k] for k in FULL_AXES}, policy)
        reasons = _cls_gate(metrics, baseline_metrics)
        candidates.append({"id": method+":"+name, "task": "classification", "eligible": not reasons,
                           "reasons": reasons, "fit_metrics": fit_metrics, "gate_metrics": metrics,
                           "method": method, "classifier_weights": policy["classification"]["weights"],
                           "all_fit_candidates": [{"name": item[2], "AUC": item[0]} for item in method_fit]})
        if not reasons:
            classifier_candidates.append((fit_metrics["score_auc"], method, policy))
    # Stable ties prefer the simpler method order: probability, logit, ECDF.
    cls_winner = max(classifier_candidates, key=lambda x: x[0])[2] if classifier_candidates else baseline
    selected = _policy(members, final_reg, cls_winner["classification"], "selected_local_postprocess")
    selected_metrics = _metrics_all(axes, selected)
    # This second assertion protects implementation changes to combined policies.
    final_violation = _worse_than_tree(selected_metrics, tree_metrics)
    if final_violation:
        raise AssertionError("final regression safety fallback failed: "+str(final_violation))
    return {"format": "r16_local_postprocess_selection_v1",
            "protocol": {"numerical_tolerance": TOL, "min_subgroup_n": MIN_SLICE_N,
                         "regression_epsilon_grid": list(EPSILONS), "D0_mass_candidates": [.5, .6, .4],
                         "regression_search": "independent_one_change_candidates_only",
                         "classifier_fit": "one_best_fixed_weight_candidate_per_transform_then_gate",
                         "classifier_transform_tie_order": list(METHODS),
                         "gate_use": "policy selection; not untouched final evaluation",
                         "report_Test_used": False, "score_share_weight": 0.0},
            "policies": {"baseline": baseline, "tree": tree, "selected": selected},
            "candidates": candidates, "tree_fit_candidates": tree_fit_audit,
            "subgroup_audit": slice_audit,
            "axis_audit": {**{k: {"n": len(bundle[k]["target_share"]), "used_as_gate": True,
                                   "source": k, "kind": "full_panel"} for k in FULL_AXES},
                           **slice_audit},
            "gate_metrics": {"baseline": baseline_metrics, "tree": tree_metrics, "selected": selected_metrics},
            "selection": {"regression_before_fallback": winner["name"],
                          "baseline_worse_than_tree_axes": _worse_than_tree(baseline_metrics, tree_metrics),
                          "regression_tree_fallback": bool(fallback_axes), "fallback_trigger_axes": fallback_axes,
                          "classification_source": cls_winner["classification"]["source"],
                          "classification_method": cls_winner["classification"]["method"],
                          "fallback_rmse_changes_vs_proposed": {
                              k: selected_metrics[k]["rmse"]-proposed_reg_metrics[k]["rmse"] for k in axes}}}
