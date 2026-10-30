"""Fit-only four-family-policy comparison on declared local selection panels.

No I/O, model inference, Test/report inputs or cohort-dependent transforms.
Gates select policies: they are development data, not independent final tests.
The historical postprocessing implementation is imported but never modified.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.optimize import minimize

try:
    from .r16_postprocess_select import (
        EPSILONS, FULL_AXES, REGIMES, REQUIRED, TOL, MIN_SLICE_N,
        apply_policy, _validate_panel, _weight, _flatten_prob, _subpanel,
        _regression_spec, _policy, _metrics, _metrics_all, _reg_gate,
        _cls_gate, _worse_than_tree,
    )
    from .r16_postprocess_math import (
        METHODS, fit_score_transform, validate_convex_weights,
        classifier_weight_candidates,
    )
except ImportError:
    from r16_postprocess_select import (
        EPSILONS, FULL_AXES, REGIMES, REQUIRED, TOL, MIN_SLICE_N,
        apply_policy, _validate_panel, _weight, _flatten_prob, _subpanel,
        _regression_spec, _policy, _metrics, _metrics_all, _reg_gate,
        _cls_gate, _worse_than_tree,
    )
    from r16_postprocess_math import (
        METHODS, fit_score_transform, validate_convex_weights,
        classifier_weight_candidates,
    )


CATEGORIES = ("capped", "free", "neural_only", "tree_only")
FREE_RIDGE = 1e-4
NEURAL_REQUIRED_RELATIVE_GAIN = .01


def prespec():
    """JSON-safe fixed protocol, to register before prediction collection."""
    return {
        "format": "r16_fourway_prespec_v1", "categories": list(CATEGORIES),
        "regression_fit": "D0/H0 separately; fit-only weighted MSE, SLSQP simplex",
        "capped_bounds": "LGB>=0.5, each NN<=0.25; Cat unrestricted within simplex",
        "free_bounds": "all five nonnegative, sum=1; no Tree floor or NN cap",
        "free_ridge": FREE_RIDGE, "free_ridge_reference": "uniform five members",
        "neural_only": "only last three members; nonnegative sum=1",
        "tree_only": "only first two members; nonnegative sum=1",
        "regression_candidates_per_category": ["identity", "D0=.6", "D0=.4", "epsilon=.005", "epsilon=.01", "epsilon=.02"],
        "regression_candidate_rule": "One-change variants of category fit weights, no DH/epsilon cross product",
        "regression_gate": "strict local_gate RMSE gain>1e-6 vs original; no available axis RMSE/absolute bias degradation>1e-6",
        "epsilon_guard": "weak/long positive underprediction mean cannot increase>1e-6",
        "regression_choice": "lowest eligible local_gate RMSE; ties within 1e-6 prefer declared candidate order",
        "immutable_control": "Original identity always available without strict improvement; preserve if Tree-safe and no better eligible candidate",
        "neural_only_extra_gate": "local_gate RMSE at least 1% lower than fit-selected Tree and nonworse all four full axes",
        "fallback": "no eligible candidate nonworse than Tree on all four main axes => Tree; fallback tradeoffs disclosed",
        "classifier_transforms": list(METHODS), "classifier_logit_clip": 1e-6,
        "classifier_grid": "finite family-specific weights; shared family weights D0/H0 50:50; no optimized score weights",
        "classifier_grid_sizes": {"capped": 6, "free": 11, "neural_only": 7, "tree_only": 5},
        "classifier_fit": "one fit-AUC winner per category/transform before gates; stable declared-grid ties",
        "classifier_gate": "strict local_gate AUC gain>1e-6 vs original and no full-axis AUC decrease>1e-6",
        "classifier_choice": "highest fit AUC among gate-eligible transform winners; declared order ties",
        "score_share_weight": 0., "query_cohort_statistics": False,
        "min_subgroup_n": MIN_SLICE_N, "numerical_tolerance": TOL,
        "selection_axes": list(FULL_AXES),
        "subgroup_axes": ["local_gate_weak_positive", "local_gate_short_positive", "local_gate_long_positive", "native12_long_positive"],
        "gate_role": "policy selection labels, not untouched tests",
        "neural_submission_rule": "Among eligible free policies with D/H-weighted NN mass>=0.5 or eligible neural_only policies: Tree-relative local_gate RMSE gain>=1%, no main-axis RMSE worse than Tree; lowest local_gate RMSE, stable declared-order tie; use selected independent classifier. Otherwise no neural submission policy.",
        "report_Test_used": False,
    }


def _fit_regression(fit, category):
    """Convex fit; no gate arrays are accepted by this function."""
    y = np.asarray(fit["target_share"], float)
    sw = _weight(fit)
    sw = np.ones(len(y))/len(y) if sw is None else sw/sw.sum()
    bounds = [(0., 1.)]*5
    initial = np.ones(5)/5
    if category == "capped":
        bounds = [(.5, 1.), (0., .5), (0., .25), (0., .25), (0., .25)]
        initial = np.array([.5, .125, .125, .125, .125])
    elif category == "neural_only":
        bounds[:2] = [(0., 0.)]*2; initial = np.array([0., 0., 1/3, 1/3, 1/3])
    elif category == "tree_only":
        bounds[2:] = [(0., 0.)]*3; initial = np.array([.5, .5, 0., 0., 0.])
    elif category != "free":
        raise ValueError("unknown regression category")
    ridge = FREE_RIDGE if category == "free" else 0.
    prior = np.ones(5)/5
    weights, audit = {}, {}
    for regime in REGIMES:
        x = np.asarray(fit[f"share_{regime}"], float)
        def objective(w):
            error = x@w-y
            return float(np.dot(sw, error*error)+ridge*np.sum((w-prior)**2))
        def jac(w):
            return 2*x.T@(sw*(x@w-y))+2*ridge*(w-prior)
        result = minimize(objective, initial, jac=jac, bounds=bounds,
                          constraints={"type": "eq", "fun": lambda w: w.sum()-1,
                                       "jac": lambda w: np.ones(5)},
                          method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
        if not result.success:
            raise RuntimeError(f"{category}/{regime} SLSQP failed: {result.message}")
        w = np.asarray(result.x, float)
        # Remove optimizer roundoff only; substantial violations are errors.
        if abs(w.sum()-1)>1e-8 or any(v<lo-1e-8 or v>hi+1e-8 for v,(lo,hi) in zip(w,bounds)):
            raise RuntimeError("SLSQP feasibility violation")
        w = np.maximum(w, 0); w /= w.sum()
        validate_convex_weights(w, 5)
        weights[regime] = w.tolist()
        audit[regime] = {"success": True, "iterations": int(result.nit),
                         "objective": objective(w), "ridge": ridge,
                         "weighted_mse": float(np.dot(sw, (x@w-y)**2)),
                         "weights": w.tolist(), "fitted_on": "fit_only"}
    return _regression_spec(weights, source=category+"_fit"), audit


def _classifier_grid(category, members):
    """Small fixed grids, never generated from query or gate statistics."""
    lgb=np.array([1.,0,0,0,0]); cat=np.array([0.,1,0,0,0])
    trees=(lgb+cat)/2; nn=np.array([0.,0,1/3,1/3,1/3])
    if category == "capped":
        return [(k,np.asarray(v,float)) for k,v in classifier_weight_candidates(members).items()]
    if category == "tree_only":
        return [(f"cat_{c}",(1-c)*lgb+c*cat) for c in (0.,.25,.5,.75,1.)]
    if category == "neural_only":
        rows=[("neural_uniform",nn)]
        rows += [(f"neural_{j}",np.eye(5)[j]) for j in (2,3,4)]
        rows += [(f"neural_pair_{a}_{b}",(np.eye(5)[a]+np.eye(5)[b])/2)
                 for a,b in ((2,3),(2,4),(3,4))]
        return rows
    if category == "free":
        return [("all_uniform",np.ones(5)/5), ("tree_uniform",trees),
                ("neural_uniform",nn), *[(f"member_{j}",np.eye(5)[j]) for j in range(5)],
                ("lgb_neural_half",(lgb+nn)/2), ("cat_neural_half",(cat+nn)/2),
                ("tree_neural_half",(trees+nn)/2)]
    raise ValueError("unknown classifier category")


def _axes(bundle):
    axes={k:bundle[k] for k in FULL_AXES}; audit={}
    gate=bundle["local_gate"]; y=np.asarray(gate["target_share"],float)
    masks={"local_gate_weak_positive":(y>0)&(y<=.03)}
    if "support_minutes" in gate:
        s=np.asarray(gate["support_minutes"],float)
        masks.update(local_gate_short_positive=(y>0)&(s<=3),
                     local_gate_long_positive=(y>0)&(s>=12))
    for name,mask in masks.items():
        n=int(mask.sum()); use=n>=MIN_SLICE_N
        audit[name]={"n":n,"used_as_gate":use,"source":"local_gate",
                     "reason":"n>=20" if use else "n<20_not_selection_axis"}
        if use: axes[name]=_subpanel(gate,mask)
    native=bundle["native12_VS_hf_high_negative"]
    if "support_minutes" in native:
        mask=(np.asarray(native["target_share"])>0)&(np.asarray(native["support_minutes"])>=12)
        n=int(mask.sum()); use=n>=MIN_SLICE_N
        audit["native12_long_positive"]={"n":n,"used_as_gate":use,
                "source":"native12_VS_hf_high_negative","reason":"n>=20" if use else "n<20_not_selection_axis"}
        if use: axes["native12_long_positive"]=_subpanel(native,mask)
    for name in ("local_gate_short_positive","local_gate_long_positive","native12_long_positive"):
        audit.setdefault(name,{"used_as_gate":False,"reason":"support_minutes_not_supplied"})
    return axes,audit


def _pick_stable(items, value, maximize=False):
    """Prespecified order breaks numerical ties within TOL."""
    best=items[0]
    for item in items[1:]:
        delta=value(item)-value(best)
        if (delta>TOL if maximize else delta<-TOL): best=item
    return best


def select(bundle, baselinepolicy):
    """Fit finite policies and choose with explicitly labeled local gates.

    Only fit + the four exact named gates are accepted. Category policies are
    fit-selected comparisons, even when a gate rejects their adoption. The
    final policy has separately selected regression and classifier components.
    """
    if set(bundle)!=REQUIRED:
        raise ValueError("bundle must contain exactly fit and four local gates; no Test/report")
    members=list(baselinepolicy["member_order"])
    if len(members)!=5 or members[:2]!=["lightgbm5","catboost5"] or len(set(members))!=5:
        raise ValueError("expected ordered LGB, Cat, three unique neural members")
    if baselinepolicy.get("D0_H0_weights")!=[.5,.5] or baselinepolicy.get("score_share_weight")!=0:
        raise ValueError("baseline D/H or independent score contract differs")
    for panel in bundle.values(): _validate_panel(panel,members)
    fit=bundle["fit"]; axes,slice_audit=_axes(bundle)
    transform=fit_score_transform(_flatten_prob(fit),"probability")
    rw={r:validate_convex_weights(baselinepolicy["experts"][r]["regression_weights"],5).tolist() for r in REGIMES}
    cw=np.concatenate([.5*validate_convex_weights(baselinepolicy["experts"][r]["independent_classifier_weights"],5) for r in REGIMES])
    baseline=_policy(members,_regression_spec(rw),{"source":"baseline","method":"probability",
                       "weights":cw.tolist(),"transform":transform},"immutable_identity")
    bm=_metrics_all(axes,baseline)
    policies={"baseline":baseline}; fit_audit={}; cls_audit=[]; cls_winners=[]; reg_candidates=[]
    # All numerical regression/classification fitting precedes gate selection.
    for category in CATEGORIES:
        reg,fa=_fit_regression(fit,category); fit_audit[category]=fa
        per_transform=[]
        for method in METHODS:
            state=fit_score_transform(_flatten_prob(fit),method)
            trials=[]
            for name,w in _classifier_grid(category,members):
                cls={"source":category+":"+name,"method":method,
                     "weights":np.r_[.5*w,.5*w].tolist(),"transform":state}
                p=_policy(members,reg,cls,category+":"+method+":"+name)
                fm=_metrics(fit,p)
                if fm["score_auc"] is None: raise ValueError("fit classifier requires both classes")
                trials.append({"policy":p,"fit_metrics":fm,"grid_name":name})
            win=_pick_stable(trials,lambda t:t["fit_metrics"]["score_auc"],True)
            per_transform.append(win)
            cls_winners.append((category,method,win))
            cls_audit.append({"category":category,"method":method,"winner":win["grid_name"],
                   "fit_grid":[{"name":t["grid_name"],"AUC":t["fit_metrics"]["score_auc"]} for t in trials]})
        raw_cls=_pick_stable(per_transform,lambda t:t["fit_metrics"]["score_auc"],True)
        policies[category]=_policy(members,reg,raw_cls["policy"]["classification"],category+"_fit_comparison")
    tree=deepcopy(policies["tree_only"]); tree["name"]="fit_selected_tree_baseline"; policies["tree"]=tree
    tm=_metrics_all(axes,tree)
    candidates=[{"id":"original_identity","task":"regression","category":"baseline",
                   "eligible":True,"reasons":["immutable_reference"],
                   "fit_metrics":_metrics(fit,baseline),"gate_metrics":bm,
                   "worse_than_tree_axes":_worse_than_tree(bm,tm)}]
    reg_candidates=[(baseline,candidates[0])]
    category_gates={}
    for category in CATEGORIES:
        base=policies[category]; reg=base["regression"]; w=reg["member_weights"]
        variants=[("identity",deepcopy(reg))]
        variants += [(f"DH_{d}",_regression_spec(w,d0_weight=d,source=f"{category}:DH_{d}")) for d in (.6,.4)]
        # Tree masks cannot change: record only the identity comparison, not duplicates.
        if category!="tree_only":
            variants += [(f"epsilon_{e}",_regression_spec(w,epsilon=e,source=f"{category}:epsilon_{e}")) for e in EPSILONS[1:]]
        cat_rows=[]
        for variant,spec in variants:
            p=_policy(members,spec,base["classification"],category+":"+variant)
            gm=_metrics_all(axes,p); fm=_metrics(fit,p)
            reasons=_reg_gate(gm,bm,epsilon=spec["epsilon"])
            worse_tree=_worse_than_tree(gm,tm)
            if category=="neural_only":
                tr=tm["local_gate"]["rmse"]; gain=(tr-gm["local_gate"]["rmse"])/max(tr,1e-15)
                if gain<NEURAL_REQUIRED_RELATIVE_GAIN: reasons.append("neural_only:Tree_relative_RMSE_gain_below_1pct")
                if worse_tree: reasons.append("neural_only:full_axis_worse_than_Tree")
            row={"id":p["name"],"task":"regression","category":category,"variant":variant,
                 "eligible":not reasons,"reasons":reasons,"fit_metrics":fm,"gate_metrics":gm,
                 "regression_spec":spec,"worse_than_tree_axes":worse_tree}
            candidates.append(row); cat_rows.append(row)
            if not reasons: reg_candidates.append((p,row))
        category_gates[category]={"identity":cat_rows[0],"eligible_variant_ids":[r["id"] for r in cat_rows if r["eligible"]]}
    eligible_tree_safe=[v for v in reg_candidates if not v[1]["worse_than_tree_axes"]]
    if eligible_tree_safe:
        reg_winner,reg_row=_pick_stable(eligible_tree_safe,lambda v:v[1]["gate_metrics"]["local_gate"]["rmse"])
        fallback=False
    else:
        reg_winner=tree; reg_row=None; fallback=True
    proposed=_pick_stable(reg_candidates,lambda v:v[1]["gate_metrics"]["local_gate"]["rmse"])[0] if reg_candidates else baseline
    proposed_metrics=_metrics_all(axes,proposed)
    passing_cls=[]
    for category,method,win in cls_winners:
        p=win["policy"]; gm=_metrics_all({k:bundle[k] for k in FULL_AXES},p)
        reasons=_cls_gate(gm,bm)
        row={"id":p["name"],"task":"classification","category":category,"method":method,
             "eligible":not reasons,"reasons":reasons,"fit_metrics":win["fit_metrics"],
             "gate_metrics":gm,"classifier_weights":p["classification"]["weights"]}
        candidates.append(row)
        category_gates[category].setdefault("classifier_transform_candidates",[]).append(row)
        if not reasons: passing_cls.append((p,row))
    cls_winner=_pick_stable(passing_cls,lambda v:v[1]["fit_metrics"]["score_auc"],True)[0] if passing_cls else baseline
    selected=_policy(members,reg_winner["regression"],cls_winner["classification"],"selected_fourway")
    policies["selected"]=selected
    sm=_metrics_all(axes,selected)
    if _worse_than_tree(sm,tm): raise AssertionError("final Tree safety fallback violated")
    neural_qualifying=[]; neural_audit=[]
    for p,row in reg_candidates:
        category=row["category"]
        if category not in ("free","neural_only"): continue
        reg=p["regression"]
        nn_mass=sum(d*sum(reg["member_weights"][r][2:]) for d,r in zip(reg["D0_H0_weights"],REGIMES))
        gain=(tm["local_gate"]["rmse"]-row["gate_metrics"]["local_gate"]["rmse"])/max(tm["local_gate"]["rmse"],1e-15)
        reasons=[]
        if category=="free" and nn_mass<.5-1e-12: reasons.append("free_NN_mass_below_0.5")
        if gain<NEURAL_REQUIRED_RELATIVE_GAIN: reasons.append("Tree_relative_RMSE_gain_below_1pct")
        if row["worse_than_tree_axes"]: reasons.append("main_axis_RMSE_worse_than_Tree")
        neural_audit.append({"id":row["id"],"NN_mass":float(nn_mass),
                             "Tree_relative_local_gate_RMSE_gain":float(gain),
                             "qualifies":not reasons,"reasons":reasons})
        if not reasons: neural_qualifying.append((p,row))
    neural_submission=None; neural_id=None
    if neural_qualifying:
        npolicy,nrow=_pick_stable(neural_qualifying,lambda v:v[1]["gate_metrics"]["local_gate"]["rmse"])
        neural_submission=_policy(members,npolicy["regression"],cls_winner["classification"],"neural_centered_fourway")
        neural_id=nrow["id"]
    return {"format":"r16_fourway_selection_v1","protocol":prespec(),"policies":policies,
            "neural_submission_policy":neural_submission,
            "neural_submission_selection":{"candidate_id":neural_id,"audit":neural_audit,
                "gate_metrics":None if neural_submission is None else _metrics_all(axes,neural_submission)},
            "candidates":candidates,"category_gates":category_gates,
            "regression_fit_audit":fit_audit,"classifier_fit_audit":cls_audit,
            "subgroup_audit":slice_audit,
            "axis_audit":{**{k:{"n":len(bundle[k]["target_share"]),"source":k,"used_as_gate":True,"kind":"full_panel"} for k in FULL_AXES},**slice_audit},
            "gate_metrics":{k:_metrics_all(axes,p) for k,p in policies.items()},
            "selection":{"regression_source":reg_winner["name"],"regression_before_safety":proposed["name"],
               "regression_tree_fallback":fallback,"fallback_trigger_axes":_worse_than_tree(proposed_metrics,tm),
               "eligible_tree_safe_ids":[row["id"] for _,row in eligible_tree_safe],
               "classification_source":cls_winner["classification"]["source"],
               "classification_method":cls_winner["classification"]["method"],
               "fallback_rmse_changes_vs_proposed":{k:sm[k]["rmse"]-proposed_metrics[k]["rmse"] for k in axes},
               "selected_candidate_id":None if reg_row is None else reg_row["id"],
               "note":"All gates are selection data. Fallback can trade subgroup bias or RMSE against the proposed policy; see per-axis deltas."}}
