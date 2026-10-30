"""Local synthetic contract fixtures; no production/Test files read."""
from copy import deepcopy
import json
import unittest
import numpy as np

from r16_postprocess_select import FULL_AXES, EPSILONS, apply_policy, select


NAMES = ["lightgbm5", "catboost5", "dasheng_base", "dinov2_small", "timemoe"]


def baseline():
    return {"member_order": NAMES, "D0_H0_weights": [.5, .5], "score_share_weight": 0.0,
            "experts": {r: {"regression_weights": [.5, 0, 1/6, 1/6, 1/6],
                             "independent_classifier_weights": [1, 0, 0, 0, 0]}
                        for r in ("D0", "H0")}}


def panel(n=100):
    # 25 negatives plus three equally-sized positive support/target strata.
    y = np.tile([0, .02, .2, .5], n//4)
    support = np.tile([0, 2, 5, 13], n//4)
    shares = np.repeat(y[:, None], 5, axis=1)
    probability = np.repeat(np.where(y>0, .8, .2)[:, None], 5, axis=1)
    out = {"target_share": y, "support_minutes": support, "sample_weight": np.ones(n)}
    for regime in ("D0", "H0"):
        out["share_"+regime] = shares.copy()
        out["classifier_"+regime] = probability.copy()
        out["stretch_"+regime] = np.repeat(shares[:, None], 4, axis=1)
    return out


def bundle(p=None):
    return {k: deepcopy(panel() if p is None else p) for k in ("fit", *FULL_AXES)}


def set_share(p, regime, columns, values):
    p["share_"+regime][:, columns] = np.asarray(values)[:, None]
    p["stretch_"+regime] = np.repeat(p["share_"+regime][:, None], 4, axis=1)


class SelectionFixtures(unittest.TestCase):
    def test_identity_is_exact_and_state_is_json_safe(self):
        b = bundle(); result = select(b, baseline())
        policy = json.loads(json.dumps(result))["policies"]["selected"]
        out = apply_policy(b["local_gate"], policy)
        np.testing.assert_allclose(out["share"], b["local_gate"]["target_share"], atol=1e-15)
        self.assertFalse(result["selection"]["regression_tree_fallback"])
        self.assertEqual(result["selection"]["classification_method"], "probability")
        self.assertEqual(result["subgroup_audit"]["local_gate_long_positive"]["n"], 25)

    def test_tree_fallback_detects_neural_underperformance(self):
        p = panel(); y = p["target_share"]
        for r in ("D0", "H0"):
            set_share(p, r, slice(2, 5), np.minimum(y+.1, 1))
        b = bundle(p); result = select(b, baseline())
        self.assertTrue(result["selection"]["regression_tree_fallback"])
        self.assertEqual(set(result["selection"]["fallback_trigger_axes"]), set(FULL_AXES))
        self.assertLess(result["gate_metrics"]["selected"]["local_gate"]["rmse"], 1e-14)
        self.assertEqual(result["selection"]["classification_method"], "probability")

    def test_positive_short_slice_does_not_include_zero_support_negatives(self):
        result = select(bundle(), baseline())
        audit = result["subgroup_audit"]
        self.assertEqual(audit["local_gate_short_positive"]["n"], 25)
        self.assertEqual(audit["local_gate_weak_positive"]["n"], 25)

    def test_small_slices_are_disclosed_as_unavailable(self):
        result = select(bundle(panel(40)), baseline())
        self.assertFalse(result["subgroup_audit"]["local_gate_long_positive"]["used_as_gate"])
        self.assertNotIn("local_gate_long_positive", result["gate_metrics"]["baseline"])

    def test_explicitly_rejects_Test_report_and_identity_drift(self):
        b = bundle(); b["Test"] = deepcopy(b["fit"])
        with self.assertRaises(ValueError): select(b, baseline())
        b = bundle(); b["fit"]["stretch_D0"][0, 0, 0] += .001
        with self.assertRaises(ValueError): select(b, baseline())
        b = bundle(); b["fit"]["stretch_D0"][0, 2, 0] += .001
        with self.assertRaises(ValueError): select(b, baseline())

    def test_weak_underprediction_blocks_stretch_even_if_global_improves(self):
        p = panel(); y = p["target_share"]
        for r in ("D0", "H0"):
            # Baseline weak prediction already underestimates; other positives overestimate.
            predicted = y.copy(); predicted[y==.02] = .01; predicted[y>=.2] += .05
            set_share(p, r, slice(0, 5), predicted)
            for j, epsilon in enumerate(EPSILONS[1:], 1):
                p["stretch_"+r][:, j, 2:] = np.maximum(predicted[:, None]-epsilon, 0)
        result = select(bundle(p), baseline())
        eps = [c for c in result["candidates"] if c["id"] == "epsilon_0.005"][0]
        self.assertFalse(eps["eligible"])
        self.assertIn("local_gate_weak_positive:underprediction_mean_worsened", eps["reasons"])

    def test_application_is_query_cohort_invariant(self):
        b = bundle(); result = select(b, baseline()); policy = result["policies"]["selected"]
        full = apply_policy(b["local_gate"], policy)
        p = b["local_gate"]
        first = {k: np.asarray(v)[:1] for k, v in p.items()}
        solo = apply_policy(first, policy)
        for k in ("share", "score"):
            np.testing.assert_allclose(solo[k], full[k][:1], atol=1e-15)

    def test_frozen_application_requires_no_dummy_query_labels(self):
        b = bundle(); policy = select(b, baseline())["policies"]["selected"]
        labeled = apply_policy(b["local_gate"], policy)
        target_free = {k: v for k, v in b["local_gate"].items()
                       if k not in ("target_share", "support_minutes", "sample_weight")}
        out = apply_policy(target_free, policy)
        for k in ("share", "score"):
            np.testing.assert_array_equal(out[k], labeled[k])
        b["fit"] = target_free
        with self.assertRaises(KeyError): select(b, baseline())

    def test_DH_candidate_can_pass_all_axes_and_be_selected(self):
        p = panel(); y = p["target_share"]
        for r in ("D0", "H0"):
            set_share(p, r, slice(0, 2), y+.1)
            set_share(p, r, slice(2, 5), np.maximum(y-.02, 0) if r=="D0" else y+.02)
        result = select(bundle(p), baseline())
        self.assertEqual(result["selection"]["regression_before_fallback"], "DH_0.6")
        self.assertFalse(result["selection"]["regression_tree_fallback"])
        self.assertEqual(result["policies"]["selected"]["regression"]["D0_H0_weights"], [.6, .4])

    def test_classifier_improvement_is_adopted_and_ties_prefer_probability(self):
        p = panel(); y = p["target_share"]
        for r in ("D0", "H0"):
            p["classifier_"+r][:, :2] = .5
            p["classifier_"+r][:, 2:] = np.where(y>0, .8, .2)[:, None]
        result = select(bundle(p), baseline())
        self.assertEqual(result["selection"]["classification_method"], "probability")
        self.assertEqual(result["selection"]["classification_source"], "lightgbm0.75_neural_uniform")
        self.assertEqual(result["gate_metrics"]["selected"]["local_gate"]["score_auc"], 1)
        self.assertAlmostEqual(result["gate_metrics"]["baseline"]["local_gate"]["score_auc"], .5)

    def test_fit_AUC_winner_is_rejected_if_any_full_gate_degrades(self):
        b = bundle(); y = b["fit"]["target_share"]
        for name, p in b.items():
            for r in ("D0", "H0"):
                p["classifier_"+r][:, :2] = .5
                sign = 1 if name=="fit" else -1
                p["classifier_"+r][:, 2:] = np.where(y>0, .5+.3*sign, .5-.3*sign)[:, None]
        result = select(b, baseline())
        self.assertEqual(result["selection"]["classification_source"], "baseline")
        proposed = [c for c in result["candidates"] if c["task"]=="classification"]
        self.assertTrue(all(not c["eligible"] for c in proposed))

    def test_native_positive_harm_cannot_hide_in_improved_mixed_panel(self):
        b = bundle()
        for p in b.values():
            predicted = p["target_share"]+.05
            for r in ("D0", "H0"):
                set_share(p, r, slice(0, 5), predicted)
                for j, epsilon in enumerate(EPSILONS[1:], 1):
                    p["stretch_"+r][:, j, 2:] = np.maximum(predicted[:, None]-epsilon, 0)
        p = b["native12_VS_hf_high_negative"]
        p["target_share"] = np.r_[np.full(50, .2), np.zeros(50)]
        p["support_minutes"] = np.r_[np.full(50, 12), np.zeros(50)]
        predicted = np.where(p["target_share"]>0, .18, .08)
        for r in ("D0", "H0"):
            set_share(p, r, slice(0, 5), predicted)
            p["classifier_"+r][:] = np.where(p["target_share"]>0, .8, .2)[:, None]
            for j, epsilon in enumerate(EPSILONS[1:], 1):
                p["stretch_"+r][:, j, 2:] = np.maximum(predicted[:, None]-epsilon, 0)
        result = select(b, baseline())
        candidate = next(c for c in result["candidates"] if c["id"]=="epsilon_0.01")
        old = result["gate_metrics"]["baseline"]["native12_VS_hf_high_negative"]
        new = candidate["gate_metrics"]["native12_VS_hf_high_negative"]
        self.assertLess(new["rmse"], old["rmse"])
        self.assertLess(abs(new["bias"]), abs(old["bias"]))
        self.assertEqual(result["axis_audit"]["native12_long_positive"]["n"], 50)
        self.assertIn("native12_long_positive:underprediction_mean_worsened", candidate["reasons"])
        self.assertIn("native12_long_positive:RMSE_worsened", candidate["reasons"])
        self.assertFalse(candidate["eligible"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
