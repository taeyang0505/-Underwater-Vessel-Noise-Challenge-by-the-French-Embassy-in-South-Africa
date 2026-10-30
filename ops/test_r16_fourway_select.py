"""Meaningful synthetic fixtures only; no production/Test files are read."""
from copy import deepcopy
import json
import unittest
import numpy as np

from r16_fourway_select import (
    CATEGORIES, FULL_AXES, EPSILONS, apply_policy, prespec, select,
    _fit_regression, _classifier_grid,
)

NAMES=["lightgbm5","catboost5","dasheng_base","dinov2_small","timemoe"]


def baseline():
    return {"member_order":NAMES,"D0_H0_weights":[.5,.5],"score_share_weight":0.,
            "experts":{r:{"regression_weights":[.5,0,1/6,1/6,1/6],
                           "independent_classifier_weights":[1,0,0,0,0]} for r in ("D0","H0")}}


def panel(n=100):
    y=np.tile([0.,.02,.2,.5],n//4)
    p={"target_share":y,"support_minutes":np.tile([0,2,5,13],n//4),
       "sample_weight":np.ones(n)}
    for r in ("D0","H0"):
        p["share_"+r]=np.repeat(y[:,None],5,axis=1)
        p["classifier_"+r]=np.repeat(np.where(y>0,.8,.2)[:,None],5,axis=1)
        sync(p,r)
    return p


def sync(p,r):
    p["stretch_"+r]=np.repeat(p["share_"+r][:,None],4,axis=1)


def bundle(p=None):
    return {k:deepcopy(panel() if p is None else p) for k in ("fit",*FULL_AXES)}


class FourWayFixtures(unittest.TestCase):
    def test_prespec_has_no_fixed_tree_floor_in_free_category(self):
        p=prespec()
        self.assertEqual(p["categories"],list(CATEGORIES))
        self.assertEqual(p["free_ridge"],1e-4)
        self.assertFalse(p["query_cohort_statistics"])
        self.assertEqual(p["score_share_weight"],0)

    def test_fits_respect_four_distinct_family_constraints(self):
        p=panel(); y=p["target_share"]
        for r in ("D0","H0"):
            p["share_"+r][:,:2]=np.minimum(y[:,None]+.3,1)
            sync(p,r)
        for category in CATEGORIES:
            reg,audit=_fit_regression(p,category)
            for r in ("D0","H0"):
                w=np.asarray(reg["member_weights"][r])
                self.assertAlmostEqual(w.sum(),1)
                self.assertTrue((w>=0).all())
                self.assertTrue(audit[r]["success"])
                if category=="capped":
                    self.assertGreaterEqual(w[0],.5-1e-8)
                    self.assertLessEqual(w[2:].max(),.25+1e-8)
                if category=="free": self.assertLess(w[:2].sum(),.01)
                if category=="neural_only": self.assertEqual(w[:2].sum(),0)
                if category=="tree_only": self.assertEqual(w[2:].sum(),0)

    def test_fit_weights_do_not_change_when_gate_labels_change(self):
        a=bundle(); b=deepcopy(a)
        for name in FULL_AXES:
            b[name]["target_share"]*=.3
        ra=select(a,baseline()); rb=select(b,baseline())
        self.assertEqual(ra["regression_fit_audit"],rb["regression_fit_audit"])
        self.assertEqual(ra["classifier_fit_audit"],rb["classifier_fit_audit"])

    def test_free_can_use_neural_signal_without_half_tree_floor(self):
        p=panel(); y=p["target_share"]
        for r in ("D0","H0"):
            p["share_"+r][:,:2]=y[:,None]+.1
            sync(p,r)
        result=select(bundle(p),baseline())
        self.assertLess(result["gate_metrics"]["free"]["local_gate"]["rmse"],.001)
        self.assertGreater(result["gate_metrics"]["capped"]["local_gate"]["rmse"],.049)
        self.assertFalse(result["selection"]["regression_tree_fallback"])
        self.assertLess(result["gate_metrics"]["selected"]["local_gate"]["rmse"],.001)
        self.assertIsNotNone(result["neural_submission_policy"])
        neural=result["neural_submission_policy"]
        self.assertEqual(neural["classification"],result["policies"]["selected"]["classification"])
        self.assertLess(apply_policy(p,neural)["share"].mean(),y.mean()+.001)

    def test_original_identity_is_preserved_if_tree_safe_and_no_candidate_improves(self):
        p=panel(); y=p["target_share"]
        # Baseline half Tree and half neural errors cancel exactly; category
        # fits must not force a Tree downgrade solely because no strict gain.
        offset=np.minimum(y/2,.1)
        for r in ("D0","H0"):
            p["share_"+r][:,:2]=(y+offset)[:,None]
            p["share_"+r][:,2:]=(y-offset)[:,None]
            sync(p,r)
        result=select(bundle(p),baseline())
        self.assertFalse(result["selection"]["regression_tree_fallback"])
        self.assertEqual(result["selection"]["regression_source"],"immutable_identity")
        self.assertIsNone(result["neural_submission_policy"])

    def test_tree_fallback_when_neural_generalization_fails(self):
        b=bundle()
        for name,p in b.items():
            y=p["target_share"]
            for r in ("D0","H0"):
                # Tree wins stress axes; neural wins local_gate. No policy
                # preserves every baseline axis, so fallback must expose a
                # local_gate tradeoff instead of claiming universal gain.
                p["share_"+r][:,:2]=y[:,None]+.05 if name in ("fit","local_gate") else y[:,None]
                p["share_"+r][:,2:]=y[:,None] if name in ("fit","local_gate") else y[:,None]+.2
                sync(p,r)
        result=select(b,baseline())
        self.assertTrue(result["selection"]["regression_tree_fallback"])
        for name in FULL_AXES[1:]:
            self.assertLess(result["gate_metrics"]["selected"][name]["rmse"],1e-12)
        self.assertGreater(result["selection"]["fallback_rmse_changes_vs_proposed"]["local_gate"],0)
        neural=[r for r in result["candidates"] if r.get("category")=="neural_only" and r["task"]=="regression"]
        self.assertTrue(all(not r["eligible"] for r in neural))

    def test_no_Test_report_API_and_no_identity_drift(self):
        b=bundle(); b["Test"]=deepcopy(b["fit"])
        with self.assertRaises(ValueError): select(b,baseline())
        b=bundle(); b["fit"]["stretch_D0"][0,0,2]+=.001
        with self.assertRaises(ValueError): select(b,baseline())

    def test_target_free_batch_invariant_application(self):
        b=bundle(); result=select(b,baseline())
        policy=json.loads(json.dumps(result))["policies"]["selected"]
        query={k:v for k,v in b["local_gate"].items() if k.startswith(("share_","classifier_","stretch_"))}
        full=apply_policy(query,policy)
        for index in (0,17,90):
            one={k:v[index:index+1] for k,v in query.items()}
            out=apply_policy(one,policy)
            for k in ("share","score"):
                np.testing.assert_allclose(out[k],full[k][index:index+1],atol=1e-14)

    def test_classification_is_not_regression_share(self):
        p=panel()
        for r in ("D0","H0"):
            p["classifier_"+r][:,:2]=.5
        result=select(bundle(p),baseline())
        self.assertEqual(result["gate_metrics"]["selected"]["local_gate"]["score_auc"],1)
        policy=result["policies"]["selected"]
        old=apply_policy(p,policy)["score"]
        p=deepcopy(p)
        for r in ("D0","H0"):
            p["share_"+r][:]=.9; sync(p,r)
        np.testing.assert_array_equal(apply_policy(p,policy)["score"],old)
        self.assertEqual(policy["score_share_weight"],0)

    def test_family_classifier_grids_are_simplexes_and_finite(self):
        counts=[]
        for category in CATEGORIES:
            rows=_classifier_grid(category,NAMES); counts.append(len(rows))
            for _,w in rows:
                self.assertAlmostEqual(w.sum(),1)
                self.assertTrue((w>=0).all())
                if category=="tree_only": self.assertEqual(w[2:].sum(),0)
                if category=="neural_only": self.assertEqual(w[:2].sum(),0)
        self.assertLessEqual(max(counts),12)

    def test_native_long_positive_damage_cannot_hide_in_mixed_gain(self):
        b=bundle()
        for p in b.values():
            y=p["target_share"]
            for r in ("D0","H0"):
                p["share_"+r][:]=y[:,None]+.05; sync(p,r)
                for j,e in enumerate(EPSILONS[1:],1):
                    p["stretch_"+r][:,j,2:]=np.maximum(y[:,None]+.05-e,0)
        p=b["native12_VS_hf_high_negative"]
        p["target_share"]=np.r_[np.full(50,.2),np.zeros(50)]
        p["support_minutes"]=np.r_[np.full(50,12),np.zeros(50)]
        pred=np.where(p["target_share"]>0,.18,.08)
        for r in ("D0","H0"):
            p["share_"+r][:]=pred[:,None]; sync(p,r)
            p["classifier_"+r][:]=np.where(p["target_share"]>0,.8,.2)[:,None]
            for j,e in enumerate(EPSILONS[1:],1):
                p["stretch_"+r][:,j,2:]=np.maximum(pred[:,None]-e,0)
        result=select(b,baseline())
        row=next(r for r in result["candidates"] if r["id"]=="neural_only:epsilon_0.01")
        self.assertEqual(result["axis_audit"]["native12_long_positive"]["n"],50)
        self.assertIn("native12_long_positive:underprediction_mean_worsened",row["reasons"])
        self.assertFalse(row["eligible"])


if __name__=="__main__": unittest.main(verbosity=2)
