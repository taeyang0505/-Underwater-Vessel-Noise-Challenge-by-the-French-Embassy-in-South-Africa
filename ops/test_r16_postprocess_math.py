"""Meaningful math/contract fixtures; no training or Test.csv access."""
import json
import unittest
import numpy as np

from r16_postprocess_math import (
    METHODS, apply_score_policy, apply_score_transform, classifier_weight_candidates,
    convex_pool, fit_score_transform, pool_mask_shares, select_classifier_policy,
    stretched_mask, stretched_share_grid, summary_metrics, weighted_auc,
)


class PostprocessFixtures(unittest.TestCase):
    def test_query_batch_order_and_singleton_invariance(self):
        fit = np.array([[0, .2], [.1, .2], [.1, .2], [.8, .2], [1, .2]])
        query = np.array([[.7, .99], [0, .2], [.1, 0], [1, 1]])
        for method in METHODS:
            state = fit_score_transform(fit, method)
            # A JSON round-trip demonstrates frozen state can be persisted.
            state = json.loads(json.dumps(state))
            before = json.dumps(state, sort_keys=True)
            all_at_once = apply_score_policy(query, [.7, .3], state)
            singletons = np.concatenate([apply_score_policy(row[None, :], [.7, .3], state)
                                         for row in query])
            np.testing.assert_allclose(all_at_once, singletons, rtol=0, atol=1e-14)
            order = [2, 0, 3, 1]
            np.testing.assert_allclose(apply_score_policy(query[order], [.7, .3], state),
                                       all_at_once[order], rtol=0, atol=1e-14)
            self.assertEqual(before, json.dumps(state, sort_keys=True))

    def test_ecdf_fixed_reference_mid_ties_and_endpoints(self):
        state = fit_score_transform([[0], [0], [1], [1]], "ecdf")
        actual = apply_score_transform([[0], [.5], [1]], state)[:, 0]
        np.testing.assert_array_equal(actual, [.25, .5, .75])
        # Inserting other query rows must not change the reference ranks.
        self.assertEqual(apply_score_transform([[.5]], state)[0, 0], .5)
        state = fit_score_transform([[.2], [.8]], "ecdf")
        np.testing.assert_array_equal(apply_score_transform([[0], [1]], state)[:, 0], [0, 1])

    def test_logit_uses_fit_statistics_and_constant_column_is_finite(self):
        state = fit_score_transform([[.25, .2], [.75, .2]], "standardized_logit")
        np.testing.assert_allclose(state["center"][0], 0, atol=1e-15)
        np.testing.assert_allclose(state["scale"][0], np.log(3), atol=1e-15)
        self.assertEqual(state["scale"][1], 1)
        np.testing.assert_allclose(apply_score_transform([[.25, .2], [.75, .2]], state),
                                   [[-1, 0], [1, 0]], atol=1e-14)
        self.assertTrue(np.isfinite(apply_score_policy([[0, 1], [1, 0]], [.5, .5], state)).all())

    def test_weighted_auc_against_manual_pairwise_oracle(self):
        labels = [1, 1, 0, 0]; scores = [1, .5, .2, .5]; weights = [2, 4, 1, 3]
        # Positive score1 contributes8; score.5 contributes4*(1+3/2)=10.
        self.assertAlmostEqual(weighted_auc(labels, scores, weights), 18 / 24)
        self.assertEqual(weighted_auc(labels, [1, 1, 1, 1], weights), .5)
        self.assertEqual(weighted_auc(labels, [0, 0, 1, 1]), 0)
        self.assertIsNone(weighted_auc([0, 0], [.2, .9]))

    def test_finite_classifier_grid_and_tie_preference(self):
        names = ["lightgbm5", "catboost5", "dasheng_base", "dinov2_small", "timemoe"]
        candidates = classifier_weight_candidates(names)
        self.assertEqual(len(candidates), 6)
        for weights in candidates.values():
            self.assertAlmostEqual(sum(weights), 1)
            self.assertTrue(all(v >= 0 for v in weights))
            self.assertFalse(weights[1] > 0 and any(v > 0 for v in weights[2:]))
        y = np.array([0, .2, 0, .3])
        probabilities = np.tile([.1, .9, .2, .8], (5, 1)).T
        for method in METHODS:
            policy = select_classifier_policy(probabilities, y, names, method=method)
            self.assertEqual(policy["candidate"], "lightgbm_only")
            self.assertEqual(policy["score_share_weight"], 0)
            self.assertEqual(policy["fit_auc"], 1)

    def test_stretch_can_remove_real_weak_positives(self):
        q = np.array([0, .01, .02, .04, .5, 1.0])
        result = stretched_mask(q, .04)
        np.testing.assert_allclose(result[:4], 0)
        self.assertAlmostEqual(result[4], .4791666666666667)
        self.assertEqual(result[5], 1)
        np.testing.assert_array_equal(stretched_mask(q, 0), q)
        self.assertTrue(((q - result) >= 0).all())
        self.assertTrue(((q - result) <= .04 + 1e-15).all())

    def test_pool_preserves_physical_weights_and_not_arithmetic_mean(self):
        q = np.array([[.9, .1]])
        w = np.array([[.01, .99]])
        original_w = w.copy()
        self.assertAlmostEqual(pool_mask_shares(q, w)[0], .108)
        self.assertNotAlmostEqual(pool_mask_shares(q, w)[0], q.mean())
        grid = stretched_share_grid(q, w, [0, .02, .04])
        self.assertEqual(grid.shape, (1, 3))
        self.assertTrue((np.diff(grid, axis=1) <= 0).all())
        np.testing.assert_array_equal(w, original_w)

    def test_final_fixed_ensemble_reduction_bound_is_attainable(self):
        # LGB .5 + NN .5; q==epsilon makes each NN lose exactlyepsilon.
        q = np.full((3, 15, 25), .04)
        w = np.full_like(q, 1 / 375)
        raw = pool_mask_shares(q, w)
        reduced = pool_mask_shares(q, w, epsilon=.04)
        weights = [.5, 0, 1 / 6, 1 / 6, 1 / 6]
        before = np.column_stack([np.full(3, .3), np.full(3, .4), raw, raw, raw])
        after = np.column_stack([np.full(3, .3), np.full(3, .4), reduced, reduced, reduced])
        difference = convex_pool(before, weights) - convex_pool(after, weights)
        np.testing.assert_allclose(difference, .02, atol=1e-14)
        # A fixedD0/H0 50:50 mean cannot enlarge that per-regime upperbound.
        np.testing.assert_allclose(.5 * difference + .5 * difference, .02, atol=1e-14)

    def test_summary_metrics_weighted_known_values(self):
        result = summary_metrics([0, 1], [.2, .8], [.1, .9], [1, 3])
        self.assertAlmostEqual(result["rmse"], .2)
        self.assertAlmostEqual(result["bias"], -.1)
        self.assertAlmostEqual(result["prediction_mean"], .65)
        self.assertAlmostEqual(result["prediction_rms"], .7)
        self.assertEqual(result["share_auc"], 1)
        self.assertEqual(result["score_auc"], 1)

    def test_invalid_contracts_rejected_instead_of_silent_renormalization(self):
        with self.assertRaises(ValueError): convex_pool([[.1, .2]], [.5, .4])
        with self.assertRaises(ValueError): convex_pool([[.1, .2]], [1.1, -.1])
        with self.assertRaises(ValueError): pool_mask_shares([[.1, .2]], [[.1, .1]])
        with self.assertRaises(ValueError): stretched_mask([.2], .05)
        with self.assertRaises(ValueError): fit_score_transform([[np.nan]], "ecdf")
        with self.assertRaises(ValueError): fit_score_transform([[.2]], "query_rank")
        state = fit_score_transform([[.2, .8]], "probability")
        with self.assertRaises(ValueError): apply_score_transform([[.2]], state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
