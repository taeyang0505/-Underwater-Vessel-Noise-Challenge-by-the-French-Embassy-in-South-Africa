"""Tiny synthetic CPU LightGBM and group-gated ensemble fixtures only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from pipeline.trees_ensemble_r9 import (
    CODE_NAMES, PhysicsLGBM, apply_frozen_ensemble, array_sha, canonical_sha,
    feature_schema, file_sha, fit_safe_ensemble, freeze_ensemble,
    transform_features,
)

ROOT = Path(__file__).resolve().parents[1]


def reject(fn):
    try:
        fn()
    except (ValueError, TypeError, FileExistsError):
        return True
    raise AssertionError('Invalid operation accepted')


def fake_data(n, seed, role):
    rng = np.random.default_rng(seed)
    background = np.exp(rng.normal(0, .3, (n, 15, 25)))
    vessel_shape = np.exp(-np.arange(25)/10.)
    temporal = np.exp(-((np.arange(15)-7)/3.)**2)
    amplitude = rng.uniform(.05, 2., n)
    amplitude[::5] = 0
    added = amplitude[:, None, None]*temporal[None, :, None]*vessel_shape[None, None, :]
    energy = background+added
    observed = np.ones((n, 15, 25), bool)
    observed[::7, :3] = False
    observed[::11, :, 5] = False
    db = np.where(observed, 80+10*np.log10(energy), np.nan)
    share = (np.where(observed, added, 0).sum((1, 2))/np.where(observed, energy, 0).sum((1, 2))).astype(np.float32)
    q = np.where(observed, added/energy, 0).astype(np.float32)
    return {'db': db, 'share': share, 'presence': (amplitude > 0).astype(np.float32),
            'cell_share': q, 'role': role}


def run():
    start = time.perf_counter()
    source_names = list(CODE_NAMES)+['ops/verify_trees_ensemble_r9.py']
    before = {name: file_sha(ROOT/name) for name in source_names}
    train = fake_data(128, 20261026, 'train')
    inner = fake_data(32, 20261027, 'selection')
    raw_hashes = {role: {key: array_sha(data[key]) for key in ('db', 'share', 'presence', 'cell_share')}
                  for role, data in [('train', train), ('inner', inner)]}
    binding = {'fixture': True, 'source': 'synthetic_random_numbers_only',
               'train_sha256': raw_hashes['train'], 'inner_sha256': raw_hashes['inner']}
    settings = {'maximum_trees': 10, 'early_stopping_patience': 3, 'threads': 2,
                'minimum_leaf_rows': 5, 'feature_batch_size': 32}
    with tempfile.TemporaryDirectory(prefix='r9_tree_ensemble_fixture_') as temporary:
        folder = Path(temporary)
        model = PhysicsLGBM(settings, seed=20261026)
        summary = model.fit(train, inner, folder/'tree', binding, fixture=True)
        p = model.predict(inner['db'], batch_size=32)
        restored = PhysicsLGBM.load(folder/'tree', expected_binding=binding, allow_fixture=True)
        by_row = restored.predict(inner['db'], batch_size=1)
        by_seven = restored.predict(inner['db'], batch_size=7)
        errors = {key: max(float(np.max(np.abs(p[key]-other[key]))) for other in (by_row, by_seven))
                  for key in p}
        assert max(errors.values()) == 0
        assert np.array_equal(p['score'], .5*p['share']+.5*p['score_classifier'])
        assert all(np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all() for a in p.values())
        x32 = transform_features(inner['db'], batch_size=32)
        x1 = transform_features(inner['db'], batch_size=1)
        assert x32.shape == (32, 3827) and np.array_equal(x32, x1)
        tree_guards = {
            'query_labels_rejected': reject(lambda: restored.predict(inner)),
            'output_overwrite_rejected': reject(lambda: model.fit(train, inner, folder/'tree', binding, fixture=True)),
            'binding_change_rejected': reject(lambda: PhysicsLGBM.load(folder/'tree', expected_binding={'different': True}, allow_fixture=True)),
            'fixture_as_production_rejected': reject(lambda: PhysicsLGBM.load(folder/'tree', expected_binding=binding)),
            'train_cap_rejected': reject(lambda: PhysicsLGBM({'maximum_training_rows': 100001})),
            'tree_cap_rejected': reject(lambda: PhysicsLGBM({'maximum_trees': 1501})),
        }
        weights_path = folder/'tree/share.txt'
        original_weights = weights_path.read_bytes()
        weights_path.write_bytes(original_weights+b'\nchanged\n')
        tree_guards['weight_tamper_rejected'] = reject(
            lambda: PhysicsLGBM.load(folder/'tree', expected_binding=binding, allow_fixture=True))
        weights_path.write_bytes(original_weights)
        tree_result = {
            'train_rows': 128, 'inner_rows': 32, 'settings': settings,
            'features': 3827, 'schema_sha256': canonical_sha(feature_schema()),
            'maximum_boosting_rounds': 10, 'best_iteration': summary['best_iteration'],
            'presence_best_iteration': summary['presence_best_iteration'],
            'score_policy': summary['score_policy'],
            'prediction_reload_chunk_max_abs': errors,
            'feature_chunk_exact': True, 'predictions_finite_bounded': True,
            'score_identity_exact': True, 'guards': tree_guards,
            'artifact_sha256': summary['artifact_sha256'],
            'lightgbm_version': summary['lightgbm_version'],
            'feature_matrix_bytes': summary['feature_matrix_bytes'],
            'elapsed_seconds': summary['elapsed_seconds'],
            'scope': 'two tiny synthetic boosters, not acoustic-dataset training',
        }

        # Finite prespecified candidates reject a poisoned tree while retaining
        # two useful neural masks. Source groups never enter model features.
        n = 120
        y = np.linspace(.15, .85, n)
        groups = np.repeat([f'source_{i}' for i in range(6)], 20)
        fit, gate = np.arange(60), np.arange(60, n)
        error = .04*np.cos(np.arange(n))
        predmap = {'cnn_a': y+error, 'cnn_b': y-error, 'lgbm_physics': 1-y}
        policy = fit_safe_ensemble(y, predmap, groups, fit_indices=fit, gate_indices=gate)
        assert policy['candidate_admitted']
        assert policy['weights']['lgbm_physics'] == 0
        assert policy['selected_name'] == 'equal/non_tree'
        archive_sha = freeze_ensemble(policy, folder/'policy.json')
        prediction = apply_frozen_ensemble(json.loads((folder/'policy.json').read_text()), predmap)
        assert np.allclose(prediction['share'], y, atol=3e-8, rtol=0)
        assert policy['gate_candidates_evaluated'] == 1
        assert not policy['gate_weight_refit']

        # A fit-improving blend is rejected on disjoint source groups. The
        # fallback was chosen on fit rows, not retrospectively on gate rows.
        a, b = y+.04, y-.05
        b[gate] = y[gate]+.12
        failure = fit_safe_ensemble(y, {'cnn_a': a, 'cnn_b': b}, groups,
                                    fit_indices=fit, gate_indices=gate)
        assert failure['candidate_fit_improves'] and not failure['candidate_admitted']
        assert failure['selected_name'] == failure['fallback_name']
        assert failure['gate_candidates_evaluated'] == 1
        invalid = dict(policy)
        invalid['weights'] = dict(policy['weights'], lgbm_physics=.2)
        shared_groups = groups.copy()
        shared_groups[gate[0]] = groups[fit[0]]
        ensemble_guards = {
            'overlapping_source_group_rejected': reject(lambda: fit_safe_ensemble(
                y, predmap, shared_groups, fit_indices=fit, gate_indices=gate)),
            'overlapping_row_rejected': reject(lambda: fit_safe_ensemble(
                y, predmap, groups, fit_indices=fit, gate_indices=np.r_[fit[0], gate])),
            'policy_tamper_rejected': reject(lambda: apply_frozen_ensemble(invalid, predmap)),
            'member_set_change_rejected': reject(lambda: apply_frozen_ensemble(policy, {'cnn_a': y})),
            'policy_overwrite_rejected': reject(lambda: freeze_ensemble(policy, folder/'policy.json')),
        }
        # Scalar neural blends commute with common-w pooling. A tree has no
        # q-grid, so this identity is deliberately not claimed for tree blends.
        rng = np.random.default_rng(9)
        w = rng.uniform(.01, 1., (4, 15, 25))
        w /= w.sum((1, 2), keepdims=True)
        q1, q2 = rng.uniform(0, 1, (2, 4, 15, 25))
        scalar = .5*(q1*w).sum((1, 2))+.5*(q2*w).sum((1, 2))
        pooled = ((.5*q1+.5*q2)*w).sum((1, 2))
        pool_error = float(np.max(np.abs(scalar-pooled)))
        assert pool_error < 1e-14
        ensemble_result = {
            'source_groups': 6, 'fit_groups': policy['fit_groups'], 'gate_groups': policy['gate_groups'],
            'fit_rows': 60, 'gate_rows': 60,
            'poisoned_tree_weight': policy['weights']['lgbm_physics'],
            'selected_name': policy['selected_name'], 'weights': policy['weights'],
            'one_candidate_gate_only': True, 'gate_failed_candidate_returns_fit_fallback': True,
            'admitted_policy': policy, 'rejected_candidate_policy': failure,
            'serialized_policy_sha256': archive_sha,
            'neural_common_weight_pool_linearity_max_abs': pool_error,
            'tree_cell_mask_equivalence_claimed': False, 'guards': ensemble_guards,
        }
    after = {name: file_sha(ROOT/name) for name in source_names}
    assert before == after
    for role, data in [('train', train), ('inner', inner)]:
        assert raw_hashes[role] == {key: array_sha(data[key]) for key in raw_hashes[role]}
    return {
        'format': 'r9_trees_ensemble_verification_v1', 'status': 'PASS',
        'synthetic_only': True, 'real_acoustic_data_read': False,
        'Test_or_old_OUTER_read': False, 'GPU_used': False,
        'threads': 2, 'source_code_sha256': before, 'source_code_unchanged': True,
        'input_arrays_unchanged': True, 'tree': tree_result, 'ensemble': ensemble_result,
        'elapsed_seconds': time.perf_counter()-start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/R9_TREES_ENSEMBLE_VERIFICATION.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous fixture evidence')
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output),
                      'feature_count': result['tree']['features'],
                      'ensemble_weights': result['ensemble']['weights']}))


if __name__ == '__main__':
    main()
