"""Generated-array CPU checks for R5 physics13; no actual fold/Test access."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.physics_features_r5 import (
    SCHEMA, FEATURE_NAMES, FIXED_FEATURE_NAMES, EXISTING_MAPPING,
    LOW_BANDS, HIGH_BANDS, MID_BANDS, fit_physics_pca,
    fixed_physics_features_r5, physics_features_r5, physics_pca_payload_sha256,
    validate_physics_pca, save_physics_pca, load_physics_pca,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rejected(action, errors=(ValueError, TypeError, FileExistsError)):
    try:
        action()
    except errors:
        return True
    raise AssertionError('An unsafe/invalid operation was accepted')


def run(output):
    checks = {}; rng = np.random.default_rng(20261025)
    constant = np.full((1, 15, 25), 100., dtype=np.float32)
    fixed = fixed_physics_features_r5(constant)
    assert fixed.shape == (1, 12) and np.array_equal(fixed, np.zeros_like(fixed))
    checks['constant_fixed_features_zero'] = True

    bump = constant.copy(); bump[0, 7, HIGH_BANDS] += 10
    f, d = fixed_physics_features_r5(bump, return_diagnostics=True)
    assert np.isclose(f[0, 0], 10., atol=1e-6) and d['high_complete_minutes'][0] == 15
    checks['analytic_high_power_bump_db'] = float(f[0, 0])

    corr = np.tile(constant, (2, 1, 1)); time = np.arange(15, dtype=float)
    for i in range(2):
        corr[i, :, LOW_BANDS] = (100 + time)[None, :]
        corr[i, :, HIGH_BANDS] = (100 + (2 if i == 0 else -2) * time)[None, :]
    f = fixed_physics_features_r5(corr)
    assert np.allclose(f[:, 1], [1., -1.], atol=1e-6)
    corr[0, 2, LOW_BANDS[0]] = np.nan; corr[0, 3, HIGH_BANDS[0]] = np.nan
    f, d = fixed_physics_features_r5(corr, return_diagnostics=True)
    assert d['corr_paired_minutes'][0] == 13 and np.isclose(f[0, 1], 1., atol=1e-6)
    checks['pair_mask_correlation_analytic'] = f[:, 1].tolist()
    sparse = constant.copy(); sparse[0, 2:] = np.nan
    f, d = fixed_physics_features_r5(sparse, return_diagnostics=True)
    assert f[0, 1] == 0 and not d['corr_defined'][0]
    checks['fewer_than_three_pairs_uses_zero_placeholder'] = True

    peak = constant.copy(); peak[0, 5, 12] += 10
    f, d = fixed_physics_features_r5(peak, return_diagnostics=True)
    expected = 9. / (15 * 25 + 9.)
    column = 2 + int(np.flatnonzero(MID_BANDS == 12)[0])
    assert d['mid_peak_minute_zero_based'][0] == 5 and np.isclose(f[0, column], expected, atol=1e-8)
    assert np.count_nonzero(f[0, 2:]) == 1
    checks['analytic_peak_mid_excess_fraction'] = {'actual': float(f[0, column]), 'expected': expected}
    gain_error = float(np.max(np.abs(fixed_physics_features_r5(peak + 13.) - f)))
    assert gain_error < 1e-6
    checks['fixed_feature_common_gain_max_error'] = gain_error

    # A rank-three spectral shape family with an independent fourth direction.
    raw_basis = rng.normal(size=(25, 4)); raw_basis -= raw_basis.mean(axis=0)
    basis, _ = np.linalg.qr(raw_basis)
    coefficients = rng.normal(size=(96, 3)) * [2., 1.5, 1.]
    spectra = 100. + coefficients @ basis[:, :3].T
    train = np.broadcast_to(spectra[:, None, :], (96, 15, 25)).astype(np.float32).copy()
    train_before = train.copy()
    binding = {'train_npz_sha256': 'a' * 64, 'data_package_hash': 'b' * 64, 'source_hash': 'c' * 64,
        'seed': 42, 'fold': 0, 'train_examples': 96, 'quantization_decimals': None,
        'preprocessing_policy_sha256': 'd' * 64}
    state = fit_physics_pca(train, binding)
    state_before = json.dumps(state, sort_keys=True)
    in_family = physics_features_r5(train[:3], state, expected_binding=binding)
    novel = train[:1].copy() + (4 * basis[:, 3])[None, None, :]
    novel_features = physics_features_r5(novel, state, expected_binding=binding)
    assert in_family[:, -1].max() < 1e-4 and novel_features[0, -1] > .7
    checks['PCA_reconstruction_residual_known_subspace'] = {
        'training_family_max_rms_db': float(in_family[:, -1].max()),
        'orthogonal_shape_rms_db': float(novel_features[0, -1]),
        'effective_rank': state['effective_rank'],
        'nominal_shape_dimensions_before_float32_rounding': 3,
        'effective_rank_includes_float32_rounding': True,
    }
    assert np.array_equal(train, train_before)

    query = rng.normal(100, 5, size=(9, 15, 25)).astype(np.float32)
    query[rng.random(query.shape) < .2] = np.nan
    query[0] = np.nan; query[1] = 100.; query[2, :, 12] = np.nan
    query_before = query.copy()
    all_features, diagnostics = physics_features_r5(query, state, expected_binding=binding, return_diagnostics=True)
    chunks = np.concatenate([physics_features_r5(query[:1], state, expected_binding=binding),
        physics_features_r5(query[1:4], state, expected_binding=binding),
        physics_features_r5(query[4:], state, expected_binding=binding)])
    chunk_error = float(np.max(np.abs(all_features - chunks)))
    alone = physics_features_r5(query[3:4], state, expected_binding=binding)
    changed = query.copy(); changed[:3] = 25.; changed[4:] = 175.
    companion_error = float(np.max(np.abs(alone - physics_features_r5(changed, state, expected_binding=binding)[3:4])))
    assert chunk_error < 1e-6 and companion_error < 1e-6
    assert all_features.shape == (9, 13) and np.isfinite(all_features).all()
    assert np.array_equal(all_features[0], np.zeros(13)) and diagnostics['pca_observed_bands'][0] == 0
    assert np.array_equal(query, query_before, equal_nan=True)
    assert json.dumps(state, sort_keys=True) == state_before
    checks['missing_constant_input_safe_and_nonmutation'] = True
    checks['query_chunk_max_abs_error'] = chunk_error
    checks['changed_query_companions_max_abs_error'] = companion_error
    assert physics_features_r5(query[:0], state, expected_binding=binding).shape == (0, 13)
    constant_state = fit_physics_pca(np.repeat(constant, 96, axis=0), binding)
    assert constant_state['effective_rank'] == 0
    assert physics_features_r5(constant, constant_state, expected_binding=binding)[0, -1] < 1e-6
    checks['rank_zero_TRAIN_PCA_safe'] = True
    checks['all_missing_TRAIN_fit_rejected'] = rejected(lambda: fit_physics_pca(np.full_like(train, np.nan), binding))
    unseen_band_train = train.copy(); unseen_band_train[:, :, 24] = np.nan
    unseen_state = fit_physics_pca(unseen_band_train, binding)
    unseen_features = physics_features_r5(query, unseen_state, expected_binding=binding)
    assert unseen_state['observed_band_training_counts'][24] == 0
    assert unseen_state['shape_mean_db'][24] == 0 and np.isfinite(unseen_features).all()
    checks['unobserved_TRAIN_band_has_explicit_zero_shape_mean'] = True
    infinity = constant.copy(); infinity[0, 0, 0] = np.inf
    checks['infinite_input_rejected'] = rejected(lambda: fixed_physics_features_r5(infinity))

    gain = physics_features_r5(query[3:4] + 11., state, expected_binding=binding)
    full_gain_error = float(np.max(np.abs(gain - alone)))
    assert full_gain_error < 2e-5
    checks['all_13_common_gain_max_error'] = full_gain_error
    checks['nonTRAIN_fit_rejected'] = all(rejected(lambda s=s: fit_physics_pca(train, binding, split=s)) for s in ['validation', 'outer_validation', 'Test'])
    checks['labels_dictionary_rejected'] = rejected(lambda: physics_features_r5({'db': query, 'share': np.zeros(len(query))}, state, expected_binding=binding))
    bad = dict(binding, train_examples=95)
    checks['partial_TRAIN_row_binding_rejected'] = rejected(lambda: fit_physics_pca(train, bad))
    bad = dict(binding, quantization_decimals=2)
    checks['raw_rounded_preprocessing_binding_mismatch_rejected'] = rejected(lambda: physics_features_r5(query, state, expected_binding=bad))
    bad = dict(binding, preprocessing_policy_sha256='e' * 64)
    checks['preprocessing_policy_SHA_mismatch_rejected'] = rejected(lambda: physics_features_r5(query, state, expected_binding=bad))
    bad = dict(binding, query_label=.5)
    checks['undeclared_label_binding_rejected'] = rejected(lambda: fit_physics_pca(train, bad))
    tampered = copy.deepcopy(state); tampered['shape_mean_db'][0] += .01
    checks['payload_mutation_rejected'] = rejected(lambda: validate_physics_pca(tampered, expected_binding=binding))
    with tempfile.TemporaryDirectory(prefix='r5-physics-fixture-') as temp:
        path = Path(temp) / 'physics_pca.json'
        receipt = save_physics_pca(path, state, expected_binding=binding)
        restored = load_physics_pca(path, expected_binding=binding, expected_file_sha256=receipt['file_sha256'], expected_payload_sha256=receipt['payload_sha256'])
        assert np.array_equal(physics_features_r5(query, restored, expected_binding=binding), all_features)
        checks['JSON_state_reload_bitwise_equal'] = True
        checks['fitted_state_overwrite_rejected'] = rejected(lambda: save_physics_pca(path, state, expected_binding=binding))
        with path.open('a') as f:
            f.write(' ')
        checks['file_SHA_whitespace_mutation_rejected'] = rejected(lambda: load_physics_pca(path, expected_binding=binding, expected_file_sha256=receipt['file_sha256']))

    result = {'status': 'PASS', 'feature_schema': SCHEMA, 'feature_names': FEATURE_NAMES,
        'fixed_features': len(FIXED_FEATURE_NAMES), 'total_features': len(FEATURE_NAMES),
        'existing_feature_mapping': EXISTING_MAPPING, 'low_band_indices': LOW_BANDS.tolist(),
        'high_band_indices': HIGH_BANDS.tolist(), 'mid_band_indices': MID_BANDS.tolist(),
        'checks': checks, 'PCA_fit_uses_labels_or_recipes': False, 'PCA_fit_split': 'generated TRAIN fixture only',
        'PCA_target': state['definitions']['pca_input'], 'quantization_is_callers_responsibility': True,
        'PCA_payload_sha256': physics_pca_payload_sha256(state), 'real_project_datasets_opened': False,
        'Test_or_calibration_or_sealed_or_R4_history_opened': False, 'GPU_used': False,
        'actual_fold_training_executed': False, 'limits': ['Historical EDA KS is train-versus-Test discrimination, not target-performance evidence.',
            'New feature definitions are not a claimed reproduction of historical EDA KS numbers.',
            'Quantile excess and PCA residual are diagnostic features, not measured vessel-only truth.',
            'No competition RMSE improvement or independent source expansion is established by this numerical fixture.'],
        'code_sha256': {name: sha(ROOT / name) for name in ['pipeline/physics_features_r5.py', 'ops/verify_physics_features_r5.py']}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/R5_PHYSICS_FEATURES_FIXTURE.json')
    run(parser.parse_args().output)
