"""Independently recompute completed R2 OOF metrics and frozen ensembles."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(a):
    final, progress = a.final_dir.resolve(), a.progress_dir.resolve()
    paths = {name: final / name for name in ['cv_report.json', 'oof_predictions.npz', 'oof_metadata.parquet']}
    if not all(p.is_file() and not p.is_symlink() for p in paths.values()):
        raise ValueError('All completed OOF files must be regular local files')
    reported = json.loads(paths['cv_report.json'].read_text())
    if not reported.get('status', '').startswith('completed nested Group CV') or reported['n_splits'] != 5:
        raise ValueError('A complete five-fold report is required')
    if any(reported.get(k) is not False for k in ['challenge_test_accessed', 'calibration_accessed', 'sealed_accessed', 'outer_used_for_early_stopping', 'outer_used_for_ensemble_weights']):
        raise ValueError('Report fitting/access provenance is unsafe')
    with zipfile.ZipFile(paths['oof_predictions.npz']) as z:
        if z.testzip() is not None:
            raise ValueError('OOF NPZ member CRC failed')
    with np.load(paths['oof_predictions.npz'], allow_pickle=False) as z:
        data = {name: z[name] for name in z.files}
    meta = pd.read_parquet(paths['oof_metadata.parquet'])
    y = data['share_true'].astype('float64'); n = len(y)
    if n != 15000 or n != reported['oof_examples'] or len(meta) != n:
        raise ValueError('OOF row count differs from five held-out sets')
    if len(np.unique(data['sample_id'])) != n or not np.array_equal(data['sample_id'].astype(str), meta.sample_id.to_numpy(dtype=str)):
        raise ValueError('OOF sample IDs duplicate or do not align with metadata')
    if not np.isfinite(y).all() or ((y < 0) | (y > 1)).any() or not np.array_equal(data['presence_true'], (y > 0).astype('float32')):
        raise ValueError('Invalid held-out target values')
    folds = meta['fold'].to_numpy()
    if set(np.unique(folds)) != set(range(5)) or any(np.sum(folds == f) != 3000 for f in range(5)):
        raise ValueError('Unexpected fold identity/count')
    boundary = meta['boundary_artifact_candidate'].to_numpy()
    if boundary.dtype.kind != 'b':
        raise ValueError('Boundary quality flag must be an explicit boolean')
    clean = ~boundary; metrics = {}; policy_checks = []
    for kind, expected in reported['pooled_oof'].items():
        p = data[kind + '_share'].astype('float64')
        if p.shape != y.shape or not np.isfinite(p).all() or ((p < -1e-10) | (p > 1+1e-10)).any():
            raise ValueError('Invalid OOF predictions: ' + kind)
        error = p-y; rmse = float(np.sqrt(np.dot(error, error)/n))
        clean_rmse = float(np.sqrt(np.mean(error[clean]**2)))
        if not np.isclose(rmse, expected['rmse'], atol=1e-10, rtol=0):
            raise ValueError('Pooled RMSE differs: ' + kind)
        if not np.isclose(clean_rmse, expected['oof_rmse_excluding_boundary'], atol=1e-10, rtol=0):
            raise ValueError('Boundary-excluded RMSE differs: ' + kind)
        by_fold = [float(np.sqrt(np.mean(error[folds == f]**2))) for f in range(5)]
        if not np.isclose(float(np.mean(by_fold)), reported['fold_rmse_mean_std'][kind]['mean'], atol=1e-10, rtol=0):
            raise ValueError('Fold mean differs: ' + kind)
        metrics[kind] = {'rmse': rmse, 'boundary_excluded_rmse': clean_rmse,
            'mae': float(np.mean(np.abs(error))), 'bias_prediction_minus_truth': float(error.mean()),
            'fold_rmse': by_fold, 'fold_rmse_mean': float(np.mean(by_fold)),
            'fold_rmse_std': float(np.std(by_fold, ddof=1))}
    for fold in range(5):
        file = progress / f'fold_{fold:02d}' / 'ensemble_policy.json'
        policy = json.loads(file.read_text()); w = np.asarray(policy['weights'], dtype='float64')
        if policy.get('fit_split') != 'inner validation only' or policy.get('outer_used_for_fit') is not False or policy.get('affine') != [1., 0.]:
            raise ValueError('Policy was not frozen exclusively from inner data')
        if not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1., atol=1e-8, rtol=0):
            raise ValueError('Policy weights are not a valid convex combination')
        mask = folds == fold
        matrix = np.column_stack([data[k + '_share'][mask] for k in policy['models']])
        delta = float(np.max(np.abs(matrix @ w - data['ensemble_share'][mask])))
        if delta > 1e-10:
            raise ValueError('OOF ensemble differs from the frozen inner policy')
        policy_checks.append({'fold': fold, 'policy_sha256': sha(file), 'max_reconstruction_error': delta})
    result = {'status': 'passed independent complete OOF verification', 'examples': n,
        'clean_examples': int(clean.sum()), 'boundary_flagged_examples': int(boundary.sum()),
        'input_sha256': {name: sha(p) for name, p in paths.items()}, 'metrics': metrics,
        'frozen_policy_reconstruction': policy_checks,
        'new_training_or_fitting_performed': False, 'hyperparameters_or_weights_changed': False,
        'official_submission': False}
    if a.output.exists():
        raise FileExistsError('Preserve previous verification output; use a fresh path')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'examples': n,
        'ensemble': metrics['ensemble'], 'report': str(a.output)}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--final-dir', type=Path, required=True)
    p.add_argument('--progress-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    verify(p.parse_args())
