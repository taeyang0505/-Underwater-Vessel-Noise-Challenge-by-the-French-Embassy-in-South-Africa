"""One isolated CPU LightGBM experiment with a training-background dictionary.

Only original TRAIN components fit the aggregate prior. INNER features are
computed from their dB observations alone. No outer, Test or reserved data are
opened, and the frozen project remains unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np


BASELINE_INNER_RMSE = .28370051829302556


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def route(item):
    refs = item['references']
    native = [r['kind'] == 'native_contiguous_15min_crop' for r in refs]
    return 'zero' if not refs else 'native_only' if all(native) else 'mixed_native_short' if any(native) else 'single_short' if len(refs) == 1 else 'multi_short'


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, dtype='float64') - np.asarray(p, dtype='float64'))**2)))


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_aggregate_prior(prior, output):
    if not isinstance(prior, dict):
        raise TypeError('An explicit serializable aggregate prior dictionary is required')
    arrays = {}; metadata = {}
    for key, value in prior.items():
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('Object arrays are not valid prior parameters')
            arrays[key] = value
        else:
            metadata[key] = json_value(value)
    np.savez_compressed(output / 'background_prior_params.npz', **arrays)
    metadata['array_parameters'] = {key: {'shape': list(v.shape), 'dtype': str(v.dtype),
        'sha256': hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()} for key, v in arrays.items()}
    metadata['aggregate_parameter_file'] = 'background_prior_params.npz'
    (output / 'background_prior_manifest.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))


def main(a):
    project, directory, output = a.project.resolve(), a.data.resolve(), a.output.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh isolated OPS output; do not resume this experiment')
    if any(output.is_relative_to(project / part) for part in ['data', 'models', 'pipeline', 'configs']):
        raise ValueError('Pilot output must remain outside frozen project inputs/code/models')
    if a.clusters != 32 or not 1 <= a.trees <= 1000 or a.threads < 1 or a.feature_batch < 1 or a.smoke_rows < 0:
        raise ValueError('This single experiment requires 32 clusters and at most 1000 CPU trees')
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(project)]
    from background_prior_features import fit_background_prior, transform_background_prior
    from pipeline.mixtures import fixed_features, physics_features
    from pipeline.metrics import evaluate_regression
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    opened = set()
    allowed = {directory / name for name in ['readiness_manifest.json', 'train.npz', 'validation.npz',
        'validation_recipes.json', 'sources/manifest.json']}
    def audit(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            name = path.name.lower()
            if name.startswith(('outer_', 'calibration', 'sealed')) or name == 'test.csv':
                raise PermissionError('Background-prior pilot blocked forbidden data access: ' + str(path))
            if path.is_relative_to(project / 'data'):
                if path not in allowed:
                    raise PermissionError('Background-prior pilot tried an unapproved dataset file: ' + str(path))
                opened.add(str(path.relative_to(project)))
    sys.addaudithook(audit)
    output.mkdir(parents=True)
    start = time.perf_counter()
    report = {'status': 'initializing', 'configuration': json_value(vars(a)),
        'full_training_experiment': a.smoke_rows == 0, 'outer_Test_calibration_sealed_accessed': False,
        'experiment': 'Third explicit modeling experiment: original labels; add aggregate TRAIN-background prior features only.',
        'baseline_r1_physics_inner_rmse': BASELINE_INNER_RMSE,
        'baseline_booster_header': {'num_iterations': 800, 'learning_rate': .035, 'num_leaves': 31,
            'min_data_in_leaf': 50, 'lambda_l2': 0, 'max_bin': 127, 'seed': 42, 'num_threads': 4,
            'feature_fraction': .85, 'bagging_fraction': .85, 'bagging_freq': 1},
        'selection_caveat': 'Repeated inner-development evidence; not an unbiased outer or official score.'}
    def save():
        report['seconds'] = time.perf_counter() - start
        report['dataset_files_opened'] = sorted(opened)
        (output / 'manifest.json').write_text(json.dumps(json_value(report), indent=2, allow_nan=False))
    save()
    try:
        gate_path = directory / 'readiness_manifest.json'
        gate = json.loads(gate_path.read_text())
        if gate.get('ready_to_train') is not True or not gate.get('cv_protocol') or 'physics' not in gate.get('version', ''):
            raise ValueError('Verified original nested physics package required')
        for key in ['source_checks_passed', 'reference_group_and_duplicate_isolation_passed',
                'background_week_and_purge_isolation_passed', 'sealed_holdout_locked', 'nested_early_stopping']:
            if gate.get(key) is not True:
                raise ValueError('Parent isolation gate failed: ' + key)
        for key in ['challenge_test_accessed', 'existing_sealed_NOAA_accessed', 'sealed_holdout_materialized',
                'frozen_v1_data_modified', 'sealed_arrays_accessed']:
            if gate.get(key) is not False:
                raise ValueError('Parent forbidden-access provenance: ' + key)
        files = gate['file_sha256']
        package_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        if package_hash != gate['data_package_hash']:
            raise ValueError('Data package hash differs')
        inputs = {name: files[name] for name in ['train.npz', 'validation.npz', 'validation_recipes.json']}
        for name, expected in inputs.items():
            if sha(directory / name) != expected:
                raise ValueError('Input SHA changed: ' + name)
        if sha(directory / 'sources/manifest.json') != gate['source_version_manifest_sha256']:
            raise ValueError('Source version metadata changed')
        readiness_sha = sha(gate_path)
        core_sha = {p.name: sha(p) for p in (project / 'pipeline').glob('*.py')}
        helper_path = Path(sys.modules[fit_background_prior.__module__].__file__).resolve()
        helper_sha, trainer_sha = sha(helper_path), sha(__file__)
        with np.load(directory / 'train.npz', allow_pickle=False) as z:
            train = {k: z[k] for k in ['db', 'cell_share', 'share']}
        with np.load(directory / 'validation.npz', allow_pickle=False) as z:
            inner = {k: z[k] for k in ['db', 'share']}
        recipes = json.loads((directory / 'validation_recipes.json').read_text())
        routes = np.asarray([route(item) for batch in recipes['batches'] for item in batch['items']])
        if len(routes) != len(inner['share']):
            raise ValueError('INNER recipe/target rows do not align')
        if a.smoke_rows:
            train = {k: v[:a.smoke_rows].copy() for k, v in train.items()}
            inner = {k: v[:max(8, a.smoke_rows // 2)].copy() for k, v in inner.items()}
            routes = routes[:len(inner['share'])]
        report.update(status='fitting training-only aggregate background prior',
            data_package_hash=package_hash, readiness_sha256=readiness_sha, input_sha256=inputs,
            source_version_manifest_sha256=gate['source_version_manifest_sha256'],
            core_sha256=core_sha, helper_sha256=helper_sha, trainer_sha256=trainer_sha,
            train_rows=len(train['share']), inner_rows=len(inner['share']),
            route_counts={str(r): int((routes == r).sum()) for r in np.unique(routes)},
            prior_fit_split='train only', query_transform_arguments='db and aggregate prior; no query labels or cell_share',
            added_source_groups=0, target_augmentation=False)
        save()
        prior = fit_background_prior(train['db'], train['cell_share'], clusters=a.clusters, seed=a.seed)
        save_aggregate_prior(prior, output)
        report['background_prior'] = {key: prior[key] for key in ['version', 'training_rows',
            'requested_clusters', 'actual_clusters', 'minimum_cluster_members', 'cluster_counts',
            'individual_training_examples_saved', 'query_requires_labels_or_source_tags']}
        prior_names = prior['feature_names']
        if not isinstance(prior_names, list) or len(prior_names) != 54 or len(set(prior_names)) != 54:
            raise ValueError('The specified aggregate prior must provide 54 named features')
        def transform(db):
            parts = []
            for begin in range(0, len(db), a.feature_batch):
                chunk = db[begin:begin+a.feature_batch]
                fixed = fixed_features(chunk, version=3); physics = physics_features(chunk)
                if fixed.shape[1] + physics.shape[1] != 1303:
                    raise ValueError('Baseline feature count changed')
                extra = np.asarray(transform_background_prior(chunk, prior), dtype='float32')
                if extra.shape != (len(chunk), len(prior_names)) or not np.isfinite(extra).all():
                    raise ValueError('Background-prior features are nonfinite or misaligned')
                parts.append(np.concatenate([fixed, physics, extra], axis=1))
            return np.concatenate(parts)
        X = transform(train['db']); V = transform(inner['db'])
        names = [f'fixed_v3_{i}' for i in range(1258)] + [f'physics_{i}' for i in range(45)] + prior_names
        (output / 'feature_names.json').write_text(json.dumps(names, indent=2))
        params = dict(objective='regression_l2', n_estimators=a.trees, learning_rate=.035,
            num_leaves=31, min_child_samples=50, colsample_bytree=.85, subsample=.85,
            subsample_freq=1, max_bin=127, n_jobs=a.threads, random_state=a.seed, verbosity=-1)
        report.update(status='fitting fixed-hyperparameter LightGBM with prior features',
            baseline_feature_count=1303, added_prior_feature_count=X.shape[1]-1303,
            feature_count=X.shape[1], model_parameters=params,
            early_stopping_rounds=40, hyperparameter_difference_from_original_baseline='Tree budget 800 to 1000 as explicitly requested; background features added; other parameters unchanged.')
        save()
        model = LGBMRegressor(**params)
        model.fit(X, train['share'], eval_set=[(V, inner['share'])],
            feature_name=names, callbacks=[early_stopping(40, verbose=False), log_evaluation(100)])
        prediction = np.clip(model.predict(V), 0, 1)
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite INNER prediction')
        model.booster_.save_model(str(output / 'regressor.txt'))
        (output / 'feature_importance.json').write_text(json.dumps({
            'names': names, 'gain': model.booster_.feature_importance(importance_type='gain').tolist(),
            'split': model.booster_.feature_importance(importance_type='split').tolist()}, indent=2))
        np.savez_compressed(output / 'inner_predictions.npz', share_true=inner['share'], share=prediction, route=routes)
        report.update(status='completed isolated CPU background-prior pilot; INNER evidence only',
            best_iteration=int(model.best_iteration_), inner_metrics=evaluate_regression(inner['share'], prediction, db=inner['db']),
            inner_rmse=rmse(inner['share'], prediction),
            rmse_difference_from_frozen_baseline=rmse(inner['share'], prediction)-BASELINE_INNER_RMSE,
            by_route={str(r): {'n': int((routes == r).sum()), 'rmse': rmse(inner['share'][routes == r], prediction[routes == r]),
                'mean_prediction': float(prediction[routes == r].mean()), 'mean_truth': float(inner['share'][routes == r].mean())} for r in np.unique(routes)})
        for name, expected in inputs.items():
            if sha(directory / name) != expected:
                raise ValueError('Frozen data changed during pilot')
        if sha(gate_path) != readiness_sha or sha(directory / 'sources/manifest.json') != gate['source_version_manifest_sha256']:
            raise ValueError('Frozen readiness/source metadata changed')
        if any(sha(project / 'pipeline' / name) != expected for name, expected in core_sha.items()):
            raise ValueError('Frozen pipeline code changed')
        if sha(helper_path) != helper_sha or sha(__file__) != trainer_sha:
            raise ValueError('Pilot helper code changed during execution')
        save()
        print(json.dumps({k: report[k] for k in ['status', 'inner_rmse', 'best_iteration',
            'rmse_difference_from_frozen_baseline', 'by_route', 'seconds', 'dataset_files_opened']}, indent=2), flush=True)
    except BaseException as exc:
        report.update(status='failed isolated pilot; preserve logs', error={'type': type(exc).__name__, 'message': str(exc)})
        save()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--clusters', type=int, default=32)
    p.add_argument('--trees', type=int, default=1000)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--feature-batch', type=int, default=256)
    p.add_argument('--smoke-rows', type=int, default=0, help='Explicit incomplete development smoke only')
    main(p.parse_args())
