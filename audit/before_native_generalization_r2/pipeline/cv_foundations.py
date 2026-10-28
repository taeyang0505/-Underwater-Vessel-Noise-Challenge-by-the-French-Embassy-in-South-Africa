"""Two-phase, training-context-only foundation candidates for nested source CV.

Inner prediction finishes before the ensemble policy is frozen. Outer rebuilds
that training context, replays two inner predictions, then reads outer arrays.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
import torch
from pipeline.mixtures import foundation_features
from pipeline.metrics import evaluate_regression, quality_flags_from_metadata

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'source_group_nested5_week24h_v1'
CHECKPOINTS = {
    'tabicl': ('tabicl-regressor-v2-20260212.ckpt', '0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a'),
    'kumo_large': ('kumo_large_regressor.pt', '745e0e5d97e4256d8a24cb61c2eecbd1128da6c83938c196b7589ea6f62e7421'),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''): h.update(block)
    return h.hexdigest()


def verify_cv_package(directory):
    """Open only train/inner values; require explicit nested-CV provenance."""
    from pipeline.train import load_prebuilt_data
    directory = Path(directory).resolve()
    gate_path = directory / 'readiness_manifest.json'
    if gate_path.is_symlink() or not gate_path.is_file(): raise ValueError('A regular CV readiness manifest is required')
    initial_sha = sha256(gate_path)
    gate = json.loads(gate_path.read_text())
    if gate.get('version') not in ['quality_v2_nested_group_cv_prebuilt', 'quality_v2_physics_group_cv_prebuilt'] or gate.get('protocol') != PROTOCOL or gate.get('cv_protocol') != PROTOCOL:
        raise ValueError('A nested source-CV package with the expected protocol is required')
    for name in ['ready_to_train', 'nested_early_stopping', 'requires_fresh_task_fit', 'source_checks_passed',
                 'reference_group_and_duplicate_isolation_passed', 'background_week_and_purge_isolation_passed', 'sealed_holdout_locked']:
        if gate.get(name) is not True: raise ValueError('Required explicit CV gate failed: ' + name)
    for name in ['challenge_test_accessed', 'sealed_arrays_accessed', 'sealed_holdout_materialized',
                 'model_evaluated_on_sealed', 'existing_sealed_NOAA_accessed', 'frozen_v1_data_modified', 'model_training_executed']:
        if gate.get(name) is not False: raise ValueError('Forbidden or unspecified CV provenance: ' + name)
    if gate.get('validation_role') != 'inner_early_stopping' or gate.get('outer_validation_role') != 'outer_oof_reporting_only':
        raise ValueError('Independent inner and outer roles are required')
    if type(gate.get('fold')) is not int or not 0 <= gate['fold'] < 5:
        raise ValueError('A valid five-fold index is required')
    if not isinstance(gate.get('source_hash'), str) or not isinstance(gate.get('parent_cv_source_hash'), str):
        raise ValueError('Missing source-collection binding')
    gap = gate.get('minimum_cross_partition_support_gap_hours')
    if not isinstance(gap, (float, int)) or not np.isfinite(gap) or gap < 24:
        raise ValueError('At least 24 hours of cross-partition support purge is required')
    if any((directory / name).exists() for name in ['calibration.npz', 'domain_holdout.npz', 'sealed_holdout.npz']):
        raise ValueError('Reserved non-CV partitions must remain outside this package')
    data, checked_gate, verification = load_prebuilt_data(directory, cv_mode=True)
    if checked_gate != gate or verification['readiness_manifest_sha256'] != initial_sha:
        raise ValueError('CV readiness changed during gate verification')
    return data, gate, verification


def verify_file(directory, gate, name):
    """Verify the exact file before any consumer opens its values."""
    if Path(name).name != name: raise ValueError('Unsafe CV file name')
    path = Path(directory) / name
    expected = gate['file_sha256'].get(name)
    if not expected or path.is_symlink() or not path.is_file() or sha256(path) != expected:
        raise ValueError('CV file missing, linked or changed: ' + name)
    return path


def binding(gate, verification):
    return {'protocol': PROTOCOL, 'fold': gate['fold'], 'data_package_hash': gate['data_package_hash'],
            'source_hash': gate['source_hash'], 'parent_cv_source_hash': gate['parent_cv_source_hash'],
            'readiness_manifest_sha256': verification['readiness_manifest_sha256']}


def frozen_policy(path, gate, required_kinds, seed=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('An existing regular frozen policy file is required')
    checksum = sha256(path)
    policy = json.loads(path.read_text())
    for key, expected in [('data_package_hash', gate['data_package_hash']), ('source_hash', gate['source_hash']),
                          ('fold', gate['fold']), ('protocol', PROTOCOL)]:
        if policy.get(key) != expected: raise ValueError('Frozen policy belongs to a different CV package: ' + key)
    if policy.get('fit_split') != 'inner validation only' or policy.get('outer_used_for_fit') is not False:
        raise ValueError('Policy must be fitted exclusively on inner validation')
    if policy.get('affine') != [1., 0.]: raise ValueError('Outer-fitted affine calibration is forbidden')
    models = policy.get('models', [])
    if not isinstance(models, list) or len(models) != len(set(models)) or not set(required_kinds).issubset(models):
        raise ValueError('Frozen policy does not contain the requested learner(s)')
    weights = np.asarray(policy.get('weights', []), dtype=float)
    if weights.shape != (len(models),) or not len(weights) or not np.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any() or not np.isclose(weights.sum(), 1., atol=1e-8, rtol=0):
        raise ValueError('Frozen policy requires bounded sum-to-one weights')
    seeds = policy.get('seeds', [])
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)) or (seed is not None and seed not in seeds):
        raise ValueError('Requested seed is not part of the frozen policy')
    if sha256(path) != checksum: raise ValueError('Frozen policy changed while opening it')
    return policy, checksum


def load_outer(directory, gate):
    path = verify_file(directory, gate, 'outer_validation.npz')
    meta_path = verify_file(directory, gate, 'validation_metadata.parquet')
    with np.load(path, allow_pickle=False) as z:
        if not {'db', 'share', 'presence', 'cell_share'}.issubset(z.files): raise ValueError('Outer arrays lack required fields')
        data = {name: z[name] for name in ['db', 'share', 'presence', 'cell_share']}
    n = len(data['share'])
    if not n or data['db'].shape != (n, 15, 25) or data['cell_share'].shape != (n, 15, 25) or data['presence'].shape != (n,):
        raise ValueError('Invalid outer array dimensions')
    if any(v.dtype != np.float32 for v in data.values()) or np.isinf(data['db']).any(): raise ValueError('Invalid outer dtype/values')
    if not np.isfinite(data['db']).any((1, 2)).all() or not all(np.isfinite(data[k]).all() for k in ['share', 'presence', 'cell_share']):
        raise ValueError('Invalid outer observations or labels')
    if not np.array_equal(data['presence'], (data['share'] > 0).astype('float32')): raise ValueError('Outer presence/share mismatch')
    meta = pd.read_parquet(meta_path)
    if len(meta) != n or not {'sample_id', 'fold', 'is_outer_oof', 'background_group', 'reference_group'}.issubset(meta.columns):
        raise ValueError('Outer sample metadata must align with the frozen arrays')
    if not meta.sample_id.is_unique or not meta.fold.eq(gate['fold']).all() or not meta.is_outer_oof.eq(True).all():
        raise ValueError('Outer metadata belongs to a different role or fold')
    return data, meta


def code_identity():
    return {name: sha256(ROOT / 'pipeline' / name) for name in
            ['cv_foundations.py', 'mixtures.py', 'metrics.py', 'train.py', 'models.py']}


def unchanged_inputs(directory, gate, verification, names):
    if sha256(Path(directory) / 'readiness_manifest.json') != verification['readiness_manifest_sha256']:
        raise ValueError('CV readiness changed during execution')
    for name in names: verify_file(directory, gate, name)


def resolve_precision(kind, requested='auto'):
    if requested not in ['auto', 'fp32', 'fp16']:
        raise ValueError('Unsupported foundation precision')
    if kind not in CHECKPOINTS:
        raise ValueError('Unsupported foundation kind')
    return 'fp32' if kind == 'kumo_large' and requested == 'auto' else requested


def make_predictor(a, X, y, checkpoint):
    """Only vendor-original weights and this training context are used."""
    precision = resolve_precision(a.kind, getattr(a, 'precision', 'auto'))
    torch.set_float32_matmul_precision('highest')
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
    if a.kind == 'tabicl':
        from tabicl import TabICLRegressor
        model = TabICLRegressor(n_estimators=a.members, batch_size=1, kv_cache='repr',
            norm_methods=['none', 'power'], model_path=checkpoint, allow_auto_download=False,
            device=a.device, use_amp=('auto' if precision == 'auto' else precision == 'fp16'), use_fa3=False, offload_mode='auto',
            n_jobs=a.threads, random_state=a.seed)
        model.fit(X, y)
        return lambda x: np.asarray(model.predict(np.asarray(x, dtype=np.float32))).reshape(-1)
    import sdm
    model = sdm.models.KumoTabular(task='regression', size='large', pretrained=False, device='cpu')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.models[sdm.Task.regression].load_state_dict(state, assign=True)
    del state
    model.to(a.device).eval()
    model.models[sdm.Task.regression].float()
    device_type = torch.device(a.device).type
    use_autocast = device_type == 'cuda' and precision == 'fp16'
    with torch.no_grad(), torch.autocast(device_type, dtype=torch.float16, enabled=use_autocast):
        model.fit(torch.from_numpy(X).to(a.device), torch.from_numpy(y[:, None]).to(a.device),
            num_estimators=a.members, estimator_batch_size=1,
            generator=torch.Generator(device=a.device).manual_seed(a.seed))
    def predict(x):
        with torch.no_grad(), torch.autocast(device_type, dtype=torch.float16, enabled=use_autocast):
            return model.predict(torch.from_numpy(np.asarray(x, dtype=np.float32)).to(a.device)).numerical.mean(-1).cpu().numpy().reshape(-1)
    return predict


def run(a):
    if a.output.exists(): raise FileExistsError('A foundation CV phase is immutable; choose a fresh output')
    if min(a.context, a.members, a.batch, a.threads) < 1 or a.kind not in CHECKPOINTS:
        raise ValueError('Invalid foundation kind or positive execution budget')
    if a.phase not in ['inner', 'outer']: raise ValueError('Unknown foundation phase')
    if a.phase == 'outer' and (a.inner_result is None or a.policy is None):
        raise ValueError('Outer phase requires --inner-result and --policy')
    if a.phase == 'inner' and (a.inner_result is not None or a.policy is not None):
        raise ValueError('Inner phase cannot consume outer/policy artifacts')
    data, gate, verification = verify_cv_package(a.data)
    if gate.get('version') == 'quality_v2_physics_group_cv_prebuilt' and a.feature_version != 3:
        raise ValueError('The physics data version requires feature-version 3')
    filename, expected_weights = CHECKPOINTS[a.kind]
    checkpoint = ROOT / 'models/pretrained' / filename
    if checkpoint.is_symlink() or sha256(checkpoint) != expected_weights:
        raise ValueError('Pinned foundation checkpoint checksum differs')
    ids = np.random.default_rng(a.seed).choice(len(data['train']['share']), min(a.context, len(data['train']['share'])), replace=False)
    identity = {**binding(gate, verification), 'kind': a.kind, 'seed': a.seed,
        'requested_context': a.context, 'context_rows': len(ids), 'context_indices': ids.tolist(),
        'context_sha256': gate['file_sha256']['train.npz'], 'context_source': 'train.npz only',
        'members': a.members, 'device': a.device, 'batch': a.batch, 'threads': a.threads,
        'checkpoint_sha256': expected_weights, 'code_sha256': code_identity(), 'feature_version': a.feature_version,
        'requested_precision': getattr(a, 'precision', 'auto'),
        'resolved_precision': resolve_precision(a.kind, getattr(a, 'precision', 'auto')),
        'float32_matmul_precision': 'highest', 'feature_dtype': 'float32'}
    policy_sha, saved_inner, inner_manifest_sha, saved_inner_sha = None, None, None, None
    if a.phase == 'outer':
        inner_manifest = a.inner_result / 'manifest.json'
        if inner_manifest.is_symlink() or not inner_manifest.is_file(): raise ValueError('Missing completed inner foundation manifest')
        inner = json.loads(inner_manifest.read_text())
        inner_manifest_sha = sha256(inner_manifest)
        if inner.get('status') != 'completed foundation CV phase' or inner.get('phase') != 'inner':
            raise ValueError('Outer phase requires a completed inner phase')
        if any(inner.get(k) != v for k, v in identity.items()): raise ValueError('Inner foundation context/configuration/code differs')
        if any(inner.get(k) is not False for k in ['test_accessed', 'calibration_accessed', 'sealed_accessed', 'outer_accessed', 'weight_training']):
            raise ValueError('Inner result has unsafe or unspecified fitting/access provenance')
        saved_path = a.inner_result / 'validation_predictions.npz'
        if saved_path.is_symlink() or not saved_path.is_file() or sha256(saved_path) != inner.get('file_sha256', {}).get(saved_path.name):
            raise ValueError('Saved inner predictions changed')
        saved_inner_sha = sha256(saved_path)
        with np.load(saved_path, allow_pickle=False) as z: saved_inner = {k: z[k] for k in ['share_true', a.kind + '_share']}
        if not np.array_equal(saved_inner['share_true'], data['validation']['share']):
            raise ValueError('Saved inner predictions do not match this inner split')
        _, policy_sha = frozen_policy(a.policy, gate, [a.kind], a.seed)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    X, y = foundation_features(data['train']['db'][ids], version=a.feature_version), data['train']['share'][ids]
    start = time.perf_counter()
    predict = make_predictor(a, X, y, checkpoint)
    fit_seconds = time.perf_counter() - start
    replay_error = None
    if a.phase == 'outer':
        check_count = min(2, len(data['validation']['share']))
        replay = np.clip(predict(foundation_features(data['validation']['db'][:check_count], version=a.feature_version)), 0, 1)
        expected = saved_inner[a.kind + '_share'][:check_count]
        if replay.shape != expected.shape or not np.allclose(replay, expected, atol=3e-5, rtol=1e-5):
            raise ValueError('Fresh context cannot reproduce saved inner predictions')
        replay_error = float(np.max(np.abs(replay - expected)))
        if sha256(a.policy) != policy_sha: raise ValueError('Frozen policy changed before outer access')
        if sha256(a.inner_result / 'manifest.json') != inner_manifest_sha or sha256(a.inner_result / 'validation_predictions.npz') != saved_inner_sha:
            raise ValueError('Saved inner artifacts changed before outer access')
        query, meta = load_outer(a.data, gate)
        split = 'outer_validation'
    else:
        query, meta, split = data['validation'], None, 'validation'
        inner_metadata = a.data / 'inner_validation_metadata.parquet'
        if inner_metadata.exists():
            verify_file(a.data, gate, inner_metadata.name)
            meta = pd.read_parquet(inner_metadata)
            if len(meta) != len(query['share']) or not meta.sample_id.is_unique:
                raise ValueError('Inner diagnostic metadata do not align with validation')
    features = foundation_features(query['db'], version=a.feature_version)
    pred = np.concatenate([predict(features[i:i + a.batch]) for i in range(0, len(features), a.batch)])
    one_count = min(2, len(features))
    one = predict(features[:one_count])
    if not np.allclose(pred[:one_count], one, atol=3e-5, rtol=1e-5):
        raise ValueError('Query dependence detected: foundation CV candidate rejected')
    pred = np.clip(pred, 0, 1)
    if pred.shape != query['share'].shape or not np.isfinite(pred).all(): raise ValueError('Invalid foundation predictions')
    names = ['train.npz', 'validation.npz']
    if a.phase == 'inner' and meta is not None:
        names.append('inner_validation_metadata.parquet')
    if a.phase == 'outer':
        names += ['outer_validation.npz', 'validation_metadata.parquet']
        if sha256(a.policy) != policy_sha: raise ValueError('Frozen policy changed during outer prediction')
        if sha256(a.inner_result / 'manifest.json') != inner_manifest_sha or sha256(a.inner_result / 'validation_predictions.npz') != saved_inner_sha:
            raise ValueError('Saved inner artifacts changed during outer prediction')
    unchanged_inputs(a.data, gate, verification, names)
    if code_identity() != identity['code_sha256'] or sha256(checkpoint) != expected_weights:
        raise ValueError('Foundation code or original checkpoint changed during execution')
    out = {'share_true': query['share'], 'presence_true': query['presence'], a.kind + '_share': pred, a.kind + '_score': pred}
    if meta is not None: out['sample_id'] = meta.sample_id.to_numpy(dtype=str)
    a.output.mkdir(parents=True, exist_ok=False)
    output_file = a.output / (split + '_predictions.npz')
    np.savez_compressed(output_file, **out)
    info = {**identity, 'status': 'completed foundation CV phase', 'phase': a.phase, 'fit_seconds': fit_seconds,
        'weight_training': False, 'epochs': None, 'optimizer': None,
        'policy_sha256_before_outer': policy_sha, 'inner_replay_max_abs_error': replay_error,
        'inner_result_manifest_sha256': inner_manifest_sha, 'inner_prediction_sha256': saved_inner_sha,
        'file_sha256': {output_file.name: sha256(output_file)},
        'results': {split: evaluate_regression(query['share'], pred, query['presence'], pred, query['db'],
            quality_flags=quality_flags_from_metadata(meta) if meta is not None else None)},
        'test_accessed': False, 'calibration_accessed': False, 'sealed_accessed': False,
        'outer_accessed': a.phase == 'outer', 'per_query_independence_checked': True,
        'score_semantics': 'Bounded share proxy, not a calibrated presence probability'}
    (a.output / 'manifest.json').write_text(json.dumps(info, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in info.items() if k not in ['context_indices', 'results', 'code_sha256']}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', choices=sorted(CHECKPOINTS), required=True)
    p.add_argument('--feature-version', type=int, choices=[1,3], default=1)
    p.add_argument('--precision', choices=['auto', 'fp32', 'fp16'], default='auto')
    p.add_argument('--phase', choices=['inner', 'outer'], default='inner')
    p.add_argument('--inner-result', type=Path)
    p.add_argument('--policy', type=Path)
    p.add_argument('--context', type=int, default=4096)
    p.add_argument('--members', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch', type=int, default=256)
    run(p.parse_args())
