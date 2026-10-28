"""Explicit nested Group CV, inner-fitted ensemble and untouched outer OOF.

The default action is a read-only preflight. --execute starts all configured
folds. No challenge inference, official submission or sealed evaluation occurs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pipeline.metrics import evaluate_regression, quality_flags_from_metadata

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def project_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def inner_weights(predictions, truth):
    """One constrained fit, never an outer-target fit or automatic model search."""
    k = predictions.shape[1]
    equal = np.ones(k) / k
    fit = minimize(lambda w: np.mean((predictions @ w - truth) ** 2) + 1e-4 * np.sum((w - equal) ** 2),
        equal, method='SLSQP', bounds=[(0., 1.)] * k,
        constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
        options={'ftol': 1e-12, 'maxiter': 300})
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError('Inner ensemble fit failed: ' + str(fit.message))
    return fit.x


def training_command(config, data, output, seed, device):
    t = config['training']
    cmd = [sys.executable, '-u', '-m', 'pipeline.train', '--data-dir', str(data),
        '--run-dir', str(output), '--tag', 'nested_cv', '--cv-mode', '--seed', str(seed),
        '--device', device, '--models', *config['models']]
    mapping = {'epochs': '--epochs', 'patience': '--patience', 'optimizer': '--optimizer',
        'learning_rate': '--lr', 'weight_decay': '--weight-decay', 'dropout': '--dropout',
        'scheduler': '--scheduler', 'batch': '--batch', 'width': '--width', 'k': '--k',
        'trees': '--trees', 'threads': '--threads', 'contrastive_weight': '--contrastive-weight',
        'contrastive_temperature': '--contrastive-temperature'}
    for key, flag in mapping.items():
        cmd += [flag, str(t[key])]
    cmd += ['--feature-version', str(t.get('feature_version', 1))]
    for key, flag in {'domain_augmentation':'--domain-augmentation',
            'augmentation_probability':'--augmentation-probability',
            'augmentation_strength_db':'--augmentation-strength-db',
            'cnn_prior_strength':'--cnn-prior-strength','cnn_prior_clip':'--cnn-prior-clip',
            'cnn_prior_residual_weight':'--cnn-prior-residual-weight','cnn_prior_clusters':'--cnn-prior-clusters',
            'cnn_head_mode':'--cnn-head-mode','cell_loss_weight':'--cell-loss-weight',
            'presence_loss_weight':'--presence-loss-weight'}.items():
        if t.get(key) is not None:
            cmd += [flag, str(t[key])]
    if type(t.get('cnn_background_prior',False)) is not bool:
        raise ValueError('cnn_background_prior must be an explicit configuration boolean')
    if t.get('cnn_background_prior',False):cmd += ['--cnn-background-prior']
    return cmd


def preflight(config, cv_dir):
    from pipeline.train import load_prebuilt_data, build_parser, validate_arguments
    cv_dir = Path(cv_dir).resolve()
    cv = json.loads((cv_dir / 'cv_manifest.json').read_text())
    if not cv.get('ready_to_train') or cv['n_splits'] != 5 or len(cv['folds']) != 5:
        raise ValueError('A completed 5-fold source package is required')
    if cv['protocol'] != config['protocol'] or cv.get('challenge_test_accessed') is not False:
        raise ValueError('Unexpected CV protocol or forbidden input provenance')
    if cv.get('nested_early_stopping') is not True or cv.get('outer_validation_used_for_early_stopping') is not False:
        raise ValueError('Separate inner validation and outer OOF are required')
    if config['protocol']=='source_group_nested5_vtuad_passage_r4':
        t=config['training'];s=config.get('ensemble_safety_r4',{})
        if config['models']!=['lgbm_retrieval','cnn','fdy','itransformer'] or config['foundation_models']:
            raise ValueError('R4 requires the four predeclared task models and no foundation candidates')
        if not s.get('enabled') or s.get('outer_gate_changes_predictions') is not False:
            raise ValueError('R4 requires an INNER safety policy and reporting-only OUTER gate')
        if (t.get('cnn_head_mode')!='direct_mask' or t.get('cnn_prior_residual_weight')!=0
                or t.get('cell_loss_weight')!=.25 or t.get('presence_loss_weight')!=.05):
            raise ValueError('R4 direct-mask and loss policy differ from the requested settings')
    if len(set(config['seeds'])) != len(config['seeds']) or not config['seeds']:
        raise ValueError('Seeds must be nonempty and unique')
    all_ids, checks = [], []
    for fold in cv['folds']:
        data_dir = project_path(fold['data_dir'])
        if cv_dir not in data_dir.parents:
            raise ValueError('A fold must live inside the frozen CV package')
        data, gate, verified = load_prebuilt_data(data_dir, cv_mode=True)
        del data
        if gate['data_package_hash'] != fold['data_package_hash'] or gate['source_hash'] != fold['source_hash']:
            raise ValueError('CV index and fold readiness disagree')
        if gate.get('version') in ['quality_v2_physics_group_cv_prebuilt','quality_v4_vtuad_passage_group_cv_prebuilt'] and config['training'].get('feature_version') != 3:
            raise ValueError('Physics V3 data require the six-channel feature-version 3 pipeline')
        for name, value in gate['file_sha256'].items():
            if Path(name).name != name or digest(data_dir / name) != value:
                raise ValueError('Fold file changed: ' + name)
        if gate.get('parent_cv_source_hash') != cv['source_hash']:
            raise ValueError('Unexpected parent source collection')
        outer = np.load(data_dir / fold['outer_validation_file'])
        meta = pd.read_parquet(data_dir / fold['validation_metadata_file'])
        if len(meta) != len(outer['share']) or len(meta) != fold['validation_examples']:
            raise ValueError('OOF metadata/count mismatch')
        if not {'sample_id', 'background_group', 'reference_group'}.issubset(meta.columns):
            raise ValueError('Missing source group metadata')
        all_ids.extend(meta.sample_id.astype(str).tolist())
        for seed in config['seeds']:
            args = build_parser().parse_args(training_command(config, data_dir, ROOT / 'models' / 'preflight_only', seed, 'cpu')[4:])
            validate_arguments(args)
        checks.append({'fold': fold['fold'], 'data_directory': str(data_dir),
            'readiness_sha256': digest(data_dir / 'readiness_manifest.json'),
            'data_package_hash': gate['data_package_hash'], 'outer_examples': len(meta),
            'training_verification': verified})
        outer.close()
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('An OOF example appears in more than one outer fold')
    # Original vendor checkpoints are not implicitly downloaded by the launcher.
    from pipeline.cv_foundations import sha256, resolve_precision
    checkpoints = {'tabicl': ('tabicl-regressor-v2-20260212.ckpt', '0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a'),
        'kumo_large': ('kumo_large_regressor.pt', '745e0e5d97e4256d8a24cb61c2eecbd1128da6c83938c196b7589ea6f62e7421')}
    for kind in config['foundation_models']:
        resolve_precision(kind, config['foundation_models'][kind].get('precision', 'auto'))
        filename, expected = checkpoints[kind]
        if sha256(ROOT / 'models/pretrained' / filename) != expected:
            raise ValueError('Foundation checkpoint checksum mismatch: ' + kind)
    return cv, {'ready': True, 'full_training_executed': False,
        'n_splits': 5, 'oof_examples': len(all_ids), 'checks': checks,
        'cv_manifest_sha256': digest(cv_dir / 'cv_manifest.json')}


def call(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    with log_path.open('w') as f:
        subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def completed_task_run(path):
    p = path / 'run_manifest.json'
    return p.exists() and json.loads(p.read_text()).get('status', '').startswith('completed ')


def immutable_step(target, label, outputs, action):
    """Reuse only complete artifacts recorded by this exact execution."""
    ledger_path = target / 'artifact_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    if label in ledger:
        for name, expected in ledger[label].items():
            if digest(target / name) != expected:
                raise ValueError('Recorded CV artifact changed: ' + name)
        return
    if any(p.exists() for p in outputs):
        raise ValueError('Unrecorded/partial artifacts cannot be reused: ' + label)
    action()
    files = []
    for path in outputs:
        if not path.exists():
            raise RuntimeError('Step did not create its required output: ' + str(path))
        files.extend([path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()])
    if not files:
        raise RuntimeError('Empty CV step: ' + label)
    ledger[label] = {str(p.relative_to(target)): digest(p) for p in files}
    temporary = ledger_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(ledger, indent=2))
    temporary.replace(ledger_path)


def execute(config, cv, cv_dir, output, device, resume):
    if output.exists() and not resume:
        raise FileExistsError('Use a fresh output directory, or --resume for the same frozen configuration')
    output.mkdir(parents=True, exist_ok=True)
    identity = {'configuration': config, 'device': device,
        'cv_manifest_sha256': digest(cv_dir / 'cv_manifest.json'),
        'fold_readiness_sha256': {str(f['fold']): digest(project_path(f['data_dir']) / 'readiness_manifest.json') for f in cv['folds']},
        'code_sha256': {p.name: digest(p) for p in (ROOT / 'pipeline').glob('*.py')}}
    identity_file = output / 'execution_identity.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError('Resume configuration/data differs from the original CV run')
    identity_file.write_text(json.dumps(identity, indent=2))
    kinds = config['models'] + list(config['foundation_models'])
    all_truth, all_presence, all_predictions, all_scores, all_meta, fold_reports = [], [], [], [], [], []
    for fold in cv['folds']:
        number = fold['fold']
        data = project_path(fold['data_dir'])
        target = output / f'fold_{number:02d}'
        target.mkdir(exist_ok=True)
        inner_by_seed, outer_by_seed = [], []
        # First finish all training and inner predictions. Outer evaluation
        # starts only after the ensemble policy has been written and hashed.
        for seed in config['seeds']:
            run = target / f'seed_{seed}'
            command = training_command(config, data, run, seed, device)
            immutable_step(target, f'task_inner_seed_{seed}', [run],
                lambda: call(command, target / f'seed_{seed}_training.log'))
            if not completed_task_run(run):
                raise RuntimeError('Task learner did not finish successfully')
            task_manifest = json.loads((run / 'run_manifest.json').read_text())
            if task_manifest['data_package']['data_package_hash'] != fold['data_package_hash']:
                raise ValueError('Task learner belongs to a different source package')
            from pipeline.train import build_parser
            expected_args = json.loads(json.dumps(vars(build_parser().parse_args(command[4:])), default=str))
            if task_manifest['arguments'] != expected_args:
                raise ValueError('Task learner arguments/seed differ from this execution')
            inside = dict(np.load(run / 'validation_predictions.npz'))
            for kind, settings in config['foundation_models'].items():
                fdir = target / f'{kind}_seed_{seed}_inner'
                immutable_step(target, f'{kind}_inner_seed_{seed}', [fdir], lambda: call(
                    [sys.executable, '-m', 'pipeline.cv_foundations', '--phase', 'inner', '--data', str(data), '--output', str(fdir),
                        '--kind', kind, '--context', str(settings['context']), '--members', str(settings['members']),
                        '--device', device, '--seed', str(seed), '--threads', str(config['training']['threads']),
                        '--feature-version', str(config['training'].get('feature_version', 1)),
                        '--precision', settings.get('precision', 'auto')],
                        target / f'{kind}_seed_{seed}_inner.log'))
                with np.load(fdir / 'validation_predictions.npz') as z:
                    if not np.array_equal(z['share_true'], inside['share_true']):
                        raise ValueError('Foundation and task inner predictions are not aligned')
                    inside[kind + '_share'], inside[kind + '_score'] = z[kind + '_share'], z[kind + '_score']
            inner_by_seed.append(inside)
        inner = np.stack([np.mean([s[k + '_share'] for s in inner_by_seed], 0) for k in kinds], 1)
        safety_settings=config.get('ensemble_safety_r4',{})
        safety_enabled=safety_settings.get('enabled',False)
        safety_detail=None
        if safety_enabled:
            from pipeline.ensemble_safety_r4 import select_inner_policy
            inner_meta=pd.read_parquet(data/'inner_validation_metadata.parquet')
            if len(inner_meta)!=len(inner):raise ValueError('INNER metadata count differs from predictions')
            weights,safety_detail=select_inner_policy(inner,inner_by_seed[0]['share_true'],kinds,
                inner_meta['reference_group'].astype(str).to_numpy(),safety_settings)
            safety_detail.update({'inner_metadata_sha256':digest(data/'inner_validation_metadata.parquet'),
                'inner_prediction_sha256':{str(seed):digest(target/f'seed_{seed}'/'validation_predictions.npz') for seed in config['seeds']},
                'selection_code_sha256':digest(ROOT/'pipeline/ensemble_safety_r4.py')})
        else:weights = inner_weights(inner, inner_by_seed[0]['share_true'])
        policy = {'models': kinds, 'weights': weights.tolist(), 'affine': [1., 0.],
            'fit_split': 'inner validation only', 'outer_used_for_fit': False,
            'seeds': config['seeds'], 'data_package_hash': fold['data_package_hash'],
            'source_hash': fold['source_hash'], 'fold': number, 'protocol': cv['protocol']}
        if safety_enabled:policy['inner_safety_gate']=safety_detail
        policy_file = target / 'ensemble_policy.json'
        immutable_step(target, 'frozen_inner_policy', [policy_file],
            lambda: policy_file.write_text(json.dumps(policy, indent=2)))
        if json.loads(policy_file.read_text()) != policy:
            raise ValueError('Recomputed inner policy differs from the frozen policy')
        if safety_enabled:
            safety_file=target/'ensemble_safety.json'
            immutable_step(target,'frozen_inner_safety_detail',[safety_file],
                lambda:safety_file.write_text(json.dumps(safety_detail,indent=2,allow_nan=False)))
        for seed in config['seeds']:
            run = target / f'seed_{seed}'
            outer_file = run / 'outer_validation_predictions.npz'
            immutable_step(target, f'task_outer_seed_{seed}', [outer_file, outer_file.with_suffix('.metrics.json')],
                lambda: call([sys.executable, '-m', 'pipeline.evaluate_cv_outer', '--run', str(run), '--data', str(data),
                    '--output', str(outer_file), '--policy', str(policy_file), '--device', device,
                    '--threads', str(config['training']['threads'])], target / f'seed_{seed}_outer.log'))
            outside = dict(np.load(outer_file))
            for kind, settings in config['foundation_models'].items():
                fdir = target / f'{kind}_seed_{seed}_outer'
                inner_dir = target / f'{kind}_seed_{seed}_inner'
                immutable_step(target, f'{kind}_outer_seed_{seed}', [fdir], lambda: call(
                    [sys.executable, '-m', 'pipeline.cv_foundations', '--phase', 'outer', '--data', str(data),
                        '--output', str(fdir), '--inner-result', str(inner_dir), '--policy', str(policy_file),
                        '--kind', kind, '--context', str(settings['context']), '--members', str(settings['members']),
                        '--device', device, '--seed', str(seed), '--threads', str(config['training']['threads']),
                        '--feature-version', str(config['training'].get('feature_version', 1)),
                        '--precision', settings.get('precision', 'auto')],
                        target / f'{kind}_seed_{seed}_outer.log'))
                with np.load(fdir / 'outer_validation_predictions.npz') as z:
                    if not np.array_equal(z['share_true'], outside['share_true']):
                        raise ValueError('Foundation and task outer predictions are not aligned')
                    outside[kind + '_share'], outside[kind + '_score'] = z[kind + '_share'], z[kind + '_score']
            outer_by_seed.append(outside)
        # All seeds share the same frozen outer rows; average within model.
        outer = np.stack([np.mean([s[k + '_share'] for s in outer_by_seed], 0) for k in kinds], 1)
        diagnostic = np.stack([np.mean([s[k + '_score'] for s in outer_by_seed], 0) for k in kinds], 1)
        y = outer_by_seed[0]['share_true']
        z = outer_by_seed[0]['presence_true']
        meta = pd.read_parquet(data / fold['validation_metadata_file'])
        if not np.array_equal(outer_by_seed[0]['sample_id'].astype(str), meta.sample_id.to_numpy(dtype=str)):
            raise ValueError('OOF sample order changed')
        groups = {k: meta[k].astype(str).to_numpy() for k in ['background_group', 'reference_group']}
        quality = quality_flags_from_metadata(meta)
        report = {'fold': number, 'models': {k: evaluate_regression(y, outer[:, i], z, diagnostic[:, i], groups=groups, quality_flags=quality)
            for i, k in enumerate(kinds)}, 'ensemble': evaluate_regression(y, outer @ weights, z, diagnostic @ weights, groups=groups, quality_flags=quality)}
        if safety_enabled:
            anchor=np.array([0.,1/3,1/3,1/3])
            report['mask_mean']=evaluate_regression(y,outer@anchor,z,diagnostic@anchor,groups=groups,quality_flags=quality)
            report['inner_safety_gate']=safety_detail
            train_path=data/'train.npz'
            gate=json.loads((data/'readiness_manifest.json').read_text())
            if digest(train_path)!=gate['file_sha256']['train.npz']:raise ValueError('TRAIN changed before constant baseline reporting')
            with np.load(train_path,allow_pickle=False) as train_labels:
                baseline_mean=float(np.mean(train_labels['share'],dtype=np.float64))
                baseline_presence=float(np.mean(train_labels['presence'],dtype=np.float64))
            mean_prediction=np.full(len(y),baseline_mean);mean_score=np.full(len(y),baseline_presence)
            report['train_mean']=evaluate_regression(y,mean_prediction,z,mean_score,groups=groups,quality_flags=quality)
            report['train_mean_binding']={'value':baseline_mean,'presence_constant':baseline_presence,
                'fit_split':'TRAIN only','train_npz_sha256':gate['file_sha256']['train.npz']}
        (target / 'outer_metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        fold_reports.append(report)
        all_truth.append(y); all_presence.append(z)
        all_predictions.append(np.column_stack([outer,outer@weights,outer@anchor,mean_prediction]) if safety_enabled else np.column_stack([outer,outer@weights]))
        all_scores.append(np.column_stack([diagnostic,diagnostic@weights,diagnostic@anchor,mean_score]) if safety_enabled else np.column_stack([diagnostic,diagnostic@weights]));all_meta.append(meta)
        print(f'Completed CV fold {number + 1}/{cv["n_splits"]}', flush=True)
    y, z, P, S, meta = np.concatenate(all_truth), np.concatenate(all_presence), np.concatenate(all_predictions), np.concatenate(all_scores), pd.concat(all_meta, ignore_index=True)
    if not meta.sample_id.is_unique:
        raise ValueError('OOF sample IDs are not globally unique')
    groups = {k: meta[k].astype(str).to_numpy() for k in ['background_group', 'reference_group']}
    names = kinds + ['ensemble'] + (['mask_mean','train_mean'] if safety_enabled else [])
    fold_rmse = {k: np.asarray([f['models'][k]['rmse'] if k in kinds else f[k]['rmse'] for f in fold_reports]) for k in names}
    report = {'status': 'completed nested Group CV; synthetic source-generalization evidence, not official score',
        'folds': fold_reports, 'n_splits': cv['n_splits'], 'oof_examples': len(y), 'seeds': config['seeds'],
        'pooled_oof': {k: evaluate_regression(y, P[:, i], z, S[:, i], groups=groups,
            quality_flags=quality_flags_from_metadata(meta)) for i, k in enumerate(names)},
        'fold_rmse_mean_std': {k: {'mean': float(v.mean()), 'std': float(v.std(ddof=1))} for k, v in fold_rmse.items()},
        'outer_used_for_early_stopping': False, 'outer_used_for_ensemble_weights': False,
        'challenge_test_accessed': False, 'calibration_accessed': False, 'sealed_accessed': False,
        'limitations': ['Synthetic rows are correlated through few source components.',
            'Pooled OOF RMSE is not the arithmetic mean of fold RMSE.',
            'Comparing architectures on OOF makes it development evidence, not a permanently untouched final assessment.',
            'Reference independence is bounded by the selected source graph; augmentation does not create new recordings.' if safety_enabled else 'Independent native long references span only five connected source components.',
            'Neither convergence nor a global minimum nor leaderboard superiority is guaranteed.']}
    if safety_enabled:
        from pipeline.ensemble_safety_r4 import outer_safety_report
        report['ensemble_safety']=outer_safety_report(report['pooled_oof'])
        report['inner_safety_accepted_folds']=[f['fold'] for f in fold_reports if f['inner_safety_gate']['candidate_accepted']]
    (output / 'cv_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(output / 'oof_predictions.npz', sample_id=meta.sample_id.to_numpy(dtype=str), share_true=y,
        presence_true=z, **{k + '_share': P[:, i] for i, k in enumerate(names)}, **{k + '_score': S[:, i] for i, k in enumerate(names)})
    meta.to_parquet(output / 'oof_metadata.parquet', index=False)


def main(a):
    config = json.loads(a.config.read_text())
    if a.ablation:
        choice = next((e for e in config['explicit_ablation_candidates'] if e['name'] == a.ablation), None)
        if choice is None:
            raise ValueError('Unknown explicit ablation')
        config['training'].update({k: v for k, v in choice.items() if k != 'name'})
        if config['training']['contrastive_weight']:
            # Representations exist for the spectro-temporal neural branches.
            config['models'] = ['cnn', 'fdy', 'itransformer']
            config['foundation_models'] = {}
        config['output_dir'] += '_' + a.ablation
    if a.output:
        config['output_dir'] = str(a.output)
    cv_dir = project_path(config['cv_dir'])
    cv, report = preflight(config, cv_dir)
    if not a.execute:
        print(json.dumps(report, indent=2), flush=True)
        return
    if a.device.startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Preflight passed; full training has not started.')
    execute(config, cv, cv_dir, project_path(config['output_dir']), a.device, a.resume)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/group_cv_physics_training.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--ablation')
    main(p.parse_args())
