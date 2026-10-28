"""End-to-end checks on tiny generated numerical fixtures, no competition fit."""
from pathlib import Path
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from pipeline.train_group_cv import preflight, execute, digest, immutable_step
from verify_training_policy_fixture import fixture

ROOT = Path(__file__).resolve().parent
PROTOCOL = 'source_group_nested5_week24h_v1'


def command(args, log):
    r = subprocess.run([sys.executable, '-m', *args], cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True)
    log.write_text(r.stdout + '\n' + r.stderr)
    if r.returncode:
        raise AssertionError(log.read_text())


def main():
    checks = {}
    with tempfile.TemporaryDirectory(prefix='uvn-cv-runner-fixture-') as tmp:
        tmp = Path(tmp)
        cv_dir = tmp / 'cv'; cv_dir.mkdir()
        folds = []
        for f in range(5):
            d = cv_dir / f'fold_{f:02d}'
            g = fixture(d, cv=True)
            parts = {}
            for i, split in enumerate(['train', 'validation', 'outer_validation']):
                n = 24 if split == 'train' else 12
                rng = np.random.default_rng(1000 * f + i)
                db = rng.normal(90, 4, (n, 15, 25)).astype('float32'); db[::2, -2:] = np.nan
                y = np.resize(np.array([0., .2, .6, .9], dtype='float32'), n)
                q = np.where(np.isfinite(db), y[:, None, None], 0).astype('float32')
                np.savez_compressed(d / (split + '.npz'), db=db, share=y, presence=(y > 0).astype('float32'), cell_share=q)
                parts[split] = {'examples': n, 'positive_draw_became_zero_count': 0, 'frozen_generation_replayed': True}
            meta = pd.DataFrame({'sample_id': [f'fixture-{f}-{i}' for i in range(12)],
                'fold': f, 'is_outer_oof': True, 'validation_id': np.arange(12),
                'background_group': f'fixture-background-{f}', 'reference_group': f'fixture-reference-{f}'})
            meta.to_parquet(d / 'validation_metadata.parquet', index=False)
            files = {p.name: digest(p) for p in d.iterdir() if p.suffix in ['.npz', '.parquet']}
            g.update(version='quality_v2_nested_group_cv_prebuilt', protocol=PROTOCOL, cv_protocol=PROTOCOL,
                fold=f, parent_cv_source_hash='numerical-fixture-only', source_hash=f'fixture-source-{f}',
                nested_early_stopping=True, requires_fresh_task_fit=True,
                sealed_arrays_accessed=False, model_evaluated_on_sealed=False, model_training_executed=False,
                minimum_cross_partition_support_gap_hours=24., validation_role='inner_early_stopping',
                outer_validation_role='outer_oof_reporting_only', partitions=parts, file_sha256=files,
                data_package_hash=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
            (d / 'readiness_manifest.json').write_text(json.dumps(g))
            folds.append({'fold': f, 'data_dir': str(d), 'data_package_hash': g['data_package_hash'],
                'source_hash': g['source_hash'], 'outer_validation_file': 'outer_validation.npz',
                'validation_metadata_file': 'validation_metadata.parquet', 'validation_examples': 12})
        cv = {'protocol': PROTOCOL, 'ready_to_train': True, 'n_splits': 5, 'folds': folds,
            'source_hash': 'numerical-fixture-only', 'challenge_test_accessed': False,
            'nested_early_stopping': True, 'outer_validation_used_for_early_stopping': False}
        (cv_dir / 'cv_manifest.json').write_text(json.dumps(cv))
        config = json.loads((ROOT / 'configs/group_cv_training.json').read_text())
        config.update(cv_dir=str(cv_dir), models=['cnn'], foundation_models={}, seeds=[42])
        config['training'].update(epochs=1, batch=8, threads=1, trees=2)
        _, ready = preflight(config, cv_dir)
        assert ready['ready'] and ready['oof_examples'] == 60
        output = tmp / 'models'
        execute(config, cv, cv_dir, output, 'cpu', False)
        report = json.loads((output / 'cv_report.json').read_text())
        assert report['oof_examples'] == 60 and not report['outer_used_for_early_stopping'] and not report['outer_used_for_ensemble_weights']
        with np.load(output / 'oof_predictions.npz') as z:
            assert len(np.unique(z['sample_id'])) == 60 and np.isfinite(z['ensemble_share']).all()
        checks['real_CNN_1epoch_5fold_inner_policy_outer_OOF'] = 'passed; numerical fixture only'
        execute(config, cv, cv_dir, output, 'cpu', True)
        checks['same_configuration_resume_with_artifact_ledger'] = 'passed'
        altered = copy.deepcopy(config); altered['training']['dropout'] = .2
        try: execute(altered, cv, cv_dir, output, 'cpu', True)
        except ValueError: checks['changed_configuration_resume_rejected'] = True
        else: raise AssertionError('Changed config was accepted')
        # Both actual pretrained APIs are exercised once on tiny fake contexts.
        d = Path(folds[0]['data_dir']); gate = json.loads((d / 'readiness_manifest.json').read_text())
        foundation_policy = tmp / 'foundation_fixture_policy.json'
        foundation_policy.write_text(json.dumps({'models': ['cnn', 'tabicl', 'kumo_large'], 'weights': [1 / 3] * 3,
            'affine': [1., 0.], 'fit_split': 'inner validation only', 'outer_used_for_fit': False,
            'seeds': [42], 'data_package_hash': gate['data_package_hash'], 'source_hash': gate['source_hash'],
            'fold': 0, 'protocol': PROTOCOL}))
        for kind in ['tabicl', 'kumo_large']:
            inside, outside = tmp / (kind + '_inner'), tmp / (kind + '_outer')
            common = ['pipeline.cv_foundations', '--data', str(d), '--kind', kind,
                '--context', '24', '--members', '1', '--device', 'cpu', '--threads', '1', '--seed', '42', '--batch', '8']
            command([*common, '--phase', 'inner', '--output', str(inside)], tmp / (kind + '_inner.log'))
            command([*common, '--phase', 'outer', '--output', str(outside), '--inner-result', str(inside),
                '--policy', str(foundation_policy)], tmp / (kind + '_outer.log'))
            m = json.loads((outside / 'manifest.json').read_text())
            assert m['phase'] == 'outer' and not m['weight_training'] and m['policy_sha256_before_outer'] == digest(foundation_policy)
            checks[kind + '_two_phase_actual_API_inner_replay'] = 'passed; tiny numerical context only'
        checkpoint = output / 'fold_00/seed_42/cnn.pt'
        checkpoint.write_bytes(checkpoint.read_bytes() + b'changed-artifact')
        try: execute(config, cv, cv_dir, output, 'cpu', True)
        except ValueError: checks['changed_checkpoint_resume_rejected'] = True
        else: raise AssertionError('Changed checkpoint was accepted')
        unrecorded = tmp / 'unrecorded'; unrecorded.mkdir(); payload = unrecorded / 'run'; payload.mkdir()
        try: immutable_step(unrecorded, 'unknown', [payload], lambda: None)
        except ValueError: checks['unrecorded_artifacts_rejected'] = True
        else: raise AssertionError('Unrecorded run was accepted')
    report = {'passed': True, 'checks': checks, 'competition_training_executed': False,
        'GPU_used': False, 'challenge_test_accessed': False, 'calibration_or_sealed_accessed': False,
        'scope': 'Generated tiny numerical fixtures validate execution, not data quality, convergence or competitive performance.',
        'code_sha256': {p.name: digest(p) for p in (ROOT / 'pipeline').glob('*.py')}}
    (ROOT / 'reports/group_cv_runner_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'code_sha256'}, indent=2))


if __name__ == '__main__':
    main()
