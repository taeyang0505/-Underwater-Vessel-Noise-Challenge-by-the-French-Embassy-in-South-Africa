"""Evaluate a finished task-model fold on its untouched outer synthetic set."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
from pipeline.models import make_model
from pipeline.train import predict_deep
from pipeline.metrics import evaluate_regression
from pipeline.cv_foundations import (verify_cv_package, binding, frozen_policy,
                                     load_outer, unchanged_inputs)

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    if a.output.exists() or a.output.with_suffix('.metrics.json').exists():
        raise FileExistsError('Outer predictions are immutable; use a fresh output')
    if a.policy is None or min(a.threads, a.batch) < 1:
        raise ValueError('A frozen policy and positive execution budget are required')
    development, gate, verification = verify_cv_package(a.data)
    del development
    manifest = json.loads((a.run / 'run_manifest.json').read_text())
    if not manifest.get('status', '').startswith('completed '):
        raise ValueError('Only completed task learners can be assessed on outer data')
    args = manifest.get('arguments', {})
    kinds = args.get('models', [])
    allowed = {'lgbm', 'lgbm_physics', 'lgbm_retrieval', 'tabm', 'cnn', 'fdy', 'itransformer'}
    if not kinds or len(set(kinds)) != len(kinds) or not set(kinds).issubset(allowed):
        raise ValueError('Missing or unsupported task learner kinds')
    if args.get('cv_mode') is not True or type(args.get('seed')) is not int:
        raise ValueError('A freshly fitted CV task learner with an explicit seed is required')
    for recorded in [manifest.get('data_folder'), args.get('data_dir')]:
        if recorded is None or Path(recorded).resolve() != a.data.resolve():
            raise ValueError('Task learner points to a different data directory')
    if args.get('output_dir') is None or Path(args['output_dir']).resolve() != a.run.resolve():
        raise ValueError('Task learner run-directory binding differs')
    if manifest.get('data_package', {}).get('data_package_hash') != gate['data_package_hash'] or manifest.get('data_package', {}).get('readiness_manifest_sha256') != verification['readiness_manifest_sha256']:
        raise ValueError('Task learner belongs to a different frozen CV package')
    if manifest.get('external_manifest', {}).get('source_hash') != gate['source_hash'] or manifest.get('validation_protocol') != gate['cv_protocol']:
        raise ValueError('Task learner source/protocol binding differs')
    if manifest.get('prebuilt_readiness_manifest') != gate:
        raise ValueError('Task learner readiness provenance differs')
    code_names = ['train.py', 'models.py', 'mixtures.py', 'train_lgbm.py', 'predict_lgbm.py',
                  'prototypes.py', 'contrastive.py', 'metrics.py']
    current_code = {name: digest(ROOT / 'pipeline' / name) for name in code_names}
    recorded_code = manifest.get('code_sha256_at_start', {})
    if any(recorded_code.get(name) != value for name, value in current_code.items()):
        raise ValueError('Task learner code differs from its frozen training code')
    _, policy_sha = frozen_policy(a.policy, gate, kinds, args['seed'])
    task_manifest_sha = digest(a.run / 'run_manifest.json')
    inference_code = {name: digest(ROOT / 'pipeline' / name) for name in
                      ['evaluate_cv_outer.py', 'cv_foundations.py', *code_names]}
    # Freeze the selected checkpoints before reading any outer target.
    checkpoints = {}
    for kind in kinds:
        names = [kind + '_share.txt', kind + '_presence.txt'] if kind.startswith('lgbm') else [kind + '.pt']
        for name in names:
            if (a.run / name).is_symlink():
                raise ValueError('Linked task checkpoints cannot be assessed')
            checkpoints[name] = digest(a.run / name)
    if 'lgbm_retrieval' in kinds:
        checkpoints['prototypes.npz'] = digest(a.run / 'prototypes.npz')
        checkpoints['prototype_manifest.json'] = digest(a.run / 'prototype_manifest.json')
        prototype_manifest = json.loads((a.run / 'prototype_manifest.json').read_text())
        if prototype_manifest.get('data_package_hash') != gate['data_package_hash'] or prototype_manifest.get('training_only') is not True:
            raise ValueError('Retrieval prototypes must belong to this fold training set')
    if digest(a.policy) != policy_sha:
        raise ValueError('Frozen policy changed before outer access')
    data, meta = load_outer(a.data, gate)
    groups = {c: meta[c].astype(str).to_numpy() for c in ['background_group', 'reference_group'] if c in meta}
    out = {'share_true': data['share'], 'presence_true': data['presence'], 'sample_id': meta.sample_id.to_numpy(dtype=str)}
    tree = [k for k in kinds if k.startswith('lgbm')]
    if tree:
        with tempfile.TemporaryDirectory(prefix='underwater-cv-outer-') as tmp:
            db, pred = Path(tmp) / 'input.npy', Path(tmp) / 'pred.npz'
            np.save(db, data['db'])
            subprocess.run([sys.executable, '-m', 'pipeline.predict_lgbm', str(a.run), str(db),
                str(pred), '--kinds', *tree], check=True)
            with np.load(pred) as d:
                out.update(dict(d))
    torch.set_num_threads(a.threads)
    for kind in [k for k in kinds if not k.startswith('lgbm')]:
        saved = torch.load(a.run / (kind + '.pt'), map_location='cpu', weights_only=True)
        if saved.get('kind') != kind or saved.get('seed') != args['seed'] or saved.get('width') != args['width'] or saved.get('k') != args['k']:
            raise ValueError('Task checkpoint does not match its learner/seed/configuration')
        architecture = saved.get('architecture', {})
        if architecture.get('architecture_version') != 2 or architecture.get('dropout') != args['dropout'] or architecture.get('model_width') != (args['width'] if kind == 'tabm' else args[kind + '_width']):
            raise ValueError('Task checkpoint architecture differs from its training arguments')
        model = make_model(kind, width=saved['width'], k=saved['k'], **saved.get('architecture', {})).to(a.device)
        model.load_state_dict(saved['state_dict'])
        p, s = predict_deep(model, kind, data, torch.device(a.device), batch=a.batch)
        out[kind + '_share'], out[kind + '_score'] = p, s
        del model
    unchanged_inputs(a.data, gate, verification, ['train.npz', 'validation.npz', 'outer_validation.npz', 'validation_metadata.parquet'])
    if digest(a.policy) != policy_sha or digest(a.run / 'run_manifest.json') != task_manifest_sha:
        raise ValueError('Frozen policy or task manifest changed during outer prediction')
    if any(digest(a.run / name) != value for name, value in checkpoints.items()) or any(digest(ROOT / 'pipeline' / name) != value for name, value in inference_code.items()):
        raise ValueError('Frozen task checkpoints or inference code changed during execution')
    report = {**binding(gate, verification), 'status': 'completed frozen-policy outer evaluation',
        'seed': args['seed'], 'models': kinds, 'checkpoint_sha256_before_outer': checkpoints,
        'policy_sha256_before_outer': policy_sha, 'task_manifest_sha256': task_manifest_sha,
        'outer_array_sha256': gate['file_sha256']['outer_validation.npz'],
        'outer_metadata_sha256': gate['file_sha256']['validation_metadata.parquet'],
        'code_sha256': inference_code,
        'selection_on_outer': False, 'test_accessed': False, 'calibration_accessed': False,
        'sealed_accessed': False, 'results': {}}
    for kind in kinds:
        report['results'][kind] = evaluate_regression(data['share'], out[kind + '_share'],
            data['presence'], out[kind + '_score'], data['db'], groups)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **out)
    report['prediction_sha256'] = digest(a.output)
    a.output.with_suffix('.metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('Finished outer evaluation:', len(meta), 'examples;', len(kinds), 'task models', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch', type=int, default=256)
    run(p.parse_args())
