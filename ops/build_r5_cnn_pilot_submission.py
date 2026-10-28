"""Explicit single completed Fold-0 CNN pilot inference, not final R5 ensemble.

Requires an independently frozen INNER-only postprocessing policy before Test
IO. Never fits on Test or modifies the running GPU experiment. The five-fold
production inference gate remains unchanged; this is a separately named pilot.
"""
from pathlib import Path
import argparse
import csv
import datetime
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import sys
import time

import numpy as np
import torch

from pipeline.grid_inputs_r5 import canonical_sha256, file_sha256, load_shared_prior
from pipeline.inference_r5 import read_query_csv, original_energy_weights
from pipeline.neural_r5 import make_neural, predict_neural, restore_into

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA = 'aa222697275306cf93e5778a707f7ee5fd75610ae0a0bf643d654854ee96d46f'
TEST_SHA = '3ef819192c05c4308b66564e6307e6f4f26fe806afe96ec93e2d15f25a3474d4'
SAMPLE_SHA = 'b9ca88ef9249e34572a564adc87025c4186a59bb7b4836a53f09c4208e36693a'
PROFILE_SHA = '9f91a8bf85c2b7ba33ea85368269073a2f6d5c53a499c87dc23f0f3b67ccce80'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--expected-policy-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError('A fresh pilot destination is required')
    if file_sha256(args.policy) != args.expected_policy_sha256:
        raise ValueError('Explicit frozen postprocessing policy SHA changed')
    policy = json.loads(args.policy.read_text())
    # A version-specific pure adapter is supplied by the INNER-only experiment.
    from ops.calibrate_r5_single_cnn import apply_frozen_policy, validate_frozen_policy
    validate_frozen_policy(policy)
    if policy['checkpoint_sha256'] != CHECKPOINT_SHA:
        raise ValueError('Postprocessing policy belongs to a different checkpoint')
    profile_file = ROOT / 'reports/R5_CNN_MAC_INFERENCE_PROFILE.json'
    if file_sha256(profile_file) != PROFILE_SHA:
        raise ValueError('CPU validation profile changed before inference')
    profile = json.loads(profile_file.read_text())
    if profile['checkpoint_sha256'] != CHECKPOINT_SHA or not profile['model_state_unchanged']:
        raise ValueError('A verified complete CPU model replay is required')
    if not profile['CPU_full_FP32'] or profile['torch_threads'] != 4:
        raise ValueError('CPU inference must match the validation precision/threads')
    model_dir = ROOT / 'ops/server_backups/r5_gpu_r1/restored_first_cnn_v1/fold_00/preprocessing_pilot/raw/bg_cnn'
    fold_dir = model_dir.parents[2]
    context_path = fold_dir / 'context.json'
    context = json.loads(context_path.read_text())
    summary = json.loads((model_dir / 'summary.json').read_text())
    if summary.get('status') != 'completed' or summary.get('fixture_only') is not False:
        raise ValueError('A completed production CNN checkpoint is required')
    if summary.get('outer_or_Test_accessed') is not False or summary['best_epoch'] != 19:
        raise ValueError('Unexpected checkpoint training scope/epoch')
    if canonical_sha256(context['config']) != context['config_sha256']:
        raise ValueError('Frozen context configuration changed')
    observed_files = {str(args.policy.resolve()): args.expected_policy_sha256,
                      str(profile_file): PROFILE_SHA}
    for name, expected in context['code_sha256'].items():
        path = ROOT / name
        if file_sha256(path) != expected:
            raise ValueError('Original model/preprocessing code changed: ' + name)
        observed_files[str(path)] = expected
    checkpoint_path = model_dir / 'best.pt'
    if file_sha256(checkpoint_path) != CHECKPOINT_SHA or summary['checkpoint_sha256'] != CHECKPOINT_SHA:
        raise ValueError('Original best CNN checkpoint changed')
    observed_files[str(checkpoint_path)] = CHECKPOINT_SHA
    if file_sha256(context_path) != profile['context_sha256']:
        raise ValueError('CPU profile and checkpoint context differ')
    prior_record = context['prior']
    prior = load_shared_prior(fold_dir / prior_record['filename'],
        expected_binding=context['training_binding'], expected_file_sha256=prior_record['file_sha256'],
        expected_prior_sha256=prior_record['payload_sha256'])
    observation = context['config']['observation_candidates']['raw']
    if observation != {'quantization_decimals': None, 'endpoint_dropout_probability': 0.0}:
        raise ValueError('Original raw CNN preprocessing must remain fixed')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    start = time.perf_counter()
    model, factory_receipt = make_neural('bg_cnn', context['config']['model_settings']['bg_cnn'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['receipt'] != summary['receipt'] or checkpoint['training_settings'] != summary['settings']:
        raise ValueError('Checkpoint receipt/training configuration differs from completed model')
    restore_into(model, checkpoint, kind='bg_cnn', binding=context['training_binding'],
                 prior_sha=prior['shared_prior_payload_sha256'], policy=observation)
    source_identity = checkpoint['dynamic_source_identity']
    receipt = json.loads((model_dir / 'dynamic_source_receipts.json').read_text())
    epoch_receipt = receipt['completed_epochs'][str(checkpoint['epoch'])]
    if source_identity != summary['dynamic_source_identity'] or receipt['source_identity'] != source_identity:
        raise ValueError('Dynamic training source receipt differs')
    if epoch_receipt != checkpoint['best_epoch_source_receipt'] or not epoch_receipt['clean_primary_once']:
        raise ValueError('Best epoch does not certify a complete TRAIN traversal')
    load_seconds = time.perf_counter() - start
    # All model/precision/policy selection has finished before this boundary.
    frozen_before_query_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    test_path, sample_path = ROOT / 'data/raw/Test.csv', ROOT / 'data/raw/sample_submission.csv'
    if file_sha256(test_path) != TEST_SHA or file_sha256(sample_path) != SAMPLE_SHA:
        raise ValueError('Original competition input files differ')
    ids, db, query_receipt = read_query_csv(test_path, expected_rows=1500)
    with sample_path.open(newline='') as f:
        sample = list(csv.reader(f))
    columns = ['ID', 'share_added_vessel', 'score_added_vessel']
    if sample[0] != columns or [r[0] for r in sample[1:]] != ids:
        raise ValueError('Sample submission ID order/columns differ')
    query_started = time.perf_counter()
    raw = predict_neural(model, db, prior, context['training_binding'], observation, device='cpu', batch_size=128)
    query_seconds = time.perf_counter() - query_started
    weights = original_energy_weights(db)
    reconstructed = (weights.astype('float64') * raw['cell_mask']).sum((1, 2))
    pooling_residual = float(np.max(np.abs(reconstructed - raw['share'])))
    if pooling_residual > 3e-6:
        raise ValueError('Cell-mask energy pooling failed on submission queries')
    share = np.asarray(apply_frozen_policy(policy, raw['share']), dtype='float64')
    score = raw['score'].astype('float64')
    if share.shape != (1500,) or score.shape != share.shape:
        raise ValueError('Submission prediction shape mismatch')
    if not np.isfinite(share).all() or not np.isfinite(score).all() or np.any((share < 0) | (share > 1)) or np.any((score < 0) | (score > 1)):
        raise ValueError('Submission predictions must be finite [0,1] values')
    for name, expected in observed_files.items():
        if file_sha256(name) != expected:
            raise ValueError('Frozen input/model/policy changed during inference')
    output.mkdir(parents=True)
    csv_path = output / 'submission.csv'
    with csv_path.open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(zip(ids, share.tolist(), score.tolist()))
    with csv_path.open(newline='') as f:
        check = list(csv.reader(f))
    values = np.array([[float(x) for x in r[1:]] for r in check[1:]])
    if check[0] != columns or [r[0] for r in check[1:]] != ids or not np.array_equal(values[:, 0], share) or not np.array_equal(values[:, 1], score):
        raise ValueError('Written CSV did not round-trip exact prediction values')
    np.savez_compressed(output / 'raw_test_predictions.npz', sample_id=np.asarray(ids, dtype=str), **raw)
    shutil.copyfile(args.policy, output / 'frozen_postprocessing_policy.json')
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = {'format': 'r5_single_complete_cnn_pilot_inference_v1', 'status': 'completed',
        'model_scope': 'one completed raw CNN from fold0; not final 5fold R5 ensemble',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'model_policy_frozen_before_Test_open_utc': frozen_before_query_at,
        'checkpoint_sha256': CHECKPOINT_SHA, 'checkpoint_epoch': 19, 'fold': 0,
        'training_binding': context['training_binding'], 'context_sha256': profile['context_sha256'],
        'raw_observation_policy': observation, 'postprocessing_policy_sha256': args.expected_policy_sha256,
        'postprocessing_policy': policy, 'CPU_profile_sha256': PROFILE_SHA,
        'query': query_receipt, 'sample_submission_sha256': SAMPLE_SHA,
        'rows': 1500, 'columns': columns, 'submission_sha256': file_sha256(csv_path),
        'ID_order_sha256': canonical_sha256(ids), 'ID_order_unique_and_matches_sample': True,
        'source_weights_and_policy_unchanged': True, 'physical_pool_max_absolute_error': pooling_residual,
        'seconds': {'load_and_verify_model': load_seconds, 'Test1500_prediction': query_seconds},
        'peak_RSS_bytes': int(peak_rss if sys.platform == 'darwin' else peak_rss * 1024),
        'factory_receipt': factory_receipt, 'threads': 4, 'batch_size': 128, 'precision': 'CPU FP32',
        'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                        'platform': platform.platform(), 'machine': platform.machine()},
        'inference_code_sha256': file_sha256(__file__),
        'calibrator_code_sha256': file_sha256(ROOT / 'ops/calibrate_r5_single_cnn.py'),
        'original_code_sha256': context['code_sha256'],
        'Test_fitting_or_cohort_tuning_or_label_recovery': False, 'OUTER_accessed': False,
        'GPU_training_modified': False, 'official_submission_performed': False,
        'external_data_source_declaration_posted_by_this_action': False}
    (output / 'inference_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'saved': str(csv_path), 'rows': 1500, 'sha256': receipt['submission_sha256'],
                      'Test1500_seconds': query_seconds, 'official_submission': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
