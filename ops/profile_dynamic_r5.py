"""Profile R5 TRAIN-only CPU input preparation, without fitting a neural model.

Initial readiness/TRAIN loading, aggregate prior fitting, and global-bank SHA
checks are timed separately from a prefix of an epoch's dynamic stream. The
prefix iterator is deliberately closed without exhaustion: no completed epoch
receipt is issued, even if --samples happens to equal the clean TRAIN count.
Global bank bytes are hashed for integrity; only TRAIN slices become examples.
No INNER/OUTER arrays, Test, WAVs, models, GPU calls, SSH, or downloads are used.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def _positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError('A positive integer is required')
    return n


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/r5_training_integration.json')
    p.add_argument('--samples', type=_positive, default=2048,
                   help='Number of TRAIN prefix examples transformed; default 2048')
    p.add_argument('--microbatch', type=_positive, default=128)
    p.add_argument('--observation', default='resolution_01_endpoint',
                   help='Predeclared observation_candidates key in --config')
    p.add_argument('--output', required=True, type=Path,
                   help='Fresh JSON path; existing paths are never overwritten')
    return p


def _read_guards(allowed, np, pd):
    """Deny data reads outside an explicit TRAIN/source allowlist.

    Python open auditing also forbids all project-data writes and audio-file
    reads. Explicit NumPy/Pandas wrappers cover their native IO entry points.
    Source metadata may declare all roles; held-out observation files do not
    appear in the allowlist and are never read.
    """
    data_root = (ROOT/'data').resolve()
    reads = set()

    def check(value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            raise ValueError('Explicit path required for profiling data IO')
        path = Path(os.fsdecode(value)).resolve()
        if path.suffix.lower() in {'.wav', '.flac', '.mp3', '.ogg', '.aiff'}:
            raise RuntimeError('Audio files are forbidden in this CPU profile')
        if path.is_relative_to(data_root):
            if path not in allowed:
                raise RuntimeError('Non-TRAIN/non-allowlisted data read: '+str(path))
            reads.add(path.relative_to(ROOT).as_posix()
                      if path.is_relative_to(ROOT) else str(path))
        return path

    def audit(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'}:
            raise RuntimeError('Network/subprocess execution is forbidden in this profile')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = check(args[0])
            if path.is_relative_to(data_root):
                flags = args[2]
                if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                    raise RuntimeError('Project source-data writes are forbidden')

    sys.addaudithook(audit)
    original_np_load, original_parquet = np.load, pd.read_parquet

    def checked_load(path, *args, **kwargs):
        check(path)
        return original_np_load(path, *args, **kwargs)

    def checked_parquet(path, *args, **kwargs):
        check(path)
        return original_parquet(path, *args, **kwargs)

    np.load, pd.read_parquet = checked_load, checked_parquet
    return reads


def _host_info():
    # platform.platform()/platform.processor() may spawn `uname` on Linux.
    # os.uname is a direct system call and keeps the subprocess ban intact.
    u = os.uname() if hasattr(os, 'uname') else None
    return {'platform': ' '.join([u.sysname, u.release, u.machine]) if u else os.name,
            'system': u.sysname if u else os.name,
            'release': u.release if u else None,
            'kernel_version': u.version if u else None,
            'machine': u.machine if u else None,
            'python': sys.version, 'logical_CPU_count': os.cpu_count(),
            'thread_environment': {k: os.environ.get(k) for k in
                ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
            'collection_uses_subprocess': False}


def _failure_receipt(args, error, progress):
    """Write only a fresh diagnostic JSON, without reopening any dataset."""
    output = _resolve(args.output)
    if output.exists() or output.is_symlink() or output.is_relative_to((ROOT/'data').resolve()):
        return None
    receipt = {'status': 'failed TRAIN-only CPU dynamic-input timing profile',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'error_type': type(error).__name__, 'error': str(error),
        'progress': progress, 'profile_completed': False,
        'completed_epoch_receipt_issued': False, 'partial_profile_reusable_as_success': False,
        'additional_dataset_reads_by_failure_handler': False,
        'neural_or_tree_training_GPU_SSH_execution_by_profile': False,
        'retry_requires_fresh_output_path': True}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
        f.write('\n')
    return output


def profile(args, progress=None):
    progress = {} if progress is None else progress
    progress['stage'] = 'configuration'
    total_start = time.perf_counter()
    output = _resolve(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Fresh output JSON required: '+str(output))
    if output.is_relative_to((ROOT/'data').resolve()):
        raise ValueError('Profile reports must not be written into source data')
    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text())
    if config.get('protocol') != 'source_group_nested5_vtuad_passage_r4':
        raise ValueError('Frozen R4 source protocol required')
    if args.observation not in config['observation_candidates']:
        raise ValueError('Unknown predeclared observation candidate: '+args.observation)
    policy = config['observation_candidates'][args.observation]
    seed, epoch, fold_number = int(config['seed']), 1, 0
    folder = _resolve(Path(config['dataset'])/'fold_00')
    allowed = {folder/name for name in [
        'readiness_manifest.json', 'train.npz', 'train_sample_metadata.parquet',
        'training_background_order.npy', 'sources/manifest.json',
        'sources/background_manifest.parquet', 'sources/reference_manifest.parquet']}
    parent = ROOT/'data/synthetic/group_cv_physics_v2_r1/fold_00/sources'
    allowed.update(parent/name for name in ['manifest.json', 'background_manifest.parquet'])
    allowed.update(ROOT/name for name in [
        'data/versions/vtuad_reference_r4_v1/manifest.json',
        'data/processed/sa_windows_db.npy', 'data/processed/external_segments_power.npy'])
    allowed = {p.resolve() for p in allowed}
    timing = {'configuration_seconds': time.perf_counter()-total_start}
    progress['timing_seconds'] = timing
    progress['stage'] = 'imports'
    start = time.perf_counter()
    import numpy as np
    import pandas as pd
    from pipeline.fold_data_r5 import FoldDataR5
    from pipeline.grid_inputs_r5 import fit_shared_prior, shared_grid_inputs, file_sha256, canonical_sha256
    from pipeline.dynamic_passage_r5 import DynamicPassageEpochSource
    from pipeline.observation_augmentation_r5 import EndpointDropoutPolicy, augment_training_observations
    timing['imports_seconds'] = time.perf_counter()-start
    reads = _read_guards(allowed, np, pd)
    progress['stage'] = 'readiness_and_binding'
    start = time.perf_counter()
    fold = FoldDataR5(folder, seed)
    timing['readiness_binding_and_TRAIN_NPZ_hash_seconds'] = time.perf_counter()-start
    progress['stage'] = 'static_TRAIN_load_and_checks'
    start = time.perf_counter()
    train, metadata = fold.load('train')
    timing['static_TRAIN_load_metadata_hash_and_label_checks_seconds'] = time.perf_counter()-start
    progress['stage'] = 'static_TRAIN_shared_prior_fit'
    start = time.perf_counter()
    prior = fit_shared_prior(train, fold.binding, split='train')
    timing['static_TRAIN_shared_prior_fit_seconds'] = time.perf_counter()-start
    del train, metadata
    progress['stage'] = 'dynamic_source_bank_and_slice_integrity_checks'
    start = time.perf_counter()
    source = DynamicPassageEpochSource(folder, seed, expected_binding=fold.binding)
    timing['dynamic_source_readiness_metadata_bank_hash_and_TRAIN_slice_checks_seconds'] = time.perf_counter()-start
    if args.samples > source.n_examples:
        raise ValueError('--samples exceeds the once-per-epoch clean TRAIN count')
    dropout = EndpointDropoutPolicy(probability=policy['endpoint_dropout_probability'],
                                    max_drop=2, min_observed_minutes=3)
    stage_times = {'generation': [], 'observation': [], 'common_grid8': [], 'checks_and_digest': []}
    counters = {k: 0 for k in ['requested_events', 'applied_events',
        'minimum_support_protection_skips', 'positive_to_zero_after_removing_known_support']}
    digest = {k: hashlib.sha256() for k in ['db', 'share', 'presence', 'cell_share', 'grid8', 'physical_weights']}
    iterator = source.iter_batches(epoch, args.microbatch, with_grid=False)
    processed = generated = 0
    seen = set()
    max_target_residual = 0.
    stream_start = time.perf_counter()
    progress.update(stage='stream_prefix', samples_processed=0, examples_generated=0)
    try:
        while processed < args.samples:
            start = time.perf_counter()
            batch = next(iterator)
            stage_times['generation'].append(time.perf_counter()-start)
            generated += len(batch['db'])
            take = min(len(batch['db']), args.samples-processed)
            batch = {k: v[:take] for k, v in batch.items()}
            start = time.perf_counter()
            changed, audit = augment_training_observations(
                batch['db'], batch['cell_share'], epoch=epoch,
                sample_indices=batch['background_index'], seed=seed, policy=dropout,
                split='train', quantization_decimals=policy['quantization_decimals'])
            stage_times['observation'].append(time.perf_counter()-start)
            start = time.perf_counter()
            grid, _ = shared_grid_inputs(changed['feature_db'], prior, expected_binding=fold.binding)
            weights = changed['energy_weights']
            stage_times['common_grid8'].append(time.perf_counter()-start)
            start = time.perf_counter()
            if grid.shape != (take, 8, 15, 25) or not np.isfinite(grid).all():
                raise ValueError('Invalid shared grid8 profile output')
            recovered = (weights.astype(float)*changed['cell_share']).sum((1, 2))
            residual = float(np.max(np.abs(recovered-changed['share'])))
            if residual > 3e-6:
                raise ValueError('Raw-energy q/y conservation failed')
            max_target_residual = max(max_target_residual, residual)
            ids = [int(v) for v in batch['background_index']]
            if seen.intersection(ids) or len(set(ids)) != len(ids):
                raise ValueError('Repeated primary background in TRAIN profile prefix')
            seen.update(ids)
            for name in counters:
                counters[name] += int(audit[name])
            for name in ['db', 'share', 'presence', 'cell_share']:
                digest[name].update(np.ascontiguousarray(changed['physical_db'] if name == 'db' else changed[name]).tobytes())
            digest['grid8'].update(np.ascontiguousarray(grid).tobytes())
            digest['physical_weights'].update(np.ascontiguousarray(weights).tobytes())
            processed += take
            progress.update(samples_processed=processed, examples_generated=generated)
            stage_times['checks_and_digest'].append(time.perf_counter()-start)
    finally:
        iterator.close()
    timing['stream_prefix_wall_seconds'] = time.perf_counter()-stream_start
    progress['stage'] = 'final_identity_checks_and_report'
    if source.completed_epochs:
        raise AssertionError('A prefix must not be certified as a completed epoch')
    start = time.perf_counter()
    source.verify_source_integrity()
    timing['final_source_identity_stat_checks_seconds'] = time.perf_counter()-start
    if fold.opened_roles != ['train']:
        raise AssertionError('A non-TRAIN role was opened')
    sums = {name: sum(values) for name, values in stage_times.items()}
    report = {
        'status': 'passed TRAIN-only CPU dynamic-input timing profile',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'fold': fold_number, 'epoch': epoch, 'samples_processed': processed,
        'examples_generated_before_optional_final_batch_truncation': generated,
        'microbatch': args.microbatch, 'batches': len(stage_times['generation']),
        'observation_name': args.observation, 'observation_policy': policy,
        'timing_seconds': timing,
        'stream_stage_seconds': sums,
        'stream_seconds_per_processed_example': timing['stream_prefix_wall_seconds']/processed,
        'generation_seconds_per_generated_example': sums['generation']/generated,
        'observation_seconds_per_processed_example': sums['observation']/processed,
        'common_grid8_seconds_per_processed_example': sums['common_grid8']/processed,
        'first_next_seconds_including_full_epoch_schedule_and_first_batch': stage_times['generation'][0],
        'batch_stage_seconds': {name: {'minimum': min(v), 'median': float(np.median(v)),
             'p90': float(np.quantile(v, .9)), 'maximum': max(v)} for name, v in stage_times.items()},
        'linear_CPU_one_epoch_stream_seconds_estimate': source.n_examples*timing['stream_prefix_wall_seconds']/processed,
        'estimate_limit': 'Same-host linear CPU input-preparation estimate only; excludes GPU/model/optimizer/evaluation/checkpoint time and parallel overlap. Hashing warms the filesystem cache before streaming.',
        'initialization_limit': 'Each helper performs its actual integrity checks; repeated TRAIN hashes are included in their respective measured stages. The static prior is ephemeral and not exported for production reuse.',
        'clean_TRAIN_primary_count': source.n_examples,
        'unique_profile_primary_count': len(seen),
        'profile_prefix_primary_indices_unique': True,
        'source_completed_epochs': source.completed_epochs,
        'epoch_iterator_fully_exhausted': False,
        'completed_epoch_receipt_issued': False,
        'partial_iterator_warning': 'The iterator was closed after the requested prefix, not advanced to StopIteration; this is not a completed TRAIN epoch.',
        'observation_augmentation_counts': counters,
        'maximum_raw_energy_weighted_target_residual': max_target_residual,
        'shared_prior_payload_sha256': prior['shared_prior_payload_sha256'],
        'prior_training_rows': prior['training_rows'], 'prior_actual_clusters': prior['actual_clusters'],
        'training_binding': fold.binding, 'dynamic_source_identity': source.identity,
        'read_data_paths_allowlisted': sorted(reads),
        'opened_static_array_roles': fold.opened_roles,
        'source_bank_scope': 'Whole global banks are integrity-hashed; only TRAIN background/reference slices become sampled examples. Other-role rows may occur in source metadata, never in held-out NPZ reads.',
        'aggregate_TRAIN_prior_CPU_fit_performed': True,
        'neural_or_tree_training_GPU_SSH_network_audio_Test_INNER_OUTER_arrays_used': False,
        'dataset_or_pipeline_source_modified': False,
        'profile_output_array_sha256': {k: v.hexdigest() for k, v in digest.items()},
        'configuration_file_sha256': file_sha256(config_path),
        'configuration_payload_sha256': canonical_sha256(config),
        'code_sha256': {name: file_sha256(ROOT/name) for name in [
            'ops/profile_dynamic_r5.py', 'pipeline/dynamic_passage_r5.py', 'pipeline/grid_inputs_r5.py',
            'pipeline/fold_data_r5.py', 'pipeline/observation_augmentation_r5.py',
            'pipeline/passage_sampler_r4.py', 'pipeline/mixtures.py', 'pipeline/background_prior_features.py']},
        'host': _host_info(),
        'total_seconds_excluding_report_write': time.perf_counter()-total_start}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': report['status'], 'report': str(output),
        'samples': processed, 'stream_seconds_per_example': report['stream_seconds_per_processed_example'],
        'completed_epoch_receipt_issued': False}, indent=2))


if __name__ == '__main__':
    args = _parser().parse_args()
    progress = {}
    try:
        profile(args, progress)
    except Exception as error:
        try:
            receipt = _failure_receipt(args, error, progress)
            if receipt is not None:
                print('Failed timing profile receipt: '+str(receipt), file=sys.stderr)
        except Exception as receipt_error:
            print('Could not write fresh failure receipt: '+type(receipt_error).__name__, file=sys.stderr)
        raise
