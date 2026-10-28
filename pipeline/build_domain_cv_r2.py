"""Versioned training-only component augmentation; keep heldouts byte-identical.

The parent saved component fractions define the reconstruction. This is domain
randomization of measured mixtures, not additional independent field sources.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_hashes(data):
    return {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in data.items()}


def create(a):
    from pipeline.energy_domain_augmentation import augment_components
    parent, output = a.parent.resolve(), a.output.resolve()
    if output.exists():
        raise FileExistsError('A new immutable data directory is required')
    if parent == output or parent in output.parents:
        raise ValueError('Output must be a separate data version')
    old = json.loads((parent / 'cv_manifest.json').read_text())
    if old.get('n_splits') != 5 or old.get('ready_to_train') is not True:
        raise ValueError('Verified five-fold parent required')
    policy = {'version': 'energy_domain_r2', 'mode': a.mode, 'probability': a.probability,
              'strength_db': a.strength_db, 'seed': a.seed,
              'module_sha256': sha(ROOT / 'pipeline/energy_domain_augmentation.py'),
              'parent_cv_manifest_sha256': sha(parent / 'cv_manifest.json'),
              'scope': 'train only; saved parent component reconstruction; no new source groups'}
    source_hash = canonical(policy)
    initial_bindings = {}
    output.mkdir(parents=True)
    report = {'ready_to_train': False, 'policy': policy, 'folds': [],
              'outer_values_read': False, 'challenge_test_accessed': False}
    cv = {**old, 'ready_to_train': False, 'source_hash': source_hash,
          'parent_cv_source_hash': old['source_hash'], 'parent_cv_dir': str(parent.relative_to(ROOT)),
          'parent_cv_manifest_sha256': policy['parent_cv_manifest_sha256'],
          'augmentation_version': 'energy_domain_r2', 'domain_augmentation_policy': policy, 'folds': []}
    try:
        for entry in old['folds']:
            number = entry['fold']; src = parent / f'fold_{number:02d}'; dst = output / src.name
            gate = json.loads((src / 'readiness_manifest.json').read_text())
            if gate.get('protocol') != old.get('protocol') or gate.get('fold') != number or gate.get('ready_to_train') is not True:
                raise ValueError('Parent fold role/readiness binding differs')
            for flag in ['source_checks_passed', 'reference_group_and_duplicate_isolation_passed',
                         'background_week_and_purge_isolation_passed', 'sealed_holdout_locked',
                         'nested_early_stopping']:
                if gate.get(flag) is not True:
                    raise ValueError('Parent isolation gate failed: ' + flag)
            for flag in ['challenge_test_accessed', 'sealed_arrays_accessed', 'sealed_holdout_materialized',
                         'existing_sealed_NOAA_accessed', 'frozen_v1_data_modified']:
                if gate.get(flag) is not False:
                    raise ValueError('Parent forbidden-access provenance: ' + flag)
            if gate.get('source_hash') != entry['source_hash'] or gate.get('data_package_hash') != entry['data_package_hash'] or gate.get('parent_cv_source_hash') != old['source_hash']:
                raise ValueError('Parent CV index and fold disagree')
            if hashlib.sha256(json.dumps(gate['file_sha256'], sort_keys=True).encode()).hexdigest() != gate['data_package_hash']:
                raise ValueError('Parent package hash differs')
            if sha(src / 'sources/manifest.json') != gate['source_version_manifest_sha256']:
                raise ValueError('Parent source-version manifest changed')
            initial_bindings[number] = {'readiness': sha(src / 'readiness_manifest.json'),
                'source_version': sha(src / 'sources/manifest.json'), 'files': gate['file_sha256']}
            for name, expected in gate['file_sha256'].items():
                if Path(name).name != name or sha(src / name) != expected:
                    raise ValueError('Parent file changed: ' + name)
            # Copies are opaque bytes. Outer arrays/labels are never loaded here.
            shutil.copytree(src, dst)
            with np.load(src / 'train.npz', allow_pickle=False) as z:
                train = {k: z[k] for k in ['db', 'share', 'presence', 'cell_share']}
            seed = a.seed + number * 100003
            augmented, audit = augment_components(train['db'], train['cell_share'], np.random.default_rng(seed),
                mode=a.mode, probability=a.probability, strength_db=a.strength_db)
            replay, _ = augment_components(train['db'], train['cell_share'], np.random.default_rng(seed),
                mode=a.mode, probability=a.probability, strength_db=a.strength_db)
            if array_hashes(augmented) != array_hashes(replay):
                raise ValueError('Deterministic training transform replay failed')
            if not np.array_equal(np.isnan(train['db']), np.isnan(augmented['db'])):
                raise ValueError('Training observation masks changed')
            if not np.array_equal(train['presence'], augmented['presence']):
                raise ValueError('A positive draw became zero or a zero became positive')
            # Recompute the target independently of augmentation implementation.
            x = augmented['db'].astype('float64'); valid = np.isfinite(x)
            peak = np.nanmax(x, axis=(1, 2), keepdims=True)
            e = np.where(valid, 10 ** ((x - peak) / 10), 0.)
            recovered = (e * augmented['cell_share']).sum((1, 2)) / e.sum((1, 2))
            error = float(np.max(np.abs(recovered - augmented['share'])))
            if error > 3e-6:
                raise ValueError('Augmented energy/target identity failed')
            np.savez_compressed(dst / 'train.npz', **augmented)
            # Keep the parent measurement/generation stage verbatim, while
            # making the current metadata describe the saved transformed pair.
            import pandas as pd
            metadata_name = 'train_metadata.parquet'
            parent_metadata_name = 'parent_train_metadata.parquet'
            if metadata_name not in gate['file_sha256']:
                raise ValueError('Parent training metadata is not fingerprinted')
            if (dst / parent_metadata_name).exists():
                raise ValueError('Parent-stage metadata already exists; an untransformed parent is required')
            shutil.copy2(src / metadata_name, dst / parent_metadata_name)
            parent_metadata_sha = sha(dst / parent_metadata_name)
            if parent_metadata_sha != gate['file_sha256'][metadata_name]:
                raise ValueError('Parent-stage metadata copy changed')
            metadata = pd.read_parquet(src / metadata_name)
            if len(metadata) != len(augmented['share']):
                raise ValueError('Parent training metadata is not row-aligned')
            parent_fields = ['requested_share', 'effective_share', 'support_coverage',
                'global_snr_db', 'active_snr_db', 'support_cap', 'cap_applied',
                'support_cap_applied', 'minute_snr_cap_applied',
                'max_minute_snr_db_actual', 'max_minute_snr_db', 'route']
            renamed = {name: 'parent_' + name for name in parent_fields if name in metadata}
            if any(name in metadata for name in renamed.values()):
                raise ValueError('Ambiguous parent-stage metadata columns')
            metadata = metadata.rename(columns=renamed)
            added = e * augmented['cell_share'].astype('float64')
            background = e * (1 - augmented['cell_share'].astype('float64'))
            minute_added = added.sum(2); minute_background = background.sum(2)
            observed = valid.any(2); active = (minute_added > 0) & observed
            if np.any((minute_added > 0) & (minute_background <= 0)):
                raise ValueError('Saved positive component has undefined infinite minute SNR')
            def snr_db(numerator, denominator):
                result = np.full(numerator.shape, np.nan, dtype='float64')
                use = (numerator > 0) & (denominator > 0)
                result[use] = 10 * np.log10(numerator[use] / denominator[use])
                return result
            minute_snr = snr_db(minute_added, minute_background)
            maximum_minute_snr = np.max(np.where(active, minute_snr, -np.inf), axis=1)
            maximum_minute_snr[~active.any(1)] = np.nan
            metadata['effective_share'] = recovered
            metadata['saved_share'] = augmented['share'].astype('float64')
            metadata['support_coverage'] = active.sum(1) / observed.sum(1)
            metadata['global_snr_db'] = snr_db(minute_added.sum(1), minute_background.sum(1))
            metadata['active_snr_db'] = snr_db(np.where(active, minute_added, 0).sum(1),
                np.where(active, minute_background, 0).sum(1))
            metadata['max_minute_snr_db_actual'] = maximum_minute_snr
            # A reference-only response aims to restore the parent global
            # share before caps. A common response has no requested share.
            metadata['requested_share'] = train['share'].astype('float64') if a.mode == 'reference_smooth' else np.nan
            metadata['metadata_stage'] = 'domain_component_transform_r2'
            for name in ['cap_applied', 'support_cap_applied', 'minute_snr_cap_applied']:
                metadata[name] = pd.Series(pd.NA, index=metadata.index, dtype='boolean')
            metadata.to_parquet(dst / metadata_name, index=False)
            metadata_semantics = {'stage': 'domain_component_transform_r2',
                'parent_metadata_file': parent_metadata_name, 'parent_metadata_sha256': parent_metadata_sha,
                'transformed_metadata_file': metadata_name, 'transformed_metadata_sha256': sha(dst / metadata_name),
                'renamed_parent_fields': renamed,
                'source_identity_QC_and_observed_minutes': 'Parent columns retained without alteration.',
                'effective_share': 'Float64 sum(q * E) / sum(E) from saved float32 db/q; saved_share records the NPZ scalar.',
                'support_coverage': 'Minutes with positive observed added energy / minutes with any observed cell.',
                'SNR': 'Float64 10*log10(added/background) on observed cells; active SNR restricts background to positive-added minutes.',
                'undefined_SNR': 'Parquet NaN for zero added energy; a positive-added zero-background minute is rejected.',
                'requested_share': 'Parent NPZ global share before reference response/caps.' if a.mode == 'reference_smooth' else 'Parquet NaN: common-response randomization has no requested global share.',
                'per_row_transform_cap_flags': 'Nullable unknown; the helper exposes aggregate cap counts, not per-row flags.',
                'aggregate_transform_audit': audit, 'rows': len(metadata),
                'max_saved_share_reconstruction_error': error}
            (dst / 'train_metadata_semantics.json').write_text(json.dumps(metadata_semantics, indent=2, allow_nan=False))
            lineage = {**policy, 'fold': number, 'fold_seed': seed,
                'parent_train_npz_sha256': gate['file_sha256']['train.npz'],
                'parent_recipe_sha256': gate['file_sha256'].get('train_recipes.json'),
                'parent_saved_components_are_transform_inputs': True,
                'array_sha256': array_hashes(augmented), 'exact_replay_checked': True,
                'max_saved_cell_target_error': error, 'audit': audit,
                'train_metadata_stage': 'domain_component_transform_r2',
                'train_metadata_semantics_file': 'train_metadata_semantics.json',
                'parent_train_metadata_sha256': parent_metadata_sha}
            (dst / 'domain_transform.json').write_text(json.dumps(lineage, indent=2, allow_nan=False))
            # Original recipes remain source lineage; transform is an explicit
            # second stage, rather than falsely relabeling them as raw replay.
            ng = dict(gate)
            ng.update({'source_hash': canonical({'parent': gate['source_hash'], 'policy': policy, 'fold': number}),
                'parent_cv_source_hash': source_hash, 'parent_fold_readiness_sha256': sha(src / 'readiness_manifest.json'),
                'augmentation_version': 'energy_domain_r2', 'domain_augmentation_policy': policy,
                'augmentation_scope': 'train only, component transform after parent recipe; heldout bytes unchanged',
                'train_recipe_semantics': 'Parent source recipe plus domain_transform.json deterministic second-stage transform',
                'train_metadata_stage': 'domain_component_transform_r2',
                'train_metadata_semantics_file': 'train_metadata_semantics.json',
                'domain_transform_replay_checked': True,
                'source_version_dir': str((dst / 'sources').relative_to(ROOT))})
            ng['partitions'] = {**gate['partitions'], 'train': {**gate['partitions']['train'],
                'npz_bytes': (dst / 'train.npz').stat().st_size, 'max_saved_cell_target_error': error,
                'domain_transform_applied': True, 'domain_transform_replayed': True}}
            ng['file_sha256'] = {p.name: sha(p) for p in dst.iterdir()
                if p.is_file() and p.name not in ['readiness_manifest.json', 'manifest.json']}
            ng['files_sha256'] = ng['file_sha256']
            # Match the existing loader's exact JSON serialization contract.
            ng['data_package_hash'] = hashlib.sha256(json.dumps(ng['file_sha256'], sort_keys=True).encode()).hexdigest()
            (dst / 'readiness_manifest.json').write_text(json.dumps(ng, indent=2, allow_nan=False))
            frozen = ['validation.npz', 'outer_validation.npz', 'validation_recipes.json',
                      'outer_validation_recipes.json', 'inner_validation_metadata.parquet', 'validation_metadata.parquet']
            frozen_hashes = {}
            for name in frozen:
                if (src / name).exists():
                    if sha(src / name) != sha(dst / name):
                        raise ValueError('Frozen evaluation bytes changed: ' + name)
                    frozen_hashes[name] = sha(dst / name)
            cv['folds'].append({**entry, 'data_dir': str(dst.relative_to(ROOT)),
                'source_hash': ng['source_hash'], 'data_package_hash': ng['data_package_hash']})
            report['folds'].append({'fold': number, 'train_examples': len(augmented['share']),
                'max_saved_cell_target_error': error, 'exact_replay_checked': True,
                'frozen_heldout_sha256': frozen_hashes, 'audit': audit})
            del train, augmented, replay, x, e, recovered, added, background, metadata
        if sha(parent / 'cv_manifest.json') != policy['parent_cv_manifest_sha256']:
            raise ValueError('Parent changed during transform')
        if sha(ROOT / 'pipeline/energy_domain_augmentation.py') != policy['module_sha256']:
            raise ValueError('Component augmentation code changed during transform')
        for number, checks in initial_bindings.items():
            src = parent / f'fold_{number:02d}'
            if sha(src / 'readiness_manifest.json') != checks['readiness'] or sha(src / 'sources/manifest.json') != checks['source_version']:
                raise ValueError('Parent fold metadata changed during transform')
            if any(sha(src / name) != expected for name, expected in checks['files'].items()):
                raise ValueError('Parent arrays/recipes/metadata changed during transform')
        report['ready_to_train'] = True; cv['ready_to_train'] = True
    finally:
        (output / 'domain_build_verification.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        (output / 'cv_manifest.json').write_text(json.dumps(cv, indent=2, allow_nan=False))
    print(json.dumps({'ready_to_train': report['ready_to_train'], 'output': str(output),
                      'folds': len(report['folds']), 'source_hash': source_hash}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, default=ROOT / 'data/synthetic/group_cv_physics_v2_r1')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['reference_smooth', 'common_response'], required=True)
    p.add_argument('--probability', type=float, default=.75)
    p.add_argument('--strength-db', type=float, default=12.)
    p.add_argument('--seed', type=int, default=3407)
    create(p.parse_args())
