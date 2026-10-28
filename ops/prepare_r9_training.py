#!/usr/bin/env python3
"""Fresh R9 fixed-data materialization and source/schedule QA; never model fit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.dataset_r9 import (
    VERSION, DATA_CODE, ROLES, R9Sources, DynamicR9Dataset, FixedR9Dataset,
    canonical, digest, sha, array_sha, make_schedule, schedule_audit, generate_fixed,
    validate_bundle, _generate,
)
from pipeline.sampler_r9 import synthesize, energy_weights
from pipeline.source_split_r6 import apply_minute_qc
from pipeline.relative_grid_r6_v2 import relative_grid_v2


def write(path, value):
    path = Path(path)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def progress(folder, value):
    p = Path(folder)/'preparation_progress.json'
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    t.replace(p)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def verify_cache(sources):
    """Every admitted source, all three families: raw2s/cached bitwise parity."""
    bg_row = sources.role_metadata(0, 'train').iloc[0].to_dict()
    bg = sources.background(bg_row)
    bank = np.load(sources.reference_path, mmap_mode='r', allow_pickle=False)
    cases = 0
    for idx, r in enumerate(sources.references.itertuples(index=False)):
        raw = bank[int(r.segment_start):int(r.segment_start)+int(r.segment_count)]
        for family_no, family in enumerate(sources.config['generator']['family_names']):
            seed = 4719+idx*31+family_no
            a = synthesize(bg, raw, np.random.default_rng(seed), family=family)
            b = synthesize(bg, None, np.random.default_rng(seed), family=family,
                           reference_minute_power=sources.reference_minutes[idx])
            for key in ('db', 'unrounded_db', 'cell_share', 'share', 'presence'):
                if not np.array_equal(a[key], b[key], equal_nan=True):
                    raise AssertionError('Direct/cached output differs')
            if canonical(a['recipe']) != canonical(b['recipe']):
                raise AssertionError('Direct/cached recipe differs')
            cases += 1
    return {'passed': True, 'reference_count': 509, 'families': 3, 'cases': cases,
            'db_q_y_presence_recipe_bitwise_equal': True}


def verify_background_masks(sources):
    frames = [sources.role_metadata(f, role) for f in range(5) for role in ('train',)+ROLES]
    all_rows = pd.concat(frames, ignore_index=True)
    uniqueness = all_rows.groupby('base_window_index').agg(
        mask_counts=('boundary_mask_bits', 'nunique'), qc_counts=('qc_observed_minutes', 'nunique'))
    if (uniqueness.mask_counts != 1).any() or (uniqueness.qc_counts != 1).any():
        raise AssertionError('A source window has role-dependent minute QC')
    unique = all_rows.drop_duplicates('base_window_index')
    masked, maxmask = 0, 0
    bank = sources.background_bank()
    for begin in range(0, len(unique), 512):
        part = unique.iloc[begin:begin+512]
        raw = np.asarray(bank[part.base_window_index.to_numpy(np.int64)])
        q = apply_minute_qc(raw, part.boundary_mask_bits.to_numpy(np.int64))
        o = np.isfinite(q)
        counts = o.any(2).sum(1)
        if (np.isinf(q).any() or not np.array_equal(o.any(2), o.all(2))
                or not np.array_equal(counts, part.qc_observed_minutes.to_numpy()) or (counts < 3).any()
                or not np.isnan(q)[np.isnan(raw)].all()):
            raise AssertionError('Background finite/missing/min3/QC mismatch')
        removed = np.isfinite(raw).any(2).sum(1)-counts
        masked += int(removed.sum())
        maxmask = max(maxmask, int(removed.max()))
    separation = []
    for fold in range(5):
        role_groups = {role: set(sources.role_metadata(fold, role).background_group) for role in ('train',)+ROLES}
        if any(role_groups[a] & role_groups[b] for a in role_groups for b in role_groups if a < b):
            raise AssertionError('Background source-group overlap')
        fm = json.loads((sources.bg_dir/f'fold_{fold:02d}/manifest.json').read_text())
        if fm['cross_role_support_gap_under24h_violations'] or fm['minimum_cross_role_support_gap_hours'] < 24.:
            raise AssertionError('Cross-role calendar support purge failed')
        separation.append({'fold': fold, 'group_counts': {k: len(v) for k, v in role_groups.items()},
                           'minimum_cross_role_support_gap_hours': fm['minimum_cross_role_support_gap_hours'],
                           'group_overlap_violations': 0})
    return {'passed': True, 'unique_source_windows_checked': len(unique),
            'actual_removed_minutes': masked, 'max_removed_minutes_per_window': maxmask,
            'minimum_observed_minutes': int(unique.qc_observed_minutes.min()),
            'original_missing_preserved': True, 'split_specific_QC_violations': 0,
            'background_group_and_24h_purge': separation}


def check_arrays(arrays, metadata, recipes):
    n = len(arrays['share'])
    if len(metadata) != n or len(recipes) != n or len(set(arrays['sample_id'])) != n:
        raise AssertionError('Fixed ID/order/recipe lengths invalid')
    if any(a.dtype.kind == 'O' for a in arrays.values()):
        raise AssertionError('Object array forbidden')
    for k in ('sample_id', 'base_window_index', 'site', 'background_group', 'reference_index', 'explicit_no_source'):
        if not np.array_equal(arrays[k], metadata[k].to_numpy()):
            raise AssertionError('Metadata/NPZ alignment differs: '+k)
    mask = np.isfinite(arrays['db'])
    q = arrays['cell_share'].astype(np.float64)
    y = arrays['share'].astype(np.float64)
    if (not np.array_equal(mask, np.isfinite(arrays['unrounded_db']))
            or not np.array_equal(mask.any(2), mask.all(2)) or (mask.any(2).sum(1) < 3).any()
            or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any()
            or np.any(q[~mask] != 0) or not np.isfinite(y).all() or (y < 0).any() or (y > 1).any()
            or not np.array_equal(arrays['presence'], y > 0)):
        raise AssertionError('Masks/known component labels invalid')
    raw_error = max(abs(float((energy_weights(raw)*trueq).sum())-float(truth))
                    for raw, trueq, truth in zip(arrays['unrounded_db'], q, y))
    if raw_error > 2e-7:
        raise AssertionError('Saved raw energy identity exceeds float32 label allowance')
    x, w = relative_grid_v2(arrays['db'])
    rounded_error = np.abs((w.astype(np.float64)*q).sum((1,2))-y)
    bounds = np.asarray([r['rounding_bound'] for r in recipes])
    if (not np.isfinite(x).all() or not np.isfinite(w).all()
            or not np.array_equal(x[:,5] > 0, mask) or np.any(x.transpose(0,2,3,1)[~mask] != 0)
            or not np.array_equal(x[:,7], np.sqrt(w)) or (rounded_error > bounds+2e-7).any()):
        raise AssertionError('Common8 input/rounded physical pooling invalid')
    zero = arrays['explicit_no_source']
    if int(zero.sum()) != round(.2*n) or np.any(y[zero] != 0) or np.any(q[zero] != 0):
        raise AssertionError('Explicit no-source invariant failed')
    return {'passed': True, 'examples': n, 'zero_requests': int(zero.sum()),
            'positive_requests_with_zero_observed_addition': int(((~zero) & (y == 0)).sum()),
            'observed_positive_count': int((y > 0).sum()),
            'raw_energy_label_max_error': float(raw_error),
            'rounded_fixed_q_pool_max_gap': float(rounded_error.max()),
            'maximum_allowed_rounding_bound': float(bounds.max()),
            'missing_cells': int((~mask).sum()),
            'target_moments_generation_QA_only': {'mean': float(y.mean()), 'std': float(y.std()),
                                               'min': float(y.min()), 'max': float(y.max())},
            'family_counts': {k: int(v) for k, v in metadata.requested_family.value_counts().items()},
            'observed_minute_counts': {str(k): int(v) for k, v in metadata.qc_observed_minutes.value_counts().items()},
            'site_counts': {str(k): int(v) for k, v in metadata.site.value_counts().items()}}


def prepare(config_path='configs/r9_full_training_v1.json', output=None):
    start = time.monotonic()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT/config_path
    config = json.loads(config_path.read_text())
    output = Path(output or config['data_output'])
    if not output.is_absolute():
        output = ROOT/output
    if output.exists():
        raise FileExistsError('R9 preparation is fresh-only; partial outputs must not be reused')
    output.mkdir(parents=True)
    pins = {name: sha(ROOT/name) for name in DATA_CODE+('ops/prepare_r9_training.py',)}
    config_sha = sha(config_path)
    write(output/'preparation_plan.json', {'version': VERSION, 'config_file': config_path.relative_to(ROOT).as_posix(),
          'config_sha256': config_sha, 'code_sha256': pins,
          'roles': config['fixed_examples'], 'source_draw': 'one anchor per source group then uniform without replacement if possible',
          'background_primary_once_per_TRAIN_epoch': True, 'all_509_references_per_TRAIN_epoch': True,
          'reference_shared_across_background_folds': True,
          'report_target_moments_generation_QA_only': True,
          'Test_or_waveform_or_model_access': False})
    progress(output, {'phase': 'source_verification', 'ready_to_train': False})
    try:
        sources = R9Sources(config_path, verify_banks=True)
        cache = verify_cache(sources)
        masks = verify_background_masks(sources)
        train_audits = []
        for fold in config['folds']:
            ds = DynamicR9Dataset(config_path, fold, sources=sources, verify_banks=False)
            train_folder = output/f'fold_{fold:02d}'
            train_folder.mkdir()
            ds.hard_negative_metadata.to_parquet(train_folder/'train_hard_negative_candidates.parquet', index=False)
            first_sha, second_sha = None, None
            for epoch in range(1,101):
                s = make_schedule(len(ds), epoch, fold, config['seed'], hard_negative_candidates=ds.hard_negative_candidates)
                a = schedule_audit(s)
                if (not a['primary_once'] or a['explicit_no_source_count'] != round(.2*len(ds))
                        or a['unique_references'] != 509 or a['reference_use_max']-a['reference_use_min'] > 1
                        or np.any(s['hard_negative'] & ~s['explicit_no_source'])
                        or a['explicit_hard_negative_count'] != min(round(.1*len(ds)), int(ds.hard_negative_candidates.sum()))):
                    raise AssertionError('Full TRAIN epoch schedule coverage failed')
                if epoch == 1: first_sha = a['schedule_sha256']
                if epoch == 2: second_sha = a['schedule_sha256']
            if first_sha == second_sha:
                raise AssertionError('Different epochs require different schedules')
            train_audits.append({'fold': fold, 'backgrounds_per_epoch': len(ds), 'epochs_scheduled': 100,
                                'references_per_epoch': 509, 'all_primary_once': True,
                                'reference_max_minus_min_use_at_most_one': True,
                                'explicit_no_source_per_epoch': round(.2*len(ds)),
                                'hard_negative_candidate_count': int(ds.hard_negative_candidates.sum()),
                                'explicit_hard_negative_per_epoch': a['explicit_hard_negative_count'],
                                'hard_negative_quota_shortfall_without_duplication': a['hard_negative_shortfall_without_duplicate_primary'],
                                'site_weights_mean': float(ds.loss_weights.mean(dtype=np.float64)),
                                'site_weight_min': float(ds.loss_weights.min()), 'site_weight_max': float(ds.loss_weights.max()),
                                'epoch1_schedule_sha256': first_sha, 'epoch2_schedule_sha256': second_sha,
                                'identity': ds.identity})
        write(output/'source_verification.json', {'cache': cache, 'background_masks': masks, 'train_schedules': train_audits})
        folds = []
        for fold in config['folds']:
            folder = output/f'fold_{fold:02d}'
            folder.mkdir(exist_ok=True)
            roles = {}
            for role, count in config['fixed_examples'].items():
                progress(output, {'phase': 'materialize_fixed', 'fold': fold, 'role': role, 'ready_to_train': False})
                arrays, metadata, recipes, audit = generate_fixed(sources, fold, role, count)
                check = check_arrays(arrays, metadata, recipes)
                # Replay fixed-index examples, including original mask and recipe.
                lookup = sources.role_metadata(fold, role).set_index('base_window_index')
                for i in sorted(set([0, count//2, count-1])):
                    recipe = recipes[i]
                    row = lookup.loc[int(arrays['base_window_index'][i])].to_dict()
                    row['base_window_index'] = int(arrays['base_window_index'][i])
                    replay = _generate(sources, row, fold=fold, epoch=1, index=i,
                              tag=101+ROLES.index(role), reference_index=int(arrays['reference_index'][i]),
                              family_index=config['generator']['family_names'].index(metadata.iloc[i].requested_family),
                              no_source=bool(arrays['explicit_no_source'][i]), role=role)
                    if any(not np.array_equal(replay[k], arrays[k][i], equal_nan=True)
                           for k in ('db', 'unrounded_db', 'share', 'presence', 'cell_share')) or canonical(replay['recipe']) != canonical(recipe):
                        raise AssertionError('Fixed example exact replay failed')
                path = folder/role
                np.savez_compressed(str(path)+'.npz', **arrays)
                metadata.to_parquet(str(path)+'_metadata.parquet', index=False)
                write(str(path)+'_recipes.json', recipes)
                roles[role] = dict(audit, numeric_checks=check, exact_replay_examples=3,
                                  numeric_or_unicode_only=True)
            write(folder/'generation_audit.json', {'fold': fold, 'roles': roles,
                  'report_labels_for_generation_QA_only_no_model_predictions': True})
            folds.append({'fold': fold, 'dynamic_train_rows': train_audits[fold]['backgrounds_per_epoch'], 'roles': roles})
        sources.assert_unchanged(verify_banks=True)
        if sha(config_path) != config_sha or any(sha(ROOT/n) != value for n, value in pins.items()):
            raise AssertionError('Config/data generation code changed during preparation')
        files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
                 if p.is_file() and p.name not in ('preparation_progress.json',)}
        manifest = {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'config_file': config_path.relative_to(ROOT).as_posix(), 'config_sha256': config_sha,
                    'code_sha256': pins, 'source_identity': sources.identity,
                    'background_source_hash': sources.bg_manifest['source_hash'],
                    'reference_source_hash': sources.ref_manifest['source_hash'],
                    'file_sha256': files, 'folds': folds,
                    'fixed_examples_total': sum(config['fixed_examples'].values())*5,
                    'reference_count': 509, 'shared_reference_CV_not_independent': True,
                    'weak_background_archive_references_included': int(sources.references.is_weak_background_label_in_vessel_archive.sum()),
                    'ambient_controls_not_in_TRAIN_reference_pool': 16,
                    'preparation_elapsed_seconds': time.monotonic()-start,
                    'model_fit_or_Test_or_waveform_access': False,
                    'old_sources_arrays_models_modified': False}
        manifest['source_hash'] = digest(manifest)
        write(output/'manifest.json', manifest)
        write(output/'readiness.json', {'version': VERSION, 'ready_to_train': True,
              'manifest_sha256': sha(output/'manifest.json'), 'source_hash': manifest['source_hash'],
              'source_checks_passed': True, 'raw_q_y_identity_passed': True,
              'all_509_TRAIN_references_each_epoch_passed': True,
              'background_group_and_24h_purge_passed': True,
              'shared_reference_CV_not_independent': True,
              'report_access_requires_global_model_and_policy_freeze': True,
              'Test_or_waveform_or_model_fit_accessed': False})
        validate_bundle(output, verify_all=True)
        # These are selection/fit/gate roles only. Report loader must fail closed.
        for fold in config['folds']:
            for role in ROLES[:-1]:
                ds = FixedR9Dataset(output, fold, role)
                if len(ds) != config['fixed_examples'][role]:
                    raise AssertionError('Prepared role count changed')
                ds[0]
            try:
                FixedR9Dataset(output, fold, 'report')
            except ValueError:
                pass
            else:
                raise AssertionError('Report opened without a frozen-policy gate')
        progress(output, {'phase': 'completed', 'ready_to_train': True,
                          'source_hash': manifest['source_hash'], 'elapsed_seconds': time.monotonic()-start})
        return manifest
    except Exception as error:
        # A failed attempt is permanent. Never repair data or silently resume it.
        write(output/'FAILED.json', {'ready_to_train': False, 'error_type': type(error).__name__,
                                   'error': str(error), 'elapsed_seconds': time.monotonic()-start})
        if (output/'readiness.json').exists():
            (output/'readiness.json').unlink()
        progress(output, {'phase': 'failed', 'ready_to_train': False, 'error': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/r9_full_training_v1.json')
    parser.add_argument('--output', help='Fresh destination; default is frozen config data_output')
    args = parser.parse_args()
    result = prepare(args.config, args.output)
    print(json.dumps({'status': 'passed', 'source_hash': result['source_hash'],
                      'fixed_examples': result['fixed_examples_total'],
                      'elapsed_seconds': result['preparation_elapsed_seconds']}))


if __name__ == '__main__':
    main()
