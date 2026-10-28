"""TRAIN-only HF replacement matched on site and exact effective minute mask.

Follow-up control, declared after detecting a mask/class association in R12.
Original v3 and its ongoing fits stay immutable. This script does not fit
models/bases, open validation values, or read Test/locked panel values.
"""
from __future__ import annotations

import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from pipeline.dataset_r12 import sha, digest, R9Sources, synthesize_r12, validate_r12_bundle
from pipeline.source_split_r6 import apply_minute_qc

PARENT = ROOT/'data/synthetic/r12_cpu_ablation_v3'
DEFAULT_OUTPUT = 'data/synthetic/r12_mask_matched_control_v2'
VERSION = 'r12_site_sensor_recorder_mask_matched_HF_control_v2'
SEED = 2026102713


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(json.dumps([a.dtype.str, list(a.shape)]).encode()+a.tobytes()).hexdigest()


def mask_bits(db):
    observed = np.isfinite(db)
    if (np.isinf(db).any() or (observed.any(2) != observed.all(2)).any()):
        raise ValueError('Complete measured/NaN minutes required')
    return (observed.all(2).astype(np.int64)*(1 << np.arange(15))).sum(1)


def fixed_auc(presence, score):
    positive = np.asarray(score)[np.asarray(presence) > 0]
    negative = np.sort(np.asarray(score)[np.asarray(presence) == 0])
    lo = np.searchsorted(negative, positive, side='left')
    hi = np.searchsorted(negative, positive, side='right')
    return float(np.sum(lo+.5*(hi-lo))/(len(positive)*len(negative)))


def count_table(site, bits, presence, indices):
    f = pd.DataFrame({'site': np.asarray(site)[indices], 'effective_mask_bits': bits[indices],
                      'presence': presence[indices]})
    return [{'site': str(s), 'effective_mask_bits': int(b), 'presence': int(y), 'n': int(n)}
            for (s, b, y), n in f.groupby(['site', 'effective_mask_bits', 'presence'], sort=True).size().items()]


def build(output=DEFAULT_OUTPUT):
    out = (ROOT/output).resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        raise FileExistsError('Fresh project-contained control output only')
    out.mkdir(parents=True)
    parent = validate_r12_bundle(PARENT)
    old_entry = parent['arms']['L1_H0']['train']
    h1_entry = parent['arms']['L1_H1']['train']
    paths = [PARENT/'manifest.json', PARENT/'readiness.json', PARENT/old_entry['path'],
        PARENT/old_entry['metadata_path'], PARENT/old_entry['recipes_path'],
        PARENT/h1_entry['metadata_path'], PARENT/'TRAIN_source_features.parquet',
        PARENT/parent['cnn_subset']['path'], Path(__file__), ROOT/'pipeline/dataset_r12.py']
    pins = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    plan = {'format': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'seed': SEED, 'parent_data_hash': parent['data_hash'], 'parent_arm': 'L1_H0',
        'requested_slots': 'original v3 H1 special_zero1200 indexed positions;600 inside fixed CNNsubset',
        'matching': 'exact original site, sensor_group, recorder label and effective QC observed-minute15bit pattern, not merely Nobs or boundary-candidate bits',
        'candidate_law': 'R9 Fold0 TRAIN only, HF18..24 raw-QC bump strictly above site×sensor×recorder×effective-mask stratum TRAIN q90',
        'selection': 'SHA256(seed,stratum,base_window_index) order without replacement; exclude ALL original12000baseIDs',
        'shortfall': 'retain original zero row and record shortage; do not lower q90, duplicate sources or alter labels',
        'invariants': ['all positive common NPZ field bytes identical', 'all-row site/mask/presence identical',
            'overall20percentzero', 'site×mask×presence tables identical in full12000 and CNN6000',
            'mask-only fixed-direction AUC delta exactly0'],
        'basis': 'reuse original TRAIN-only R12 basis and exact original basis_training_binding',
        'followup_not_original_eight_arm_preregistration': True,
        'prefit_recorder_refinement': 'Initial same-site/mask v1 preserved. Site uniquely determines sensor_group here, but RichardsBay/StCroix contain3recorderlabels each; v1 changed449recorderlabels. Add exact recorder and recompute TRAIN-only conditionalq90 before any v2-control-model fit. Partial v1 control fit was stopped by parent and is never reused; this correction uses source metadata, not model outcomes.',
        'model_fit_inference_Test_validation_locked_panels_or_network': False,
        'input_sha256': pins}
    write(out/'predeclared_plan.json', plan)
    try:
        sources = R9Sources()
        frame = pd.read_parquet(PARENT/'TRAIN_source_features.parquet').copy().reset_index(drop=True)
        original_train = sources.role_metadata(0, 'train')
        if not np.array_equal(frame.base_window_index, original_train.base_window_index):
            raise ValueError('TRAIN feature metadata order/source changed')
        effective = []
        for start in range(0, len(frame), 512):
            f = frame.iloc[start:start+512]
            db = apply_minute_qc(np.asarray(sources.background_bank()[f.base_window_index.to_numpy(int)]),
                                 f.boundary_mask_bits.to_numpy(np.int64))
            effective.extend(mask_bits(db).tolist())
        frame['effective_mask_bits'] = effective
        if not np.array_equal(np.asarray([int(x).bit_count() for x in effective]), frame.qc_observed_minutes):
            raise AssertionError('Effective bit mask and source QC counts differ')
        strata = ['site', 'sensor_group', 'recorder', 'effective_mask_bits']
        grouped = frame.groupby(strata, sort=True)
        frame['stratum_HF_q90_db'] = grouped.hf_bump_2_8_db.transform(lambda x: x.quantile(.9))
        frame['matched_HF_candidate'] = frame.hf_bump_2_8_db > frame.stratum_HF_q90_db
        summaries = []
        for (site, sensor, recorder, bits), g in frame.groupby(strata, sort=True):
            summaries.append({'site': str(site), 'effective_mask_bits': int(bits),
                'sensor_group': int(sensor), 'recorder': str(recorder),
                'Nobs': int(bits).bit_count(), 'TRAIN_rows': len(g),
                'q90_db': float(g.stratum_HF_q90_db.iloc[0]),
                'strictly_above_q90_candidates': int(g.matched_HF_candidate.sum())})
        with np.load(PARENT/old_entry['path'], allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        original = {k: v.copy() for k, v in arrays.items()}
        metadata = pd.read_parquet(PARENT/old_entry['metadata_path']).copy()
        h1_meta = pd.read_parquet(PARENT/h1_entry['metadata_path'])
        recipes = json.loads((PARENT/old_entry['recipes_path']).read_text())
        subset = np.load(PARENT/parent['cnn_subset']['path'], allow_pickle=False)
        requested = np.flatnonzero(h1_meta.special_zero.to_numpy(bool))
        if len(requested) != 1200 or np.isin(subset, requested).sum() != 600:
            raise AssertionError('Original special slots/CNN600 membership differ')
        if np.any(arrays['presence'][requested] != 0):
            raise AssertionError('Replacement must never touch a positive example')
        original_bits = mask_bits(arrays['db'])
        lookup = frame.set_index('base_window_index', drop=False)
        if not np.array_equal(original_bits, lookup.loc[arrays['base_window_index'], 'effective_mask_bits']):
            raise AssertionError('Original TRAIN input mask provenance differs')
        blocked = set(map(int, arrays['base_window_index']))
        candidates = frame[frame.matched_HF_candidate & ~frame.base_window_index.isin(blocked)].copy()
        queues = {}
        for (site, sensor, recorder, bits), g in candidates.groupby(strata, sort=True):
            keys = [hashlib.sha256(f'R12MATCH|{SEED}|{site}|{sensor}|{recorder}|{bits}|{int(i)}'.encode()).hexdigest()
                    for i in g.base_window_index]
            queues[(str(site), int(sensor), str(recorder), int(bits))] = g.iloc[np.argsort(keys, kind='stable')].to_dict('records')
        cursors, receipts = {}, []
        metadata['special_zero'] = False
        metadata['matched_control_requested'] = False
        metadata['original_base_window_index'] = arrays['base_window_index']
        for i in requested:
            i = int(i)
            metadata.at[i, 'matched_control_requested'] = True
            site, bits = str(arrays['site'][i]), int(original_bits[i])
            source_record = lookup.loc[int(original['base_window_index'][i])]
            sensor, recorder = int(source_record.sensor_group), str(source_record.recorder)
            key = (site, sensor, recorder, bits); cursor = cursors.get(key, 0); pool = queues.get(key, [])
            if cursor >= len(pool):
                receipts.append({'row_index': i, 'replaced': False, 'reason': 'same_site_exact_mask_q90_pool_shortfall',
                    'site': site, 'effective_mask_bits': bits,
                    'sensor_group': sensor, 'recorder': recorder,
                    'original_base_window_index': int(original['base_window_index'][i])})
                continue
            candidate = pool[cursor]; cursors[key] = cursor+1
            db = sources.background(candidate)
            item, receipt = synthesize_r12(db, None, recipes[i]['seed'], 'no_source', no_source=True)
            for k in ('db', 'unrounded_db', 'share', 'presence', 'cell_share'):
                arrays[k][i] = item[k]
            new_index = int(candidate['base_window_index'])
            arrays['base_window_index'][i] = new_index
            arrays['source_id'][i] = 'SA:'+str(new_index)
            arrays['background_group'][i] = str(candidate['background_group'])
            # Site, exact effective mask, row order and sampleID stay unchanged.
            for k in ('source_id', 'base_window_index', 'background_group'):
                metadata.at[i, k] = arrays[k][i]
            for k in ('sensor_group', 'start_export_clock', 'qc_observed_minutes', 'boundary_mask_bits'):
                metadata.at[i, k] = candidate[k]
            metadata.at[i, 'special_zero'] = True
            matched = {'row_index': i, 'replaced': True, 'site': site,
                'sensor_group': sensor, 'recorder': recorder,
                'effective_mask_bits': bits, 'Nobs': bits.bit_count(),
                'original_base_window_index': int(original['base_window_index'][i]),
                'new_base_window_index': new_index,
                'stratum_HF_q90_db': float(candidate['stratum_HF_q90_db']),
                'new_HF_bump_db': float(candidate['hf_bump_2_8_db'])}
            receipts.append(matched)
            recipes[i] = {**recipes[i], **receipt, 'background_index': new_index,
                'source_id': arrays['source_id'][i], 'boundary_mask_bits': int(candidate['boundary_mask_bits']),
                'r12_site_mask_matched_followup': matched}
        positive = arrays['presence'] > 0
        positive_byte_hashes = {}
        for k in original:
            old_sha, new_sha = array_sha(original[k][positive]), array_sha(arrays[k][positive])
            if old_sha != new_sha:
                raise AssertionError('Positive field changed: '+k)
            positive_byte_hashes[k] = old_sha
        new_bits = mask_bits(arrays['db'])
        if (not np.array_equal(new_bits, original_bits)
                or not np.array_equal(arrays['site'], original['site'])
                or not np.array_equal(arrays['presence'], original['presence'])
                or not np.array_equal(arrays['share'], original['share'])
                or not np.array_equal(arrays['cell_share'], original['cell_share'])
                or len(set(arrays['base_window_index'])) != len(arrays['share'])):
            raise AssertionError('Matched site/mask/target/source uniqueness failed')
        for column in ('sensor_group', 'recorder'):
            if not np.array_equal(lookup.loc[arrays['base_window_index'], column].to_numpy(),
                                  lookup.loc[original['base_window_index'], column].to_numpy()):
                raise AssertionError('All-row source sensor/recorder provenance changed')
        if (np.count_nonzero(arrays['share'] == 0) != 2400
                or np.any(arrays['cell_share'][~positive] != 0)):
            raise AssertionError('Physical known A=0 target changed')
        qas = {}
        for role, indices in [('TRAIN12000', np.arange(12000)), ('CNN6000', subset)]:
            a = count_table(original['site'], original_bits, original['presence'], indices)
            b = count_table(arrays['site'], new_bits, arrays['presence'], indices)
            if a != b:
                raise AssertionError('Conditional class table changed: '+role)
            before_auc = fixed_auc(original['presence'][indices], -np.asarray([int(x).bit_count() for x in original_bits[indices]]))
            after_auc = fixed_auc(arrays['presence'][indices], -np.asarray([int(x).bit_count() for x in new_bits[indices]]))
            if before_auc != after_auc:
                raise AssertionError('Mask-only AUC changed')
            qas[role] = {'rows': len(indices), 'zero_count': int((arrays['presence'][indices] == 0).sum()),
                'site_effective_mask_presence_table_before_after_identical': True,
                'conditional_table_sha256': digest(a), 'conditional_table': b,
                'AUC_negative_Nobs_before': before_auc, 'AUC_negative_Nobs_after': after_auc,
                'AUC_delta': after_auc-before_auc,
                'AUC_positive_Nobs_before': 1-before_auc, 'AUC_positive_Nobs_after': 1-after_auc}
        replaced = np.flatnonzero(metadata.special_zero.to_numpy(bool))
        np.savez_compressed(out/'train.npz', **arrays)
        metadata.to_parquet(out/'train_metadata.parquet', index=False)
        write(out/'train_recipes.json', recipes)
        candidates.to_parquet(out/'TRAIN_matched_candidate_metadata.parquet', index=False)
        write(out/'stratum_HF_thresholds.json', summaries)
        write(out/'replacement_receipts.json', receipts)
        # Preserve the exact original row-index file, not a new subset draw.
        with (out/'cnn_subset_indices.npy').open('xb') as f:
            f.write((PARENT/parent['cnn_subset']['path']).read_bytes())
        with np.load(out/'train.npz', allow_pickle=False) as z:
            for k in arrays:
                if z[k].dtype.kind == 'O' or not np.array_equal(z[k], arrays[k], equal_nan=z[k].dtype.kind == 'f'):
                    raise AssertionError('No-pickle saved-array parity failed')
        audit = {'status': 'passed', 'requested_replacements': 1200,
            'actual_replacements': len(replaced), 'shortfall': 1200-len(replaced),
            'requested_CNN_replacements': 600, 'actual_CNN_replacements': int(np.isin(subset, replaced).sum()),
            'positive_common_field_bytes_unchanged': True, 'positive_field_sha256': positive_byte_hashes,
            'all_rows_site_and_exact_mask_and_targets_unchanged': True,
            'all_rows_sensor_and_recorder_unchanged': True,
            'source_primary_unique_count': len(set(arrays['base_window_index'])),
            'new_replacements_overlap_original_baseIDs': 0,
            'known_zero_energy_q_and_share_exact': True, 'before_after': qas,
            'HF_candidate_truth': 'high natural HF variation only, not naturally ship-absent truth',
            'remaining_limit': 'site/mask class ratios are controlled, but new background texture/source-group distributions can differ; this is a follow-up TRAIN control, not new independent validation'}
        write(out/'audit.json', audit)
        sources.assert_unchanged()
        for relative, h in pins.items():
            if sha(ROOT/relative) != h:
                raise AssertionError('Original pinned source changed: '+relative)
        manifest = {'version': VERSION, 'status': 'completed', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'seed': SEED, 'input_sha256': pins, 'parent_manifest_file': (PARENT/'manifest.json').relative_to(ROOT).as_posix(),
            'parent_data_hash': parent['data_hash'], 'source_identity': sources.identity,
            'basis_training_binding': parent['basis_training_binding'],
            'train': {'path': 'train.npz', 'sha256': sha(out/'train.npz'),
                'metadata_path': 'train_metadata.parquet', 'metadata_sha256': sha(out/'train_metadata.parquet'),
                'recipes_path': 'train_recipes.json', 'recipes_sha256': sha(out/'train_recipes.json')},
            'cnn_subset': {'path': 'cnn_subset_indices.npy', 'sha256': sha(out/'cnn_subset_indices.npy'), 'examples': 6000},
            'validation_fit_unchanged_reference': {'project_path': (PARENT/parent['common_validation']['validation_fit']['path']).relative_to(ROOT).as_posix(),
                'sha256': parent['common_validation']['validation_fit']['sha256'], 'values_opened': False},
            'audit': audit, 'original_v3_and_R9_sources_unchanged': True,
            'new_basis_or_model_fit_inference_Test_validation_locked_access': False,
            'file_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
        manifest['data_hash'] = digest(manifest)
        write(out/'manifest.json', manifest)
        write(out/'readiness.json', {'version': VERSION, 'ready_to_train': True,
            'manifest_sha256': sha(out/'manifest.json'), 'data_hash': manifest['data_hash'],
            'source_checks_passed': True, 'training_executed': False})
        return manifest
    except BaseException as exc:
        write(out/'FAILED.json', {'status': 'failed_preserved', 'error': repr(exc),
            'ready_to_train': False, 'partial_reuse': False})
        raise


def load_control(folder=ROOT/DEFAULT_OUTPUT):
    """Restore the new TRAIN only; validation_fit remains in immutable v3."""
    folder = Path(folder).resolve()
    m = json.loads((folder/'manifest.json').read_text())
    r = json.loads((folder/'readiness.json').read_text())
    if (m.get('version') != VERSION or r.get('ready_to_train') is not True
            or sha(folder/'manifest.json') != r['manifest_sha256']
            or digest({k: v for k, v in m.items() if k != 'data_hash'}) != m['data_hash']
            or r['data_hash'] != m['data_hash']):
        raise ValueError('Matched control readiness/hash invalid')
    for name, h in m['file_sha256'].items():
        if sha(folder/name) != h: raise ValueError('Control artifact changed: '+name)
    for name, h in m['input_sha256'].items():
        if sha(ROOT/name) != h: raise ValueError('Control original source changed: '+name)
    with np.load(folder/m['train']['path'], allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, m


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    a = p.parse_args()
    result = build(a.output)
    print(json.dumps({'status': 'completed', 'output': a.output, 'data_hash': result['data_hash'],
        'actual_replacements': result['audit']['actual_replacements'],
        'actual_CNN_replacements': result['audit']['actual_CNN_replacements'],
        'audit': result['audit']['before_after']}, ensure_ascii=False))
