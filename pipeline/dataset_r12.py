"""Prospective R12 CPU data ablation, separate from immutable R9/R11.

Known added *whole recording* energy is the target. Natural vessel absence,
source-independent reference CV and hidden competition-generator parity are
not asserted. Positive placement is explicitly observation-support conditioned.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.dataset_r9 import R9Sources, array_sha, canonical, digest, sha
from pipeline.sampler_r9 import energy_weights, smooth_eq
from pipeline.relative_grid_r6 import quantization_pooling_bound
from pipeline.relative_grid_r6_v2 import relative_grid_v2
from pipeline.source_split_r6 import apply_minute_qc

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'r12_cpu_data_ablation_v1'
ARMS = ('L0_H0', 'L0_H1', 'L1_H0', 'L1_H1')
VAL_ROLES = ('validation_fit', 'validation_gate', 'eval')
CODE = ('pipeline/dataset_r12.py', 'ops/build_r12_cpu_data.py')


@dataclass(frozen=True)
class R12DataConfig:
    seed: int = 2026102712
    fold: int = 0
    train_examples: int = 12000
    cnn_subset_examples: int = 6000
    validation_fit_examples: int = 1000
    validation_gate_examples: int = 500
    eval_examples: int = 500
    basis_backgrounds: int = 2000
    explicit_zero_fraction: float = .2
    train_long_positive_fraction: float = .3
    validation_long_positive_fraction: float = .5
    native_long_fraction: float = .25
    special_zero_fraction: float = .5
    eq_max_db: float = 6.
    snr_db: tuple = (-10., 10.)
    high_snr_long_examples: int = 500
    high_snr_db: tuple = (10., 20.)

    def validate(self):
        if (self.fold != 0 or self.train_examples != 12000 or self.cnn_subset_examples != 6000
                or (self.validation_fit_examples, self.validation_gate_examples,
                    self.eval_examples) != (1000, 500, 500)
                or self.basis_backgrounds != 2000
                or self.explicit_zero_fraction != .2 or self.eq_max_db != 6.
                or self.snr_db != (-10., 10.)
                or self.train_long_positive_fraction != .3
                or self.validation_long_positive_fraction != .5
                or self.native_long_fraction != .25
                or self.special_zero_fraction != .5):
            raise ValueError('Use the predeclared R12 data law, not outcome tuning')
        return self


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def rank(seed, tag, value):
    return hashlib.sha256(f'R12|{seed}|{tag}|{value}'.encode()).hexdigest()


def choose(frame, count, seed, tag):
    if count > len(frame):
        raise ValueError(f'Unique source shortage: {tag}, need {count}, have {len(frame)}')
    keys = [rank(seed, tag, int(i)) for i in frame.base_window_index]
    return frame.iloc[np.argsort(keys, kind='stable')[:count]].copy().reset_index(drop=True)


def hf_bump(db):
    """TRAIN-defined 1995..7943 Hz linear band-sum max-minus-median dB."""
    x = np.asarray(db, np.float64)
    o = np.isfinite(x).all(2)
    v = x[:, :, 18:25]
    top = np.max(np.where(o[:, :, None], v, 0.), axis=2)
    shifted = np.where(o[:, :, None], v-top[:, :, None], 0.)
    broad = top + 10*np.log10(np.power(10., shifted/10).sum(2))
    broad[~o] = np.nan
    return np.nanmax(broad, axis=1)-np.nanmedian(broad, axis=1)


def source_features(sources, frame):
    values = []
    for begin in range(0, len(frame), 512):
        f = frame.iloc[begin:begin+512]
        db = apply_minute_qc(np.asarray(sources.background_bank()[f.base_window_index.to_numpy(int)]),
                             f.boundary_mask_bits.to_numpy(np.int64))
        values.extend(hf_bump(db).tolist())
    out = frame.copy()
    out['hf_bump_2_8_db'] = values
    return out


def r11_used_sources(root):
    used, groups, witnesses = set(), set(), {}
    # Existing diagnostic metadata only: no prediction or Test/report arrays.
    for p in sorted((root/'reports/r11_diagnostics').rglob('*.parquet')):
        f = pd.read_parquet(p)
        if 'base_window_index' not in f:
            continue
        used.update(map(int, f.base_window_index))
        if 'background_group' in f:
            groups.update(map(str, f.background_group))
        witnesses[p.relative_to(root).as_posix()] = sha(p)
    # R11-3 also audited the already saved R9 selection1000 predictions.
    # Their source metadata live outside the R11 directory and must not be
    # accidentally advertised as new diagnostic backgrounds.
    p = root/'data/synthetic/group_cv_minute_full_r9_v1/fold_00/selection_metadata.parquet'
    f = pd.read_parquet(p)
    used.update(map(int, f.base_window_index))
    groups.update(map(str, f.background_group))
    witnesses[p.relative_to(root).as_posix()] = sha(p)
    return used, groups, witnesses


def _nearest_seconds(values, other):
    other = np.sort(other)
    if not len(other):
        return np.full(len(values), np.inf)
    p = np.searchsorted(other, values)
    return np.minimum(abs(values-other[np.clip(p-1, 0, len(other)-1)]),
                      abs(values-other[np.clip(p, 0, len(other)-1)]))


def minimum_role_gap(frame, role_column='r12_role'):
    t = pd.to_datetime(frame.start_export_clock).astype('int64').to_numpy()/1e9
    gap = np.full(len(frame), np.inf)
    labels = frame[role_column].to_numpy()
    for ix in frame.groupby('sensor_group', sort=False).indices.values():
        ix = np.asarray(ix)
        for role in sorted(set(labels[ix])):
            a, b = ix[labels[ix] == role], ix[labels[ix] != role]
            gap[a] = np.maximum(0., _nearest_seconds(t[a], t[b])-900)/3600
    return gap


def partition_validation(frame, train, seed):
    """Whole selection source groups, then eval>gate>fit 24h support purge."""
    f = frame.copy().reset_index(drop=True)
    sizes = f.groupby('background_group').size().to_dict()
    groups = sorted(sizes, key=lambda g: rank(seed, 'validation-group', g))
    targets = np.asarray([.5, .25, .25])*len(f)
    counts = np.zeros(3)
    assignment = {}
    for g in groups:
        role = int(np.argmax((targets-counts)/targets))
        assignment[g] = VAL_ROLES[role]
        counts[role] += sizes[g]
    f['r12_role'] = f.background_group.map(assignment)
    times = pd.to_datetime(f.start_export_clock).astype('int64').to_numpy()/1e9
    labels = f.r12_role.to_numpy()
    keep = np.ones(len(f), bool)
    priority = {'validation_fit': ('validation_gate', 'eval'),
                'validation_gate': ('eval',), 'eval': ()}
    for ix in f.groupby('sensor_group', sort=False).indices.values():
        ix = np.asarray(ix)
        for role, higher in priority.items():
            a = ix[labels[ix] == role]
            b = ix[np.isin(labels[ix], higher)]
            if len(a) and len(b):
                keep[a] &= _nearest_seconds(times[a], times[b])-900 >= 24*3600
    excluded = f[~keep].copy()
    admitted = f[keep].copy().reset_index(drop=True)
    admitted['r12_cross_role_gap_hours'] = minimum_role_gap(admitted)
    if (admitted.r12_cross_role_gap_hours < 24-1e-9).any():
        raise AssertionError('Validation internal 24h support purge failed')
    if set(train.base_window_index) & set(admitted.base_window_index):
        raise AssertionError('TRAIN/validation source ID overlap')
    if set(train.background_group) & set(admitted.background_group):
        raise AssertionError('TRAIN/validation source group overlap')
    combined = pd.concat([train.assign(r12_role='train'), admitted], ignore_index=True)
    all_gaps = minimum_role_gap(combined)
    if (all_gaps < 24-1e-9).any():
        raise AssertionError('TRAIN/validation 900s+24h support separation failed')
    stats = {r: {'source_windows': int((admitted.r12_role == r).sum()),
                 'source_groups': int(admitted[admitted.r12_role == r].background_group.nunique())}
             for r in VAL_ROLES}
    return admitted, excluded, {'roles': stats, 'purged_windows': len(excluded),
        'all_active_roles_minimum_support_gap_hours': float(np.min(all_gaps)),
        'TRAIN_validation_source_ID_overlap': 0, 'TRAIN_validation_group_overlap': 0,
        'internal_validation_group_overlap': 0,
        'clock_basis': 'original export clock, not independently verified UTC'}


def reference_audit(sources):
    f = sources.references.copy()
    begin, end = pd.to_datetime(f.utc_begin, utc=True), pd.to_datetime(f.utc_end, utc=True)
    pairs = []
    for i, row in f.iterrows():
        js = f.index[f.hull_id.eq(row.hull_id) & begin.eq(end.iloc[i])]
        for j in js:
            if i != j:
                pairs.append({'first_reference_index': int(i), 'second_reference_index': int(j),
                    'hull_id': str(row.hull_id), 'combined_native_minutes': int(
                        row.native_complete_minute_count+f.iloc[j].native_complete_minute_count)})
    minutes = np.asarray([len(p) for p in sources.reference_minutes])
    return {'native_ge15_indices': np.flatnonzero(minutes >= 15).tolist(),
        'native_5_to14_indices': np.flatnonzero((minutes >= 5) & (minutes <= 14)).tolist(),
        'same_hull_exact_UTC_adjacency_pairs': pairs, 'joined_references_used': False,
        'join_exclusion_reason': 'No per-clip common recorder/hydrophone acquisition identity; adjacent AIS/UTC cuts alone do not establish continuous acquisition. No fabricated continuity.',
        'all_509_shared_reference_pool': True, 'reference_independent_validation': False}


def _cycle(indices, n, rng):
    return np.resize(rng.permutation(np.asarray(indices, np.int64)), n)


def schedule(n, seed, tag, long_fraction, ref_info):
    if n % 5:
        raise ValueError('Exact20percent no-source requires N divisible by5')
    rng = np.random.default_rng(np.random.SeedSequence([seed, tag, 11]))
    zero = np.zeros(n, bool)
    zero[rng.permutation(n)[:n//5]] = True
    positive = np.flatnonzero(~zero)
    refs = np.full(n, -1, np.int64)
    refs[positive] = _cycle(np.arange(509), len(positive), rng)
    family = rng.choice(np.asarray(['native_inside', 'native_short', 'native_edge']), n,
                        p=[.5, .25, .25]).astype('U32')
    family[zero] = 'no_source'
    long_rows = rng.permutation(positive)[:round(long_fraction*len(positive))]
    native_n = round(.25*len(long_rows))
    native, smooth = long_rows[:native_n], long_rows[native_n:]
    refs[native] = _cycle(ref_info['native_ge15_indices'], len(native), rng)
    refs[smooth] = _cycle(ref_info['native_5_to14_indices'], len(smooth), rng)
    family[native], family[smooth] = 'long_native', 'long_smooth_bell'
    # Store seeds rather than consuming variable amounts of RNG during placement.
    return {'zero': zero, 'reference_index': refs, 'family': family,
            'item_seed': np.asarray([int(x) for x in rng.integers(0, 2**63-1, n)], np.int64)}


def _support_starts(length, observed, edge=False):
    starts = np.arange(-length+1, 15) if edge else np.arange(16-length)
    if edge:
        starts = starts[(starts < 0) | (starts+length > 15)]
    valid = []
    for start in starts:
        a, b = max(0, int(start)), min(15, int(start)+length)
        if b > a and observed[a:b].any():
            valid.append(int(start))
    return valid


def supported_placement(native, rng, family, observed):
    """Enumerate finite geometry; keep reference/EQ/SNR, never retry labels."""
    n = len(native)
    fallback = None
    crop = 0
    requested = family
    if family == 'native_short':
        length = int(rng.integers(1, min(3, n)+1))
        crop = int(rng.integers(n-length+1))
        part = native[crop:crop+length]
    elif family in ('native_inside', 'native_edge'):
        length = n if family == 'native_edge' else min(n, 15)
        if family == 'native_inside':
            crop = int(rng.integers(n-length+1))
        part = native[crop:crop+length]
    elif family == 'long_native':
        if n < 15:
            raise ValueError('Real >=15 native minutes required')
        length = int(rng.integers(12, 16))
        crop = int(rng.integers(n-length+1))
        part = native[crop:crop+length]
    elif family == 'long_smooth_bell':
        if not 5 <= n <= 14:
            raise ValueError('Only actual5..14minute references can enter this statistical arm')
        length = int(rng.integers(12, 16))
        coords = np.linspace(0, n-1, length)
        part = np.stack([np.interp(coords, np.arange(n), native[:, j]) for j in range(25)], axis=1)
        part *= (.6+.4*np.sin(np.pi*(np.arange(length)+.5)/length)**2)[:, None]
    else:
        raise ValueError('Unknown family')
    starts = _support_starts(len(part), observed, edge=(family == 'native_edge'))
    if not starts:
        if family != 'native_edge':
            raise AssertionError('Inside placement must reach at least one observed minute')
        # A short passage cannot reach a middle observed block while straddling
        # a window edge. The same real reference is relocated, never relabelled.
        family = 'native_inside'
        fallback = 'edge_has_no_observed_support_same_reference_inside'
        length = min(15, n)
        crop = int(rng.integers(n-length+1))
        part = native[crop:crop+length]
        starts = _support_starts(length, observed)
    start = int(rng.choice(starts))
    placed = np.zeros((15, 25), np.float64)
    left, right = max(0, start), min(15, start+len(part))
    placed[left:right] = part[left-start:right-start]
    if not placed[observed].sum() > 0:
        raise AssertionError('Positive support failed; no retry/relabel is permitted')
    # For native families alpha uses the complete native reference peak as R9;
    # for long augmentation it uses the declared newly constructed trajectory.
    full_peak = float((part if requested.startswith('long_') else native).sum(1).max())
    return placed, full_peak, {'requested_family': requested, 'effective_family': family,
        'native_minutes': n, 'rendered_trajectory_minutes': len(part),
        'crop_begin_minute': crop, 'window_start_minute': start,
        'valid_placement_candidates': len(starts), 'same_reference_fallback': fallback,
        'mask_conditioned_placement': True,
        'statistical_time_resampling_not_new_WAV': requested == 'long_smooth_bell',
        'joined_recordings': False,
        'reference_peak_scope': 'full EQ native trajectory before crop/mask' if not requested.startswith('long_')
             else 'constructed EQ long trajectory before placement/mask'}


def synthesize_r12(background_db, native, seed, family, *, no_source=False,
                   snr_range=(-10., 10.)):
    x = np.asarray(background_db, np.float64)
    observed = np.isfinite(x)
    if (x.shape != (15, 25) or np.isinf(x).any()
            or (observed.any(1) != observed.all(1)).any() or observed.all(1).sum() < 3):
        raise ValueError('Three complete actual background minutes required')
    top = float(np.nanmax(x))
    b = np.where(observed, 10**((x-top)/10), 0.)
    bmedian = float(np.median(b.sum(1)[observed.all(1)]))
    rng = np.random.default_rng(int(seed))
    eq = smooth_eq(rng, 6.)
    level = float(rng.uniform(*snr_range))
    alpha, full_peak, route = 0., None, {}
    added = np.zeros_like(b)
    if not no_source:
        transformed = np.asarray(native, np.float64)*10**(eq[None, :]/10)
        placed, full_peak, route = supported_placement(transformed, rng, family, observed.all(1))
        alpha = float(10**(level/10)*bmedian/full_peak)
        added = np.where(observed, alpha*placed, 0.)
    e = b+added
    q = np.divide(added, e, out=np.zeros_like(e), where=observed)
    y = float(added.sum()/e.sum())
    raw = np.full_like(x, np.nan)
    raw[observed] = 10*np.log10(e[observed])+top
    rounded = np.round(raw, 1).astype(np.float32)
    raw_error = abs(float((energy_weights(raw)*q).sum())-y)
    gap = abs(float((energy_weights(rounded)*q).sum())-y)
    bound = quantization_pooling_bound(.05+float(np.max(abs(np.spacing(rounded[observed]))))/2+1e-12)
    if (raw_error > 1e-11 or gap > bound+2e-7
            or bool(y > 0) != (not no_source)
            or not np.array_equal(np.isfinite(rounded), observed)):
        raise AssertionError('Known physical mixture/positive support/mask/rounding failed')
    return {'db': rounded, 'unrounded_db': raw, 'cell_share': q.astype(np.float32),
        'share': np.float32(y), 'presence': np.int64(y > 0)}, {
        'version': VERSION, 'no_source': bool(no_source), 'seed': int(seed),
        'family': 'no_source' if no_source else family, 'route': route,
        'reference_eq_db': eq.tolist() if not no_source else [0.]*25,
        'requested_full_peak_snr_db': None if no_source else level,
        'alpha': alpha, 'full_reference_peak_after_EQ': full_peak,
        'background_median_observed_minute_power': bmedian,
        'observed_added_minutes': int((added.sum(1) > 0).sum()),
        'observed_minutes': int(observed.all(1).sum()),
        'raw_label_identity_error': raw_error, 'rounded_pool_gap': gap,
        'rounding_bound': bound,
        'interpretation': 'known added whole measured recording energy, not naturally ship-absent background or isolated ship truth'}


def _special_backgrounds(base, pool, zero, seed, tag):
    """Replace half zero slots, without duplicates or positive-row changes."""
    zero_rows = np.flatnonzero(zero)
    slots = np.random.default_rng(np.random.SeedSequence([seed, tag, 98])).permutation(zero_rows)[:len(zero_rows)//2]
    # Excluding ALL original primaries also handles a finite-candidate
    # shortfall: an unfilled slot cannot duplicate a replacement elsewhere.
    blocked = set(base.base_window_index)
    candidates = pool[pool.r12_special_zero_candidate & ~pool.base_window_index.isin(blocked)]
    sf = candidates[candidates.site.str.contains('francis', case=False)]
    first = choose(sf, min(len(slots)//2, len(sf)), seed, f'{tag}-SF')
    remain = candidates[~candidates.base_window_index.isin(first.base_window_index)]
    high = remain[remain.hf_above_TRAIN_q90]
    second = choose(high, min(len(slots)-len(first), len(high)), seed, f'{tag}-HF')
    selected = pd.concat([first, second], ignore_index=True)
    # Fill only with other declared dense SF/HF candidates; never invent one.
    remaining = candidates[~candidates.base_window_index.isin(selected.base_window_index)]
    extra = choose(remaining, min(len(slots)-len(selected), len(remaining)), seed, f'{tag}-candidate')
    selected = pd.concat([selected, extra], ignore_index=True)
    out = base.copy()
    out['r12_special_zero_selected'] = False
    for slot, (_, row) in zip(slots, selected.iterrows()):
        for col in base.columns:
            out.at[int(slot), col] = row[col]
        out.at[int(slot), 'r12_special_zero_selected'] = True
    if out.base_window_index.duplicated().any():
        raise AssertionError('HF-zero replacement duplicated a primary source')
    return out, {'requested_special_zeros': len(slots), 'actual_special_zeros': len(selected),
        'shortfall_without_duplication': len(slots)-len(selected),
        'selected_SF_dense': int(selected.site.str.contains('francis', case=False).sum()),
        'selected_above_TRAIN_HF_q90': int(selected.hf_above_TRAIN_q90.sum())}


def materialize(sources, frame, spec, role, *, snr_range=(-10., 10.)):
    arrays, recipes, meta = [], [], []
    for i, row in frame.iterrows():
        ref, zero, family = int(spec['reference_index'][i]), bool(spec['zero'][i]), str(spec['family'][i])
        item, recipe = synthesize_r12(sources.background(row), None if zero else sources.reference_minutes[ref],
                                      int(spec['item_seed'][i]), family, no_source=zero, snr_range=snr_range)
        source_id = f'SA:{int(row.base_window_index)}'
        item.update(site=str(row.site), source_id=source_id,
            sample_id=f'r12_{role}_{i:06d}', family='no_source' if zero else family,
            base_window_index=np.int64(row.base_window_index), background_group=str(row.background_group),
            reference_index=np.int64(ref), explicit_no_source=np.bool_(zero))
        recipe.update(reference_index=ref, reference_clip_id=None if zero else str(sources.references.iloc[ref].clip_id),
            source_id=source_id, sample_id=item['sample_id'], role=role,
            background_index=int(row.base_window_index), boundary_mask_bits=int(row.boundary_mask_bits))
        arrays.append(item)
        recipes.append(recipe)
        meta.append({'sample_id': item['sample_id'], 'source_id': source_id,
            'base_window_index': int(row.base_window_index), 'background_group': str(row.background_group),
            'site': str(row.site), 'sensor_group': int(row.sensor_group),
            'start_export_clock': str(row.start_export_clock), 'qc_observed_minutes': int(row.qc_observed_minutes),
            'boundary_mask_bits': int(row.boundary_mask_bits), 'family': item['family'],
            'reference_index': ref, 'explicit_no_source': zero,
            'special_zero': bool(row.get('r12_special_zero_selected', False)),
            'share': float(item['share']), 'presence': int(item['presence'])})
    keys = arrays[0].keys()
    result = {k: np.asarray([a[k] for a in arrays]) for k in keys}
    if any(a.dtype.kind == 'O' for a in result.values()):
        raise AssertionError('No object/pickle arrays')
    return result, pd.DataFrame(meta), recipes


def profile(arrays, recipes):
    n = len(arrays['share']); zero = arrays['explicit_no_source']; y = arrays['share'].astype(float)
    if (int(zero.sum())*5 != n or np.any(y[~zero] <= 0) or np.any(y[zero] != 0)
            or not np.array_equal(arrays['presence'], (~zero).astype(np.int64))
            or len(set(arrays['sample_id'])) != n or len(set(arrays['base_window_index'])) != n):
        raise AssertionError('Exact zero fraction/unique primary/positive support failed')
    o = np.isfinite(arrays['db'])
    if (np.isinf(arrays['db']).any() or (o.any(2) != o.all(2)).any()
            or not np.isfinite(arrays['cell_share']).all()
            or np.any(arrays['cell_share'][~o] != 0)):
        raise AssertionError('Invalid mask or physical target')
    return {'examples': n, 'unique_backgrounds': len(set(arrays['base_window_index'])),
        'source_groups': len(set(arrays['background_group'])), 'no_source_count': int(zero.sum()),
        'positive_count': int((~zero).sum()), 'target_mean': float(y.mean()),
        'target_RMS': float(np.sqrt(np.mean(y*y))), 'target_min_max': [float(y.min()), float(y.max())],
        'site_counts': {str(k): int(v) for k, v in pd.Series(arrays['site']).value_counts().items()},
        'family_counts': {str(k): int(v) for k, v in pd.Series(arrays['family']).value_counts().items()},
        'unique_positive_references': len(set(arrays['reference_index'][~zero])),
        'same_reference_edge_fallback_count': sum(bool(r['route'].get('same_reference_fallback')) for r in recipes),
        'raw_identity_error_max': max(r['raw_label_identity_error'] for r in recipes),
        'rounded_pool_gap_max': max(r['rounded_pool_gap'] for r in recipes)}


def _save_dataset(out, stem, arrays, metadata, recipes, files):
    paths = {'path': stem+'.npz', 'metadata_path': stem+'_metadata.parquet', 'recipes_path': stem+'_recipes.json'}
    np.savez_compressed(out/paths['path'], **arrays)
    metadata.to_parquet(out/paths['metadata_path'], index=False)
    write_new(out/paths['recipes_path'], recipes)
    # An actual no-pickle reload guards against Unicode/object serialization errors.
    with np.load(out/paths['path'], allow_pickle=False) as z:
        for k in arrays:
            if z[k].dtype.kind == 'O' or not np.array_equal(z[k], arrays[k], equal_nan=z[k].dtype.kind == 'f'):
                raise AssertionError('Serialized data round-trip failed')
    for name in paths.values():
        files[name] = sha(out/name)
    return paths | {'sha256': files[paths['path']], 'metadata_sha256': files[paths['metadata_path']],
                    'recipes_sha256': files[paths['recipes_path']], 'profile': profile(arrays, recipes)}


def _basis_frame(train, n, seed):
    groups = {str(site): g for site, g in train.groupby('site', sort=True)}
    quota = {s: 0 for s in groups}
    for _ in range(n):
        eligible = [s for s in groups if quota[s] < len(groups[s])]
        s = min(eligible, key=lambda x: (quota[x], x))
        quota[s] += 1
    return pd.concat([choose(groups[s], quota[s], seed, 'PCA-background-'+s) for s in groups],
                     ignore_index=True), quota


def build_r12(output='data/synthetic/r12_cpu_ablation_v3', config=None, project_root=ROOT):
    c = (config or R12DataConfig()).validate()
    root = Path(project_root).resolve(); out = (root/output).resolve()
    if not out.is_relative_to(root) or out.exists():
        raise FileExistsError('A fresh project-contained R12 output is mandatory')
    out.mkdir(parents=True)
    plan = {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'configuration': asdict(c), 'arms': list(ARMS), 'base_source': 'immutable R9 Fold0 TRAIN/selection',
        'long_law': '30percent TRAIN positives;25percent of long native>=15,75percent real5..14min resampled12..15 bell(.6+.4sin^2); no unverified join',
        'validation_law': 'positive standard/long50:50; exact20percentzeros; half-zero HF/SFdense if available; TRAINthreshold frozen',
        'HF_law': 'replace half explicit zero slots with actual TRAIN Nobs>=13 SFdense or HF18..24bump>TRAINdenseq90; not natural ship-absence labels',
        'positive_support': 'finite candidate geometry conditioned on observed minutes; unsupported edge falls back to inside same reference. No seed retries or label mutation.',
        'source_split': 'R9 background TRAIN vs selection with existing900s+24h purge; validation groups further split then24h support purged',
        'R11_exclusion': 'exclude all existing R11 source baseIDs; group overlap with R11 is disclosed, not blind',
        'CNN_subset': {'seed_sequence': [c.seed, 6000],
            'law': 'same6000indices in all4arms:1200zeros(half specialHF slots),4800positive(30percent long-designated slots). Tree uses full12000.',
            'selection': 'independent NumPy seeded permutations within predeclared zero/HF/long metadata strata, then one seeded shuffle; no model output or validation metric'},
        'high_snr_panel': 'additional500 unused eval-role sources if available, allpositive long;SNRUniform[10,20], explicitzero20percent; report-only challenge',
        'no_Test_report_values_model_predictions_or_fitting_used': True,
        'code_sha256': {n: sha(root/n) for n in CODE}}
    write_new(out/'predeclared_plan.json', plan)
    try:
        sources = R9Sources(project_root=root)
        train = sources.role_metadata(c.fold, 'train')
        selection = sources.role_metadata(c.fold, 'selection')
        used, usedgroups, witnesses = r11_used_sources(root)
        available = selection[~selection.base_window_index.isin(used)].copy().reset_index(drop=True)
        valpool, purged, split_stats = partition_validation(available, train, c.seed)
        ref_info = reference_audit(sources)
        if len(ref_info['native_ge15_indices']) != 7 or len(ref_info['native_5_to14_indices']) != 106:
            raise ValueError('Audited native reference support changed')
        train = source_features(sources, train)
        dense = train.qc_observed_minutes.ge(13)
        threshold = float(np.quantile(train.loc[dense, 'hf_bump_2_8_db'], .9))
        def flags(frame):
            f = frame.copy(); f['hf_above_TRAIN_q90'] = f.qc_observed_minutes.ge(13) & f.hf_bump_2_8_db.gt(threshold)
            f['r12_special_zero_candidate'] = f.qc_observed_minutes.ge(13) & (
                f.hf_above_TRAIN_q90 | f.site.str.contains('francis', case=False))
            return f
        train = flags(train); valpool = flags(source_features(sources, valpool))
        train.to_parquet(out/'TRAIN_source_features.parquet', index=False)
        valpool.to_parquet(out/'validation_source_roles.parquet', index=False)
        purged.to_parquet(out/'validation_purged_metadata.parquet', index=False)
        source_audit = {'TRAIN_windows': len(train), 'selection_windows': len(selection),
            'R11_used_selection_IDs': len(set(selection.base_window_index)&used),
            'new_selection_IDs_before_internal_purge': len(available),
            'R11_unexposed_selection_groups': len(set(selection.background_group)-usedgroups),
            'R11_groups_shared_development_not_blind': True,
            'R11_source_metadata_witnesses': witnesses, 'role_separation': split_stats,
            'HF_TRAIN_dense_population': int(dense.sum()), 'HF_TRAIN_q90_db': threshold,
            'TRAIN_special_zero_candidates': int(train.r12_special_zero_candidate.sum()),
            'references': ref_info, 'reference_independent_CV': False,
            'source_identity': sources.identity, 'source_hash': sources.source_hash}
        write_new(out/'source_audit.json', source_audit)
        files = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        # All four arms use the same indexed source draw; only explicit HF-zero
        # slots and the predeclared long-positive slots may differ.
        base = choose(train, c.train_examples, c.seed, 'TRAIN-primary')
        spec0 = schedule(len(base), c.seed, 1, 0., ref_info)
        spec1 = schedule(len(base), c.seed, 1, c.train_long_positive_fraction, ref_info)
        if not np.array_equal(spec0['zero'], spec1['zero']):
            raise AssertionError('Long arms changed no-source slots')
        hardbase, hardaudit = _special_backgrounds(base, train, spec0['zero'], c.seed, 1)
        if np.any(base.loc[~spec0['zero'], 'base_window_index'].to_numpy() != hardbase.loc[~spec0['zero'], 'base_window_index'].to_numpy()):
            raise AssertionError('HF changed positive source draw')
        rng = np.random.default_rng(np.random.SeedSequence([c.seed, 6000]))
        hard_zero = np.flatnonzero(hardbase.r12_special_zero_selected.to_numpy(bool))
        other_zero = np.flatnonzero(spec0['zero'] & ~hardbase.r12_special_zero_selected.to_numpy(bool))
        long_pos = np.flatnonzero(np.char.startswith(spec1['family'], 'long_'))
        standard_pos = np.flatnonzero(~spec1['zero'] & ~np.char.startswith(spec1['family'], 'long_'))
        if len(hard_zero) < 600 or len(other_zero) < 600:
            raise ValueError('TRAIN CNNsubset requires600special+600ordinaryzero sources')
        subset = np.r_[rng.permutation(hard_zero)[:600], rng.permutation(other_zero)[:600],
                       rng.permutation(long_pos)[:1440], rng.permutation(standard_pos)[:3360]]
        subset = subset[rng.permutation(len(subset))].astype(np.int64)
        if len(subset) != 6000 or len(set(subset)) != 6000 or spec0['zero'][subset].sum() != 1200:
            raise AssertionError('Shared balanced CNNsubset failed')
        np.save(out/'cnn_subset_indices.npy', subset, allow_pickle=False)
        files['cnn_subset_indices.npy'] = sha(out/'cnn_subset_indices.npy')
        arms = {}; profiles = {}
        for long_flag, spec in ((0, spec0), (1, spec1)):
            for hf_flag, frame in ((0, base), (1, hardbase)):
                arm = f'L{long_flag}_H{hf_flag}'
                arrays, metadata, recipes = materialize(sources, frame, spec, 'train_'+arm)
                desc = _save_dataset(out, 'train_'+arm, arrays, metadata, recipes, files)
                arms[arm] = {'train': desc}; profiles[arm] = desc['profile']
                del arrays, metadata, recipes
        common = {}; selected_eval_ids = set()
        for j, role in enumerate(VAL_ROLES):
            n = (c.validation_fit_examples, c.validation_gate_examples, c.eval_examples)[j]
            pool = valpool[valpool.r12_role.eq(role)].reset_index(drop=True)
            frame = choose(pool, n, c.seed, role+'-primary')
            spec = schedule(n, c.seed, 100+j, c.validation_long_positive_fraction, ref_info)
            frame, z_audit = _special_backgrounds(frame, pool, spec['zero'], c.seed, 100+j)
            arrays, metadata, recipes = materialize(sources, frame, spec, role)
            common[role] = _save_dataset(out, role, arrays, metadata, recipes, files) | {'special_zero_audit': z_audit}
            if role == 'eval': selected_eval_ids = set(map(int, frame.base_window_index))
        extra = valpool[valpool.r12_role.eq('eval') & ~valpool.base_window_index.isin(selected_eval_ids)].reset_index(drop=True)
        high = {'created': False, 'reason': 'Insufficient distinct eval-role sources after standardeval; no resampling'}
        if len(extra) >= c.high_snr_long_examples:
            frame = choose(extra, c.high_snr_long_examples, c.seed, 'highSNRlong-primary')
            spec = schedule(len(frame), c.seed, 501, 1., ref_info)
            arrays, metadata, recipes = materialize(sources, frame, spec, 'high_snr_long', snr_range=c.high_snr_db)
            high = {'created': True, **_save_dataset(out, 'high_snr_long', arrays, metadata, recipes, files)}
        basis, quotas = _basis_frame(train, c.basis_backgrounds, c.seed)
        basis_arrays = {'db': np.stack([sources.background(r) for _, r in basis.iterrows()]),
            'base_window_index': basis.base_window_index.to_numpy(np.int64),
            'source_id': np.asarray(['SA:'+str(i) for i in basis.base_window_index]),
            'site': basis.site.to_numpy(dtype=str)}
        np.savez_compressed(out/'background_basis.npz', **basis_arrays)
        basis.to_parquet(out/'background_basis_metadata.parquet', index=False)
        for name in ('background_basis.npz', 'background_basis_metadata.parquet'):
            files[name] = sha(out/name)
        for arm in arms:
            arms[arm].update(common)
        sources.assert_unchanged(verify_banks=True)
        for name, value in plan['code_sha256'].items():
            if sha(root/name) != value: raise AssertionError('R12 code changed during preparation')
        manifest = {'version': VERSION, 'status': 'completed', 'configuration': asdict(c),
            'plan_sha256': sha(out/'predeclared_plan.json'), 'code_sha256': plan['code_sha256'],
            'source_identity': sources.identity, 'source_audit': source_audit,
            'source_audit_sha256': sha(out/'source_audit.json'), 'arms': arms,
            'common_validation': common, 'high_snr_long': high,
            'background_basis': {'path': 'background_basis.npz', 'sha256': files['background_basis.npz'],
                'metadata_path': 'background_basis_metadata.parquet',
                'metadata_sha256': files['background_basis_metadata.parquet'], 'examples': len(basis),
                'site_counts': quotas, 'role': 'raw actual R9 TRAIN SA backgrounds only, no added recording'},
            'basis_training_binding': {'version': VERSION, 'role': 'train', 'fold': c.fold,
                'R9_source_hash': sources.source_hash,
                'background_basis_npz_sha256': files['background_basis.npz'],
                'source_manifest_sha256': sha(sources.bg_dir/f'fold_{c.fold:02d}/train_manifest.parquet'),
                'background_basis_metadata_sha256': files['background_basis_metadata.parquet'],
                'R12_plan_sha256': sha(out/'predeclared_plan.json'),
                'raw_SA_no_added_recording': True, 'unique_source_IDs': len(basis)},
            'TRAIN_special_zero_audit': hardaudit, 'file_sha256': files,
            'cnn_subset': {'path': 'cnn_subset_indices.npy', 'sha256': files['cnn_subset_indices.npy'],
                'examples': len(subset), 'same_indices_all_four_arms': True,
                'seed_sequence': [c.seed, 6000],
                'selection_method': 'seeded metadata-stratified permutation only; no model predictions, labels fitted to validation, or performance-based selection',
                'strata': {'special_zero': 600, 'other_zero': 600,
                           'long_designated_positive': 1440, 'other_positive': 3360},
                'explicit_zero_count': 1200, 'positive_count': 4800,
                'long_positive_count_when_L1': 1440,
                'special_zero_count_when_H1': 600},
            'all_original_source_banks_and_R9_code_unchanged': True,
            'model_fitting_or_Test_or_unused_report_access': False,
            'validation_semantics': 'TRAIN-separated background groups with24h purge; baseIDs not used inR11, but all25selection groups were exposed inR11. Development, not blind/OOF.',
            'positive_support_semantics': 'same-reference support-conditioned placements, unlike mask-independent R9; actual observedzero equals explicit20percent by design'}
        manifest['data_hash'] = digest(manifest)
        write_new(out/'manifest.json', manifest)
        write_new(out/'readiness.json', {'version': VERSION, 'ready_to_train': True,
            'source_checks_passed': True, 'manifest_sha256': sha(out/'manifest.json'),
            'data_hash': manifest['data_hash'], 'original_sources_unchanged': True,
            'actual_model_fit_executed': False})
        return manifest
    except BaseException as exc:
        write_new(out/'FAILED.json', {'status': 'failed_preserved', 'error': repr(exc),
            'ready_to_train': False, 'partial_reuse_permitted': False})
        raise


def validate_r12_bundle(folder, *, project_root=ROOT, verify_all=True):
    folder, root = Path(folder).resolve(), Path(project_root).resolve()
    m = json.loads((folder/'manifest.json').read_text()); r = json.loads((folder/'readiness.json').read_text())
    if (m.get('version') != VERSION or r.get('ready_to_train') is not True
            or r.get('source_checks_passed') is not True or sha(folder/'manifest.json') != r['manifest_sha256']
            or digest({k: v for k, v in m.items() if k != 'data_hash'}) != m['data_hash']
            or r['data_hash'] != m['data_hash']):
        raise ValueError('Unready/changed R12 data bundle')
    if verify_all:
        for name, h in m['file_sha256'].items():
            if sha(folder/name) != h: raise ValueError('R12 artifact changed: '+name)
        for name, h in m['code_sha256'].items():
            if sha(root/name) != h: raise ValueError('R12 generator source changed: '+name)
    return m


def load_r12_arrays(folder, arm, role, *, verify_all=True):
    if arm not in ARMS or role not in ('train',)+VAL_ROLES:
        raise ValueError('Explicit R12 arm and stage role required')
    m = validate_r12_bundle(folder, verify_all=verify_all)
    entry = m['arms'][arm][role]
    if sha(Path(folder)/entry['path']) != entry['sha256']:
        raise ValueError('Dataset SHA changed')
    with np.load(Path(folder)/entry['path'], allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    return a


class R12Dataset:
    def __init__(self, bundle, arm, role):
        self.arrays = load_r12_arrays(bundle, arm, role)
        self.identity = {'version': VERSION, 'data_hash': validate_r12_bundle(bundle, verify_all=False)['data_hash'],
                         'arm': arm, 'role': role, 'rows': len(self.arrays['share'])}
    def __len__(self): return len(self.arrays['share'])
    def raw_arrays(self): return self.arrays.copy()
    def __getitem__(self, index):
        a = self.arrays; x, w = relative_grid_v2(a['db'][index:index+1])
        return {'x': x[0], 'w': w[0], **{k: a[k][index] for k in
            ('share', 'cell_share', 'presence', 'site', 'sample_id', 'source_id', 'family')}}


def numeric_fixture():
    """No source files/models: finite placement and physical target oracles."""
    count = fallback = 0
    for minutes in (1, 2, 5, 6, 7, 14, 15, 23):
        native = (.1+np.arange(minutes*25).reshape(minutes, 25)/100.)
        for pattern in (np.arange(15), np.arange(5), np.arange(5, 10), np.arange(10, 15)):
            bg = np.full((15, 25), np.nan); bg[pattern] = 75.+np.linspace(-10, 10, 25)
            families = ['native_inside', 'native_short', 'native_edge']
            if minutes >= 15: families.append('long_native')
            if 5 <= minutes <= 14: families.append('long_smooth_bell')
            for family in families:
                out, recipe = synthesize_r12(bg, native, 42, family)
                if not out['share'] > 0 or not np.array_equal(np.isfinite(out['db']), np.isfinite(bg)):
                    raise AssertionError('Fixture support/mask failed')
                fallback += bool(recipe['route']['same_reference_fallback']); count += 1
            out, _ = synthesize_r12(bg, None, 42, 'no_source', no_source=True)
            if out['share'] != 0 or out['presence'] != 0 or out['cell_share'].any():
                raise AssertionError('Exact no-addition fixture failed')
    if fallback == 0: raise AssertionError('Impossible middle-mask edge case not exercised')
    base = pd.DataFrame({'base_window_index': np.arange(10), 'site': ['X']*10,
                         'r12_special_zero_candidate': [True]*10,
                         'hf_above_TRAIN_q90': [True]*10})
    zero = np.asarray([True, True]+[False]*8)
    unchanged, audit = _special_backgrounds(base, base, zero, 42, 1)
    if audit['shortfall_without_duplication'] != 1 or unchanged.base_window_index.duplicated().any():
        raise AssertionError('Finite candidate shortfall must preserve unique sources')
    return {'passed': True, 'positive_cases': count, 'same_reference_edge_fallback_cases': fallback,
        'zero_masks_checked': 32, 'HF_shortfall_preserves_source_uniqueness': True,
        'no_original_sources_or_models_opened': True}
