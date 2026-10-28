"""R9 acoustic data only: dynamic TRAIN and separately frozen evaluation roles.

The 509 measured VTUAD vessel-archive recordings are shared across background
folds, including 22 weak-background classes. Their known *added recording*
energy is the target; neither natural ship truth nor isolated vessel energy
is asserted. Sixteen dedicated ambient files are excluded from this pool.
No Test file, waveform file, learned prior, or source-identification input is
read. Query features accept db alone. Raw component labels stay unrounded;
model weights use the same rounded observations as the eight input channels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.relative_grid_r6_v2 import relative_grid_v2
from pipeline.reference_pool_r9 import reference_indices_for_epoch
from pipeline.sampler_r9 import native_minutes, synthesize
from pipeline.source_split_r6 import apply_minute_qc
from pipeline.hard_negatives_r6_v2 import transient_features, candidate_mask, POLICY as HARD_NEGATIVE_POLICY

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'r9_minute_full_dataset_v1'
ROLES = ('selection', 'ensemble_fit', 'ensemble_gate', 'report')
DATA_CODE = ('pipeline/dataset_r9.py', 'pipeline/sampler_r9.py',
             'pipeline/reference_pool_r9.py', 'pipeline/source_split_r6.py',
             'pipeline/hard_negatives_r6_v2.py',
             'pipeline/relative_grid_r6.py', 'pipeline/relative_grid_r6_v2.py')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(canonical({'shape': list(a.shape), 'dtype': str(a.dtype)}) + a.tobytes()).hexdigest()


def _inside(root, name):
    p = Path(name)
    p = p.resolve() if p.is_absolute() else (root / p).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('Path must remain inside this project')
    return p


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError('Integer ' + name + ' required')
    return int(value)


def _version(root, relative, expected, verify_banks):
    folder = _inside(root, relative)
    m = json.loads((folder / 'manifest.json').read_text())
    ready = json.loads((folder / 'readiness.json').read_text())
    if m.get('version') != expected or ready.get('version') != expected or ready.get('ready') is not True:
        raise ValueError('Unready source version: ' + relative)
    if sha(folder / 'manifest.json') != ready['manifest_sha256']:
        raise ValueError('Source manifest SHA mismatch')
    for name, value in m['input_sha256'].items():
        if sha(_inside(root, name)) != value:
            raise ValueError('Source input changed: ' + name)
    for name, value in m['file_sha256'].items():
        if sha(_inside(folder, name)) != value:
            raise ValueError('Source artifact changed: ' + name)
    bank = m.get('background_bank', m.get('bank'))
    if verify_banks and sha(_inside(root, bank['path'])) != bank['sha256']:
        raise ValueError('Numeric source bank SHA mismatch')
    return folder, m


def _config(path):
    c = json.loads(Path(path).read_text())
    g = c['generator']
    expected = {'minute_first_fraction': 1., 'explicit_no_source_fraction': .2,
                'smooth_eq_max_db': 6., 'time_warp': False,
                'force_boundary_fade': False, 'waveform_repeat_or_gap_fill': False,
                'quantization_decimals': 1, 'labels_from_unrounded_components': True,
                'model_weights_from_rounded_observations': True,
                'parameters_from_Test_statistics': False}
    if (c.get('version') != 'r9_full_training_v1' or c.get('folds') != list(range(5))
            or c.get('official_submission') is not False
            or g.get('family_names') != ['native_inside', 'native_short', 'native_edge']
            or g.get('family_probabilities') != [.5, .25, .25]
            or g.get('snr_db_uniform') != [-10., 10.]
            or any(g.get(k) != v for k, v in expected.items())
            or c.get('fixed_examples') != dict(selection=1000, ensemble_fit=500, ensemble_gate=500, report=1000)):
        raise ValueError('Frozen R9 data policy differs')
    return c


class R9Sources:
    """Verified numeric banks and a once-per-process native-minute cache.

    SHA streams cover whole files for integrity; statistical reads use only
    admitted background indices and the 509 admitted reference slices. The
    mmap is reopened by spawned workers without changing the source bytes.
    """
    def __init__(self, config_path='configs/r9_full_training_v1.json', project_root=ROOT,
                 verify_banks=True):
        self.root = Path(project_root).resolve()
        self.config_path = _inside(self.root, config_path)
        self.config = _config(self.config_path)
        self.bg_dir, self.bg_manifest = _version(self.root, self.config['background_version'], 'background_cv_r9_v1', verify_banks)
        self.ref_dir, self.ref_manifest = _version(self.root, self.config['reference_version'], 'vtuad_full_r9_v1', verify_banks)
        self.background_path = _inside(self.root, self.bg_manifest['background_bank']['path'])
        self.reference_path = _inside(self.root, self.ref_manifest['bank']['path'])
        self.references = pd.read_parquet(self.ref_dir / 'reference_manifest.parquet').sort_values('reference_row_index').reset_index(drop=True)
        if (len(self.references) != 509 or self.references.reference_row_index.tolist() != list(range(509))
                or not self.references.reference_eligible.all()
                or set(self.references.archive_role) != {'vessel_passage'}
                or self.references.sha256.duplicated().any()):
            raise ValueError('Exactly 509 admitted unique vessel-archive references required')
        bank = np.load(self.reference_path, mmap_mode='r', allow_pickle=False)
        if list(bank.shape) != self.ref_manifest['bank']['shape'] or str(bank.dtype) != 'float32':
            raise ValueError('Reference bank shape/dtype differs')
        cache = []
        for r in self.references.itertuples(index=False):
            start, count = int(r.segment_start), int(r.segment_count)
            if start < 0 or count < 30 or start + count > len(bank):
                raise ValueError('Reference slice bounds invalid')
            a = np.asarray(bank[start:start+count])
            if hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() != r.expected_segment_slice_sha256:
                raise ValueError('Reference slice SHA mismatch: ' + r.clip_id)
            m = native_minutes(a)
            if len(m) != int(r.native_complete_minute_count):
                raise ValueError('Native complete-minute count differs')
            m.setflags(write=False)
            cache.append(m)
        self.reference_minutes = tuple(cache)
        self._background = None
        self._hard_negative_cache = {}
        b = self.background_bank()
        if list(b.shape) != self.bg_manifest['background_bank']['shape'] or str(b.dtype) != 'float32':
            raise ValueError('Background bank shape/dtype differs')
        self.code_sha256 = {n: sha(self.root / n) for n in DATA_CODE}
        self.identity = {'version': VERSION, 'config_sha256': sha(self.config_path),
                         'background_source_hash': self.bg_manifest['source_hash'],
                         'reference_source_hash': self.ref_manifest['source_hash'],
                         'background_manifest_sha256': sha(self.bg_dir / 'manifest.json'),
                         'reference_manifest_sha256': sha(self.ref_dir / 'manifest.json'),
                         'background_bank_sha256': self.bg_manifest['background_bank']['sha256'],
                         'reference_bank_sha256': self.ref_manifest['bank']['sha256'],
                         'minute_cache_sha256': digest([array_sha(a) for a in cache]),
                         'code_sha256': self.code_sha256,
                         'reference_count': 509, 'reference_independent_CV': False}
        self.source_hash = digest(self.identity)

    def background_bank(self):
        if self._background is None:
            self._background = np.load(self.background_path, mmap_mode='r', allow_pickle=False)
        return self._background

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_background'] = None
        return state

    def role_metadata(self, fold, role):
        fold = _integer(fold, 'fold')
        if fold not in self.config['folds'] or role not in ('train',) + ROLES:
            raise ValueError('Unknown background fold/role')
        frame = pd.read_parquet(self.bg_dir / f'fold_{fold:02d}' / (role + '_manifest.parquet'))
        required = {'base_window_index', 'site', 'background_group', 'boundary_mask_bits', 'qc_observed_minutes'}
        if (not required <= set(frame) or not len(frame) or frame.base_window_index.duplicated().any()
                or (frame.qc_observed_minutes < 3).any()):
            raise ValueError('Unique minute-QC eligible background source rows required')
        return frame.reset_index(drop=True)

    def background(self, row):
        idx = int(row['base_window_index'])
        raw = np.asarray(self.background_bank()[idx:idx+1])
        db = apply_minute_qc(raw, np.asarray([int(row['boundary_mask_bits'])], np.int64))[0]
        o = np.isfinite(db)
        if (np.isinf(db).any() or (o.any(1) != o.all(1)).any()
                or int(o.any(1).sum()) != int(row['qc_observed_minutes']) or o.any(1).sum() < 3):
            raise ValueError('Source minute mask/QC count differs')
        return db

    def hard_negative_metadata(self, fold):
        """TRAIN-only candidate definition; use hash-bound prepared results.

        The LF q90 threshold is fit separately on each fold's TRAIN only.
        Candidate != naturally ship-absent. The exact known A=0 is imposed
        only by an explicit no-addition draw, never by natural labels.
        """
        if fold in self._hard_negative_cache:
            return self._hard_negative_cache[fold]
        train = self.role_metadata(fold, 'train')
        bundle = _inside(self.root, self.config['data_output'])
        relative = f'fold_{fold:02d}/train_hard_negative_candidates.parquet'
        if (bundle/'readiness.json').exists():
            m = validate_bundle(bundle, project_root=self.root)
            if (m['config_sha256'] != self.identity['config_sha256']
                    or m['source_identity'] != self.identity
                    or sha(bundle/relative) != m['file_sha256'][relative]):
                raise ValueError('TRAIN hard-negative metadata binding changed')
            frame = pd.read_parquet(bundle/relative)
        else:
            parts = []
            bank = self.background_bank()
            for begin in range(0, len(train), 512):
                part = train.iloc[begin:begin+512]
                db = apply_minute_qc(np.asarray(bank[part.base_window_index.to_numpy(np.int64)]),
                                     part.boundary_mask_bits.to_numpy(np.int64))
                parts.append(pd.DataFrame(transient_features(db)))
            features = pd.concat(parts, ignore_index=True)
            mask, threshold = candidate_mask(features)
            frame = pd.concat([train[['base_window_index', 'site', 'background_group']].reset_index(drop=True), features], axis=1)
            frame['hard_negative_candidate'] = mask
            frame['lf_train_q90_threshold_db'] = threshold
        if (len(frame) != len(train) or not np.array_equal(frame.base_window_index, train.base_window_index)
                or not np.array_equal(frame.site, train.site)
                or frame.hard_negative_candidate.isna().any()):
            raise ValueError('TRAIN hard-negative candidate order differs')
        self._hard_negative_cache[fold] = frame
        return frame

    def assert_unchanged(self, verify_banks=True):
        if sha(self.config_path) != self.identity['config_sha256']:
            raise ValueError('Config changed during preparation')
        for n, value in self.code_sha256.items():
            if sha(self.root / n) != value:
                raise ValueError('Data code changed: ' + n)
        _version(self.root, self.config['background_version'], 'background_cv_r9_v1', verify_banks)
        _version(self.root, self.config['reference_version'], 'vtuad_full_r9_v1', verify_banks)


def make_schedule(n, epoch, fold, seed, *, reference_count=509, tag=0, hard_negative_candidates=None):
    """Value-independent indexed schedule; independent RNG streams per role.

    Explicit zero requests are exactly round(.2*N). A positive request may
    produce zero observed addition when its real trajectory falls in missing
    minutes; that result is preserved instead of retried or relabelled.
    """
    n, epoch, fold, seed = (_integer(v, k, m) for v, k, m in
                            [(n, 'N', 1), (epoch, 'epoch', 1), (fold, 'fold', 0), (seed, 'seed', 0)])
    if fold > 4:
        raise ValueError('Fold must be 0..4')
    def rng(stream):
        return np.random.default_rng(np.random.SeedSequence([seed, fold, epoch, tag, stream]))
    order = rng(1).permutation(n)
    candidate = np.zeros(n, bool)
    hard = np.zeros(n, bool)
    if hard_negative_candidates is not None:
        candidate_source = np.asarray(hard_negative_candidates)
        if tag != 0 or candidate_source.shape != (n,) or candidate_source.dtype != bool:
            raise ValueError('TRAIN-only boolean hard-negative source mask required')
        candidate = candidate_source[order]
        available = np.flatnonzero(candidate)
        chosen = rng(4).permutation(available)[:min(round(.1*n), len(available))]
        hard[chosen] = True
    zero = hard.copy()
    remainder = np.flatnonzero(~zero)
    zero[rng(2).permutation(remainder)[:round(.2*n)-int(zero.sum())]] = True
    refs = np.full(n, -1, np.int64)
    refs[~zero] = reference_indices_for_epoch(reference_count, int((~zero).sum()), epoch, fold, seed+tag)
    family = rng(3).choice(3, n, p=[.5, .25, .25]).astype(np.int8)
    return dict(primary_order=order, explicit_no_source=zero, reference_index=refs, family_index=family,
                hard_negative_candidate=candidate, hard_negative=hard)


def schedule_audit(schedule, reference_count=509):
    n = len(schedule['primary_order'])
    refs = schedule['reference_index'][~schedule['explicit_no_source']]
    counts = np.bincount(refs, minlength=reference_count)
    return {'examples': n, 'primary_once': bool(np.array_equal(np.sort(schedule['primary_order']), np.arange(n))),
            'explicit_no_source_count': int(schedule['explicit_no_source'].sum()),
            'hard_negative_candidate_count': int(schedule['hard_negative_candidate'].sum()),
            'explicit_hard_negative_count': int(schedule['hard_negative'].sum()),
            'hard_negative_desired_count': round(.1*n),
            'hard_negative_shortfall_without_duplicate_primary': max(0, round(.1*n)-int(schedule['hard_negative'].sum())),
            'positive_request_count': len(refs), 'unique_references': int((counts > 0).sum()),
            'reference_use_min': int(counts.min()), 'reference_use_max': int(counts.max()),
            'reference_count': reference_count,
            'schedule_sha256': digest({k: array_sha(v) for k, v in schedule.items()})}


def _generate(sources, row, *, fold, epoch, index, tag, reference_index, family_index, no_source, role):
    db = sources.background(row)
    c = sources.config
    ref = int(reference_index)
    rng = np.random.default_rng(np.random.SeedSequence([int(c['seed']), fold, epoch, index, tag, 0x5249]))
    out = synthesize(db, None, rng,
                     reference_minute_power=None if no_source else sources.reference_minutes[ref],
                     family=c['generator']['family_names'][int(family_index)],
                     snr_range=tuple(c['generator']['snr_db_uniform']),
                     eq_max_db=c['generator']['smooth_eq_max_db'], no_source=bool(no_source))
    out.update(site=str(row['site']), background_group=str(row['background_group']),
               base_window_index=np.int64(row['base_window_index']), reference_index=np.int64(ref),
               explicit_no_source=np.bool_(no_source),
               sample_id=f'r9_f{fold:02d}_{role}_e{epoch:03d}_{index:07d}')
    out['recipe'].update(background_index=int(row['base_window_index']),
                         boundary_mask_bits=int(row['boundary_mask_bits']),
                         reference_index=ref,
                         reference_clip_id=None if ref < 0 else str(sources.references.iloc[ref].clip_id),
                         sample_id=out['sample_id'], fold=fold, role=role, epoch=epoch,
                         deterministic_seed=[int(c['seed']), fold, epoch, index, tag, 0x5249])
    return out


class DynamicR9Dataset:
    """Every admitted TRAIN primary exactly once, new deterministic view/epoch."""
    def __init__(self, config_path='configs/r9_full_training_v1.json', fold=0, epoch=1,
                 *, sources=None, verify_banks=True):
        self.sources = sources or R9Sources(config_path, verify_banks=verify_banks)
        self.fold = _integer(fold, 'fold')
        self.metadata = self.sources.role_metadata(self.fold, 'train')
        self.rows = self.metadata[['base_window_index', 'site', 'background_group',
                                   'boundary_mask_bits', 'qc_observed_minutes']].to_dict('records')
        self.hard_negative_metadata = self.sources.hard_negative_metadata(self.fold)
        self.hard_negative_candidates = self.hard_negative_metadata.hard_negative_candidate.to_numpy(bool)
        counts = self.metadata.site.value_counts()
        self.loss_weights = np.asarray([len(self.metadata)/(len(counts)*counts[s]) for s in self.metadata.site], np.float32)
        if not np.isclose(float(self.loss_weights.mean(dtype=np.float64)), 1., atol=1e-6):
            raise ValueError('TRAIN site weights must have mean one')
        self.identity = dict(self.sources.identity, role='train', fold=self.fold,
                             rows=len(self.metadata), epoch_policy='each primary once; fixed-index stateless epoch seed',
                             train_manifest_sha256=sha(self.sources.bg_dir / f'fold_{self.fold:02d}/train_manifest.parquet'),
                             site_weights_sha256=array_sha(self.loss_weights),
                             hard_negative_candidate_sha256=array_sha(self.hard_negative_candidates),
                             hard_negative_threshold_db=float(self.hard_negative_metadata.lf_train_q90_threshold_db.iloc[0]),
                             hard_negative_policy=HARD_NEGATIVE_POLICY,
                             hard_negative_allocation='min(round(.1*N), candidates) explicit zero, no primary duplication; rest of .2*N zeros uniform',
                             site_counts={str(k): int(v) for k, v in counts.items()})
        self.set_epoch(epoch)

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = _integer(epoch, 'epoch', 1)
        self.schedule = make_schedule(len(self), self.epoch, self.fold, int(self.sources.config['seed']),
                                      hard_negative_candidates=self.hard_negative_candidates)
        audit = self.epoch_audit()
        if not audit['primary_once'] or audit['unique_references'] != 509:
            raise ValueError('TRAIN primary/reference coverage invariant failed')

    def epoch_audit(self):
        return dict(schedule_audit(self.schedule), epoch=self.epoch, fold=self.fold,
                    no_source_policy='explicit request; natural background ships are not labelled',
                    coverage_kind='scheduled, not proof a partially consumed iterator completed')

    def raw_item(self, index):
        index = _integer(index, 'index')
        if index >= len(self):
            raise IndexError(index)
        source_index = int(self.schedule['primary_order'][index])
        out = _generate(self.sources, self.rows[source_index], fold=self.fold, epoch=self.epoch,
                        index=index, tag=0, reference_index=int(self.schedule['reference_index'][index]),
                        family_index=int(self.schedule['family_index'][index]),
                        no_source=bool(self.schedule['explicit_no_source'][index]), role='train')
        out['loss_weight'] = self.loss_weights[source_index]
        out['hard_negative_candidate'] = self.schedule['hard_negative_candidate'][index]
        out['hard_negative'] = self.schedule['hard_negative'][index]
        out['recipe'].update(hard_negative_candidate=bool(out['hard_negative_candidate']),
                             explicit_hard_negative=bool(out['hard_negative']))
        return out

    def __getitem__(self, index):
        row = self.raw_item(index)
        x, w = relative_grid_v2(row['db'][None])
        return {k: row[k] for k in ('share', 'cell_share', 'presence', 'site', 'sample_id', 'loss_weight')} | {'x': x[0], 'w': w[0]}


def fixed_background_indices(frame, count, seed):
    """Predeclared metadata-only group coverage, then uniform row sampling.

    At least one window per available group; the remaining rows are sampled
    without replacement when possible. If N exceeds the source pool, all
    rows are used before additional replacement. Unique counts are reported.
    No signal, target, report performance, or Test value affects the choice.
    """
    rng = np.random.default_rng(seed)
    anchors = [int(rng.choice(g.index.to_numpy())) for _, g in frame.groupby('background_group', sort=True)]
    if count < len(anchors):
        raise ValueError('Fixed role cannot cover its admitted source groups')
    remainder = np.setdiff1d(np.arange(len(frame)), np.asarray(anchors), assume_unique=False)
    remaining = count-len(anchors)
    selected = anchors + rng.permutation(remainder)[:remaining].tolist()
    if len(selected) < count:
        selected += rng.choice(len(frame), count-len(selected), replace=True).tolist()
    return np.asarray(selected, np.int64)[rng.permutation(count)]


def generate_fixed(sources, fold, role, count):
    if role not in ROLES:
        raise ValueError('Fixed role required')
    tag = 101 + ROLES.index(role)
    frame = sources.role_metadata(fold, role)
    picks = fixed_background_indices(frame, count, np.random.SeedSequence([sources.config['seed'], fold, tag]))
    schedule = make_schedule(count, 1, fold, int(sources.config['seed']), tag=tag)
    # The selected source order is independently randomized above. Its own
    # permutation is then consumed once, without changing the reference law.
    items, recipes, meta = [], [], []
    for i in range(count):
        row = frame.iloc[int(picks[int(schedule['primary_order'][i])])].to_dict()
        item = _generate(sources, row, fold=fold, epoch=1, index=i, tag=tag,
                         reference_index=int(schedule['reference_index'][i]),
                         family_index=int(schedule['family_index'][i]),
                         no_source=bool(schedule['explicit_no_source'][i]), role=role)
        items.append(item)
        recipes.append(item['recipe'])
        meta.append({k: item[k].item() if isinstance(item[k], np.generic) else item[k]
                     for k in ('site', 'sample_id', 'background_group', 'base_window_index', 'reference_index', 'explicit_no_source')}
                    | {'fold': fold, 'role': role, 'boundary_mask_bits': int(row['boundary_mask_bits']),
                       'qc_observed_minutes': int(row['qc_observed_minutes']),
                       'requested_family': sources.config['generator']['family_names'][int(schedule['family_index'][i])]})
    keys = ('db', 'unrounded_db', 'share', 'presence', 'cell_share', 'site', 'sample_id',
            'background_group', 'base_window_index', 'reference_index', 'explicit_no_source')
    arrays = {k: np.asarray([row[k] for row in items]) for k in keys}
    if any(a.dtype.kind == 'O' for a in arrays.values()):
        raise ValueError('Pickle/object array forbidden')
    return arrays, pd.DataFrame(meta), recipes, dict(schedule_audit(schedule),
          source_pool_windows=len(frame), source_pool_groups=int(frame.background_group.nunique()),
          unique_primary_backgrounds=int(arrays['base_window_index'].size and len(np.unique(arrays['base_window_index']))),
          unique_background_groups=int(len(np.unique(arrays['background_group']))),
          replacement_examples=count-len(np.unique(arrays['base_window_index'])))


def validate_bundle(folder, *, project_root=ROOT, verify_all=False):
    root, folder = Path(project_root).resolve(), Path(folder).resolve()
    m = json.loads((folder/'manifest.json').read_text())
    g = json.loads((folder/'readiness.json').read_text())
    if (m.get('version') != VERSION or g.get('version') != VERSION
            or g.get('ready_to_train') is not True or g.get('source_checks_passed') is not True
            or sha(folder/'manifest.json') != g.get('manifest_sha256')
            or g.get('source_hash') != m.get('source_hash')):
        raise ValueError('Prepared R9 bundle readiness/hash gate failed')
    content = {k: v for k, v in m.items() if k != 'source_hash'}
    if digest(content) != m['source_hash']:
        raise ValueError('Prepared bundle canonical hash failed')
    if verify_all:
        for name, value in m['file_sha256'].items():
            if sha(_inside(folder, name)) != value:
                raise ValueError('Prepared artifact changed: ' + name)
        for name, value in m['code_sha256'].items():
            if sha(_inside(root, name)) != value:
                raise ValueError('Data preparation code changed: ' + name)
        if sha(_inside(root, m['config_file'])) != m['config_sha256']:
            raise ValueError('Prepared config changed')
    return m


def _report_gate(root, gate, fold, bundle_hash):
    if isinstance(gate, (str, Path)):
        gate = json.loads(Path(gate).read_text())
    if (not isinstance(gate, dict) or gate.get('fold') != fold
            or gate.get('all_models_frozen') is not True or gate.get('ensemble_policy_frozen') is not True
            or gate.get('data_bundle_hash') != bundle_hash
            or len(gate.get('model_summary_files', {})) != 5):
        raise ValueError('Report requires this fold frozen five-model and ensemble-policy gate')
    for name, value in gate['model_summary_files'].items():
        p = _inside(root, name)
        if sha(p) != value:
            raise ValueError('Frozen model summary changed')
        summary = json.loads(p.read_text())
        if summary.get('status') != 'completed' or summary.get('fixture_only') is not False:
            raise ValueError('Production completed model summaries required')
    if sha(_inside(root, gate['ensemble_policy_file'])) != gate.get('ensemble_policy_sha256'):
        raise ValueError('Frozen ensemble policy changed')
    return digest(gate)


class FixedR9Dataset:
    def __init__(self, bundle_dir, fold, role, *, report_gate=None, project_root=ROOT):
        self.root, self.folder = Path(project_root).resolve(), Path(bundle_dir).resolve()
        self.fold, self.role = _integer(fold, 'fold'), role
        if role not in ROLES or self.fold not in range(5):
            raise ValueError('Explicit fixed role/fold required')
        self.manifest = validate_bundle(self.folder, project_root=self.root)
        self.data_bundle_hash = self.manifest['source_hash']
        gate_hash = None
        if role == 'report':
            # Gate is checked BEFORE np.load or metadata reads for this role.
            gate_hash = _report_gate(self.root, report_gate, fold, self.data_bundle_hash)
        prefix = f'fold_{fold:02d}/{role}'
        for suffix in ('.npz', '_metadata.parquet', '_recipes.json'):
            name = prefix+suffix
            if sha(_inside(self.folder, name)) != self.manifest['file_sha256'][name]:
                raise ValueError('Fixed role artifact changed: ' + name)
        with np.load(self.folder/(prefix+'.npz'), allow_pickle=False) as z:
            self.arrays = {k: z[k] for k in z.files}
        for a in self.arrays.values():
            if a.dtype.kind == 'O':
                raise ValueError('Pickle/object array forbidden')
            a.setflags(write=False)
        self.identity = {'version': VERSION, 'data_bundle_hash': self.data_bundle_hash,
                         'role': role, 'fold': fold, 'rows': len(self),
                         'npz_sha256': self.manifest['file_sha256'][prefix+'.npz'],
                         'metadata_sha256': self.manifest['file_sha256'][prefix+'_metadata.parquet'],
                         'config_sha256': self.manifest['config_sha256'],
                         'report_access_gate_sha256': gate_hash}

    def __len__(self):
        return len(self.arrays['share'])

    def raw_arrays(self):
        """Immutable numeric/Unicode views of this explicitly opened role only."""
        return self.arrays.copy()

    def raw_item(self, index):
        return {k: v[index] for k, v in self.arrays.items()}

    def __getitem__(self, index):
        row = self.raw_item(index)
        x, w = relative_grid_v2(row['db'][None])
        return {k: row[k] for k in ('share', 'cell_share', 'presence', 'site', 'sample_id')} | {'x': x[0], 'w': w[0]}
