"""R9 CPU relative-grid LightGBM and explicit INNER source-group safety gate.

No challenge-query I/O, model fitting on Test, prior retrieval, or old model
weights occur here. The tree uses the existing R6 3,827-column transform.
Tree early stopping counts boosting rounds; the neural 100-epoch/no-ES
policy does not apply to LightGBM. The ensemble is a finite predeclared
candidate comparison, not a claim that fixed equal weights are optimal.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURE_COUNT = 3827
TREE_SCORE_POLICY = 'half_clipped_share_half_presence_classifier_v1'
TREE_DEFAULTS = {
    'maximum_trees': 1500, 'early_stopping_patience': 40, 'threads': 4,
    'learning_rate': .035, 'num_leaves': 31, 'minimum_leaf_rows': 50,
    'maximum_training_rows': 100000, 'feature_batch_size': 512,
}
CODE_NAMES = (
    'pipeline/trees_ensemble_r9.py', 'pipeline/cpu_ablation_r6_v2.py',
    'pipeline/relative_grid_r6_v2.py', 'pipeline/relative_grid_r6.py',
    'pipeline/physics_features_r6_v2.py',
)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    if a.dtype.kind == 'O':
        raise TypeError('Object arrays cannot define numeric row identity')
    h = hashlib.sha256(json.dumps({'dtype': a.dtype.str, 'shape': list(a.shape)},
                                 sort_keys=True).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def feature_schema():
    from pipeline.relative_grid_r6_v2 import FEATURE_NAMES as channels
    from pipeline.physics_features_r6_v2 import FEATURE_NAMES as physics
    names = [f'grid/{name}/t{t:02d}/f{f:02d}' for name in channels
             for t in range(15) for f in range(25)]
    names += [f'{stat}/{name}/f{f:02d}' for stat in ('mean', 'std_ddof0', 'max', 'min')
              for name in channels for f in range(25)]
    names += ['physics/'+name for name in physics]
    assert len(names) == FEATURE_COUNT and len(set(names)) == FEATURE_COUNT
    return {
        'format': 'r9_r6_relative_tree_features_v1', 'feature_count': FEATURE_COUNT,
        'feature_names': names,
        'transform': "pipeline.cpu_ablation_r6_v2.tree_features(db,'relative8',None)",
        'feature_parts': {'flattened_relative8': 3000, 'temporal_channel_statistics': 800,
                          'physics_r6_v2': 27},
        'quantization_decimals': 1, 'query_inputs': ['db'],
        'statistics_missing_policy': 'statistics of the existing zero-filled relative grid',
        'learned_feature_state': False,
    }


def _db(value):
    if isinstance(value, Mapping):
        raise TypeError('Prediction/feature transform accepts db only, never a labelled dictionary')
    a = np.asarray(value)
    if a.ndim != 3 or a.shape[1:] != (15, 25) or not len(a) or a.dtype.kind != 'f':
        raise ValueError('Nonempty floating db[N,15,25] required')
    if np.isinf(a).any() or not np.isfinite(a).any(axis=(1, 2)).all():
        raise ValueError('Infinite values and wholly missing windows are invalid')
    return a


def transform_features(db, *, batch_size=512):
    from pipeline.cpu_ablation_r6_v2 import tree_features
    a = _db(db)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Positive feature batch required')
    result = np.empty((len(a), FEATURE_COUNT), dtype=np.float32)
    for start in range(0, len(a), batch_size):
        part = tree_features(a[start:start+batch_size], 'relative8', None)
        if part.shape != (min(batch_size, len(a)-start), FEATURE_COUNT) or not np.isfinite(part).all():
            raise ValueError('R6 tree feature schema/finite-value mismatch')
        result[start:start+len(part)] = part
    return result


def _dataset(value, role):
    if not isinstance(value, Mapping) or not {'db', 'share', 'presence'} <= set(value):
        raise TypeError('Explicit db/share/presence dataset dictionary required')
    allowed_roles = {'train'} if role == 'train' else {'inner', 'selection'}
    if 'role' in value and value['role'] not in allowed_roles:
        raise ValueError('Unexpected dataset role')
    db = _db(value['db'])
    y = np.asarray(value['share'], dtype=np.float64)
    z = np.asarray(value['presence'], dtype=np.float64)
    if y.shape != (len(db),) or z.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(z).all():
        raise ValueError('Finite aligned labels required')
    if ((y < 0) | (y > 1)).any() or not np.isin(z, [0., 1.]).all():
        raise ValueError('Share in [0,1] and binary controlled-addition presence required')
    if ((z == 0) & (y != 0)).any():
        raise ValueError('Absent added component must have zero energy share')
    return db, y, z


class PhysicsLGBM:
    """Two deterministic CPU boosters; .fit returns the completed summary.

    Full float32 train/INNER feature matrices remain resident. At the 100k
    TRAIN cap the train matrix alone is 1,530,800,000 bytes (1.426 GiB),
    excluding inputs, INNER, histogram bins and LightGBM working memory.
    Chunked transform bounds temporary grids; it is not out-of-core training.
    """
    def __init__(self, settings=None, *, seed=20261026):
        if settings is not None and (not isinstance(settings, Mapping) or set(settings)-set(TREE_DEFAULTS)):
            raise ValueError('Unsupported LightGBM settings')
        self.settings = dict(TREE_DEFAULTS, **(settings or {}))
        for key in set(TREE_DEFAULTS)-{'learning_rate'}:
            if type(self.settings[key]) is not int or self.settings[key] < 1:
                raise ValueError('Positive integer tree settings required')
        if self.settings['maximum_training_rows'] > 100000 or self.settings['maximum_trees'] > 1500:
            raise ValueError('Predeclared 100k-row/1500-tree caps exceeded')
        if not np.isfinite(self.settings['learning_rate']) or not 0 < self.settings['learning_rate'] <= 1:
            raise ValueError('Finite positive learning rate <=1 required')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Unsigned 32-bit integer seed required')
        self.seed = seed
        self.regressor = self.classifier = None
        self.summary = None

    def fit(self, train, inner, output_dir, binding, *, fixture=False):
        import lightgbm as lgb
        td, ty, tz = _dataset(train, 'train')
        vd, vy, vz = _dataset(inner, 'inner')
        if len(td) > self.settings['maximum_training_rows']:
            raise ValueError('TRAIN exceeds the fixed memory cap; no automatic subsampling')
        if len(vd) > 100000:
            raise ValueError('INNER exceeds the fixed matrix memory cap')
        if len(np.unique(tz)) != 2:
            raise ValueError('Both TRAIN presence classes required; no silent constant classifier')
        if not isinstance(binding, Mapping) or not binding:
            raise ValueError('Explicit nonempty source/split binding required')
        canonical_sha(binding)
        folder = Path(output_dir)
        if folder.exists():
            raise FileExistsError('Fresh output required; partial model reuse is forbidden')
        folder.mkdir(parents=True)
        start = time.perf_counter()
        try:
            x = transform_features(td, batch_size=self.settings['feature_batch_size'])
            v = transform_features(vd, batch_size=self.settings['feature_batch_size'])
            args = dict(n_estimators=self.settings['maximum_trees'],
                        learning_rate=self.settings['learning_rate'],
                        num_leaves=self.settings['num_leaves'],
                        min_child_samples=self.settings['minimum_leaf_rows'],
                        max_bin=127, n_jobs=self.settings['threads'],
                        random_state=self.seed, verbosity=-1,
                        colsample_bytree=.85, subsample=.85, subsample_freq=1,
                        deterministic=True, force_col_wise=True,
                        data_random_seed=self.seed, feature_fraction_seed=self.seed,
                        bagging_seed=self.seed)
            reg = lgb.LGBMRegressor(objective='regression_l2', metric='None', **args)
            clf = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', **args)
            def clipped_share_rmse(truth, prediction):
                return 'clipped_share_rmse', float(np.sqrt(np.mean((np.clip(prediction, 0, 1)-truth)**2))), False
            reg.fit(x, ty, eval_set=[(v, vy)], eval_metric=clipped_share_rmse, callbacks=[
                lgb.early_stopping(self.settings['early_stopping_patience'], verbose=False), lgb.log_evaluation(0)])
            clf.fit(x, tz, eval_set=[(v, vz)], callbacks=[
                lgb.early_stopping(self.settings['early_stopping_patience'], verbose=False), lgb.log_evaluation(0)])
            self.regressor, self.classifier = reg.booster_, clf.booster_
            self.regressor.save_model(str(folder/'share.txt'))
            self.classifier.save_model(str(folder/'presence.txt'))
            schema = feature_schema()
            _write_json(folder/'feature_schema.json', schema)
            pred = self._predict_features(v)
            np.savez_compressed(folder/'inner_predictions.npz', **pred)
            artifact_names = ['share.txt', 'presence.txt', 'feature_schema.json', 'inner_predictions.npz']
            self.summary = {
                'format': 'r9_physics_lgbm_v1', 'status': 'completed',
                'kind': 'lgbm_physics', 'fixture_only': bool(fixture),
                'training_binding': dict(binding), 'settings': self.settings, 'seed': self.seed,
                'n_features': FEATURE_COUNT, 'feature_schema_sha256': canonical_sha(schema),
                'score_policy': TREE_SCORE_POLICY, 'selection': 'fixed INNER pooled clipped-share RMSE',
                'presence_selection': 'fixed INNER binary logloss',
                'best_iteration': int(reg.best_iteration_),
                'presence_best_iteration': int(clf.best_iteration_),
                'train_rows': len(td), 'inner_rows': len(vd),
                'feature_matrix_bytes': {'train': x.nbytes, 'inner': v.nbytes},
                'input_sha256': {
                    'train': {'db': array_sha(td), 'share': array_sha(ty), 'presence': array_sha(tz)},
                    'inner': {'db': array_sha(vd), 'share': array_sha(vy), 'presence': array_sha(vz)},
                },
                'best_inner_rmse': float(np.sqrt(np.mean((pred['share'].astype(float)-vy)**2))),
                'artifact_sha256': {name: file_sha(folder/name) for name in artifact_names},
                'code_sha256': {name: file_sha(ROOT/name) for name in CODE_NAMES},
                'lightgbm_version': lgb.__version__, 'device': 'cpu',
                'elapsed_seconds': time.perf_counter()-start,
                'Test_or_outer_accessed': False,
                'NN_100_epoch_policy_applicable': False,
            }
            _write_json(folder/'summary.json', self.summary)
            return self.summary
        except BaseException as error:
            _write_json(folder/'FAILED.json', {'error_type': type(error).__name__, 'error': str(error),
                                             'partial_reuse_allowed': False})
            self.regressor = self.classifier = None
            raise

    def _predict_features(self, x):
        if self.regressor is None or self.classifier is None:
            raise ValueError('Completed trained/restored boosters required')
        p = np.clip(self.regressor.predict(x, num_threads=self.settings['threads']), 0, 1).astype(np.float32)
        z = np.clip(self.classifier.predict(x, num_threads=self.settings['threads']), 0, 1).astype(np.float32)
        if not np.isfinite(p).all() or not np.isfinite(z).all():
            raise FloatingPointError('Nonfinite tree prediction')
        return {'share': p, 'score': .5*p+.5*z, 'score_classifier': z}

    def predict(self, db, *, batch_size=256):
        a = _db(db)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Positive prediction batch required')
        pieces = [self._predict_features(transform_features(a[start:start+batch_size], batch_size=batch_size))
                  for start in range(0, len(a), batch_size)]
        return {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}

    @classmethod
    def load(cls, output_dir, *, expected_binding, allow_fixture=False):
        import lightgbm as lgb
        folder = Path(output_dir)
        summary = json.loads((folder/'summary.json').read_text())
        if (summary.get('format') != 'r9_physics_lgbm_v1' or summary.get('status') != 'completed'
                or summary.get('kind') != 'lgbm_physics' or type(summary.get('fixture_only')) is not bool
                or summary.get('n_features') != FEATURE_COUNT or (folder/'FAILED.json').exists()):
            raise ValueError('Complete unfailed R9 tree required')
        if summary.get('fixture_only') and not allow_fixture:
            raise ValueError('Fixture weights are not production weights')
        if summary.get('training_binding') != expected_binding or summary.get('score_policy') != TREE_SCORE_POLICY:
            raise ValueError('Tree source/split/score binding differs')
        expected_files = {'share.txt', 'presence.txt', 'feature_schema.json', 'inner_predictions.npz'}
        if set(summary.get('artifact_sha256', {})) != expected_files or set(summary.get('code_sha256', {})) != set(CODE_NAMES):
            raise ValueError('Incomplete artifact/code closure')
        for name, digest in summary['artifact_sha256'].items():
            if file_sha(folder/name) != digest:
                raise ValueError('Tree artifact changed: '+name)
        for name, digest in summary['code_sha256'].items():
            if file_sha(ROOT/name) != digest:
                raise ValueError('Feature/model source code changed: '+name)
        schema = json.loads((folder/'feature_schema.json').read_text())
        if schema != feature_schema() or canonical_sha(schema) != summary['feature_schema_sha256']:
            raise ValueError('Feature column semantics differ')
        obj = cls(summary['settings'], seed=summary['seed'])
        obj.regressor = lgb.Booster(model_file=str(folder/'share.txt'))
        obj.classifier = lgb.Booster(model_file=str(folder/'presence.txt'))
        if obj.regressor.num_feature() != FEATURE_COUNT or obj.classifier.num_feature() != FEATURE_COUNT:
            raise ValueError('Booster feature dimensions differ')
        obj.summary = summary
        return obj


def _predictions(predmap, expected_rows=None):
    if not isinstance(predmap, Mapping) or not predmap or any(not isinstance(k, str) or not k for k in predmap):
        raise TypeError('Nonempty named prediction mapping required')
    names = sorted(predmap)
    shares, scores = [], []
    for name in names:
        value = predmap[name]
        if isinstance(value, Mapping):
            if 'share' not in value:
                raise ValueError('Member share prediction missing')
            p = np.asarray(value['share'], dtype=np.float64)
            s = np.asarray(value.get('score', value['share']), dtype=np.float64)
        else:
            p = np.asarray(value, dtype=np.float64)
            s = p
        if p.ndim != 1 or not len(p) or s.shape != p.shape or (expected_rows is not None and len(p) != expected_rows):
            raise ValueError('Aligned one-dimensional member predictions required')
        if not np.isfinite(p).all() or not np.isfinite(s).all() or ((p < 0) | (p > 1) | (s < 0) | (s > 1)).any():
            raise ValueError('Finite share/diagnostic score in [0,1] required')
        expected_rows = len(p)
        shares.append(p)
        scores.append(s)
    return names, np.stack(shares, axis=1), np.stack(scores, axis=1)


def _indices(value, n):
    a = np.asarray(value)
    if a.ndim != 1 or not len(a) or a.dtype.kind not in 'iu' or (a < 0).any() or (a >= n).any() or len(set(a.tolist())) != len(a):
        raise ValueError('Unique nonempty explicit row indices within bounds required')
    return a.astype(np.int64)


def fit_safe_ensemble(own_inner_y, predmap, groups, *, fit_indices, gate_indices,
                      epsilon_rmse=1e-4, tree_member='lgbm_physics'):
    """Freeze one fit-selected candidate, then evaluate that candidate once.

    Prespecified candidates: every single model, equal-all, equal-non-tree.
    The fallback is the best fit-row single non-tree model if available,
    otherwise the only tree. This prevents tree-only validation success from
    bypassing the disjoint source-group gate. No gate search/refit occurs.
    Gate admission is validation-based selection, NOT untouched evaluation.
    An independent outer/report partition belongs to the calling runner.
    """
    y = np.asarray(own_inner_y, dtype=np.float64)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
        raise ValueError('Finite own-INNER energy-share targets in [0,1] required')
    names, p, _ = _predictions(predmap, len(y))
    g = np.asarray(groups)
    if g.shape != y.shape or g.dtype.kind not in 'USiu' or any(str(v) == '' for v in g):
        raise ValueError('Explicit source-group identities required (never prediction features)')
    g = g.astype(str)
    fit, gate = _indices(fit_indices, len(y)), _indices(gate_indices, len(y))
    if np.intersect1d(fit, gate).size or set(g[fit]) & set(g[gate]):
        raise ValueError('Fit and safety gate must be disjoint by row AND source group')
    if not np.isfinite(epsilon_rmse) or epsilon_rmse < 0:
        raise ValueError('Finite predeclared nonnegative improvement threshold required')
    count = len(names)
    one = np.eye(count)
    candidates = {'single/'+name: one[i] for i, name in enumerate(names)}
    if count >= 2:
        candidates['equal/all'] = np.ones(count)/count
    non_tree = [i for i, name in enumerate(names) if name != tree_member]
    if len(non_tree) >= 2:
        w = np.zeros(count)
        w[non_tree] = 1/len(non_tree)
        candidates['equal/non_tree'] = w
    rmse = lambda ix, w: float(np.sqrt(np.mean((p[ix]@w-y[ix])**2)))
    fallback_choices = non_tree or list(range(count))
    fallback_index = min(fallback_choices, key=lambda i: (rmse(fit, one[i]), names[i]))
    fallback_name, fallback_w = 'single/'+names[fallback_index], one[fallback_index]
    fit_metrics = {name: rmse(fit, w) for name, w in candidates.items()}
    candidate_name = min(candidates, key=lambda name: (fit_metrics[name], name))
    candidate_w = candidates[candidate_name]
    # Candidate identity and all coefficients are determined before gate rows
    # or gate targets enter a metric. Exactly this one candidate is tested.
    frozen_candidate = {'name': candidate_name, 'weights': dict(zip(names, candidate_w.tolist()))}
    candidate_hash = canonical_sha(frozen_candidate)
    fallback_fit = fit_metrics[fallback_name]
    fit_improves = fit_metrics[candidate_name] < fallback_fit-epsilon_rmse
    gate_result = None
    admitted = False
    if fit_improves:
        candidate_gate = rmse(gate, candidate_w)
        fallback_gate = rmse(gate, fallback_w)
        admitted = candidate_gate < fallback_gate-epsilon_rmse
        gate_result = {'candidate_rmse': candidate_gate, 'fallback_rmse': fallback_gate,
                       'candidate_improves': bool(admitted)}
    final_w = candidate_w if admitted else fallback_w
    final_name = candidate_name if admitted else fallback_name
    policy = {
        'format': 'r9_safe_inner_ensemble_v1', 'status': 'frozen',
        'member_names': names, 'weights': dict(zip(names, final_w.tolist())),
        'selected_name': final_name, 'fallback_name': fallback_name,
        'fallback_policy': 'best_fit_non_tree_single_when_available',
        'candidate_definition': 'all_singles_plus_equal_all_plus_equal_non_tree',
        'candidate_name': candidate_name, 'candidate_before_gate': frozen_candidate,
        'candidate_before_gate_sha256': candidate_hash,
        'candidate_fit_rmse': fit_metrics[candidate_name], 'fallback_fit_rmse': fallback_fit,
        'all_candidate_fit_rmse': fit_metrics, 'epsilon_rmse': float(epsilon_rmse),
        'candidate_fit_improves': bool(fit_improves), 'candidate_admitted': bool(admitted),
        'gate': gate_result, 'gate_candidates_evaluated': int(fit_improves),
        'gate_weight_refit': False, 'gate_hyperparameter_search': False,
        'affine': [1., 0.], 'share_policy': 'nonnegative_simplex_scalar_average',
        'score_policy': 'same_weights_average_member_diagnostic_scores_not_new_calibration',
        'source_group_disjoint': True, 'own_inner_only': True,
        'gate_is_validation_selection_not_blind_evaluation': True,
        'outer_or_Test_accessed': False,
        'fit_rows': len(fit), 'gate_rows': len(gate),
        'unselected_rows': int(len(y)-len(fit)-len(gate)),
        'fit_groups': sorted(set(g[fit])), 'gate_groups': sorted(set(g[gate])),
        'input_sha256': {'y': array_sha(y), 'groups': array_sha(g),
                         'fit_indices': array_sha(fit), 'gate_indices': array_sha(gate),
                         'member_share_matrix': array_sha(p)},
        'code_sha256': {'pipeline/trees_ensemble_r9.py': file_sha(Path(__file__))},
    }
    policy['policy_sha256'] = canonical_sha(policy)
    return policy


def apply_frozen_ensemble(policy, predmap):
    """Apply the frozen scalar simplex weights; accepts predictions only.

    A blend containing a tree has no cell mask interpretation. For neural-only
    masks using identical per-window w, scalar averaging is mathematically
    equivalent to pooling the same weighted average of their q masks.
    """
    payload = {k: v for k, v in policy.items() if k != 'policy_sha256'}
    if policy.get('format') != 'r9_safe_inner_ensemble_v1' or policy.get('status') != 'frozen' or canonical_sha(payload) != policy.get('policy_sha256'):
        raise ValueError('Frozen unmodified ensemble policy required')
    if policy.get('code_sha256') != {'pipeline/trees_ensemble_r9.py': file_sha(Path(__file__))}:
        raise ValueError('Ensemble source code differs')
    names, p, score = _predictions(predmap)
    if names != policy['member_names'] or set(policy['weights']) != set(names):
        raise ValueError('Frozen ensemble members differ')
    w = np.asarray([policy['weights'][name] for name in names], dtype=np.float64)
    if not np.isfinite(w).all() or (w < 0).any() or abs(w.sum()-1) > 1e-12 or policy.get('affine') != [1., 0.]:
        raise ValueError('Nonnegative simplex weights without affine fitting required')
    return {'share': np.clip(p@w, 0, 1).astype(np.float32),
            'score': np.clip(score@w, 0, 1).astype(np.float32)}


apply_ensemble = apply_frozen_ensemble


def freeze_ensemble(policy, path):
    """Write one immutable policy; callers must freeze before outer access."""
    path = Path(path)
    if path.exists():
        raise FileExistsError('Refusing to overwrite a frozen ensemble policy')
    payload = {k: v for k, v in policy.items() if k != 'policy_sha256'}
    if canonical_sha(payload) != policy.get('policy_sha256'):
        raise ValueError('Policy payload hash mismatch')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(policy, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return file_sha(path)
