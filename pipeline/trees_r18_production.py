"""R18 single-split production trees, with a small registered level-prior ablation.

The default relative regressor is compared with ONE auxiliary branch on clean_ES.
No challenge query, evaluation source, site identifier, PCA or LF-anchor is fitted.
The independent classifier exactly restores R18 profile_energy (2,408 features).
"""
from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np

from pipeline.training_r16_multisupport import tree_features
from pipeline.trees_r16_production import schemas as old_schemas, db_check, sha, write, token
from pipeline.relative_grid_r6_v2 import relative_grid_v2
from ops.r16_temporal_features import temporal_statistics

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026103018
AUX_NAMES = [
    'aux/common_window_median_div100', 'aux/common_window_mean_div100',
    'aux/common_mean_energy_db_div100', 'aux/q05_integral_share_estimate',
    'aux/q20_integral_share_estimate', 'aux/TRAIN_global_B_to_E_log10',
    'aux/TRAIN_global_integral_share_estimate', 'aux/TRAIN_conditional_B_to_E_log10',
    'aux/TRAIN_conditional_integral_share_estimate',
    'aux/TRAIN_q20_shape_centroid_rmse_div20',
    'aux/TRAIN_conditional_B_mean_db_div100', 'aux/TRAIN_global_B_mean_db_div100',
]


def schemas():
    extra = temporal_statistics(np.ones((1, 15, 25), np.float32))[1]
    old = old_schemas()
    original_reg = old['regression'] + ['F1/' + n for n in extra]
    # The regression schema does NOT contain the classifier's 25 observed
    # fractions. Remove its actual 493 explicit mask/validity/count columns.
    removed_reg = [i for i, n in enumerate(original_reg)
                   if '/observed_mask/' in n or
                   (n.startswith('physics/') and
                    (n.endswith('_defined') or n.endswith('_valid_minutes')))]
    reg_idx = np.setdiff1d(np.arange(len(original_reg)), removed_reg)
    original_clf = old['classification'] + ['F1/' + n for n in extra]
    clf_idx = np.asarray([i for i, n in enumerate(original_clf)
                          if not n.startswith('observed_fraction/')], int)
    clf = [original_clf[i] for i in clf_idx]
    clf += [f'restored_Ch4_band_profile/f{f:02d}' for f in range(25)]
    clf += [f'restored_Ch7_sqrtw/t{t:02d}/f{f:02d}' for t in range(15) for f in range(25)]
    reg = [original_reg[i] for i in reg_idx]
    assert (len(original_reg), len(removed_reg), len(reg), len(clf)) == (5160, 493, 4667, 2408)
    return dict(regression_relative=reg, regression_auxiliary=reg + AUX_NAMES,
                classification_profile_energy=clf,
                regression_removed_explicit_mask=[original_reg[i] for i in removed_reg],
                classification_removed_explicit_mask=[n for n in original_clf if n.startswith('observed_fraction/')],
                regression_indices=reg_idx.tolist(), classification_indices=clf_idx.tolist())


_SCHEMA = None


def schema():
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = schemas()
    return _SCHEMA


def energy(d):
    d = np.asarray(d, np.float64)
    return np.where(np.isfinite(d), 10. ** (np.nan_to_num(d, nan=0.) / 10.), 0.)


def profile_restore(db):
    """Identical values/order to ops.r18_classifier_audit.extra_features."""
    d = np.round(np.asarray(db, np.float64), 1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(d, axis=1)
        whole = np.nanmedian(d, axis=(1, 2))
    _, w = relative_grid_v2(d)
    profile = np.nan_to_num((med - whole[:, None]) / 20., nan=0.)
    return np.concatenate([profile, np.sqrt(w).reshape(len(d), -1)], axis=1).astype(np.float32)


def fit_background_prior(train_backgrounds, folder):
    """Fit metadata-free 16-shape templates from unique original TRAIN B only.

    Source indices/QC bits identify supervised TRAIN components at fitting time.
    Predict accepts only db. No exact-source matching, labels or site lookup is
    exposed at inference. These are auxiliary estimates, never physical bounds.
    """
    import pandas as pd
    from sklearn.cluster import KMeans
    from pipeline.source_split_r6 import apply_minute_qc

    path = Path(train_backgrounds)
    meta = pd.read_parquet(path).drop_duplicates('base_window_index').reset_index(drop=True)
    if len(meta) < 100 or not {'base_window_index', 'boundary_mask_bits'} <= set(meta):
        raise ValueError('Unique TRAIN background source/QC table required')
    bank_path = ROOT / 'data/processed/sa_windows_db.npy'
    bank = np.load(bank_path, mmap_mode='r')
    profiles, shapes = [], []
    for start in range(0, len(meta), 512):
        part = meta.iloc[start:start + 512]
        d = apply_minute_qc(np.asarray(bank[part.base_window_index.to_numpy(int)]),
                            part.boundary_mask_bits.to_numpy(int))
        o = np.isfinite(d)
        power = energy(d).sum(1) / np.maximum(o.sum(1), 1)
        profiles.append(10 * np.log10(np.maximum(power, 1e-200)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            q = np.nanquantile(d, .2, axis=1)
        q = np.where(np.isfinite(q), q, np.nanmedian(d, axis=(1, 2))[:, None])
        shapes.append(q - np.median(q, axis=1)[:, None])
    profiles, shapes = np.concatenate(profiles), np.concatenate(shapes)
    km = KMeans(n_clusters=16, random_state=SEED, n_init=5).fit(shapes)
    templates = np.stack([np.median(profiles[km.labels_ == i], axis=0) for i in range(16)])
    prior = dict(format='r18_TRAIN_background_shape16_v1',
                 centers=km.cluster_centers_.tolist(), templates_db=templates.tolist(),
                 global_template_db=np.median(profiles, axis=0).tolist(),
                 cluster_counts=np.bincount(km.labels_, minlength=16).tolist(),
                 unique_TRAIN_backgrounds=len(meta), source_metadata_sha256=sha(path),
                 SA_bank_sha256=sha(bank_path),
                 fit_inputs='Unique original TRAIN B rows after original boundary minute QC; no evaluation backgrounds',
                 query_inputs=['db'], source_IDs_at_inference=False,
                 no_PCA_or_LF_anchor=True, no_bounds_or_forced_share_targets=True)
    write(Path(folder) / 'TRAIN_background_prior.json', prior)
    return prior


def auxiliary(db, prior):
    if prior is None:
        raise ValueError('Auxiliary branch requires a frozen TRAIN-only prior')
    d = np.round(np.asarray(db, np.float64), 1)
    o = np.isfinite(d)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        whole = np.nanmedian(d, axis=(1, 2))
        mean = np.nanmean(d, axis=(1, 2))
        q05, q20 = np.nanquantile(d, [.05, .2], axis=1)
    q05 = np.where(np.isfinite(q05), q05, whole[:, None])
    q20 = np.where(np.isfinite(q20), q20, whole[:, None])
    E = energy(d).sum((1, 2))
    n = o.sum((1, 2))
    count_f = o.sum(1)
    shape = q20 - np.median(q20, axis=1)[:, None]
    centers = np.asarray(prior['centers'])
    distances = ((shape[:, None] - centers[None]) ** 2).mean(2)
    closest = np.argmin(distances, axis=1)
    conditional = np.asarray(prior['templates_db'])[closest]
    global_db = np.asarray(prior['global_template_db'])
    Bcond = (10 ** (conditional / 10.) * count_f).sum(1)
    Bglobal = (10 ** (global_db[None] / 10.) * count_f).sum(1)
    B05 = (10 ** (q05 / 10.) * count_f).sum(1)
    B20 = (10 ** (q20 / 10.) * count_f).sum(1)
    result = np.column_stack([
        whole / 100., mean / 100., (10 * np.log10(E / n)) / 100.,
        np.clip(1 - B05 / E, 0, 1), np.clip(1 - B20 / E, 0, 1),
        np.clip(np.log10(Bglobal / E), -6, 6), np.clip(1 - Bglobal / E, 0, 1),
        np.clip(np.log10(Bcond / E), -6, 6), np.clip(1 - Bcond / E, 0, 1),
        np.sqrt(distances[np.arange(len(d)), closest]) / 20.,
        10 * np.log10(Bcond / n) / 100., 10 * np.log10(Bglobal / n) / 100.,
    ]).astype(np.float32)
    assert result.shape == (len(d), len(AUX_NAMES)) and np.isfinite(result).all()
    return result


def transform(db, kind, prior=None, batch_size=256):
    d = db_check(db)
    names = {'relative': 'regression_relative', 'auxiliary': 'regression_auxiliary',
             'classification': 'classification_profile_energy'}
    if kind not in names:
        raise ValueError(kind)
    result = np.empty((len(d), len(schema()[names[kind]])), np.float32)
    idx = schema()['classification_indices' if kind == 'classification' else 'regression_indices']
    for start in range(0, len(d), batch_size):
        part = d[start:start + batch_size]
        original = tree_features(part, 'classification' if kind == 'classification' else 'regression')
        cleaned = original[:, idx]
        if kind == 'classification':
            cleaned = np.concatenate([cleaned, profile_restore(part)], axis=1)
        elif kind == 'auxiliary':
            cleaned = np.concatenate([cleaned, auxiliary(part, prior)], axis=1)
        if not np.isfinite(cleaned).all():
            raise ValueError('Nonfinite production tree features')
        result[start:start + len(part)] = cleaned
    return result


def predict(folder, db, branches=('selected', 'relative', 'auxiliary'), batch_size=256):
    import json
    import lightgbm as lgb
    folder = Path(folder)
    summary = json.loads((folder / 'summary.json').read_text())
    if summary['status'] != 'completed':
        raise ValueError('Only a completed model may predict')
    for n, h in summary['artifact_sha256'].items():
        if sha(folder / n) != h:
            raise ValueError('Production tree artifact changed: ' + n)
    prior = json.loads((folder / 'TRAIN_background_prior.json').read_text())
    outputs = {}
    required = set(summary['selected_regression'] if b == 'selected' else b for b in branches)
    models = {b: lgb.Booster(model_file=str(folder / (b + '.txt'))) for b in required}
    models['classification'] = lgb.Booster(model_file=str(folder / 'classification.txt'))
    temp = {b: [] for b in models}
    for start in range(0, len(db), batch_size):
        d = db[start:start + batch_size]
        for b, model in models.items():
            x = transform(d, b, prior=prior, batch_size=batch_size)
            p = model.predict(x, num_threads=2)
            temp[b].append(np.clip(p, 0, 1).astype(np.float32))
    for b in branches:
        key = summary['selected_regression'] if b == 'selected' else b
        outputs['share_' + b] = np.concatenate(temp[key])
    outputs['classifier_prob'] = np.concatenate(temp['classification'])
    return outputs
