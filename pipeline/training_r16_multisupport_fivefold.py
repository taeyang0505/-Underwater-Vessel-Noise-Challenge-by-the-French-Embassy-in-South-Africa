"""Versioned Fold1..4 extension; never edits/restarts the running Fold0 run."""
from pathlib import Path
import gc
import os
import random
import shutil
import signal
import sys
import time
import json
import numpy as np

from pipeline.r16_multisupport_data import ROOT, OUT, REGIMES, KINDS, read, write, sha, save, role, guard
from pipeline import training_r16_multisupport as base

MODELS = ROOT / 'models/r16_multisupport_fivefold_r1'
CACHE = ROOT / 'data/cache/r16_multisupport_fivefold_r1'
OPS = ROOT / 'ops/r16_multisupport_fivefold_r1'
NEW_SOURCES = ('pipeline/training_r16_multisupport_fivefold.py',
               'ops/run_r16_multisupport_fivefold.py',
               'ops/finalize_r16_multisupport_fivefold.py')


def codes():
    return {**base.codes(), **{n: sha(ROOT / n) for n in NEW_SOURCES}}


def jobs():
    return [(f, k, r) for f in range(1, 5) for k in KINDS for r in REGIMES]


def folder_for(fold, regime, kind):
    if fold not in range(5) or regime not in REGIMES or kind not in KINDS:
        raise ValueError('Unknown fold/regime/architecture')
    if fold == 0:
        return base.MODELS / 'neural' / regime / kind
    return MODELS / 'neural' / f'fold_{fold:02d}' / regime / kind


def validate_fold_manifest(fold):
    """Own train/CV metadata only; no CV signals/labels enter optimization."""
    import pandas as pd
    from pipeline.r16_production_data import _crossrole
    d = OUT / f'fold_{fold:02d}'
    tr = pd.read_parquet(d / 'train_backgrounds.parquet')
    cv = pd.read_parquet(d / 'cv_backgrounds.parquet')
    proof = _crossrole(tr, {'own_cv': cv})
    a = pd.read_parquet(d / 'train_references.parquet')
    b = pd.read_parquet(d / 'cv_references.parquet')
    if set(a.reference_component) & set(b.reference_component):
        raise ValueError('Own-fold reference overlap')
    return {'fold': fold, 'train_rows': len(tr), 'cv_rows': len(cv),
            'background_purge': proof, 'reference_component_overlap': 0,
            'validation_labels_used_for_optimization': False}


def cache_fold(fold):
    if fold not in range(1, 5):
        raise ValueError('Only new folds may create an extension cache')
    target = CACHE / f'fold_{fold:02d}'
    if target.exists():
        raise FileExistsError('No partial cache reuse')
    if shutil.disk_usage(ROOT).free < 6 * 1024**3:
        raise RuntimeError('Need >=6 GiB free before building one fold cache')
    pins = codes()
    proof = validate_fold_manifest(fold)
    target.mkdir(parents=True)
    for regime in REGIMES:
        data = role(fold, regime, 'train')
        n = len(data['share'])
        folder = target / regime
        folder.mkdir()
        x = np.lib.format.open_memmap(folder / 'x.npy', mode='w+', dtype=np.float32, shape=(n, 17, 15, 25))
        w = np.lib.format.open_memmap(folder / 'w.npy', mode='w+', dtype=np.float32, shape=(n, 15, 25))
        for start in range(0, n, 256):
            xx, ww = base.features(data['db'][start:start + 256])
            x[start:start + 256], w[start:start + 256] = xx, ww
            if not np.isfinite(xx).all() or not np.isfinite(ww).all():
                raise ValueError('Nonfinite cached features')
            if np.max(abs((data['cell_share'][start:start + 256] * ww).sum((1, 2))
                          - data['share'][start:start + 256])) > .0058:
                raise ValueError('Raw-target/rounded-weight bound failed')
        x.flush(); w.flush()
        del x, w
        for key in ('share', 'presence', 'cell_share', 'sample_id'):
            np.save(folder / (key + '.npy'), data[key], allow_pickle=False)
        write(folder / 'manifest.json', {'status': 'completed', 'fold': fold, 'regime': regime,
              'n': n, 'code_sha256': pins, 'data_manifest_sha256': sha(OUT / 'manifest.json'),
              'artifact_sha256': {p.name: sha(p) for p in folder.glob('*.npy')}})
        del data
        gc.collect()
    base.verify(pins)
    write(MODELS / f'fold_{fold:02d}_cache_receipt.json', {'status': 'completed',
          'fold': fold, 'source_proof': proof, 'regime_cache_manifests':
          {r: read(target / r / 'manifest.json') for r in REGIMES}, 'ephemeral_cache': True})


def read_cache(fold, regime):
    folder = CACHE / f'fold_{fold:02d}' / regime
    m = read(folder / 'manifest.json')
    if (m['fold'] != fold or m['regime'] != regime or m['code_sha256'] != codes()
            or m['data_manifest_sha256'] != sha(OUT / 'manifest.json')):
        raise ValueError('Wrong fold/cache binding')
    for name, h in m['artifact_sha256'].items():
        if sha(folder / name) != h:
            raise ValueError('Cached bytes changed')
    return base.cache_read(folder)


def clean_cache(fold):
    """Remove only this extension's derived npy cache after all 8 models pass."""
    for kind in KINDS:
        for regime in REGIMES:
            validate_completed(folder_for(fold, regime, kind), fold, regime, kind)
    folder = CACHE / f'fold_{fold:02d}'
    if folder.is_symlink() or folder.parent.resolve() != CACHE.resolve():
        raise ValueError('Refuse cache path escape')
    expected = {'x.npy', 'w.npy', 'share.npy', 'presence.npy', 'cell_share.npy', 'sample_id.npy'}
    for regime in REGIMES:
        part = folder / regime
        m = read(part / 'manifest.json')
        if set(m['artifact_sha256']) != expected:
            raise ValueError('Unexpected cache artifacts; cleanup stopped')
        for name, h in m['artifact_sha256'].items():
            path = part / name
            if path.is_symlink() or sha(path) != h:
                raise ValueError('Cache cleanup identity failed')
            path.unlink()
        write(part / 'CACHE_RELEASED.json', {'status': 'released', 'fold': fold,
              'reason': 'All eight model artifacts verified; raw data/checkpoints retained'})


def validate_completed(folder, fold, regime, kind):
    s = read(folder / 'summary.json')
    expected_codes = base.codes() if fold == 0 else codes()
    if (s['status'] != 'completed' or s['epochs'] != 100 or s['regime'] != regime
            or s['kind'] != kind or s['code_sha256'] != expected_codes):
        raise ValueError('Incomplete or mismatched neural member')
    if fold and (s['fold'] != fold or s['fresh_initialization'] is not True):
        raise ValueError('New folds must initialize independently')
    for name, h in s['artifact_sha256'].items():
        if sha(folder / name) != h:
            raise ValueError('Neural member bytes changed')
    rows = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
    if [r['epoch'] for r in rows] != list(range(1, 101)):
        raise ValueError('Exactly 100 completed epochs required')
    return s


def train_one(fold, kind, regime):
    import torch
    from pipeline.models_r16_multisupport import MultiSupportModel, decoupled_loss, clip_decoupled_gradients
    from pipeline.training_r16_production import rng_state
    if (fold, kind, regime) not in jobs():
        raise ValueError('Unapproved new-fold job')
    sys.addaudithook(guard)
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        raise RuntimeError('Explicit MPS available/fallback0 required')
    plan = read(OPS / 'PLAN.json')
    pins = codes()
    if plan['code_sha256'] != pins:
        raise ValueError('Queued sources changed')
    base.verify(pins)
    datahash = sha(OUT / 'manifest.json')
    pf = read(base.MODELS / 'ACTUAL_PREFLIGHT.json')
    if pf['code_sha256'] != base.codes() or pf['data_manifest_sha256'] != datahash:
        raise ValueError('Original actual-device preflight differs')
    folder = folder_for(fold, regime, kind)
    if folder.exists():
        raise FileExistsError('No failed partial run/automatic replay')
    tr = read_cache(fold, regime)
    va = base.cache_read(base.cache_role('D0', 'selection'))
    device, batch = pf['models'][kind]['device'], pf['models'][kind]['batch']
    seed = base.SEED + 10007 * fold
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = MultiSupportModel(kind).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
    folder.mkdir(parents=True)
    write(folder / 'PLAN.json', {'fold': fold, 'kind': kind, 'regime': regime,
          'fresh_initialization': True, 'warm_start_from_fold0': False, 'seed': seed,
          'epochs': 100, 'optimizer': 'Adam', 'early_stopping': False,
          'device': device, 'batch': batch, 'code_sha256': pins, 'data_manifest_sha256': datahash,
          'architecture': model.architecture(), 'validation': 'predeclared shared local_fit only',
          'CV_or_locked_panels_or_Test_for_optimization': False})
    pause = {'requested': False}
    signal.signal(signal.SIGUSR1, lambda *_: pause.update(requested=True))
    best, bc, be, bce, steps = float('inf'), -1., 0, 0, 0
    for epoch in range(1, 101):
        base.verify(pins)
        started = time.perf_counter(); seen = 0; sse = cell = clf = 0.
        order = np.random.default_rng([seed, epoch]).permutation(len(tr['share']))
        model.train()
        for off in range(0, len(order), batch):
            ix = order[off:off + batch]
            x = torch.tensor(np.asarray(tr['x'][ix]), device=device)
            w = torch.tensor(np.asarray(tr['w'][ix]), device=device)
            target = tuple(torch.tensor(np.asarray(tr[k][ix]), dtype=torch.float32, device=device)
                           for k in ('share', 'presence', 'cell_share'))
            opt.zero_grad(set_to_none=True)
            loss, parts = decoupled_loss(model(x, w), target, w)
            loss.backward(); clip_decoupled_gradients(model); opt.step()
            steps += 1; seen += len(ix)
            sse += float(parts['share_mse'].detach().sum().cpu())
            cell += float(parts['weighted_cell_mse'].detach().sum().cpu())
            clf += float(parts['classification_bce'].detach().sum().cpu())
            if off == 0 or steps % 100 == 0:
                write(MODELS / 'progress.json', {'fold': fold, 'kind': kind, 'regime': regime,
                      'epoch': epoch, 'rows_seen': seen, 'total_rows': len(order), 'optimizer_steps': steps})
        p, c = base.predict_model(model, va['x'], va['w'], device, batch)
        m = base.metric(va['share'], p, c)
        ri, ci = m['rmse'] < best, m['classifier_auc'] > bc
        if ri: best, be = m['rmse'], epoch
        if ci: bc, bce = m['classifier_auc'], epoch
        lr = opt.param_groups[0]['lr']; scheduler.step()
        ck = {'format': 'r16_multisupport_fivefold_checkpoint_v1', 'fold': fold,
              'kind': kind, 'regime': regime, 'epoch': epoch, 'fresh_initialization': True,
              'state_dict': model.state_dict(), 'architecture': model.architecture(),
              'optimizer_state_dict': opt.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
              'rng': rng_state(device), 'optimizer_steps': steps, 'best_regression_epoch': be,
              'best_regression_rmse': best, 'best_classifier_epoch': bce, 'best_classifier_auc': bc,
              'code_sha256': pins, 'data_manifest_sha256': datahash}
        if ri:
            base.atomic_pt(folder / 'best_regression.pt', ck)
            save(folder / 'selection_regression.npz', {'share': p, 'sample_id': va['sample_id'], 'target_share': va['share']})
        if ci:
            base.atomic_pt(folder / 'best_classifier.pt', ck)
            save(folder / 'selection_classification.npz', {'classifier_prob': c, 'sample_id': va['sample_id'], 'target_share': va['share']})
        base.atomic_pt(folder / 'latest.pt', ck)
        row = {'fold': fold, 'kind': kind, 'regime': regime, 'epoch': epoch,
               'train_rmse': float(np.sqrt(sse / seen)), 'train_cell_mse': cell / seen,
               'train_classifier_BCE': clf / seen, 'selection': m, 'best_RMSE': best,
               'best_regression_epoch': be, 'best_classifier_AUC': bc, 'best_classifier_epoch': bce,
               'learning_rate': lr, 'epoch_seconds': time.perf_counter() - started}
        with (folder / 'epochs.jsonl').open('a') as f: f.write(json.dumps(row, allow_nan=False) + '\n')
        write(folder / 'progress.json', row)
        print(json.dumps(row), flush=True)
        if pause['requested'] or (MODELS / 'PAUSE_AFTER_EPOCH').exists():
            write(MODELS / 'PAUSED.json', {'fold': fold, 'regime': regime, 'kind': kind,
                  'epoch': epoch, 'latest_sha256': sha(folder / 'latest.pt'), 'explicit_resume_required': True})
            raise RuntimeError('Paused after saved epoch; no automatic continuation')
    base.verify(pins)
    write(folder / 'summary.json', {'status': 'completed', 'fold': fold, 'regime': regime,
          'kind': kind, 'epochs': 100, 'early_stopping': False, 'fresh_initialization': True,
          'best_regression_epoch': be, 'best_rmse': best, 'best_classifier_epoch': bce,
          'best_classifier_auc': bc, 'code_sha256': pins, 'data_manifest_sha256': datahash,
          'artifact_sha256': {p.name: sha(p) for p in folder.iterdir() if p.is_file()}})
    validate_completed(folder, fold, regime, kind)
