"""R18 clean-production pretrained adapters; no changes to frozen R16 sources.

Both tasks consume the same complete 17-channel db-only representation. Each
task owns a separate pretrained object, stem, LoRA, norms and head. Training
reads only the R18 TRAIN and clean_ES roles. No Test/population fitting occurs.
The finite feature cache lives in RAM, never as a 1.5-GB disk artifact, and is
built once per process rather than recomputed at every epoch.
"""
from __future__ import annotations

from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'models/r18_production_r1/neural'
OPS = ROOT / 'ops/r18_production_r1/neural'
REPORT = ROOT / 'reports/r18_production_r1/neural'
DATA = ROOT / 'data/synthetic/r18_production_v1/manifest.json'
FAMILIES = ('dinov2_small', 'dasheng_base')
REGIMES = ('D0', 'H0')
TASKS = ('regression', 'classification')
SEED = 202610300618
CODE = (
    'pipeline/training_r18_production_neural.py', 'ops/run_r18_production_neural.py',
    'pipeline/pretrained_audio2d_r16.py', 'pipeline/load_foundation_r5.py',
    'pipeline/r16_feature_masking.py', 'pipeline/r16_cuda_precision.py',
    'pipeline/relative_grid_r6.py', 'pipeline/relative_grid_r6_v2.py',
    'ops/r16_temporal_features.py',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **value)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def _cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def atomic_pt(path, value):
    import torch
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        torch.save(_cpu(value), stream)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def export_best_from_latest(folder, checkpoint):
    """latest is the atomic transaction; best is a repairable compact export.

    Including the winning adapter in latest prevents a process death between
    two file replacements from selecting stale/future best weights on resume.
    Optimizer state is needed only in latest, never in the inference export.
    """
    fields = ('format', 'family', 'regime', 'task', 'best', 'best_epoch',
              'source_sha256', 'data_manifest_sha256', 'frozen_base_sha256', 'provenance')
    best = {k: checkpoint[k] for k in fields}
    best.update(adapter_state=checkpoint['best_adapter_state'], epoch=checkpoint['best_epoch'])
    atomic_pt(Path(folder) / 'best.pt', best)
    return best


def guard_training(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        name = str(args[0]).lower()
        if Path(name).name in ('test.csv', 'sample_submission.csv'):
            raise PermissionError('Test input forbidden in R18 training/preflight')


def pins():
    return {name: sha(ROOT / name) for name in CODE}


def verify_pins(expected):
    for name, digest in expected.items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Frozen run source changed: ' + name)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def feature17(db):
    from pipeline.relative_grid_r6_v2 import relative_grid_v2
    from ops.r16_temporal_features import temporal_grid
    a, w = relative_grid_v2(db)
    b, _ = temporal_grid(db)
    return np.concatenate((a, b), axis=1), w


def materialize_ram(raw, *, label='dataset', chunk=256):
    """Bound memory to one float32 feature cache; zero feature-cache disk use."""
    n = len(raw['db'])
    x = np.empty((n, 17, 15, 25), np.float32)
    w = np.empty((n, 15, 25), np.float32)
    started = time.perf_counter()
    for begin in range(0, n, chunk):
        x[begin:begin + chunk], w[begin:begin + chunk] = feature17(raw['db'][begin:begin + chunk])
        if begin and begin % (chunk * 40) == 0:
            write(OPS / 'progress.json', {'phase': 'features_RAM_only', 'dataset': label,
                  'rows': begin, 'total': n, 'seconds': time.perf_counter() - started})
    if not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError('Nonfinite features')
    if np.max(np.abs(w.sum((1, 2)) - 1)) > 2e-6:
        raise AssertionError('Physical weights do not normalize')
    if not np.array_equal(x[:, 5] > 0, np.isfinite(raw['db'])):
        raise AssertionError('Observation channel is not original O')
    if 'cell_share' in raw:
        gap = np.max(np.abs((w * raw['cell_share']).sum((1, 2)) - raw['share']))
        if gap > .0058:
            raise AssertionError('Rounded weight/raw target consistency bound failed')
    out = {k: v for k, v in raw.items() if k in ('share', 'presence', 'cell_share', 'sample_id')}
    out.update(x=x, w=w)
    receipt = {'dataset': label, 'n': n, 'seconds': time.perf_counter() - started,
               'RAM_feature_bytes': int(x.nbytes + w.nbytes), 'feature_cache_disk_bytes': 0,
               'channels': 17, 'classification_channels': list(range(17)),
               'PCA_residual_channels': False, 'LF_anchor_channels': False}
    return out, receipt


def validate_raw(raw, *, supervised=True):
    db = raw['db']
    if db.ndim != 3 or db.shape[1:] != (15, 25) or not len(db):
        raise ValueError('Expected nonempty [N,15,25] db')
    if np.isinf(db).any() or not np.isfinite(db).any((1, 2)).all():
        raise ValueError('Invalid observations')
    finite = db[np.isfinite(db)].astype(np.float64)
    if np.max(np.abs(finite - np.round(finite, 1))) > 1e-5:
        raise ValueError('Data must be quantized to 0.1 dB')
    if supervised:
        for key in ('share', 'presence', 'cell_share', 'sample_id'):
            if len(raw[key]) != len(db):
                raise ValueError('Label/id row mismatch')
        if not np.array_equal(raw['share'] > 0, raw['presence'] > 0):
            raise ValueError('Physical presence mismatch')
        if np.any(raw['cell_share'][~np.isfinite(db)] != 0):
            raise ValueError('Unobserved cell target nonzero')
        if len(np.unique(raw['presence'])) != 2:
            raise ValueError('Supervised role must include both classes')


def freeze_plan(manifest=DATA, *, device='cuda', microbatch=16, effective_batch=64):
    manifest = resolve(manifest)
    m = read(manifest)
    if m.get('status') != 'completed' or m.get('Test_used', False):
        raise ValueError('Completed no-Test production data manifest required')
    if device not in ('cuda', 'mps', 'cpu') or not 1 <= microbatch <= effective_batch:
        raise ValueError('Invalid device/batch')
    entries = {}
    for role in ('train', 'clean_ES'):
        entries[role] = {}
        for regime in REGIMES:
            e = dict(m['roles'][role][regime])
            path = resolve(e['path'])
            if sha(path) != e['sha256']:
                raise ValueError('Data artifact SHA differs: ' + str(path))
            e['path'] = str(path.relative_to(ROOT))
            entries[role][regime] = e
    p = {'format': 'r18_clean_neural_plan_v1', 'status': 'fixed_before_training',
         'families': list(FAMILIES), 'regimes': list(REGIMES), 'tasks': list(TASKS),
         'fold_scope': 'single source-separated split; no additional k-fold',
         'manifest_path': str(manifest.relative_to(ROOT)), 'data_manifest_sha256': sha(manifest),
         'data': entries, 'source_sha256': pins(), 'device': device,
         'microbatch': microbatch, 'effective_batch': effective_batch, 'seed': SEED,
         'warmup_epochs': 3, 'stage2_max_epochs': 25, 'patience': 7,
         'warmup_lr': .001, 'stage2_lr': .0002, 'stage2_eta_min': .00001,
         'weight_decay': .001, 'dropout': .1, 'lora_rank': 8,
         'regression_input_channels': list(range(17)), 'classification_input_channels': list(range(17)),
         'selection_role': 'clean_ES only; fresh_fit/gate/report never selects epoch',
         'features': 'unchanged relative_grid_r6_v2 8 + R16 temporal_grid 9',
         'masking': 'context masks + regression profile dropout only; O,w,q,y unchanged',
         'regression_loss': 'MSE(share,y) + 0.25 * sum(w*(q_hat-q)^2)',
         'classification_loss': 'independent BCEWithLogits(presence)',
         'Test_used': False, 'disk_feature_cache': False}
    path = OPS / 'RUN_PLAN.json'
    if path.exists() and read(path) != p:
        raise FileExistsError('Existing immutable R18 plan differs')
    write(path, p)
    write(REPORT / 'NEURAL_PLAN.json', p)
    return p


def plan_read():
    p = read(OPS / 'RUN_PLAN.json')
    verify_pins(p['source_sha256'])
    if sha(resolve(p['manifest_path'])) != p['data_manifest_sha256']:
        raise ValueError('Data manifest changed')
    return p


def role_data(plan, role, regime):
    e = plan['data'][role][regime]
    path = resolve(e['path'])
    if sha(path) != e['sha256']:
        raise RuntimeError('Training/ES data SHA changed')
    raw = load_npz(path)
    validate_raw(raw)
    if len(raw['share']) != e['n']:
        raise ValueError('Manifest row count differs')
    return raw


def branch(family, task, device='cpu'):
    """Public R16 backbone loader, new R18 full17 classifier stem wrapper."""
    import torch
    from pipeline.pretrained_audio2d_r16 import create_audio2d_branch
    from pipeline.r16_cuda_precision import install_fp32_classifier_pool
    if family not in FAMILIES or task not in TASKS:
        raise ValueError('R18 includes DINO/Dasheng and two independent tasks only')
    model, receipt = create_audio2d_branch(family, task, rank=8, dropout=.1,
                                         device='cpu', dtype='float32')
    if task == 'classification':
        model.stem = torch.nn.Conv2d(17, model.width, 3, stride=1, padding=1)
        model.in_channels = 17
        model.mask_channel = 5
        model.set_stage(1)
    model, precision = install_fp32_classifier_pool(model, family, task)
    model = model.to(device=device)
    receipt.update({'r18_full17_input': True, 'input_channels': list(range(17)),
                    'observation_channel': 5, 'independent_task_parameters': True,
                    'classification_profile_and_energy_restored': task == 'classification',
                    'precision_policy': precision})
    return model, receipt


def fingerprint(model):
    import torch
    h = hashlib.sha256()
    for name, p in model.frozen_named_parameters():
        h.update(name.encode())
        h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def device_setup(device):
    import torch
    torch.set_num_threads(2)
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('Silent MPS fallback forbidden')
    if device == 'cuda':
        from pipeline.r16_cuda_precision import require_cuda_bf16
        return require_cuda_bf16()
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no implicit CPU fallback')
    return {'device': device, 'torch': torch.__version__, 'BF16': False}


def autocast(device):
    import torch
    return torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()


def sync(device):
    import torch
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def tensor_batch(data, indices, device):
    import torch
    return tuple(torch.as_tensor(np.asarray(data[k][indices]).copy(), dtype=torch.float32, device=device)
                 for k in ('x', 'w', 'share', 'cell_share', 'presence'))


def objective(logits, task, w, y, q, z):
    import torch
    logits = logits.float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite logits')
    if task == 'classification':
        if logits.shape != z.shape:
            raise ValueError('Classifier shape differs')
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, z), logits.sigmoid()
    if logits.shape != w.shape:
        raise ValueError('Mask shape differs')
    qhat = logits.sigmoid()
    yhat = (w * qhat).sum((1, 2))
    return ((yhat - y).square() + .25 * (w * (qhat - q).square()).sum((1, 2))).mean(), yhat


def auc(z, p):
    z, p = np.asarray(z, bool), np.asarray(p, float)
    if not z.any() or z.all():
        return None
    order = np.argsort(p, kind='stable')
    a, z = p[order], z[order]
    starts = np.r_[0, np.flatnonzero(a[1:] != a[:-1]) + 1]
    pos = np.add.reduceat(z.astype(float), starts)
    neg = np.add.reduceat((~z).astype(float), starts)
    return float(np.sum(pos * (np.cumsum(neg) - .5 * neg)) / (pos.sum() * neg.sum()))


def metrics(y, p, task):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if task == 'classification':
        return {'n': len(y), 'auc': auc(y > 0, p), 'probability_mean': float(p.mean()),
                'probability_ge090': float(np.mean(p >= .90)), 'probability_ge099': float(np.mean(p >= .99))}
    return {'n': len(y), 'rmse': float(np.sqrt(np.mean((p - y) ** 2))),
            'bias': float(np.mean(p - y)), 'prediction_mean': float(p.mean()),
            'prediction_rms': float(np.sqrt(np.mean(p ** 2))), 'share_auc': auc(y > 0, p)}


def predict(model, task, data, device, batch, *, masks=False):
    import torch
    model.eval()
    pp, qq = [], []
    with torch.no_grad():
        for begin in range(0, len(data['x']), batch):
            sl = slice(begin, begin + batch)
            x = torch.as_tensor(np.asarray(data['x'][sl]).copy(), dtype=torch.float32, device=device)
            w = torch.as_tensor(np.asarray(data['w'][sl]).copy(), dtype=torch.float32, device=device)
            with autocast(device):
                logits = model(x)
            if task == 'classification':
                p = logits.float().sigmoid()
            else:
                q = logits.float().sigmoid()
                p = (w * q).sum((1, 2))
                if masks:
                    qq.append(q.cpu().numpy())
            if not torch.isfinite(p).all():
                raise FloatingPointError('Nonfinite inference')
            pp.append(p.cpu().numpy())
    return np.concatenate(pp), np.concatenate(qq) if qq else None


def preflight(family, *, steps=3):
    """Disposable real R18 steps for both independent FULL17 tasks."""
    import torch
    from pipeline.r16_feature_masking import apply_r16_feature_masking, make_feature_masking_generator
    sys.addaudithook(guard_training)
    p = plan_read()
    device, batch = p['device'], p['microbatch']
    hardware = device_setup(device)
    raw = role_data(p, 'train', 'D0')
    # Small real-data preflight only; full RAM cache is created by the train job.
    subset = {k: v[:max(steps * batch, 32)] for k, v in raw.items()}
    del raw
    data, front = materialize_ram(subset, label='preflight_real_R18_D0')
    result, pointers = {}, {}
    models = []
    for task in TASKS:
        torch.manual_seed(SEED)
        model, provenance = branch(family, task, device)
        models.append(model)
        pointers[task] = {t.data_ptr() for t in model.parameters()}
        base = fingerprint(model)
        model.set_stage(1)
        if any(q.requires_grad for name, q in model.named_parameters() if name.startswith('backbone.')):
            raise AssertionError('Warmup unfroze the backbone/LoRA')
        model.set_stage(2).train()
        opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=.0002, weight_decay=.001)
        masking = make_feature_masking_generator(SEED, stream='r18_preflight_' + task)
        times, losses = [], []
        for step in range(steps):
            x, w, y, q, z = tensor_batch(data, slice(step * batch, (step + 1) * batch), device)
            original = [v.clone() for v in (x, w, y, q, z)]
            sync(device)
            began = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            xa, meta = apply_r16_feature_masking(x, task=task, training=True, generator=masking)
            if not torch.equal(xa[:, 5], x[:, 5]):
                raise AssertionError('Feature augmentation changed O')
            with autocast(device):
                logits = model(xa)
            loss, prediction = objective(logits, task, w, y, q, z)
            loss.backward()
            if not all(torch.equal(a, b) for a, b in zip((x, w, y, q, z), original)):
                raise AssertionError('Augmentation mutated physical input/targets')
            if any(v.grad is not None for _, v in model.frozen_named_parameters()):
                raise AssertionError('Frozen pretrained tensor received gradients')
            norm = torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite adaptation gradient')
            if not model.stem.weight.grad.abs().sum() > 0:
                raise AssertionError('New FULL17 stem did not receive gradients')
            if task == 'classification' and (model.stem.weight.grad[:, (4, 7)].abs().sum() <= 0):
                raise AssertionError('Restored Ch4/Ch7 disconnected from classifier')
            opt.step()
            sync(device)
            times.append(time.perf_counter() - began)
            losses.append(float(loss.detach().cpu()))
        if fingerprint(model) != base:
            raise AssertionError('Pretrained base changed')
        elapsed = float(np.median(times[1:] or times))
        result[task] = {'status': 'passed', 'family': family, 'task': task,
                        'input_shape': [batch, 17, 15, 25], 'real_data': True,
                        'steps': steps, 'losses': losses, 'step_seconds': times,
                        'steady_seconds': elapsed, 'rows_per_second': batch / elapsed,
                        'training_epoch_estimate_seconds': p['data']['train']['D0']['n'] * elapsed / batch,
                        'FULL17_classifier_Ch4_Ch7_gradient_passed': task == 'classification',
                        'O_w_q_y_unmodified': True, 'permanent_base_unchanged': True,
                        'frozen_base_sha256': base, 'provenance': provenance,
                        'parameters_total': sum(v.numel() for v in model.parameters()),
                        'parameters_trainable_stage2': sum(v.numel() for v in model.parameters() if v.requires_grad),
                        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None}
        del opt
    if pointers['regression'] & pointers['classification']:
        raise AssertionError('Regression/classification share parameter storage')
    output = {'status': 'passed', 'family': family, 'hardware': hardware, 'frontend': front,
              'source_sha256': p['source_sha256'], 'data_manifest_sha256': p['data_manifest_sha256'],
              'independent_parameter_storage_passed': True, 'models': result,
              'disposable_preflight_is_not_training': True}
    write(OPS / f'PREFLIGHT_{family}.json', output)
    write(REPORT / f'PREFLIGHT_{family}.json', output)
    del models, model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return output


def train_job(family, regime, task):
    import torch
    from pipeline.r16_feature_masking import apply_r16_feature_masking, make_feature_masking_generator
    sys.addaudithook(guard_training)
    plan = plan_read()
    if family not in FAMILIES or regime not in REGIMES or task not in TASKS:
        raise ValueError('Job outside R18 fixed scope')
    pf = read(OPS / f'PREFLIGHT_{family}.json')
    if pf['source_sha256'] != plan['source_sha256'] or pf['data_manifest_sha256'] != plan['data_manifest_sha256']:
        raise ValueError('Preflight identity differs')
    device, batch = plan['device'], plan['microbatch']
    device_setup(device)
    folder = RUN / family / regime / task
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'summary.json').exists():
        complete = read(folder / 'summary.json')
        if complete['data_manifest_sha256'] != plan['data_manifest_sha256'] or complete['source_sha256'] != plan['source_sha256']:
            raise ValueError('Completed job identity differs')
        return 0
    tr, front_tr = materialize_ram(role_data(plan, 'train', regime), label='TRAIN_' + regime)
    va, front_va = materialize_ram(role_data(plan, 'clean_ES', regime), label='clean_ES_' + regime)
    write(folder / 'FEATURE_RAM_RECEIPT.json', {'train': front_tr, 'clean_ES': front_va})
    torch.manual_seed(SEED)
    np.random.seed(SEED % (2 ** 32))
    random.seed(SEED)
    model, provenance = branch(family, task, device)
    base = fingerprint(model)
    write(folder / 'PLAN.json', {'family': family, 'regime': regime, 'task': task,
                                'run_plan': plan, 'provenance': provenance,
                                'frozen_base_sha256': base, 'train_n': len(tr['share']),
                                'clean_ES_n': len(va['share'])})
    resume = None
    if (folder / 'latest.pt').exists():
        resume = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=True)
        for name, expected in (('source_sha256', plan['source_sha256']),
                               ('data_manifest_sha256', plan['data_manifest_sha256']),
                               ('frozen_base_sha256', base)):
            if resume[name] != expected:
                raise ValueError('Resume identity differs: ' + name)
        if (resume['family'], resume['regime'], resume['task']) != (family, regime, task):
            raise ValueError('Resume task mismatch')
        model.load_adapter_state(resume['adapter_state'])
        # Restore at once, including the stage1 epoch3 -> stage2 transition.
        # Reconstructing a public model/stem above consumed initialization RNG.
        torch.set_rng_state(resume['torch_rng'])
        if device == 'cuda':
            torch.cuda.set_rng_state(resume['device_rng'])
        elif device == 'mps':
            torch.mps.set_rng_state(resume['device_rng'])
        export_best_from_latest(folder, resume)
    best = resume['best'] if resume else (math.inf if task == 'regression' else -math.inf)
    best_epoch = resume['best_epoch'] if resume else 0
    bad = resume['patience_counter'] if resume else 0
    steps = resume['optimizer_steps'] if resume else 0
    best_adapter = resume['best_adapter_state'] if resume else None
    next_epoch = resume['epoch'] + 1 if resume else 1
    total_epochs = plan['warmup_epochs'] + plan['stage2_max_epochs']
    pause = {'requested': False}
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: pause.update(requested=True))
    started, opt, scheduler, current_stage = time.perf_counter(), None, None, 0
    reason = 'maximum_epochs'
    if resume and resume['stage'] == 2 and bad >= plan['patience']:
        next_epoch = total_epochs + 1
        reason = 'early_stopping'
    for epoch in range(next_epoch, total_epochs + 1):
        stage = 1 if epoch <= plan['warmup_epochs'] else 2
        stage_epoch = epoch if stage == 1 else epoch - plan['warmup_epochs']
        verify_pins(plan['source_sha256'])
        if stage != current_stage:
            model.set_stage(stage)
            opt = torch.optim.Adam([v for v in model.parameters() if v.requires_grad],
                                   lr=plan['warmup_lr'] if stage == 1 else plan['stage2_lr'],
                                   weight_decay=plan['weight_decay'])
            scheduler = None if stage == 1 else torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=plan['stage2_max_epochs'], eta_min=plan['stage2_eta_min'])
            if resume and resume['stage'] == stage:
                opt.load_state_dict(resume['optimizer_state'])
                if scheduler:
                    scheduler.load_state_dict(resume['scheduler_state'])
                torch.set_rng_state(resume['torch_rng'])
                if device == 'cuda':
                    torch.cuda.set_rng_state(resume['device_rng'])
                elif device == 'mps':
                    torch.mps.set_rng_state(resume['device_rng'])
            elif stage == 2:
                bad = 0
            current_stage = stage
        model.train()
        began, loss_sum, seen = time.perf_counter(), 0., 0
        order = np.random.default_rng([SEED, epoch]).permutation(len(tr['share']))
        maskgen = make_feature_masking_generator(SEED, epoch=epoch, stream='r18_' + task)
        effective = plan['effective_batch']
        for begin in range(0, len(order), effective):
            group = order[begin:begin + effective]
            opt.zero_grad(set_to_none=True)
            for offset in range(0, len(group), batch):
                ids = group[offset:offset + batch]
                x, w, y, q, z = tensor_batch(tr, ids, device)
                xa, _ = apply_r16_feature_masking(x, task=task, training=True, generator=maskgen)
                with autocast(device):
                    logits = model(xa)
                loss, _ = objective(logits, task, w, y, q, z)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                (loss * (len(ids) / len(group))).backward()
                loss_sum += float(loss.detach().cpu()) * len(ids)
                seen += len(ids)
            norm = torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite adaptation gradient')
            opt.step()
            steps += 1
            if steps % 20 == 0 or begin == 0:
                write(OPS / 'progress.json', {'phase': 'training', 'family': family, 'regime': regime,
                      'task': task, 'stage': stage, 'epoch': epoch, 'stage_epoch': stage_epoch,
                      'rows_seen': seen, 'train_rows': len(order), 'mean_loss': loss_sum / seen,
                      'epoch_elapsed_seconds': time.perf_counter() - began,
                      'best': best if best_epoch else None, 'best_epoch': best_epoch,
                      'patience_counter': bad, 'optimizer_steps': steps,
                      'cuda_allocated_bytes': torch.cuda.memory_allocated() if device == 'cuda' else None})
        predictions, _ = predict(model, task, va, device, batch)
        result = metrics(va['share'], predictions, task)
        value = result['rmse'] if task == 'regression' else result['auc']
        if value is None:
            raise ValueError('clean_ES must include both classes')
        improved = value < best - 1e-6 if task == 'regression' else value > best + 1e-6
        if improved:
            best, best_epoch, bad = value, epoch, 0
            best_adapter = model.adapter_state()
        elif stage == 2:
            bad += 1
        lr = opt.param_groups[0]['lr']
        if scheduler:
            scheduler.step()
        row = {'family': family, 'regime': regime, 'task': task, 'stage': stage, 'epoch': epoch,
               'stage_epoch': stage_epoch, 'train_loss': loss_sum / seen, 'clean_ES': result,
               'best': best, 'best_epoch': best_epoch, 'patience_counter': bad,
               'learning_rate': lr, 'epoch_seconds': time.perf_counter() - began,
               'optimizer_steps': steps}
        ck = {'format': 'r18_full17_clean_adapter_checkpoint_v1', 'family': family, 'regime': regime,
              'task': task, 'stage': stage, 'epoch': epoch, 'best': best, 'best_epoch': best_epoch,
              'patience_counter': bad, 'optimizer_steps': steps,
              'adapter_state': model.adapter_state(), 'best_adapter_state': best_adapter,
              'epoch_row': row, 'optimizer_state': opt.state_dict(),
              'scheduler_state': scheduler.state_dict() if scheduler else None,
              'torch_rng': torch.get_rng_state(),
              'device_rng': torch.cuda.get_rng_state() if device == 'cuda' else (
                  torch.mps.get_rng_state() if device == 'mps' else None),
              'source_sha256': plan['source_sha256'], 'data_manifest_sha256': plan['data_manifest_sha256'],
              'frozen_base_sha256': base, 'provenance': provenance}
        atomic_pt(folder / 'latest.pt', ck)
        if improved:
            export_best_from_latest(folder, ck)
            save(folder / 'clean_ES_best.npz', {'prediction': predictions,
                 'target_share': va['share'], 'sample_id': va['sample_id']})
        with (folder / 'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        write(folder / 'progress.json', row)
        write(OPS / 'progress.json', {'phase': 'epoch_completed', **row})
        print(json.dumps(row, allow_nan=False), flush=True)
        if pause['requested'] or (OPS / 'PAUSE_AFTER_EPOCH').exists():
            write(OPS / 'PAUSED.json', {'family': family, 'regime': regime, 'task': task,
                  'epoch': epoch, 'latest_sha256': sha(folder / 'latest.pt'),
                  'resume': 'run the same train/all command; resumes at next epoch'})
            return 20
        if stage == 2 and bad >= plan['patience']:
            reason = 'early_stopping'
            break
    if fingerprint(model) != base:
        raise AssertionError('Frozen base changed during training')
    committed = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=True)
    best_ck = export_best_from_latest(folder, committed)
    model.load_adapter_state(best_ck['adapter_state'])
    pred, _ = predict(model, task, va, device, batch)
    save(folder / 'clean_ES_best.npz', {'prediction': pred, 'target_share': va['share'], 'sample_id': va['sample_id']})
    # latest.pt is the authoritative boundary even if a previous process was
    # interrupted between atomically saving it and writing its progress JSON.
    epochs_run = committed['epoch']
    summary = {'status': 'completed', 'family': family, 'regime': regime, 'task': task,
               'input_channels': 17, 'best_epoch': best_epoch, 'epochs_run': epochs_run,
               'stop_reason': reason, 'clean_ES_best': metrics(va['share'], pred, task),
               'frozen_base_unchanged': True, 'frozen_base_sha256': base,
               'source_sha256': plan['source_sha256'], 'data_manifest_sha256': plan['data_manifest_sha256'],
               'seconds_this_invocation': time.perf_counter() - started,
               'artifacts': {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != 'summary.json'},
               'fresh_fit_gate_report_Test_used_for_epoch_selection': False}
    write(folder / 'summary.json', summary)
    return 0


def predict_npz(family, regime, task, input_path, output_path, *, device=None, masks=False):
    """Frozen-checkpoint inference; no policy or calibration fitting here."""
    import torch
    p = plan_read()
    device = device or p['device']
    device_setup(device)
    folder = RUN / family / regime / task
    if not (folder / 'summary.json').exists():
        raise RuntimeError('Completed/frozen checkpoint required before panel/Test inference')
    raw = load_npz(resolve(input_path))
    validate_raw(raw, supervised='share' in raw)
    data, front = materialize_ram(raw, label='frozen_inference')
    model, _ = branch(family, task, device)
    ck = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    if fingerprint(model) != ck['frozen_base_sha256']:
        raise ValueError('Inference frozen base differs')
    model.load_adapter_state(ck['adapter_state'])
    prediction, qhat = predict(model, task, data, device, p['microbatch'], masks=masks)
    out = {'prediction': prediction, 'sample_id': raw.get('sample_id', np.arange(len(prediction)).astype(str))}
    if 'share' in raw:
        out['target_share'] = raw['share']
    if qhat is not None:
        out['q_hat'] = qhat
    output_path = resolve(output_path)
    save(output_path, out)
    receipt = {'status': 'completed', 'family': family, 'regime': regime, 'task': task,
               'input_sha256': sha(resolve(input_path)), 'checkpoint_sha256': sha(folder / 'best.pt'),
               'output_sha256': sha(output_path), 'n': len(prediction), 'frontend': front,
               'feature_masking_at_inference': False, 'fitting_performed': False,
               'metrics': metrics(raw['share'], prediction, task) if 'share' in raw else None}
    write(output_path.with_suffix('.json'), receipt)
    return receipt
