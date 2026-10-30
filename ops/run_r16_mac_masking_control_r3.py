"""Small prospective Mac masking control; never a production training runner.

Four DINOv2-small jobs: D0 / Fold 0 / regression+classification / off+both.
Only existing TRAIN and the prospective local_fit partition may be read.
No scratch continuation, Test, locked panels, or cloud activity occurs here.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
OPS = ROOT / 'ops/r16_mac_masking_control_r3'
RUN = ROOT / 'models/r16_mac_masking_control_r3'
REPORT = ROOT / 'reports/r16_mac_masking_control_r3'
CAT = ROOT / 'ops/r16_catboost_peft_cpu8_r1'
DATA = ROOT / 'data/synthetic/r16_multisupport_v1'
CACHE = ROOT / 'data/cache/r16_multisupport_full_r1'
SPLIT = ROOT / 'ops/r16_pretrained_masking_r1/LOCAL_FIT_SPLIT.json'
SEED = 202610282230
NUMPY_LEGACY_SEED = SEED % (2 ** 32)
SAMPLE_N = 2048
JOBS = [(task, variant) for task in ('regression', 'classification')
        for variant in ('off', 'both')]


def stamp():
    import datetime
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def access_guard(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
        return
    s = os.fsdecode(args[0]).lower()
    p = Path(s)
    if p.name in ('test.csv', 'sample_submission.csv'):
        raise PermissionError('No query data access in Mac control')
    if p.suffix in ('.npz', '.npy', '.parquet', '.csv') and any(
        x in s for x in ('local_gate', 'local_report', 'domain_holdout', '/r11_', '/eda_deep/')
    ):
        raise PermissionError('No locked evaluation panel access in Mac control')


def pins():
    from pipeline.training_r16_pretrained_masked import source_pins
    return {**source_pins(), str(Path(__file__).relative_to(ROOT)): sha(__file__)}


def verify_sources(plan):
    if pins() != plan['source_sha256']:
        raise ValueError('Immutable control source closure changed')
    for rel, expected in plan['input_sha256'].items():
        if sha(ROOT / rel) != expected:
            raise ValueError('Immutable control input changed: ' + rel)


def verify_caches(plan):
    """Recheck mmap array bytes at each process start, not every epoch."""
    verify_sources(plan)
    for folder in (CACHE / 'D0/train', CACHE / 'shared_selection/selection'):
        for name, expected in read(folder / 'manifest.json')['artifact_sha256'].items():
            if sha(folder / name) != expected:
                raise ValueError('Existing cache array changed: ' + str(folder / name))


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Fail closed; never signal or stop another process.


def catboost_ready():
    if not (CAT / 'state.json').is_file():
        return False, {'reason': 'CatBoost state unavailable'}
    state = read(CAT / 'state.json')
    if state.get('phase') != 'completed' or state.get('pair_count') != 10:
        return False, {'reason': 'CatBoost has not completed all 10 pairs',
                       'phase': state.get('phase'), 'completed_pairs': len(state.get('completed_pairs', []))}
    ledger = ROOT / 'models/r16_catboost_peft_cpu8_r1/ALL_PAIRS_COMPLETED.json'
    if not ledger.is_file() or read(ledger).get('pair_count') != 10:
        return False, {'reason': 'CatBoost completed ledger absent'}
    live = []
    for name, key in (('LAUNCHED.json', 'supervisor_pid'), ('worker.json', 'pid')):
        p = CAT / name
        pid = read(p).get(key) if p.exists() else None
        if alive(pid):
            live.append(pid)
    if live:
        return False, {'reason': 'CatBoost supervisor/worker PID still alive', 'pids': live}
    if (CAT / 'supervisor.lock').exists():
        return False, {'reason': 'CatBoost supervisor lock remains'}
    return True, {'reason': 'CatBoost completed and recorded processes exited',
                  'state_sha256': sha(CAT / 'state.json'), 'ledger_sha256': sha(ledger)}


def wait_for_catboost(wait=False):
    while True:
        ok, status = catboost_ready()
        if ok:
            return status
        if status.get('phase') == 'failed':
            raise RuntimeError('CatBoost failed; fix it before starting concurrent Mac MPS work')
        print(json.dumps({'phase': 'waiting_for_catboost', **status}, ensure_ascii=False), flush=True)
        if not wait:
            raise RuntimeError('MPS control waits for CatBoost completion; --wait-catboost permits waiting')
        time.sleep(30)


def proportional_source_subset(meta, n=SAMPLE_N):
    """Proportional source/operator/class/support strata, deterministic within each.

    Existing TRAIN-versus-local_fit isolation is preserved, not re-invented by
    row splitting. Sampling is metadata-only; no prediction or target magnitude.
    """
    import numpy as np
    if n < 2 or n > 4096:
        raise ValueError('Small-control sample must be 2..4096 rows')
    support = np.select([meta['active_minutes'] == 0, meta['active_minutes'] <= 4,
                         meta['active_minutes'] <= 10], [0, 1, 2], default=3)
    groups = {}
    for i, key in enumerate(zip(meta['background_group'].astype(str), meta['operator'].astype(str),
                                meta['presence'].astype(int), support)):
        groups.setdefault(tuple(key), []).append(i)
    keys = sorted(groups)
    raw = np.asarray([len(groups[k]) * n / len(support) for k in keys])
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    for j in np.argsort(-(raw - counts), kind='stable')[:remainder]:
        counts[j] += 1
    rng = np.random.default_rng(SEED)
    selected = []
    for key, count in zip(keys, counts):
        selected.extend(rng.permutation(groups[key])[:count].tolist())
    selected = np.asarray(sorted(selected), dtype=np.int64)
    if len(selected) != n or len(set(selected.tolist())) != n:
        raise AssertionError('Metadata sample allocation failed')
    return selected


def make_plan():
    import numpy as np
    import pandas as pd
    from pipeline.training_r16_pretrained_masked import selection_split
    from pipeline.r16_production_data import _crossrole
    from pipeline.r16_feature_masking import FeatureMaskingConfig
    if (OPS / 'PLAN.json').exists():
        raise FileExistsError('Control plan exists; never overwrite a fixed plan')
    manifest = read(DATA / 'manifest.json')
    entry = manifest['folds']['0']['roles']['train']['D0']
    path = ROOT / entry['path']
    if sha(path) != entry['sha256']:
        raise ValueError('TRAIN NPZ changed')
    with np.load(path, allow_pickle=False) as f:
        meta = {k: f[k] for k in ('sample_id', 'base_window_index', 'background_group',
                'site', 'operator', 'presence', 'active_minutes')}
    indices = proportional_source_subset(meta)
    split, roles = selection_split()
    train_metadata = DATA / 'fold_00/train_backgrounds.parquet'
    frame = pd.read_parquet(train_metadata)
    frame = frame[frame.base_window_index.isin(meta['base_window_index'][indices])].copy()
    fit_frame = pd.read_parquet(ROOT / split['source_metadata'])
    proof = _crossrole(frame, {'checkpoint_and_ensemble': fit_frame})
    for row in proof:
        if row.get('base_window_overlap') or row.get('background_group_overlap') or row.get('minimum_support_gap_hours', 0) < 24:
            raise ValueError('Selected TRAIN versus local_fit purge failed')
    inputs = {str((DATA / 'manifest.json').relative_to(ROOT)): sha(DATA / 'manifest.json'),
              entry['path']: entry['sha256'], str(SPLIT.relative_to(ROOT)): sha(SPLIT),
              str(train_metadata.relative_to(ROOT)): sha(train_metadata),
              split['source_metadata']: split['source_metadata_sha256']}
    for folder in (CACHE / 'D0/train', CACHE / 'shared_selection/selection'):
        m = read(folder / 'manifest.json')
        if m['data_manifest_sha256'] != sha(DATA / 'manifest.json'):
            raise ValueError('Existing cache manifest differs')
        inputs[str((folder / 'manifest.json').relative_to(ROOT))] = sha(folder / 'manifest.json')
        for name, expected in m['artifact_sha256'].items():
            if sha(folder / name) != expected:
                raise ValueError('Existing cache checksum differs')
    chosen = split['chosen']
    plan = {'format': 'r16_mac_masking_small_control_v2_seed32_compat', 'status': 'fixed_before_training',
        'created_at_kst': stamp(), 'family': 'dinov2_small', 'fold': 0, 'regime': 'D0',
        'jobs': [{'task': task, 'variant': v} for task, v in JOBS], 'small_control_only': True,
        'sample_n': len(indices), 'sample_indices': indices.tolist(),
        'sample_ids': meta['sample_id'][indices].astype(str).tolist(),
        'metadata_sampling': 'proportional background_group x operator x presence x active-support-bin',
        'selected_background_groups': sorted(set(meta['background_group'][indices].astype(str))),
        'selected_sites': sorted(set(meta['site'][indices].astype(str))),
        'selected_positive_fraction': float(meta['presence'][indices].mean()),
        'checkpoint_n': len(roles['checkpoint_selection']), 'comparison_n': len(roles['ensemble_fit']),
        'checkpoint_role': 'prospective local_fit checkpoint_selection only',
        'comparison_role': 'prospective local_fit ensemble_fit only, after checkpoint fixed',
        'checkpoint_backgrounds': chosen['checkpoint_selection']['unique_backgrounds'],
        'comparison_backgrounds': chosen['ensemble_fit']['unique_backgrounds'],
        'TRAIN_local_fit_purge': proof, 'local_fit_crossrole_purge': chosen['crossrole_proof'],
        'seed': SEED, 'numpy_legacy_seed_u32': NUMPY_LEGACY_SEED,
        'seed_contract': 'Retain 64-bit root for default_rng/Python/Torch/MPS; NumPy legacy API uses root modulo 2**32',
        'supersedes_failed_run': 'r16_mac_masking_control_r2',
        'same_initialization_and_epoch_order_across_mask_variants': True,
        'warmup_epochs': 3, 'stage2_max_epochs': 5, 'small_control_early_stopping': False,
        'production_stage2_max25_ES7_not_changed': True, 'effective_batch': 64, 'microbatch': 8,
        'device': 'mps', 'precision': 'float32', 'implicit_CPU_fallback': False,
        'optimizer': 'Adam', 'warmup_lr': .001, 'stage2_lr': .0002, 'stage2_min_lr': .00001,
        'weight_decay': .001, 'dropout': .1, 'LoRA_rank': 8, 'gradient_clip': 1.,
        'masking': {v: FeatureMaskingConfig.for_variant(v).state_dict() for v in ('off', 'both')},
        'source_sha256': pins(), 'input_sha256': inputs,
        'Test_used': False, 'locked_panel_used': False, 'comparison_used_for_epochs': False,
        'limitations': ['2,048-row / eight-epoch single-family D0 control is not full-scale convergence evidence.',
            'Cannot establish masking advantage for Dasheng or TimeMoE, H0 or five folds.',
            'Checkpoint and comparison roles are background-source separated but share some sensors and external references.',
            'Comparison is a development comparison, not a previously untouched official proxy.',
            'No claim of local-to-leaderboard equivalence or guaranteed transfer improvement.']}
    write(OPS / 'PLAN.json', plan)
    write(OPS / 'state.json', {'phase': 'planned_not_started', 'plan_sha256': sha(OPS / 'PLAN.json'),
                              'created_at_kst': stamp(), 'training_started': False})
    return plan


def fixed_data(plan):
    import numpy as np
    from pipeline.training_r16_multisupport import cache_read
    from pipeline.training_r16_pretrained_masked import selection_split
    tr = cache_read(CACHE / 'D0/train')
    ix = np.asarray(plan['sample_indices'], dtype=int)
    if tr['sample_id'][ix].astype(str).tolist() != plan['sample_ids']:
        raise ValueError('TRAIN cache sample order changed')
    subset = {k: np.asarray(v[ix]).copy() for k, v in tr.items()}
    _, split = selection_split()
    fit = cache_read(CACHE / 'shared_selection/selection')
    roles = {r: {k: np.asarray(v[ids]).copy() for k, v in fit.items()} for r, ids in split.items()}
    return subset, roles


def check_device():
    import torch
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        raise RuntimeError('MPS CPU fallback must explicitly be 0')
    if not torch.backends.mps.is_available():
        raise RuntimeError('Native MPS unavailable; no CPU fallback')
    if shutil.disk_usage(ROOT).free < 3 * 1024 ** 3:
        raise RuntimeError('Less than 3 GiB free; do not begin another job')


def seed_frameworks(include_mps=True):
    """Keep the original 64-bit root seed; adapt only NumPy's legacy API.

    default_rng metadata sampling and epoch orders keep SEED unchanged. Python,
    PyTorch and MPS accept the root seed; np.random.seed requires uint32. Both
    off/both branches use the exact same mapping, so pairing stays deterministic.
    """
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(NUMPY_LEGACY_SEED)
    torch.manual_seed(SEED)
    if include_mps:
        torch.mps.manual_seed(SEED)
    return {'root_seed_u64': SEED, 'numpy_legacy_seed_u32': NUMPY_LEGACY_SEED,
            'stateless_default_rng_seed_unchanged': True,
            'python_torch_mps_root_seed_unchanged': True}


def initialize(task):
    import numpy as np
    import torch
    from pipeline.training_r16_pretrained_masked import branch
    torch.set_num_threads(2)
    seed_frameworks(include_mps=True)
    return branch('dinov2_small', task, 'mps')


def adapter_sha(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.adapter_state().items()):
        h.update(name.encode()); h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def batch_tensors(data, indices):
    import numpy as np
    import torch
    return tuple(torch.tensor(np.asarray(data[k][indices]), dtype=torch.float32, device='mps')
                 for k in ('x', 'w', 'share', 'cell_share', 'presence'))


def preflight(wait=False):
    import numpy as np
    import torch
    from pipeline.training_r16_pretrained_masked import base_fingerprint, task_input, loss_and_values
    from pipeline.r16_feature_masking import (FeatureMaskingConfig, apply_r16_feature_masking,
                                             make_feature_masking_generator)
    ready = wait_for_catboost(wait); check_device()
    plan = read(OPS / 'PLAN.json'); verify_caches(plan)
    data, _ = fixed_data(plan); receipts = []
    for task, variant in JOBS:
        model, provenance = initialize(task); before = base_fingerprint(model); initial = adapter_sha(model)
        model.set_stage(2); model.train()
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.0002, weight_decay=.001)
        x, w, y, q, z = batch_tensors(data, slice(0, plan['microbatch']))
        originals = [v.clone() for v in (x, w, y, q, z)]
        torch.mps.synchronize(); began = time.perf_counter()
        xa, meta = apply_r16_feature_masking(x, task=task, training=True,
            generator=make_feature_masking_generator(SEED, stream='small_control_preflight'),
            config=FeatureMaskingConfig.for_variant(variant))
        loss, _, _ = loss_and_values(model(task_input(xa, task)), task, w, y, q, z)
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm): raise FloatingPointError('Invalid MPS update')
        if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for n,p in model.named_parameters() if n.startswith('stem.')):
            raise AssertionError('No input-stem adaptation gradient')
        if any(p.grad is not None for _, p in model.frozen_named_parameters()):
            raise AssertionError('Frozen base gradient')
        if not all(torch.equal(v, old) for v, old in zip((x,w,y,q,z), originals)):
            raise AssertionError('Input, weights or labels mutated')
        opt.step(); torch.mps.synchronize(); elapsed = time.perf_counter() - began
        if base_fingerprint(model) != before: raise AssertionError('Frozen base changed')
        receipts.append({'task': task, 'variant': variant, 'initial_adapter_sha256': initial,
            'single_disposable_step_seconds': elapsed, 'microbatch': plan['microbatch'],
            'loss': float(loss.detach().cpu()), 'masking': meta.log_dict(),
            'frozen_base_unchanged': True, 'provenance': provenance,
            'disposable_connection_check_not_full_training': True})
        del model,opt,x,xa,w,y,q,z,loss,originals; gc.collect(); torch.mps.empty_cache()
    for task in ('regression','classification'):
        pair = [r for r in receipts if r['task'] == task]
        if pair[0]['initial_adapter_sha256'] != pair[1]['initial_adapter_sha256']:
            raise AssertionError('Masking pair initialization differs')
    result = {'status': 'passed', 'receipts': receipts, 'catboost_guard': ready,
              'plan_sha256': sha(OPS / 'PLAN.json'), 'source_sha256': plan['source_sha256'],
              'production_training_started': False, 'created_at_kst': stamp()}
    write(OPS / 'PREFLIGHT.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def train_worker(task, variant):
    import numpy as np
    import torch
    from pipeline.training_r16_pretrained_masked import (base_fingerprint, task_input,
        loss_and_values, predict)
    from pipeline.training_r16_multisupport import atomic_pt, metric
    from pipeline.r16_multisupport_data import save
    from pipeline.r16_feature_masking import (FeatureMaskingConfig, apply_r16_feature_masking,
                                             make_feature_masking_generator)
    wait_for_catboost(False); check_device()
    plan = read(OPS / 'PLAN.json'); verify_caches(plan)
    pf = read(OPS / 'PREFLIGHT.json')
    if pf['status'] != 'passed' or pf['plan_sha256'] != sha(OPS / 'PLAN.json'):
        raise ValueError('Current actual MPS preflight required')
    folder = RUN / task / variant
    if folder.exists(): raise FileExistsError('Fresh small-control job required; no partial resume')
    folder.mkdir(parents=True)
    data, roles = fixed_data(plan); val = roles['checkpoint_selection']; hold = roles['ensemble_fit']
    model, provenance = initialize(task); before = base_fingerprint(model); initial = adapter_sha(model)
    maskcfg = FeatureMaskingConfig.for_variant(variant)
    write(folder / 'PLAN.json', {'task':task,'variant':variant,'plan_sha256':sha(OPS/'PLAN.json'),
        'initial_adapter_sha256':initial,'frozen_base_sha256':before,'provenance':provenance})
    best = math.inf if task == 'regression' else -math.inf
    best_epoch = 0; steps = 0; began = time.perf_counter()
    for stage, count in ((1,3),(2,5)):
        model.set_stage(stage)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
            lr=.001 if stage == 1 else .0002, weight_decay=.001)
        scheduler = None if stage == 1 else torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=5,eta_min=.00001)
        for local_epoch in range(1,count+1):
            epoch = local_epoch if stage == 1 else 3+local_epoch
            verify_sources(plan); model.train(); start = time.perf_counter()
            order = np.random.default_rng([SEED,epoch]).permutation(len(data['share']))
            order_sha = hashlib.sha256(order.tobytes()).hexdigest()
            gen = make_feature_masking_generator(SEED,epoch=epoch,stream='small_control_train')
            total = 0.; seen = 0
            for begin in range(0,len(order),plan['effective_batch']):
                group = order[begin:begin+plan['effective_batch']]; opt.zero_grad(set_to_none=True)
                for off in range(0,len(group),plan['microbatch']):
                    ids = group[off:off+plan['microbatch']]
                    x,w,y,q,z = batch_tensors(data,ids)
                    xa,_ = apply_r16_feature_masking(x,task=task,training=True,generator=gen,config=maskcfg)
                    loss,_,_ = loss_and_values(model(task_input(xa,task)),task,w,y,q,z)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite control loss')
                    (loss*(len(ids)/len(group))).backward()
                    total += float(loss.detach().cpu())*len(ids); seen += len(ids)
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                if not torch.isfinite(norm): raise FloatingPointError('Nonfinite control gradient')
                opt.step(); steps += 1
                if steps%8 == 0:
                    write(OPS/'progress.json',{'phase':'training','task':task,'variant':variant,
                        'epoch':epoch,'max_epochs':8,'seen':seen,'train_n':len(order),'updated_at_kst':stamp()})
            prediction = predict(model,task,val,'mps',plan['microbatch'])
            cv = metric(val['share'], prediction if task=='regression' else np.zeros_like(prediction),
                        prediction if task=='classification' else np.zeros_like(prediction))
            value = cv['rmse'] if task=='regression' else cv['classifier_auc']
            better = value < best-1e-6 if task=='regression' else value > best+1e-6
            if better:
                best=value; best_epoch=epoch
                atomic_pt(folder/'best.pt',{'format':'r16_small_control_adapter_v1','adapter_state':model.adapter_state(),
                    'task':task,'variant':variant,'epoch':epoch,'best_cv':best,
                    'plan_sha256':sha(OPS/'PLAN.json'),'base_sha256':before})
                save(folder/'checkpoint_best.npz',{'prediction':prediction,'target_share':val['share'],'sample_id':val['sample_id']})
            row={'task':task,'variant':variant,'stage':stage,'epoch':epoch,'max_epochs':8,
                 'train_loss':total/seen,'checkpoint_metric':value,'best_metric':best,'best_epoch':best_epoch,
                 'epoch_seconds':time.perf_counter()-start,'train_order_sha256':order_sha,'optimizer_steps':steps}
            if scheduler: scheduler.step()
            with (folder/'epochs.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
            write(OPS/'progress.json',{'phase':'epoch_completed',**row,'updated_at_kst':stamp()})
            print(json.dumps(row),flush=True)
    if base_fingerprint(model) != before: raise AssertionError('Frozen base changed')
    checkpoint = torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    model.load_adapter_state(checkpoint['adapter_state'])
    pred = predict(model,task,hold,'mps',plan['microbatch'])
    save(folder/'comparison.npz',{'prediction':pred,'target_share':hold['share'],'sample_id':hold['sample_id']})
    m = metric(hold['share'],pred if task=='regression' else np.zeros_like(pred),
               pred if task=='classification' else np.zeros_like(pred))
    result={'status':'completed','task':task,'variant':variant,'epochs_run':8,'best_epoch':best_epoch,
        'folder':str(folder.relative_to(ROOT)),
        'best_checkpoint_metric':best,'comparison_metrics':m,'initial_adapter_sha256':initial,
        'frozen_base_unchanged':True,'seconds':time.perf_counter()-began,
        'plan_sha256':sha(OPS/'PLAN.json'),'source_sha256':plan['source_sha256'],
        'small_control_not_production':True,'Test_used':False,'locked_panels_used':False,
        'artifact_sha256':{p.name:sha(p)for p in folder.iterdir()if p.is_file()}}
    write(folder/'summary.json',result)


def summarize():
    import numpy as np
    from pipeline.training_r16_multisupport import metric
    plan=read(OPS/'PLAN.json'); rows=[]
    for task,variant in JOBS:
        folder=RUN/task/variant; summary=read(folder/'summary.json')
        if summary['status']!='completed' or summary['plan_sha256']!=sha(OPS/'PLAN.json'):
            raise ValueError('Incomplete or stale comparison job')
        for name,h in summary['artifact_sha256'].items():
            if sha(folder/name)!=h: raise ValueError('Control artifact changed')
        rows.append(summary)
    for task in ('regression','classification'):
        pair=[r for r in rows if r['task']==task]
        if pair[0]['initial_adapter_sha256']!=pair[1]['initial_adapter_sha256']:
            raise ValueError('Pair initial weights differed')
        logs=[ [json.loads(v)['train_order_sha256']for v in (RUN/task/variant/'epochs.jsonl').read_text().splitlines()]
               for variant in ('off','both')]
        if logs[0]!=logs[1]:raise ValueError('Pair training order differed')
    combined={}
    for variant in ('off','both'):
        with np.load(RUN/'regression'/variant/'comparison.npz',allow_pickle=False)as a:
            y=a['target_share'];p=a['prediction'];ids=a['sample_id']
        with np.load(RUN/'classification'/variant/'comparison.npz',allow_pickle=False)as a:
            if not np.array_equal(ids,a['sample_id']):raise ValueError('Regression/classifier comparison order differs')
            combined[variant]=metric(y,p,a['prediction'])
    result={'status':'completed','jobs':rows,'combined_comparison':combined,
        'plan_sha256':sha(OPS/'PLAN.json'),'limitations':plan['limitations'],
        'official_submission':False,'production_training_started':False,'updated_at_kst':stamp()}
    write(REPORT/'RESULTS.json',result);write(OPS/'state.json',result)
    text=['# Mac small masking control\n','DINOv2-small / Fold 0 D0 / 2,048 TRAIN rows / eight epochs.\n',
          '| Masking | Comparison RMSE | Independent classifier AUC |','|---|---:|---:|']
    text += [f"| {v} | {m['rmse']:.6f} | {m['classifier_auc']:.6f} |"for v,m in combined.items()]
    text+=['','Checkpoint selection: 1,194 rows. Final comparison: separate 1,206 rows. No Test or locked panels.','']
    text+=['- '+s for s in plan['limitations']]
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'RESULTS.md').write_text('\n'.join(text)+'\n')
    return result


def run(wait=False):
    wait_for_catboost(wait);check_device()
    plan=read(OPS/'PLAN.json');verify_sources(plan)
    lock=OPS/'controller.lock';fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(fd,'w')as f:f.write(str(os.getpid()))
    try:
        for task,variant in JOBS:
            if (RUN/task/variant).exists():raise FileExistsError('Fresh control only; completed/partial rerun forbidden')
            write(OPS/'state.json',{'phase':'training','task':task,'variant':variant,'pid':os.getpid(),
                                  'training_started':True,'updated_at_kst':stamp()})
            with (OPS/f'{task}_{variant}.log').open('a')as f:
                child=subprocess.Popen([sys.executable,str(Path(__file__)),'worker','--task',task,'--variant',variant],
                    cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                write(OPS/'worker.json',{'pid':child.pid,'task':task,'variant':variant})
                if child.wait()!=0:raise RuntimeError('Control job failed; no automatic retry')
        summarize()
    except Exception as exc:
        write(OPS/'state.json',{'phase':'failed','error':f'{type(exc).__name__}: {exc}',
            'automatic_retry':False,'training_results_must_not_be_reused_as_complete':True,
            'updated_at_kst':stamp()})
        raise
    finally:lock.unlink(missing_ok=True)


def execute(wait=False):
    """One detached controller: CatBoost gate -> MPS preflight -> four jobs."""
    plan=read(OPS/'PLAN.json');verify_sources(plan)
    lock=OPS/'execute.lock';fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(fd,'w')as f:f.write(str(os.getpid()))
    try:
        write(OPS/'state.json',{'phase':'waiting_for_catboost','pid':os.getpid(),
            'plan_sha256':sha(OPS/'PLAN.json'),'training_started':False,'updated_at_kst':stamp()})
        wait_for_catboost(wait)
        write(OPS/'state.json',{'phase':'MPS_disposable_preflight','pid':os.getpid(),
            'training_started':False,'updated_at_kst':stamp()})
        preflight(False)
        run(False)
    except Exception as exc:
        write(OPS/'state.json',{'phase':'failed','error':f'{type(exc).__name__}: {exc}',
            'automatic_retry':False,'updated_at_kst':stamp()})
        raise
    finally:lock.unlink(missing_ok=True)


def launch():
    plan=read(OPS/'PLAN.json');verify_sources(plan)
    if any((OPS/name).exists()for name in('LAUNCHED.json','launch.lock','execute.lock','controller.lock')):
        raise RuntimeError('Control was already launched or has a lock; never duplicate/reuse partial run')
    lock=OPS/'launch.lock';fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(fd,'w')as f:f.write(str(os.getpid()))
    try:
        with (OPS/'driver.log').open('a')as f:
            child=subprocess.Popen([sys.executable,str(Path(__file__)),'execute','--wait-catboost'],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        receipt={'phase':'launched_waiting_for_catboost','pid':child.pid,'detached':True,
            'plan_sha256':sha(OPS/'PLAN.json'),'training_started':False,'updated_at_kst':stamp()}
        write(OPS/'LAUNCHED.json',receipt)
        print(json.dumps(receipt),flush=True)
    finally:lock.unlink(missing_ok=True)


def fixture():
    """CPU synthetic contract only, no public weights and no real model fit."""
    import numpy as np
    import torch
    from pipeline.r16_feature_masking import (FeatureMaskingConfig,apply_r16_feature_masking,
                                             make_feature_masking_generator)
    from pipeline.training_r16_pretrained_masked import task_input,loss_and_values
    # Reproduce the exact original failure, then exercise corrected production
    # seeding, not just the stateless default_rng used by the prior fixture.
    legacy_reproduced = False
    try:
        np.random.seed(SEED)
    except ValueError as exc:
        legacy_reproduced = 'Seed must be between 0 and 2**32 - 1' in str(exc)
    if not legacy_reproduced:
        raise AssertionError('Original legacy NumPy failure was not reproduced')
    seed_contract = seed_frameworks(include_mps=False)
    first = [random.random(), float(np.random.random()), float(torch.rand(()))]
    seed_frameworks(include_mps=False)
    second = [random.random(), float(np.random.random()), float(torch.rand(()))]
    if first != second or not 0 <= NUMPY_LEGACY_SEED < 2**32:
        raise AssertionError('Paired framework initialization is not deterministic')
    rng=np.random.default_rng(17);n=80
    meta={'background_group':np.asarray(['a','b','c','d']*(n//4)),
        'operator':np.asarray(['op0','op1']*(n//2)),
        'presence':np.asarray([0,1,1,1]*(n//4)),'active_minutes':np.asarray([0,2,7,13]*(n//4))}
    ix=proportional_source_subset(meta,40)
    assert np.array_equal(ix,proportional_source_subset(meta,40)) and len(set(ix))==40
    x=torch.tensor(rng.normal(size=(4,17,15,25)),dtype=torch.float32);x[:,5]=1
    w=torch.full((4,15,25),1/375);q=torch.tensor(rng.uniform(size=(4,15,25)),dtype=torch.float32)
    y=(w*q).sum((1,2));z=(y>0).float();original=[a.clone()for a in(x,w,y,q,z)]
    receipts=[]
    for task,variant in JOBS:
        xa,info=apply_r16_feature_masking(x,task=task,training=True,
            generator=make_feature_masking_generator(SEED,stream='fixture'),config=FeatureMaskingConfig.for_variant(variant))
        assert all(torch.equal(a,b)for a,b in zip((x,w,y,q,z),original))
        assert torch.equal(xa[:,5],x[:,5])
        assert task_input(xa,task).shape[1]==(17 if task=='regression' else 15)
        evalx,_=apply_r16_feature_masking(x,task=task,training=False,
            generator=make_feature_masking_generator(SEED),config=FeatureMaskingConfig.for_variant(variant))
        assert torch.equal(evalx,x)
        logits=torch.zeros((4,15,25)if task=='regression'else(4,),requires_grad=True)
        loss,_,_=loss_and_values(logits,task,w,y,q,z);loss.backward()
        assert torch.isfinite(loss)and torch.isfinite(logits.grad).all()
        receipts.append({'task':task,'variant':variant,'loss_finite':True,'metadata':info.log_dict()})
    result={'status':'passed','synthetic_CPU_fixture_only':True,'no_public_weights_loaded':True,
        'original_legacy_seed_failure_reproduced':legacy_reproduced,
        'corrected_production_seed_function_exercised':True, 'seed_contract':seed_contract,
        'paired_framework_random_draws_equal':first == second,
        'no_MPS_training_started':True,'no_data_or_old_plan_mutated':True,'checks':receipts,'updated_at_kst':stamp()}
    write(OPS/'CPU_FIXTURE.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('fixture','plan','preflight','run','launch','execute','worker','summarize','status'))
    p.add_argument('--task',choices=('regression','classification'))
    p.add_argument('--variant',choices=('off','both'))
    p.add_argument('--wait-catboost',action='store_true')
    a=p.parse_args();sys.addaudithook(access_guard)
    if a.command=='fixture':fixture()
    elif a.command=='plan':
        plan=make_plan();print(json.dumps({k:v for k,v in plan.items()if k not in('sample_indices','sample_ids','source_sha256','input_sha256')},ensure_ascii=False),flush=True)
    elif a.command=='preflight':preflight(a.wait_catboost)
    elif a.command=='run':run(a.wait_catboost)
    elif a.command=='execute':execute(a.wait_catboost)
    elif a.command=='launch':launch()
    elif a.command=='summarize':print(json.dumps(summarize(),ensure_ascii=False),flush=True)
    elif a.command=='status':print((OPS/'state.json').read_text()if(OPS/'state.json').exists()else'not planned')
    else:
        if not a.task or not a.variant:p.error('worker requires --task and --variant')
        train_worker(a.task,a.variant)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        # A preflight/worker failure also needs an unambiguous fresh-run status.
        if sys.argv[1:2] not in (['fixture'], ['plan'], ['status']):
            write(OPS/'LAST_FAILURE.json',{'error':f'{type(exc).__name__}: {exc}',
                'automatic_retry':False,'updated_at_kst':stamp()})
        raise
