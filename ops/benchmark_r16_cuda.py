"""Disposable CUDA throughput probe on a small, labelled real TRAIN NPZ.

Never writes model weights, changes a run plan, resumes training or reads Test.
The NPZ must contain x17,w,share,cell_share,presence from the canonical cache.
Each timed training step is a complete effective-batch64 Adam update, including
microbatch transfers, training-only feature masking, backward and clipping.
FP32 disables TF32; BF16 is a separate autocast experiment, not a run change.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(variable, '2')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(piece)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def summary(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'median': float(np.median(a)), 'mean': float(a.mean()),
            'minimum': float(a.min()), 'p90': float(np.quantile(a, .9)), 'seconds': a.tolist()}


def load_probe(path, effective):
    import numpy as np
    with np.load(path, allow_pickle=False) as stored:
        required = ('x', 'w', 'share', 'cell_share', 'presence')
        if not set(required) <= set(stored.files):
            raise ValueError('Probe NPZ requires x,w,share,cell_share,presence')
        data = {key: np.asarray(stored[key], dtype=np.float32).copy() for key in required}
    n = len(data['share'])
    if n < effective or data['x'].shape != (n, 17, 15, 25):
        raise ValueError('At least64 canonical real rows with17channels are required')
    if data['w'].shape != (n, 15, 25) or data['cell_share'].shape != data['w'].shape or data['presence'].shape != (n,):
        raise ValueError('Probe label/weight shapes differ')
    if any(not np.isfinite(value).all() for value in data.values()):
        raise ValueError('Probe must be finite, with missingness in channel5')
    observed = data['x'][:, 5]
    if not np.isin(observed, (0., 1.)).all() or not observed.any((1, 2)).all():
        raise ValueError('Original binary observation channel required')
    if np.any(data['w'] < 0) or not np.allclose(data['w'].sum((1, 2)), 1., atol=1e-6):
        raise ValueError('Physical pooling weights must be normalized')
    if (data['w'][observed == 0] != 0).any() or (data['cell_share'][observed == 0] != 0).any():
        raise ValueError('Missing cells must have zero weights and labels')
    if any(((data[key] < 0) | (data[key] > 1)).any() for key in ('share','cell_share','presence')):
        raise ValueError('Physical labels outside[0,1]')
    if not np.array_equal(data['share'] > 0, data['presence'] > 0):
        raise ValueError('Physical share/presence differs')
    gap = float(abs((data['w'] * data['cell_share']).sum((1, 2)) - data['share']).max())
    if gap > .0058:
        raise ValueError('Probe quantization-pooling bound exceeded')
    return data, {'path': str(Path(path).resolve()), 'sha256': digest(path), 'n': n,
                  'bytes': Path(path).stat().st_size, 'maximum_quantized_pooling_gap': gap,
                  'source': 'caller-provided canonical TRAIN cache subset; no query inputs'}


def probe(args, task, data):
    import numpy as np
    import torch
    from pipeline.training_r16_pretrained_masked import branch, task_input, loss_and_values, base_fingerprint
    from pipeline.r16_feature_masking import FeatureMaskingConfig, apply_r16_feature_masking, make_feature_masking_generator
    torch.manual_seed(args.seed)
    load_start = time.perf_counter()
    model, provenance = branch(args.family, task, 'cuda')
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    fingerprint = base_fingerprint(model)
    model.set_stage(2)
    model.train()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                  lr=2e-4, weight_decay=1e-3)
    config = FeatureMaskingConfig.for_variant(args.variant)
    generator = make_feature_masking_generator(args.seed, stream='disposable_cuda_benchmark')
    train_times = []
    infer_times = []
    losses = []
    rng = np.random.default_rng(args.seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    autocast = lambda: (torch.autocast('cuda', dtype=torch.bfloat16) if args.precision == 'bf16'
                       else contextlib.nullcontext())
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.warmup + args.steps):
        indices = rng.permutation(len(data['share']))[:args.effective_batch]
        torch.cuda.synchronize()
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.
        for offset in range(0, args.effective_batch, args.batch):
            ids = indices[offset:offset+args.batch]
            tensors = {key: torch.tensor(value[ids], device='cuda') for key, value in data.items()}
            x, metadata = apply_r16_feature_masking(tensors['x'], task=task, training=True,
                                                   generator=generator, config=config)
            if not torch.equal(x[:, 5], tensors['x'][:, 5]):
                raise AssertionError('Feature masking changed observation mask')
            with autocast():
                logits = model(task_input(x, task))
                loss, _, _ = loss_and_values(logits, task, tensors['w'], tensors['share'],
                                             tensors['cell_share'], tensors['presence'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite benchmark loss')
            (loss * (len(ids) / args.effective_batch)).backward()
            update_loss += float(loss.detach()) * len(ids) / args.effective_batch
        for name, parameter in model.frozen_named_parameters():
            if parameter.grad is not None:
                raise AssertionError('Frozen base received gradient:' + name)
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite benchmark gradients')
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        if step >= args.warmup:
            train_times.append(elapsed)
            losses.append(update_loss)
    train_peak = {'allocated_bytes': torch.cuda.max_memory_allocated(),
                  'reserved_bytes': torch.cuda.max_memory_reserved()}
    stem_gradient = sum(float(p.grad.float().square().sum()) for name, p in model.named_parameters()
                        if name.startswith('stem') and p.grad is not None) ** .5
    if not stem_gradient > 0:
        raise AssertionError('No nonzero stem gradient')
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    reference = None
    precision_delta = None
    with torch.no_grad():
        for step in range(2 + args.inference_steps):
            ids = np.arange(step * args.batch, (step + 1) * args.batch) % len(data['share'])
            torch.cuda.synchronize()
            begin = time.perf_counter()
            x = torch.tensor(data['x'][ids], device='cuda')
            w = torch.tensor(data['w'][ids], device='cuda')
            with autocast():
                logits = model(task_input(x, task))
                value = logits.float().sigmoid() if task == 'classification' else (logits.float().sigmoid() * w).sum((1, 2))
            if not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite inference')
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            if step >= 2:
                infer_times.append(elapsed)
            if args.precision == 'bf16' and step == 0:
                # Same post-update model and unmasked input; no adoption gate or
                # task-performance claim is inferred from this numerical delta.
                logits32 = model(task_input(x, task))
                reference = logits32.float().sigmoid() if task == 'classification' else (logits32.float().sigmoid() * w).sum((1, 2))
                precision_delta = {'maximum_absolute_prediction_difference': float((reference-value).abs().max()),
                                   'mean_absolute_prediction_difference': float((reference-value).abs().mean())}
    unchanged = base_fingerprint(model) == fingerprint
    if not unchanged:
        raise AssertionError('Frozen pretrained base bytes changed')
    train_summary = summary(train_times)
    infer_summary = summary(infer_times)
    result = {'status': 'passed', 'family': args.family, 'task': task, 'stage': 2,
              'precision': args.precision, 'TF32': False, 'variant': args.variant,
              'microbatch': args.batch, 'effective_batch': args.effective_batch,
              'warmup_updates': args.warmup, 'timed_optimizer_updates': args.steps,
              'train_update': train_summary, 'inference_microbatch': infer_summary,
              'train_rows_per_second': args.effective_batch / train_summary['median'],
              'inference_rows_per_second': args.batch / infer_summary['median'],
              'projection': {'train_rows_per_epoch': args.train_rows,
                             'train_epoch_seconds': math.ceil(args.train_rows/args.effective_batch)*train_summary['median'],
                             'checkpoint_validation_rows': 1194,
                             'checkpoint_validation_seconds': math.ceil(1194/args.batch)*infer_summary['median'],
                             'excludes': ['model load', 'checkpoint IO', 'full cache preprocessing', 'artifact hashing',
                                          'end-of-job ownCV/selection inference', 'locked panels and Test inference'],
                             'stage1_runtime_measured': False, 'early_stop_epoch_unknown': True},
              'model_load_seconds': load_seconds, 'training_peak_memory': train_peak,
              'inference_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'frozen_base_unchanged': unchanged, 'stem_gradient_L2': stem_gradient,
              'timed_losses': losses, 'bf16_vs_fp32_same_model_probe': precision_delta,
              'provenance': provenance, 'disposable_updates_not_a_trained_checkpoint': True}
    del model, tensors, logits, loss, x, w, trainable
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True, choices=('dasheng_base','dinov2_small','timemoe'))
    parser.add_argument('--task', choices=('regression','classification','both'), default='both')
    parser.add_argument('--batch', type=int, choices=(8,16), default=8)
    parser.add_argument('--precision', choices=('fp32','bf16'), default='fp32')
    parser.add_argument('--variant', choices=('off','spec','profile','both'), default='both')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--inference-steps', type=int, default=8)
    parser.add_argument('--effective-batch', type=int, default=64)
    parser.add_argument('--train-rows', type=int, default=61446)
    parser.add_argument('--seed', type=int, default=202610281901)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh benchmark output required')
    if args.effective_batch != 64 or args.warmup < 3 or args.steps < 8 or args.inference_steps < 8:
        raise ValueError('Use effective64,warmup>=3,timedupdates>=8,infersteps>=8')
    from pipeline.r16_multisupport_data import guard
    def audit(event, values):
        guard(event, values)
        if event in ('socket.connect','socket.getaddrinfo'):
            raise PermissionError('Public checkpoints must already be local; no benchmark network access')
    sys.addaudithook(audit)
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU/MPS fallback')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Native CUDA BF16 required for separate autocast probe')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    data, receipt = load_probe(args.data, args.effective_batch)
    report = {'format': 'r16_cuda_effective64_throughput_probe_v1', 'status': 'running',
              'time_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'script_sha256': digest(__file__), 'input': receipt,
              'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                              'torch_cuda': torch.version.cuda, 'GPU': torch.cuda.get_device_name(),
                              'compute_capability': list(torch.cuda.get_device_capability()),
                              'torch_arch_list': torch.cuda.get_arch_list(),
                              'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory},
              'models': {}, 'Test_used': False, 'production_checkpoints_written': False}
    write_json(args.output, report)
    try:
        for task in ('regression','classification') if args.task == 'both' else (args.task,):
            report['models'][task] = probe(args, task, data)
            write_json(args.output, report)
            print(json.dumps({'family': args.family, 'task': task, 'precision': args.precision,
                              'projection': report['models'][task]['projection']}), flush=True)
        report['status'] = 'passed'
        write_json(args.output, report)
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        write_json(args.output, report)
        raise


if __name__ == '__main__':
    main()
