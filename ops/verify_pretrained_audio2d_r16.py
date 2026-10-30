"""Synthetic-only contract / public-weight 1-step checks; never reads Test."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def frozen_hash(module):
    digest = hashlib.sha256()
    for name, p in module.frozen_named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().cpu().float().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checks(module, torch, batch, device, steps):
    def sync():
        if device.startswith('mps'):
            torch.mps.synchronize()
        elif device.startswith('cuda'):
            torch.cuda.synchronize()
    x = torch.randn(batch, module.in_channels, 15, 25, device=device)
    x[:, module.mask_channel] = 1
    x[:, module.mask_channel, 0] = 0
    result = {'task': module.task, 'batch': batch, 'device': device, 'stages': {}}
    base_before = frozen_hash(module)
    union_keys = set(module.adapter_state())
    for stage in (1, 2):
        module.set_stage(stage).train()
        optimizer = torch.optim.Adam((p for p in module.parameters() if p.requires_grad), lr=1e-3)
        sync(); start = time.monotonic()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            pred = module(x)
            expected = (batch, 15, 25) if module.task == 'regression' else (batch,)
            assert tuple(pred.shape) == expected, (pred.shape, expected)
            if module.task == 'regression':
                q = pred.sigmoid()
                w = torch.rand_like(q) * (x[:, module.mask_channel] > 0)
                w = w / w.sum((1, 2), keepdim=True)
                y_hat = (w * q).sum((1, 2))
                target_q = torch.rand_like(q)
                target_y = (w * target_q).sum((1, 2))
                loss = ((y_hat - target_y) ** 2).mean() + .25 * (w * (q - target_q) ** 2).sum((1, 2)).mean()
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, torch.zeros_like(pred))
            loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
            assert module.stem.weight.grad is not None and module.stem.weight.grad.abs().sum() > 0
            for name, p in module.frozen_named_parameters():
                assert p.grad is None, 'Frozen base received gradient: ' + name
            if stage == 1:
                assert not any(p.requires_grad for n, p in module.named_parameters() if n.startswith('backbone.'))
            optimizer.step()
        sync()
        result['stages'][str(stage)] = {
            'seconds_per_step': (time.monotonic() - start) / steps,
            'loss': float(loss.detach().cpu()),
            'trainable_parameters': sum(p.numel() for p in module.parameters() if p.requires_grad),
            'all375_tokens': True, 'finite_forward_backward': True,
        }
        assert set(module.adapter_state()) == union_keys
    base_after = frozen_hash(module)
    assert base_before == base_after
    module.eval()
    with torch.no_grad():
        reference = module(x).detach().clone()
    state = module.adapter_state()
    with torch.no_grad():
        module.stem.weight.add_(.2)
    module.load_adapter_state(state)
    with torch.no_grad():
        restored = module(x)
    assert torch.equal(reference, restored), 'Adapter reload changed eval prediction'
    bad = dict(state); bad.pop(next(iter(bad)))
    try:
        module.load_adapter_state(bad)
    except RuntimeError:
        strict_rejected = True
    else:
        raise AssertionError('Incomplete adapter accepted')
    result.update({'frozen_sha256_before': base_before, 'frozen_sha256_after': base_after,
                   'frozen_weights_unchanged': True, 'adapter_reload_exact': True,
                   'strict_missing_key_rejected': strict_rejected, 'adapter_tensors': len(state)})
    return result


def profile(module, torch, batches, device, steps):
    """Steady synthetic production-like steps, excluding exhaustive grad/SHA QA."""
    module.set_stage(2).train()
    optimizer = torch.optim.Adam((p for p in module.parameters() if p.requires_grad), lr=2e-4)
    rows = []
    def sync():
        if device.startswith('mps'): torch.mps.synchronize()
        elif device.startswith('cuda'): torch.cuda.synchronize()
    for batch in batches:
        x = torch.randn(batch, module.in_channels, 15, 25, device=device)
        x[:, module.mask_channel] = 1
        x[:, module.mask_channel, 0] = 0
        q_target = torch.rand(batch, 15, 25, device=device)
        w = torch.rand_like(q_target) * (x[:, module.mask_channel] > 0)
        w /= w.sum((1, 2), keepdim=True)
        y = (w * q_target).sum((1, 2))
        def step():
            optimizer.zero_grad(set_to_none=True)
            output = module(x)
            if module.task == 'regression':
                q = output.sigmoid()
                loss = (((q * w).sum((1, 2)) - y) ** 2).mean() + .25 * (w * (q-q_target)**2).sum((1, 2)).mean()
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(output, (y > .5).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in module.parameters() if p.requires_grad), 1.0)
            optimizer.step()
        for _ in range(2): step()
        sync(); started = time.monotonic()
        for _ in range(steps): step()
        sync(); elapsed = time.monotonic() - started
        row = {'batch': batch, 'timed_steps': steps, 'seconds_per_step': elapsed / steps,
               'samples_per_second': batch * steps / elapsed,
               'synthetic_61000_example_epoch_minutes_per_task': 61000 * elapsed / (batch * steps * 60),
               'process_max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        if device.startswith('mps'):
            row['mps_driver_allocated_bytes'] = torch.mps.driver_allocated_memory()
            row['mps_current_allocated_bytes'] = torch.mps.current_allocated_memory()
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['dasheng_base', 'dasheng_1.2b', 'dinov2_small'], default='dasheng_base')
    parser.add_argument('--task', choices=['regression', 'classification'], default='regression')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--profile-batches', default='', help='Comma-separated synthetic throughput batches')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import torch
    from pipeline.pretrained_audio2d_r16 import (Audio2DTaskBranch, OLD_CACHE,
        create_audio2d_branch, fetch_audio2d_checkpoint)
    torch.set_num_threads(args.threads)
    torch.manual_seed(3407)
    if args.download:
        if args.kind == 'dasheng_1.2b':
            raise ValueError('1.2B uses already pinned R5 loader; this CLI only downloads the smaller approved candidates')
        receipt = fetch_audio2d_checkpoint(args.kind)
        print(json.dumps({'download_verified': receipt}, ensure_ascii=False), flush=True)
    if args.download_only:
        return
    if args.fixture:
        from pipeline.load_foundation_r5 import _official_runtime
        _official_runtime(OLD_CACHE)
        from transformers.models.dinov2.configuration_dinov2 import Dinov2Config
        from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
        config = Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                              image_size=28, patch_size=14)
        a = Audio2DTaskBranch(Dinov2Model(config), 'dinov2', 'regression', 32, rank=2)
        b = Audio2DTaskBranch(Dinov2Model(config), 'dinov2', 'classification', 32, rank=2)
        assert not ({p.data_ptr() for p in a.parameters()} & {p.data_ptr() for p in b.parameters()})
        reports = [checks(m.to(args.device), torch, args.batch, args.device, args.steps) for m in (a, b)]
        result = {'fixture_only': True, 'pretrained_claim': False,
                  'regression_classifier_parameter_storage_disjoint': True, 'checks': reports}
    else:
        started = time.monotonic()
        module, provenance = create_audio2d_branch(args.kind, args.task, device=args.device)
        result = {'fixture_only': False, 'provenance': provenance,
                  'load_seconds': time.monotonic() - started,
                  'synthetic_checks': checks(module, torch, args.batch, args.device, args.steps)}
        if args.profile_batches:
            result['steady_profile'] = profile(module, torch, [int(v) for v in args.profile_batches.split(',')], args.device, args.steps)
    result['torch_version'] = torch.__version__
    result['passed'] = True
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
