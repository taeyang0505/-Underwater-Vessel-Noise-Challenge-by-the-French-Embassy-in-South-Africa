"""Bounded R16 TimeMoE PEFT verification; no real rows/Test/download/training."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1')


def guard(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise RuntimeError('No network during verifier')
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        s = str(args[0]).lower()
        if Path(s).name in ('test.csv', 'sample_submission.csv'):
            raise RuntimeError('No Test access during verifier')


def digest(parameters):
    import torch
    h = hashlib.sha256()
    for n, p in parameters:
        h.update(n.encode() + b'\0')
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def memory(device):
    import torch
    values = {'maximum_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if device == 'mps':
        torch.mps.synchronize()
        values.update({'mps_current_allocated_bytes': torch.mps.current_allocated_memory(),
                       'mps_driver_allocated_bytes': torch.mps.driver_allocated_memory(),
                       'mps_recommended_max_bytes': torch.mps.recommended_max_memory()})
    return values


def benchmark(device):
    """Only synthetic stage2 steps: B1/B4/B8, one warmup plus two timed steps."""
    import torch
    from pipeline.pretrained_timemoe_r16 import create_timemoe_branch
    torch.manual_seed(20261028)
    model, provenance = create_timemoe_branch('regression', device=device)
    model.set_stage(2).train()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    rows = []
    for batch in (1, 4, 8):
        x = torch.randn(batch, 17, 15, 25, device=device)
        x[:, 5] = 1
        x[:, :, 0] = 0
        w = (x[:, 5] > .5).float(); w /= w.sum((1, 2), keepdim=True)
        timings, memory_rows = [], []
        for repeat in range(3):
            model.zero_grad(set_to_none=True)
            if device == 'mps': torch.mps.synchronize()
            begin = time.perf_counter()
            q = model(x).sigmoid() * (w > 0)
            share = (q*w).sum((1, 2))
            loss = (share-.2).square().mean() + .25*((q-.15).square()*w).sum((1, 2)).mean()
            loss.backward()
            optimizer.step()
            if device == 'mps': torch.mps.synchronize()
            timings.append(time.perf_counter()-begin)
            if not torch.isfinite(loss): raise AssertionError('Benchmark loss not finite')
            memory_rows.append(memory(device))
            del q, share, loss
        rows.append({'batch_size': batch, 'warmup_seconds': timings[0],
                     'measured_seconds': timings[1:],
                     'mean_step_seconds': sum(timings[1:])/2,
                     'samples_per_second': batch/(sum(timings[1:])/2),
                     'memory_snapshots_after_backward_optimizer': memory_rows,
                     'memory_is_sampled_not_peak_allocator': True})
        del x, w
    return {'task': 'regression', 'device': device, 'stage': 2, 'provenance': provenance,
            'synthetic_rows_only': True, 'real_training_or_Test_used': False,
            'bounded_benchmark': rows}


def fixture(task):
    from pipeline.load_foundation_r5 import DEFAULT_CACHE, SPECS, _official_runtime, _import_pinned_module, model_directory
    from pipeline.pretrained_timemoe_r16 import TimeMoER16Branch
    _official_runtime(DEFAULT_CACHE)
    directory = model_directory('timemoe_public')
    package = 'r16_timemoe_tiny_fixture'
    spec = SPECS['timemoe_public']
    cm = _import_pinned_module(directory, package, spec['configuration_module'])
    mm = _import_pinned_module(directory, package, spec['modeling_module'])
    cfg = cm.TimeMoeConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=4,
                           num_experts=4, num_experts_per_tok=2, max_position_embeddings=32,
                           horizon_lengths=[1], input_size=1, use_cache=False)
    cfg._attn_implementation = 'eager'
    return TimeMoER16Branch(mm.TimeMoeModel(cfg), task, fixture_mode=True), {'fixture': True, 'pretrained': False}


def one(task, real, device):
    import torch
    from pipeline.pretrained_timemoe_r16 import create_timemoe_branch
    begin = time.perf_counter()
    torch.manual_seed(20261028)
    model, provenance = create_timemoe_branch(task) if real else fixture(task)
    model = model.to(device)
    load_seconds = time.perf_counter() - begin
    # An untrained LoRA branch must initially reproduce every exact frozen
    # projection, rather than silently perturbing the public checkpoint.
    from pipeline.foundation_masks_r5 import LoRALinear
    model.eval()
    parity = 0.
    with torch.no_grad():
        for module in model.backbone.modules():
            if isinstance(module, LoRALinear):
                inp = torch.randn(1, 2, module.in_features, device=device,
                                  dtype=module.base.weight.dtype)
                parity = max(parity, float((module(inp)-module.base(inp)).abs().max()))
    if parity != 0.: raise AssertionError('Zero-initialized LoRA changes base output')
    x = torch.randn(1, model.input_channels, 15, 25, device=device)
    x[:, 5] = 1
    x[:, :, 0] = 0  # all-missing minute tests safe cross-band attention.
    x[:, :, 4, 7] = 0
    before = digest(model.frozen_named_parameters())
    checks = []
    for stage in (1, 2):
        model.set_stage(stage).train()
        model.zero_grad(set_to_none=True)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        if stage == 1 and any(n.startswith('backbone.') for n in names):
            raise AssertionError('Backbone adaptation active in warmup')
        if stage == 2 and not any(n.endswith('lora_B') for n in names):
            raise AssertionError('No LoRA in stage2')
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        if device == 'mps': torch.mps.synchronize()
        t = time.perf_counter()
        logits = model(x)
        if task == 'regression':
            w = (x[:, 5] > .5).float(); w = w / w.sum((1, 2), keepdim=True)
            q = logits.sigmoid() * (w > 0)
            y = (q * w).sum((1, 2))
            loss = (y - .2).square().mean() + .25 * ((q-.15).square()*w).sum((1, 2)).mean()
            if logits.shape != (1, 15, 25): raise AssertionError('Mask shape')
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
            if logits.shape != (1,): raise AssertionError('Classifier shape')
        if not torch.isfinite(loss): raise AssertionError('Nonfinite loss')
        loss.backward()
        if device == 'mps': torch.mps.synchronize()
        elapsed = time.perf_counter() - t
        if any(p.grad is not None for _, p in model.frozen_named_parameters()):
            raise AssertionError('Frozen base gradient')
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
                raise AssertionError('Nonfinite grad: '+n)
        if model.stem.weight.grad is None or model.stem.weight.grad.norm() <= 0:
            raise AssertionError('Stem gradient missing through frozen model')
        lora_grad = [float(p.grad.norm()) for n, p in model.named_parameters()
                     if n.endswith('lora_B') and p.grad is not None]
        if stage == 2 and not any(v > 0 for v in lora_grad): raise AssertionError('LoRA B gradient missing')
        optimizer.step()
        checks.append({'stage': stage, 'loss': float(loss.detach()),
                       'forward_backward_seconds': elapsed, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                       'stem_gradient_norm': float(model.stem.weight.grad.norm()),
                       'LoRA_B_max_gradient_norm': max(lora_grad, default=0.),
                       'frozen_base_gradients_absent': True})
        del optimizer, logits, loss
    after = digest(model.frozen_named_parameters())
    if before != after: raise AssertionError('Frozen parameter bytes changed')
    model.eval()
    with torch.no_grad(): a = model(x).detach().cpu()
    adapter = model.adapter_state()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in adapter: p.zero_()
    model.load_adapter_state(adapter)
    with torch.no_grad(): b = model(x).detach().cpu()
    delta = float((a-b).abs().max())
    if delta > 1e-6: raise AssertionError('Adapter reload output differs')
    architecture = model.architecture()
    result = {'task': task, 'device': device, 'provenance': provenance, 'architecture': architecture,
              'load_seconds': load_seconds, 'stages': checks,
              'frozen_base_sha256_before': before, 'frozen_base_sha256_after': after,
              'adapter_reload_max_abs': delta, 'one_missing_minute_and_cell_finite': True,
              'initial_LoRA_vs_frozen_projection_max_abs': parity,
              'real_training_or_Test_used': False, 'actual_performance_measured': False,
              'memory': memory(device),
              'total_seconds': time.perf_counter()-begin}
    del model, adapter, x, a, b
    gc.collect()
    if device == 'mps': torch.mps.empty_cache()
    return result


def independent_task_contract():
    """Tiny official-API objects: no parameters/storage shared between tasks."""
    import torch
    reg, _ = fixture('regression')
    cls, _ = fixture('classification')
    r_ids = {id(p) for p in reg.parameters()}
    c_ids = {id(p) for p in cls.parameters()}
    if r_ids & c_ids: raise AssertionError('Task parameters are shared')
    r_storage = {p.untyped_storage().data_ptr() for p in reg.parameters()}
    c_storage = {p.untyped_storage().data_ptr() for p in cls.parameters()}
    if r_storage & c_storage: raise AssertionError('Task parameter storage shared')
    reg_before = digest(reg.named_parameters())
    cls.set_stage(2)
    x = torch.randn(1, 15, 15, 25); x[:, 5] = 1
    cls(x).sum().backward()
    torch.optim.Adam([p for p in cls.parameters() if p.requires_grad], lr=1e-4).step()
    if digest(reg.named_parameters()) != reg_before:
        raise AssertionError('Classification step modified regression state')
    state = reg.adapter_state()
    try:
        reg.load_adapter_state({**state, 'unexpected': torch.zeros(1)})
    except ValueError:
        pass
    else:
        raise AssertionError('Non-strict adapter load accepted unexpected key')
    return {'fixture_only': True, 'task_parameters_and_storage_disjoint': True,
            'classification_update_preserves_regression_bytes': True,
            'unexpected_adapter_key_rejected': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--task', choices=('regression', 'classification', 'both'), default='both')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--benchmark-only', action='store_true')
    args = parser.parse_args()
    sys.addaudithook(guard)
    import torch
    torch.set_num_threads(2)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no implicit fallback')
    tasks = ('regression', 'classification') if args.task == 'both' else (args.task,)
    if args.benchmark_only and not args.real: raise ValueError('Benchmark requires explicit --real')
    result = {'status': 'passed', 'real_public_backbone': args.real,
              'rows': [benchmark(args.device)] if args.benchmark_only else [one(t, args.real, args.device) for t in tasks]}
    if not args.real:
        result['independent_task_contract'] = independent_task_contract()
    path = args.output or ROOT / 'reports' / ('R16_TIMEMOE_PEFT_' + ('REAL_' if args.real else 'FIXTURE_') + args.device.upper() + '.json')
    if path.exists(): raise FileExistsError('Preserve existing verifier output')
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': 'passed', 'report': str(path), 'real': args.real,
                      'rows': [{k: v for k, v in r.items() if k in ('task','load_seconds','stages','total_seconds','bounded_benchmark')} for r in result['rows']]}, ensure_ascii=False))


if __name__ == '__main__': main()
