"""Synthetic-only CPU R9 forward/backprop/checkpoint contract fixture.

No acoustic dataset, validation panel, Test, prior, or old weights are read.
One Adam update is an operator fixture, not model training/performance proof.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from pipeline.models_r9 import MODEL_DEFAULTS, make_r9_model
from pipeline.relative_grid_r6_v2 import relative_grid_v2
from pipeline.relative_models_r6_v2 import fixed_presence_score
from pipeline.neural_r5 import physical_multitask_loss

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = (
    'pipeline/models_r9.py', 'pipeline/hybrid_models_r5.py', 'pipeline/models.py',
    'pipeline/grid_models_r5.py', 'pipeline/relative_grid_r6_v2.py',
    'pipeline/relative_grid_r6.py', 'pipeline/relative_models_r6_v2.py',
    'pipeline/neural_r5.py', 'ops/verify_models_r9.py',
)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expect_reject(fn):
    try:
        fn()
    except (ValueError, TypeError):
        return True
    raise AssertionError('Invalid contract accepted')


def run():
    torch.set_num_threads(2)
    before = {p: file_sha(ROOT/p) for p in SOURCE_NAMES}
    rng = np.random.default_rng(20261026)
    db = rng.normal(85., 8., (3, 15, 25))
    db[0, :3] = np.nan
    db[1, 8:] = np.nan
    db[2, :, 4] = np.nan
    db[2, 2, 10] = np.nan
    grid, weights = relative_grid_v2(db)
    x = torch.from_numpy(grid)
    w = torch.from_numpy(weights)
    target_q = torch.from_numpy(rng.uniform(.05, .95, (3, 15, 25)).astype(np.float32))
    target_q[0] = 0
    target_q = torch.where(w > 0, target_q, torch.zeros_like(target_q))
    y = (target_q*w).sum((1, 2))
    presence = torch.tensor([0., 1., 1.])
    checks = {}
    for kind in MODEL_DEFAULTS:
        start = time.perf_counter()
        torch.manual_seed(20261026)
        model = make_r9_model(kind)
        assert not any('presence_head' in k for k, _ in model.named_parameters())
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0003)
        output = model(x, w)
        loss, parts = physical_multitask_loss(output, (y, presence, target_q), w,
                                             cell_weight=.25, presence_weight=.05)
        loss.mean().backward()
        no_grad = [k for k, p in model.named_parameters() if p.grad is None]
        assert not no_grad, no_grad
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        grad_norm = sum(float(p.grad.detach().double().square().sum()) for p in model.parameters())**.5
        assert grad_norm > 0
        first_before = next(model.parameters()).detach().clone()
        optimizer.step()
        assert not torch.equal(first_before, next(model.parameters()).detach())
        model.eval()
        context_check = None
        if kind == 'bg_cnn':
            with torch.no_grad():
                hidden = model.backbone.encode(x)
                observed = x[:, 5:6]
                expected_context = (hidden*observed).sum(3, keepdim=True)/observed.sum(3, keepdim=True).clamp_min(1)
                actual_hidden = model.encode(x)
                assert torch.equal(actual_hidden, hidden+expected_context)
                assert torch.equal(expected_context[0, :, :3], torch.zeros_like(expected_context[0, :, :3]))
                context_check = {'masked_mean_oracle_exact': True,
                                 'all_missing_minute_context_exact_zero': True,
                                 'parameter_free': True, 'all25frequency_dependency': True}
        with torch.no_grad():
            full = model(x, w)
            chunks = tuple(torch.cat([model(x[i:i+1], w[i:i+1])[j] for i in range(3)]) for j in range(3))
            repeat = model(x, w)
            expected_score, expected_share, _ = fixed_presence_score(full[2], w)
            assert torch.equal(full[0], expected_share)
            assert torch.equal(full[2][w == 0], torch.zeros_like(full[2][w == 0]))
            assert ((full[2] >= 0) & (full[2] <= 1)).all()
            assert torch.allclose(full[1].sigmoid(), expected_score.clamp(1e-6, 1-1e-6), atol=1e-7, rtol=0)
            batch_error = max(float((a-b).abs().max()) for a, b in zip(full, chunks))
            assert batch_error <= 3e-6, (kind, batch_error)
            assert all(torch.equal(a, b) for a, b in zip(full, repeat))
            embedding = model.representation(x)
            assert embedding.shape[0] == 3 and embedding.ndim == 2 and torch.isfinite(embedding).all()
        architecture = model.architecture()
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        buffer = io.BytesIO()
        torch.save({'architecture': architecture, 'state_dict': state}, buffer)
        state_sha = hashlib.sha256(buffer.getvalue()).hexdigest()
        buffer.seek(0)
        checkpoint = torch.load(buffer, map_location='cpu', weights_only=True)
        restored = make_r9_model(checkpoint['architecture']['kind'], **checkpoint['architecture']['model_settings'])
        assert restored.architecture() == checkpoint['architecture']
        restored.load_state_dict(checkpoint['state_dict'], strict=True)
        restored.eval()
        with torch.no_grad():
            recovered = restored(x, w)
        reload_error = max(float((a-b).abs().max()) for a, b in zip(full, recovered))
        assert reload_error == 0
        invalid_x = x.clone()
        invalid_x[:, 7] = 0
        invalid_mask = x.clone()
        invalid_mask[:, 5] = .5
        with torch.no_grad():
            guards = {
                'legacy_channel_contract': expect_reject(lambda: model(invalid_x, w)),
                'nonbinary_mask': expect_reject(lambda: model(invalid_mask, w)),
                'unnormalized_weights': expect_reject(lambda: model(x, w*.5)),
                'wrong_channels': expect_reject(lambda: model(x[:, :6], w)),
                'unknown_settings': expect_reject(lambda: make_r9_model(kind, input_channels=6)),
            }
        checks[kind] = {
            'architecture': architecture, 'one_adam_update': True,
            'loss_mean': float(loss.detach().mean()),
            'loss_components': {k: float(v.detach().mean()) for k, v in parts.items()},
            'gradient_l2': grad_norm, 'parameters_without_gradient': no_grad,
            'all_gradients_finite': True, 'first_parameter_updated': True,
            'share_shape': list(full[0].shape), 'q_shape': list(full[2].shape),
            'representation_shape': list(embedding.shape),
            'observed_mask_zero_exact': True, 'q_bounded': True,
            'physical_pool_exact': True, 'fixed_score_verified': True,
            'repeated_batch_exact': True, 'batch_independence_max_abs': batch_error,
            'batch_independence_tolerance': 3e-6, 'checkpoint_weights_only_reload_max_abs': reload_error,
            'in_memory_fixture_checkpoint_sha256': state_sha,
            'contract_rejection_guards': guards, 'elapsed_seconds': time.perf_counter()-start,
            'wide_frequency_context': context_check,
        }
        del optimizer, model, restored
    after = {p: file_sha(ROOT/p) for p in SOURCE_NAMES}
    assert before == after
    return {
        'format': 'r9_model_adapter_verification_v1', 'status': 'PASS',
        'scope': 'Synthetic CPU operator fixtures; not full training or performance evaluation',
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'threads': 2, 'device': 'cpu', 'seed': 20261026,
        'synthetic_examples': 3, 'adam_updates_per_kind': 1,
        'source_datasets_read': False, 'Test_read': False, 'old_weights_read': False,
        'GPU_used': False, 'pretrained_weights_used': False,
        'training_policy_expected_in_runner': {'optimizer': 'Adam', 'epochs': 100,
                                             'early_stopping': False, 'dropout': .1},
        'models': checks, 'source_code_sha256': before, 'source_code_unchanged': True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/R9_MODEL_ADAPTER_VERIFICATION.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fixture output already exists; preserve prior evidence')
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output),
                      'parameters': {k: v['architecture']['total_parameters'] for k, v in report['models'].items()}}))


if __name__ == '__main__':
    main()
