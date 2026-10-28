"""CPU-only numerical/gradient fixture. No source datasets or model files read."""
import json
import math
import numpy as np
import torch
from pipeline.contrastive import common_gain_views, contrastive_regularizer, nt_xent
from pipeline.mixtures import cnn_channels
from pipeline.models import make_model


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected a clear ValueError')


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(711)
    b = rng.uniform(.01, 2, (4, 15, 25))
    r = rng.uniform(0, 1, (4, 15, 25))
    observed = rng.uniform(size=b.shape) > .12
    x = np.where(observed, 10 * np.log10(b + r), np.nan).astype(np.float32)
    original = x.copy()
    views = common_gain_views(x, 9)
    target = np.where(observed, r, 0).sum((1, 2)) / np.where(observed, b + r, 0).sum((1, 2))
    original_channels, original_weights = cnn_channels(x)
    for v in views:
        assert np.array_equal(np.isnan(v), ~observed)
        gain_db = np.nanmean(v - x, (1, 2))
        assert (np.abs(gain_db) <= 3).all()
        factor = 10 ** (gain_db[:, None, None] / 10)
        transformed_target = np.where(observed, r * factor, 0).sum((1, 2)) / np.where(observed, (b + r) * factor, 0).sum((1, 2))
        assert np.allclose(target, transformed_target, atol=1e-12)
        channels, weights = cnn_channels(v)
        assert np.allclose(channels[:, 0], original_channels[:, 0], atol=2e-6)
        assert np.allclose(weights, original_weights, atol=2e-7)
        assert np.array_equal(channels[:, 3], original_channels[:, 3])
    assert np.array_equal(x, original, equal_nan=True)
    # Independent hand-computed N=2 NT-Xent reference, including both views.
    first = torch.eye(2, requires_grad=True)
    loss = nt_xent(first, first.clone(), .2)
    assert math.isclose(float(loss.detach()), math.log1p(2 * math.exp(-5)), abs_tol=1e-7)
    same_group = nt_xent(first, first.clone(), .2, ['a', 'a'])
    assert same_group.requires_grad and float(same_group.detach()) == 0
    fixture = {}
    for kind in ['cnn', 'fdy', 'itransformer']:
        torch.manual_seed(31)
        model = make_model(kind, architecture_version=2, dropout=.1, model_width=8)
        model.train()
        initial_keys = tuple(model.state_dict())
        saved_rng = torch.get_rng_state().clone()
        result = contrastive_regularizer(model, kind, x, 'cpu', .2, 77)
        assert torch.equal(saved_rng, torch.get_rng_state())
        repeat = contrastive_regularizer(model, kind, x, 'cpu', .2, 77)
        assert torch.equal(result, repeat)
        assert result.ndim == 0 and torch.isfinite(result)
        result.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        magnitude = float(sum(g.abs().sum() for g in grads))
        assert magnitude > 0
        assert tuple(model.state_dict()) == initial_keys
        duplicates = np.repeat(x[:1], 3, axis=0)
        duplicates[1] += 1
        duplicates[2] -= 2
        zero = contrastive_regularizer(model, kind, duplicates, 'cpu', .2, 77)
        assert zero.requires_grad and float(zero.detach()) == 0
        grouped = contrastive_regularizer(model, kind, x, 'cpu', .2, 77, source_groups=['one'] * 4)
        assert float(grouped.detach()) == 0
        singleton = contrastive_regularizer(model, kind, x[:1], 'cpu', .2, 77)
        assert singleton.requires_grad and float(singleton.detach()) == 0
        fixture[kind] = {'loss': float(result.detach()), 'finite_nonzero_gradient_l1': magnitude,
                         'rng_restored': True, 'repeat_seed_equal': True,
                         'duplicate_only_loss': 0, 'same_group_only_loss': 0}
        model.eval()
        rejects(lambda: contrastive_regularizer(model, kind, x, 'cpu'))
    assert np.array_equal(x, original, equal_nan=True)
    rejects(lambda: common_gain_views(np.full((2, 15, 25), np.nan)))
    rejects(lambda: common_gain_views(np.full((2, 15, 25), np.inf)))
    rejects(lambda: nt_xent(first, first, 0))
    rejects(lambda: contrastive_regularizer(model, 'tabm', x, 'cpu'))
    print(json.dumps({'status': 'CPU synthetic fixture passed', 'real_data_read': False,
                      'cuda_tested': False, 'training_performed': False,
                      'manual_nt_xent_reference': float(loss.detach()), 'models': fixture}, indent=2))


if __name__ == '__main__':
    main()
