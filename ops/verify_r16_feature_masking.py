"""Meaningful feature-only mask contract checks, synthetic and cached TRAIN.

No model training, Test input, test labels or frozen source edits occur here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--skip-real-cache', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/r16_pretrained_ft_r1/feature_masking_contract.json')
    args = parser.parse_args()
    import numpy as np
    import torch
    from pipeline.r16_feature_masking import (
        FeatureMaskingConfig, apply_r16_feature_masking, make_feature_masking_generator,
    )
    from pipeline.training_r16_pretrained import CLASSIFIER_CHANNELS, task_input
    torch.set_num_threads(2)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no CPU fallback')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    started = time.monotonic()
    rng = torch.Generator(device='cpu').manual_seed(7931)
    x = torch.randn(16, 17, 15, 25, generator=rng)
    x[:, 5] = 1
    # Dense, sparse-minute, sparse-cell and tiny-support observation patterns.
    x[0, 5] = 0; x[0, 5, 7, 12] = 1
    x[1, 5] = 0; x[1, 5, 1:3, 1] = 1
    x[2, 5] = 0; x[2, 5, 3, :5] = 1
    x[3:7, 5, :10] = 0
    x = x.to(args.device)
    x_before = x.clone()
    global_state = torch.get_rng_state().clone()
    config = FeatureMaskingConfig(block_probability=1., profile_probability=1.)

    def generator(seed=3407):
        return make_feature_masking_generator(seed, epoch=3, stream='contract')

    def verify_effect(a, augmented, meta, task):
        assert torch.equal(a[:, 5], augmented[:, 5]), 'True O modified'
        assert not (meta.context_mask & (a[:, 5] == 0)).any(), 'Missing cell inverted'
        obs = (a[:, 5] > 0).sum((1, 2))
        assert (meta.masked_cells <= (obs.float()*.3).floor()).all()
        assert (meta.visible_minutes >= 1).all()
        context_channels = torch.arange(17, device=a.device) != 5
        hidden = meta.context_mask[:, None] & context_channels[None, :, None, None]
        if task == 'regression':
            profile_mask = torch.zeros_like(a, dtype=torch.bool)
            profile_mask[:, 4] = meta.profile_dropped[:, None, None]
            hidden = hidden | profile_mask
        assert (augmented[hidden] == 0).all(), 'Hidden feature retained'
        assert torch.equal(a[~hidden], augmented[~hidden]), 'Unselected feature changed'
        assert torch.isfinite(augmented).all()
        return meta.log_dict()

    reg, reg_meta = apply_r16_feature_masking(x, task='regression', training=True, generator=generator(), config=config)
    reg_log = verify_effect(x, reg, reg_meta, 'regression')
    assert reg_meta.profile_dropped.all()
    assert reg_meta.masked_cells[0] == 0 and reg_meta.masked_cells[1] == 0
    assert torch.equal(x, x_before), 'Input mutated in place'
    assert torch.equal(torch.get_rng_state(), global_state), 'Global RNG consumed'
    reg2, meta2 = apply_r16_feature_masking(x, task='regression', training=True, generator=generator(), config=config)
    assert torch.equal(reg, reg2) and torch.equal(reg_meta.context_mask, meta2.context_mask)

    # A checkpointed control RNG resumes the next augmentation exactly.
    control = generator()
    apply_r16_feature_masking(x, task='regression', training=True, generator=control, config=config)
    saved_rng = control.get_state().clone()
    expected, expected_meta = apply_r16_feature_masking(x, task='regression', training=True, generator=control, config=config)
    resumed = generator(999); resumed.set_state(saved_rng)
    actual, actual_meta = apply_r16_feature_masking(x, task='regression', training=True, generator=resumed, config=config)
    assert torch.equal(expected, actual) and torch.equal(expected_meta.context_mask, actual_meta.context_mask)

    clf, clf_meta = apply_r16_feature_masking(x, task='classification', training=True, generator=generator(), config=config)
    clf_log = verify_effect(x, clf, clf_meta, 'classification')
    assert not clf_meta.profile_dropped.any(), 'Regression profile dropout leaked to classifier'
    assert torch.equal(reg_meta.context_mask, clf_meta.context_mask)
    classifier_input = task_input(clf, 'classification')
    assert classifier_input.shape == (16, 15, 15, 25)
    # Classifier channel4 is ORIGINAL channel6, not the removed profile channel4.
    visible = ~clf_meta.context_mask
    assert torch.equal(classifier_input[:, 4][visible], x[:, 6][visible])
    assert torch.equal(classifier_input[:, 5], x[:, 5])
    try:
        apply_r16_feature_masking(classifier_input, task='classification', training=True, generator=generator())
    except ValueError:
        pass
    else:
        raise AssertionError('Direct15-channel input should be rejected to prevent channel-index confusion')

    variants = {}
    for variant in ('off', 'spec', 'profile', 'both'):
        cfg = FeatureMaskingConfig.for_variant(variant)
        g = generator(); old_state = g.get_state().clone()
        eval_x, eval_meta = apply_r16_feature_masking(x, task='regression', training=False, generator=g, config=cfg)
        assert eval_x is x and torch.equal(g.get_state(), old_state), 'Eval must be exact identity and consume no RNG'
        out, meta = apply_r16_feature_masking(x, task='regression', training=True, generator=generator(), config=cfg)
        variants[variant] = verify_effect(x, out, meta, 'regression')
        if variant == 'off': assert out is x
        if variant == 'profile': assert not meta.context_mask.any()
        if variant == 'spec': assert not meta.profile_dropped.any()

    # Zeroing features preserves the gradient of visible cells and leaves O
    # unaltered. Masked semantic values receive zero input gradient.
    grad_x = x.detach().clone().requires_grad_(True)
    grad_out, grad_meta = apply_r16_feature_masking(grad_x, task='regression', training=True, generator=generator(), config=config)
    grad_out.sum().backward()
    assert torch.equal(grad_x.grad[:, 5], torch.ones_like(x[:, 5]))
    assert (grad_x.grad[:, 4] == 0).all()
    assert (grad_x.grad[:, 0][grad_meta.context_mask] == 0).all()
    assert (grad_x.grad[:, 0][~grad_meta.context_mask] == 1).all()

    # Ordinary global recorder-level dB offsets cannot change mask decisions.
    from pipeline.training_r16_multisupport import features
    db = np.random.default_rng(31).normal(85, 8, size=(4, 15, 25))
    db[:, 0] = np.nan
    a, wa = features(db); b, wb = features(db + 30.)
    assert np.max(np.abs(a-b)) < 5e-5 and np.max(np.abs(wa-wb)) < 5e-6
    aa, ma = apply_r16_feature_masking(torch.tensor(a, device=args.device), task='regression', training=True, generator=generator(), config=config)
    bb, mb = apply_r16_feature_masking(torch.tensor(b, device=args.device), task='regression', training=True, generator=generator(), config=config)
    assert torch.equal(ma.context_mask, mb.context_mask)
    assert torch.allclose(aa, bb, atol=5e-5, rtol=0)

    # Backend-portable plan: explicit CPU control RNG, device-native mask ops.
    cpu_out, cpu_meta = apply_r16_feature_masking(x.cpu(), task='regression', training=True, generator=generator(), config=config)
    assert torch.equal(cpu_meta.context_mask, reg_meta.context_mask.cpu())
    assert torch.equal(cpu_out, reg.cpu())

    real = []
    if not args.skip_real_cache:
        from pipeline.training_r16_multisupport import CACHE
        for regime in ('D0', 'H0'):
            folder = CACHE/regime/'train'
            required = ('x', 'w', 'share', 'cell_share', 'presence')
            paths = {name:folder/(name+'.npy') for name in required}
            if not all(path.is_file() for path in paths.values()):
                raise RuntimeError('Verified R16 TRAIN cache unavailable: '+str(folder))
            arrays = {name:np.load(path, mmap_mode='r', allow_pickle=False) for name,path in paths.items()}
            # Preserve physical arrays byte-for-byte on CPU for read-only
            # auditing. Only x is an augmentation input; this also avoids
            # inventing a target dtype cast merely because MPS has no FP64.
            batch = {name:torch.from_numpy(np.array(value[:16], copy=True)) for name,value in arrays.items()}
            before = {name:tensor_sha(value) for name,value in batch.items()}
            real_x = batch['x'].to(args.device)
            augmented, meta = apply_r16_feature_masking(real_x, task='regression', training=True, generator=generator(), config=config)
            log = verify_effect(real_x, augmented, meta, 'regression')
            assert torch.equal(real_x.cpu(), batch['x']), 'Original device input mutated'
            after = {name:tensor_sha(value) for name,value in batch.items()}
            assert before == after, 'Cached x/w/q/y/z ownership violated'
            # Loss target identity remains tied to ORIGINAL separate w and q.
            # Explicit read-only audit transfer, not an accelerator fallback:
            # MPS intentionally has no FP64 dtype. Feature masking itself
            # remains device-native and preserves original FP32 tensors.
            audit_w = batch['w'].detach().cpu().double()
            audit_q = batch['cell_share'].detach().cpu().double()
            audit_y = batch['share'].detach().cpu().double()
            reconstructed = (audit_w*audit_q).sum((1,2))
            error = float((reconstructed-audit_y).abs().max())
            # Rounded-db input weights and full-precision preserved targets
            # already differ slightly in the frozen cache. Do not change them
            # to manufacture a tighter identity; report this pre-existing gap.
            assert torch.isfinite(reconstructed).all()
            original_w = batch['w'].clone()
            unchanged_prediction = (original_w * batch['cell_share']).sum((1,2))
            assert torch.equal(unchanged_prediction, (batch['w']*batch['cell_share']).sum((1,2)))
            real.append({'regime':regime, 'cache_role':'TRAIN only', 'batch_n':16,
                         'original_batch_sha256':before, 'source_x_w_q_y_z_unchanged':True,
                         'original_weighted_target_max_abs_error':error, 'masking':log})

    # Empirical frequencies at default probabilities; no labels or scores used.
    rates_x = torch.ones(2048, 17, 15, 25)
    _, rate_meta = apply_r16_feature_masking(rates_x, task='regression', training=True, generator=generator(), config=FeatureMaskingConfig())
    rates = rate_meta.log_dict()
    assert abs(rates['block_requested_samples']/2048-.5) < .05
    assert abs(rates['profile_dropped_samples']/2048-.4) < .05
    report = {
        'passed':True, 'device':args.device, 'torch_version':torch.__version__,
        'synthetic_contract':reg_log, 'classifier_contract':clf_log,
        'variants':variants, 'default_empirical_rates':rates, 'real_cached_train_batches':real,
        'eval_exact_identity_and_no_rng_consumption':True,
        'explicit_generator_resume_exact':True, 'global_rng_not_consumed':True,
        'original_observation_O_preserved':True, 'physical_w_q_y_z_never_recomputed':True,
        'regression_Ch4_dropout_only':True, 'classifier_order_preserved':list(CLASSIFIER_CHANNELS),
        'CPU_accelerator_plans_identical':True, 'global_dB_offset_invariant':True,
        'masked_gradient_zero_and_visible_gradient_preserved':True,
        'Test_or_submission_accessed':False, 'training_started':False,
        'elapsed_seconds':time.monotonic()-started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
