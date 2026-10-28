"""Bounded full-backbone wiring check; TRAIN only, never resumes production."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from pipeline.models_r16_f1_preflight import R16F1PreflightModel
from pipeline.models_r16_production import decoupled_loss
from pipeline.relative_grid_r6_v2 import relative_grid_v2
from ops.r16_temporal_features import temporal_grid

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'mps'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2026102817)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no automatic CPU fallback')
    source = ROOT / 'reports/r16_target_matched_support_r1/data/S1_TM_H0_F1.npz'
    pause = ROOT / 'models/r16_production_nn_fold0_r1/PAUSED.json'
    pause_sha = sha(pause)
    with np.load(source, allow_pickle=False) as d:
        active = (d['cell_share'] > 0).any(2).sum(1)
        ids = [int(np.flatnonzero(d['share'] == 0)[0]),
               int(np.flatnonzero((d['share'] > 0) & (active >= 12))[0])]
        db = d['db'][ids]
        y = torch.tensor(d['share'][ids], dtype=torch.float32)
        q = torch.tensor(d['cell_share'][ids], dtype=torch.float32)
    base, w = relative_grid_v2(db)
    extra, _ = temporal_grid(db)
    x = torch.from_numpy(np.concatenate([base, extra], 1))
    weights = torch.from_numpy(w)
    device = torch.device(args.device)
    target = (y.to(device), (y > 0).float().to(device), q.to(device))
    xx, ww = x.to(device), weights.to(device)
    rows = []
    for kind in ('bg_cnn', 'fdy', 'itransformer', 'tfgrid_fdy_conformer'):
        started = time.perf_counter()
        torch.manual_seed(2026102817)
        cpu = R16F1PreflightModel(kind).eval()
        with torch.no_grad():
            reference = tuple(t.detach() for t in cpu(x, weights))
        model = deepcopy(cpu).to(device).eval()
        with torch.no_grad():
            actual = model(xx, ww)
        delta = max(float((a.cpu() - b).abs().max()) for a, b in zip(actual, reference))
        if delta > 5e-4:
            raise AssertionError(f'{kind} CPU/device discrepancy {delta}')
        assert torch.allclose(actual[0], (actual[2] * ww).sum((1, 2)), atol=1e-7, rtol=0)
        assert torch.count_nonzero(actual[2][ww == 0]) == 0
        # Regression and presence gradients cannot flow into each other.
        reg = set(map(id, model.regressor.parameters()))
        clf = set(map(id, model.classifier.parameters()))
        assert not reg & clf
        model(xx, ww)[0].sum().backward()
        assert all(p.grad is None for p in model.classifier.parameters())
        model.zero_grad(set_to_none=True)
        model(xx, ww)[1].sum().backward()
        assert all(p.grad is None for p in model.regressor.parameters())
        model.zero_grad(set_to_none=True)
        # The independent score ignores spectral profile and pooling weights.
        with torch.no_grad():
            changed = xx.clone()
            changed[:, 4] += xx[:, 5] * 7
            changed[:, 7] *= 2
            score_delta = float((model.classifier(changed) - model.classifier(xx)).abs().max())
        assert score_delta == 0
        # One disposable Adam step confirms real physical targets connect.
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        output = model(xx, ww)
        loss, _ = decoupled_loss(output, target, ww)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if kind == 'itransformer':
            added_grad = model.regressor.backbone.embed.weight.grad[:, 8 * 15:]
        else:
            b = model.regressor.backbone
            stem = b.network.body[0] if kind == 'bg_cnn' else (b.grid.stem[0] if kind == 'tfgrid_fdy_conformer' else b.stem)
            added_grad = stem.weight.grad[:, 8:]
        f1_gradient = float(added_grad.abs().sum())
        assert f1_gradient > 0
        optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model(xx, ww)[0].cpu()
        with tempfile.TemporaryDirectory(prefix='r16_f1_preflight_') as tmp:
            path = Path(tmp) / 'disposable.pt'
            torch.save(model.cpu().state_dict(), path)
            reloaded = R16F1PreflightModel(kind).eval()
            reloaded.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            with torch.no_grad():
                reload_gap = float((reloaded(x, weights)[0] - expected).abs().max())
            assert reload_gap < 5e-4
        row = {**model.architecture(), 'device': args.device, 'loss': float(loss.detach().cpu()),
               'CPU_device_max_abs': delta, 'checkpoint_reload_max_abs': reload_gap,
               'F1_gradient_L1': f1_gradient, 'classifier_denominator_profile_delta': score_delta,
               'physical_pooling': 'passed', 'gradient_decoupling': 'passed',
               'elapsed_seconds': time.perf_counter() - started}
        rows.append(row)
        print(json.dumps({'kind': kind, 'status': 'passed', 'elapsed_seconds': row['elapsed_seconds']}), flush=True)
        del model, cpu, reloaded, optimizer, actual, output
        if args.device == 'mps':
            torch.mps.empty_cache()
    assert sha(pause) == pause_sha
    result = {'status': 'passed', 'scope': 'two TRAIN samples; disposable single Adam step per backbone',
              'torch_version': torch.__version__, 'device': args.device, 'rows': rows,
              'source_data_sha256': sha(source), 'source_indices': ids,
              'old_pause_receipt_unchanged': True, 'production_training_started': False,
              'Test_used': False, 'validation_used': False,
              'source_code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                 [Path(__file__), ROOT / 'pipeline/models_r16_f1_preflight.py', ROOT / 'ops/r16_temporal_features.py']}}
    output = ROOT / f'reports/r16_multisupport_r1/FULL_F1_PREFLIGHT_{args.device.upper()}.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(str(output), flush=True)


if __name__ == '__main__':
    main()
