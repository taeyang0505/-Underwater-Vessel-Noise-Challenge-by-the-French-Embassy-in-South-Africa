"""Disposable MPS throughput probes; never edits or resumes production learners."""
from pathlib import Path
import argparse, contextlib, gc, hashlib, json, os, statistics, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'reports/r16_mps_speed_audit_r1'


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(value, indent=2, allow_nan=False))
    q.replace(p)


def prepare():
    import numpy as np
    from pipeline.training_r16_multisupport import features, codes
    source = ROOT / 'reports/r16_multisupport_full_preflight_r1/H0.npz'
    with np.load(source, allow_pickle=False) as data:
        x, w = features(data['db'])
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / 'TRAIN_fixture.npz', x=x, w=w,
                 share=data['share'], presence=data['presence'], cell_share=data['cell_share'])
    write('INPUT.json', {'source': str(source.relative_to(ROOT)), 'rows': len(x),
          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'code_sha256': codes(), 'Test_used': False, 'production_weights_used': False})


def run():
    import numpy as np
    import torch
    from pipeline.models_r16_multisupport import MultiSupportModel, decoupled_loss, clip_decoupled_gradients
    torch.set_num_threads(2)
    assert torch.backends.mps.is_available(), 'Actual Metal device required'
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    data = np.load(OUT / 'TRAIN_fixture.npz', allow_pickle=False)
    receipt = {'status': 'running', 'torch': torch.__version__,
        'device': 'mps', 'fallback': False, 'CPU_threads': 2,
        'Test_used': False, 'production_sources_modified': False,
        'production_weights_used': False, 'disposable_training_only': True,
        'concurrent_production_stage': 'CPU Trees', 'results': []}
    write('RESULTS.json', receipt)

    def raw_forward(m, x, w):
        q = m.regressor._cell_mask(x).float()
        q = torch.where(w > 0, q, torch.zeros_like(q))
        return (q * w.float()).sum((1, 2)), m.classifier(x), q

    def raw_loss(prediction, targets, w):
        p, c, q = prediction
        y, z, tq = targets
        a = (p-y).square()
        b = ((q-tq).square() * w).sum((1, 2))
        c = torch.nn.functional.binary_cross_entropy_with_logits(c, z, reduction='none')
        return (a + .25*b).mean() + c.mean(), (a.sum(), b.sum(), c.sum())

    for kind in ('bg_cnn', 'tfgrid_fdy_conformer'):
        for mode in ('current_fp32', 'prechecked_deferred_fp32', 'prechecked_deferred_bf16'):
            try:
                torch.manual_seed(20261028)
                m = MultiSupportModel(kind)
                ix = np.arange(64) % len(data['x'])
                xc = torch.from_numpy(data['x'][ix]); wc = torch.from_numpy(data['w'][ix])
                m.regressor._validate(xc, wc)
                m.eval()
                with torch.no_grad():
                    ordinary = m(xc, wc)
                    # A disposable instance only. Full CPU fixture checks above remain intact.
                    if mode != 'current_fp32':
                        m.regressor._validate_x = lambda x: None
                        direct = raw_forward(m, xc, wc)
                        parity = max(float((a-b).abs().max()) for a, b in zip(ordinary, direct))
                        assert parity == 0, parity
                    else:
                        parity = 0.
                m = m.to('mps').train()
                opt = torch.optim.Adam(m.parameters(), lr=.001, weight_decay=.0003)
                xc = xc.to('mps'); wc = wc.to('mps')
                targets = tuple(torch.tensor(data[k][ix], dtype=torch.float32, device='mps')
                                for k in ('share', 'presence', 'cell_share'))
                latest = None

                def step():
                    nonlocal latest
                    opt.zero_grad(set_to_none=True)
                    ctx = torch.autocast('mps', dtype=torch.bfloat16) if mode.endswith('bf16') else contextlib.nullcontext()
                    with ctx:
                        pred = m(xc, wc) if mode == 'current_fp32' else raw_forward(m, xc, wc)
                        if mode == 'current_fp32':
                            loss, parts = decoupled_loss(pred, targets, wc)
                            vals = tuple(parts[k].sum() for k in ('share_mse', 'weighted_cell_mse', 'classification_bce'))
                        else:
                            loss, vals = raw_loss(pred, targets, wc)
                    loss.backward()
                    if mode == 'current_fp32':
                        clip_decoupled_gradients(m)
                    else:
                        torch.nn.utils.clip_grad_norm_(m.regressor.parameters(), 1., error_if_nonfinite=False)
                        torch.nn.utils.clip_grad_norm_(m.classifier.parameters(), 1., error_if_nonfinite=False)
                    opt.step()
                    latest = loss.detach()
                    if mode == 'current_fp32':
                        for v in vals: float(v.detach().cpu())
                    return torch.stack(vals).detach()

                for _ in range(2): step()
                torch.mps.synchronize()
                blocks = []
                for _ in range(3):
                    torch.mps.synchronize(); start = time.perf_counter()
                    acc = torch.zeros(3, device='mps')
                    for j in range(6): acc += step()
                    acc.cpu(); torch.mps.synchronize()
                    blocks.append((time.perf_counter() - start) / 6)
                finite = bool(torch.isfinite(latest).cpu()) and all(bool(torch.isfinite(p).all().cpu()) for p in m.parameters())
                assert finite, 'Nonfinite loss/weights'
                result = {'kind': kind, 'mode': mode, 'batch_size': 64,
                    'median_step_seconds': statistics.median(blocks), 'block_step_seconds': blocks,
                    'examples_per_second': 64 / statistics.median(blocks),
                    'sampled_driver_bytes': torch.mps.driver_allocated_memory(), 'memory_is_peak': False,
                    'FP32_prechecked_eval_max_abs': parity, 'finite_loss_and_parameters': finite,
                    'last_disposable_loss': float(latest.cpu()),
                    'input_transfer_timed': False, 'full_epoch_measured': False,
                    'candidate_drops_per_step_host_checks': mode != 'current_fp32'}
            except Exception as e:
                result = {'kind': kind, 'mode': mode, 'status': 'failed', 'error': repr(e)}
            receipt['results'].append(result); write('RESULTS.json', receipt)
            print(json.dumps(result), flush=True)
            m = opt = xc = wc = targets = latest = None
            gc.collect(); torch.mps.empty_cache()
    receipt['status'] = 'completed'; write('RESULTS.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('prepare', 'run'))
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else run)()
