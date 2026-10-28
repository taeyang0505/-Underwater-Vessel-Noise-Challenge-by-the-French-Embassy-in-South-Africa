"""Bounded connection/coverage tests, not an optimization experiment."""
from pathlib import Path
import csv
import json
import sys
import tempfile
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from pipeline.r16_multisupport_data import KINDS, REGIMES, OUT, load, read, write, sha
from pipeline import training_r16_multisupport as base
from pipeline.training_r16_multisupport_fivefold import OPS, codes, jobs, folder_for, validate_fold_manifest
from ops.run_r16_multisupport_fivefold import verify_csv, handoff


def main():
    began = time.perf_counter(); torch.set_num_threads(1)
    pins = codes()
    assert len(jobs()) == len(set(jobs())) == 32
    assert {f for f, k, r in jobs()} == {1, 2, 3, 4}
    assert all((f, k, r) in jobs() for f in range(1, 5) for k in KINDS for r in REGIMES)
    assert len({folder_for(f, r, k) for f in range(5) for k in KINDS for r in REGIMES}) == 40
    assert all(folder_for(0, r, k) == base.MODELS / 'neural' / r / k for k in KINDS for r in REGIMES)
    try: handoff()
    except (ValueError, FileNotFoundError): early_blocked = True
    else: early_blocked = False
    if read(ROOT / 'ops/r16_multisupport_full_r1/state.json')['phase'] != 'completed':
        assert early_blocked
    with tempfile.TemporaryDirectory(prefix='r16-5fold-contract-', dir='/private/tmp') as tmp:
        f = Path(tmp) / 'fixture.csv'
        with f.open('w', newline='') as stream:
            w = csv.writer(stream); w.writerow(['ID', 'share_added_vessel', 'score_added_vessel'])
            w.writerows([['a', .1, .2], ['b', .3, .4]])
        verify_csv(f, sha(f), expected_rows=2)
        for expected_sha, n in [('0' * 64, 2), (sha(f), 3)]:
            try: verify_csv(f, expected_sha, expected_rows=n)
            except ValueError: pass
            else: raise AssertionError('Bad CSV was accepted')
    proof = [validate_fold_manifest(f) for f in range(1, 5)]
    manifest = read(OUT / 'manifest.json')
    d = load(ROOT / manifest['folds']['1']['roles']['train']['H0']['path'])
    ix = [int(np.flatnonzero(d['share'] == 0)[0]), int(np.flatnonzero(d['share'] > 0)[0])]
    x, w = base.features(d['db'][ix])
    target = tuple(torch.tensor(d[k][ix], dtype=torch.float32) for k in ('share', 'presence', 'cell_share'))
    del d
    from pipeline.models_r16_multisupport import MultiSupportModel, decoupled_loss, clip_decoupled_gradients
    xx, ww = torch.from_numpy(x), torch.from_numpy(w)
    rows = []
    # A bounded disposable real-TRAIN gradient/reload check for each unchanged backbone.
    for kind in KINDS:
        torch.manual_seed(base.SEED + 10007)
        model = MultiSupportModel(kind).eval()
        with torch.no_grad(): p, c, q = model(xx, ww)
        assert torch.allclose(p, (q * ww).sum((1, 2)), atol=1e-7, rtol=0)
        model.train(); optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0003)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = decoupled_loss(model(xx, ww), target, ww)
        loss.backward(); clip_decoupled_gradients(model)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step(); model.eval()
        with torch.no_grad(): expected = model(xx, ww)
        with tempfile.TemporaryDirectory(prefix='r16-5fold-state-', dir='/private/tmp') as tmp:
            path = Path(tmp) / 'disposable.pt'
            base.atomic_pt(path, {'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()})
            restored = MultiSupportModel(kind).eval()
            restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict'])
            with torch.no_grad(): actual = restored(xx, ww)
            gap = max(float((a - b).abs().max()) for a, b in zip(actual, expected))
            assert gap == 0
        rows.append({'kind': kind, 'CPU_forward_backward_checkpoint_reload': 'passed',
                     'maximum_reload_error': gap, 'disposable_only': True})
    base.verify(pins)
    # Same frozen feature extractor and no changed model/optimizer hyperparameters.
    write(OPS / 'CONNECTION_VERIFICATION.json', {'status': 'passed', 'code_sha256': pins,
          '32_unique_new_jobs': True, '40_unique_full_model_paths': True,
          'fold0_completion_barrier_rejects_early_start': early_blocked,
          'bad_CSV_hash_and_row_count_rejected': True, 'fold_source_purge_proof': proof,
          'disposable_CPU_checks': rows, 'new_fold_full_training_started': False,
          'full_finalization_executed': False, 'MPS_validation': 'reuses unchanged actual Fold0 preflight after SHA check',
          'Test_used': False, 'locked_panels_used': False, 'elapsed_seconds': time.perf_counter() - began})
    print(json.dumps({'status': 'passed', 'elapsed_seconds': time.perf_counter() - began,
                      'new_jobs': len(jobs()), 'Test_used': False}), flush=True)


if __name__ == '__main__':
    main()
