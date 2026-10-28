"""Independent Tree-only candidate; frozen equal folds/regimes, no fitting.

May run while the original neural learner runs: only complete Tree artifacts
are read, and every output has a separate destination.
"""
from pathlib import Path
import gc
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from pipeline.r16_multisupport_data import OUT, REGIMES, read, write, sha, save, role
from pipeline.training_r16_multisupport import MODELS, codes, verify, tree_features, metric

DEST = ROOT / 'submissions/r16_multisupport_tree5_r1'


def registry():
    pins = codes()
    done = MODELS / 'ALL_TREES_COMPLETED.json'
    d = read(done)
    if (d['status'] != 'completed' or d['code_sha256'] != codes()
            or d['data_manifest_sha256'] != sha(OUT / 'manifest.json')
            or set(d['models']) != {f'{r}_{f}' for r in REGIMES for f in range(5)}):
        raise ValueError('All ten matching completed Tree pairs required')
    pins[str(done.relative_to(ROOT))] = sha(done)
    for r in REGIMES:
        for f in range(5):
            folder = MODELS / 'trees' / r / f'fold_{f:02d}'
            s = read(folder / 'summary.json')
            if (s['status'] != 'completed' or s['fold'] != f or s['regime'] != r
                    or s['code_sha256'] != codes()
                    or s['data_manifest_sha256'] != d['data_manifest_sha256']):
                raise ValueError('Tree source/fold binding differs')
            for name, h in s['artifact_sha256'].items():
                p = folder / name
                if sha(p) != h:
                    raise ValueError('Completed Tree bytes changed')
                pins[str(p.relative_to(ROOT))] = h
            pins[str((folder / 'summary.json').relative_to(ROOT))] = sha(folder / 'summary.json')
    for name in ('ops/infer_r16_multisupport_tree5.py', 'ops/predict_r16_production_nn.py',
                 'ops/finalize_r16_production.py'):
        pins[name] = sha(ROOT / name)
    return pins


def main():
    import lightgbm as lgb
    from ops.predict_r16_production_nn import guard_state, read_raw_csv
    from ops.finalize_r16_production import write_submission, _stats, descriptive_comparison
    state, audit = guard_state()
    sys.addaudithook(audit)
    started = time.perf_counter()
    pins = registry()
    if DEST.exists():
        raise FileExistsError('Fresh candidate only; never overwrite')
    DEST.mkdir(parents=True)
    policy = {'format': 'r16_multisupport_tree5_fixed_v1', 'members': [],
              'fold_weights_each_regime': [0.2] * 5, 'regime_weights': [0.5, 0.5],
              'regression': 'clip each raw Tree prediction to [0,1], then equal average',
              'score': 'independent Tree classifier probability only; no share/contrast',
              'postprocessing': '[0,1] clipping only', 'policy_fit': False,
              'Test_used_for_policy': False, 'official_upload': False}
    for r in REGIMES:
        for f in range(5):
            policy['members'].append({'regime': r, 'fold': f, 'weight': 0.1})
    write(DEST / 'frozen_policy.json', policy)
    pins[str((DEST / 'frozen_policy.json').relative_to(ROOT))] = sha(DEST / 'frozen_policy.json')
    write(DEST / 'GLOBAL_FREEZE.json', {'status': 'completed', 'all_models_frozen': True,
          'ensemble_policy_frozen': True, 'artifact_sha256': pins,
          'data_manifest_sha256': sha(OUT / 'manifest.json'), 'Test_used': False})
    verify(pins)
    query = (ROOT / 'data/raw/Test.csv').resolve()
    state['allowed_query'] = query
    ids, db, query_receipt = read_raw_csv(query)
    xr = tree_features(db, 'regression')
    xc = tree_features(db, 'classification')
    pr, pc, members = [], [], []
    for r in REGIMES:
        for f in range(5):
            folder = MODELS / 'trees' / r / f'fold_{f:02d}'
            a = lgb.Booster(model_file=str(folder / 'regression.txt'))
            b = lgb.Booster(model_file=str(folder / 'classification.txt'))
            pr.append(np.clip(a.predict(xr, num_threads=1), 0, 1))
            pc.append(b.predict(xc, num_threads=1))
            members.append(f'{r}_fold{f}')
            del a, b
            gc.collect()
    matrix_p, matrix_c = np.column_stack(pr), np.column_stack(pc)
    pred = {'share': np.clip(matrix_p.mean(1), 0, 1), 'score': np.clip(matrix_c.mean(1), 0, 1)}
    receipt = write_submission(DEST / 'submission_r16_tree5.csv', ids, pred)
    save(DEST / 'query_predictions.npz', {'sample_id': ids, **pred,
         'member_share': matrix_p, 'member_classifier': matrix_c, 'member_order': np.asarray(members)})
    verify(pins)
    result = {'status': 'completed', 'submission': receipt,
              'statistics': {k: _stats(v) for k, v in pred.items()},
              'historical_comparison': descriptive_comparison(ids, pred),
              'query_receipt': query_receipt, 'elapsed_seconds': time.perf_counter() - started,
              'official_upload': False, 'policy_changed_after_query': False,
              'official_RMSE_AUC': 'unmeasured; hidden Test labels unavailable',
              'neural_training_modified_or_paused': False}
    write(DEST / 'COMPLETED.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
