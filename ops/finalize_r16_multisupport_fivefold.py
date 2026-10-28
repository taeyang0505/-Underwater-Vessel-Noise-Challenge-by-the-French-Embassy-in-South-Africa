"""Freeze Tree5 + neural5 per regime, evaluate own folds, then infer Test once.

No recipe/model/weight selection uses CV, locked panels, or Test. The original
Fold0 candidate and its completed artifacts are retained byte-for-byte.
"""
from pathlib import Path
import gc
import html
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.r16_multisupport_data import OUT, PARENT, REGIMES, KINDS, read, write, sha, load, save, role
from pipeline import training_r16_multisupport as original
from pipeline.training_r16_multisupport_fivefold import MODELS, codes, folder_for, validate_completed
from ops.finalize_r16_multisupport_full import MEMBERS, aligned, fit_policy, apply, prerequisites

FINAL = MODELS / 'final'
REPORT = ROOT / 'reports/r16_multisupport_fivefold_r1'


def prerequisite_pins():
    pins = {**prerequisites(), **codes()}
    for f in range(1, 5):
        for k in KINDS:
            for r in REGIMES:
                folder = folder_for(f, r, k)
                s = validate_completed(folder, f, r, k)
                for name, h in s['artifact_sha256'].items():
                    pins[str((folder / name).relative_to(ROOT))] = h
                pins[str((folder / 'summary.json').relative_to(ROOT))] = sha(folder / 'summary.json')
    for name in ('ops/finalize_r16_multisupport_full.py', 'pipeline/ensemble_r16_production.py'):
        pins[name] = sha(ROOT / name)
    return pins


class FullPredictor:
    def __init__(self):
        import lightgbm as lgb
        self.trees = {r: [(lgb.Booster(model_file=str(original.MODELS / 'trees' / r / f'fold_{f:02d}' / 'regression.txt')),
                           lgb.Booster(model_file=str(original.MODELS / 'trees' / r / f'fold_{f:02d}' / 'classification.txt')))
                         for f in range(5)] for r in REGIMES}

    def neural(self, fold, regime, kind, x, w):
        import torch
        from pipeline.models_r16_multisupport import MultiSupportModel
        folder = folder_for(fold, regime, kind)
        model = MultiSupportModel(kind).eval()
        cp = torch.load(folder / 'best_regression.pt', map_location='cpu', weights_only=True)
        expected = original.codes() if fold == 0 else codes()
        if (cp['architecture'] != model.architecture() or cp['code_sha256'] != expected
                or cp['data_manifest_sha256'] != sha(OUT / 'manifest.json')
                or (fold and cp['fold'] != fold)):
            raise ValueError('Frozen checkpoint binding failed')
        model.load_state_dict(cp['state_dict'])
        pp = []
        with torch.no_grad():
            for start in range(0, len(x), 64):
                xx = torch.from_numpy(x[start:start + 64]); ww = torch.from_numpy(w[start:start + 64])
                p, _, q = model(xx, ww)
                if not torch.allclose(p, (q * ww).sum((1, 2)), atol=1e-7, rtol=0):
                    raise ValueError('Physical pooling differs')
                pp.append(p.numpy())
        cp = torch.load(folder / 'best_classifier.pt', map_location='cpu', weights_only=True)
        if (cp['architecture'] != model.architecture() or cp['code_sha256'] != expected
                or cp['data_manifest_sha256'] != sha(OUT / 'manifest.json')
                or (fold and cp['fold'] != fold)):
            raise ValueError('Frozen classifier binding failed')
        model.classifier.load_state_dict({k.removeprefix('classifier.'): v for k, v in cp['state_dict'].items()
                                          if k.startswith('classifier.')})
        cc = []
        with torch.no_grad():
            for start in range(0, len(x), 128):
                cc.append(model.classifier(torch.from_numpy(x[start:start + 128])).sigmoid().numpy())
        del model, cp
        gc.collect()
        return np.concatenate(pp), np.concatenate(cc)

    def predict(self, db, folds=tuple(range(5))):
        if not folds or len(set(folds)) != len(folds) or any(f not in range(5) for f in folds):
            raise ValueError('Exact nonduplicate fold set required')
        x, w = original.features(db)
        xr, xc = original.tree_features(db, 'regression'), original.tree_features(db, 'classification')
        shares, probs = {}, {}
        for r in REGIMES:
            ps = [np.mean([np.clip(self.trees[r][f][0].predict(xr, num_threads=2), 0, 1) for f in folds], 0)]
            cs = [np.mean([self.trees[r][f][1].predict(xc, num_threads=2) for f in folds], 0)]
            for k in KINDS:
                values = [self.neural(f, r, k, x, w) for f in folds]
                ps.append(np.mean([p for p, c in values], 0))
                cs.append(np.mean([c for p, c in values], 0))
            shares[r], probs[r] = np.column_stack(ps), np.column_stack(cs)
        if any(not np.isfinite(a).all() or ((a < 0) | (a > 1)).any()
               for a in [*shares.values(), *probs.values()]):
            raise ValueError('Invalid frozen output')
        return shares, probs


def finalize():
    import torch
    from ops.predict_r16_production_nn import guard_state, read_raw_csv
    from ops.finalize_r16_production import write_submission, _stats, descriptive_comparison
    from ops.r16_support_temporal_data import load_panel, load_long_panel
    from ops.r16_forward_data import OPERATORS
    state, guard = guard_state(); sys.addaudithook(guard); torch.set_num_threads(2)
    pins = prerequisite_pins()
    if FINAL.exists():
        raise FileExistsError('Never reuse partial finalization')
    FINAL.mkdir(parents=True); REPORT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    fit = role(0, 'D0', 'selection'); ids, y = fit['sample_id'], fit['share']
    ps, cs, experts = {}, {}, {}
    for r in REGIMES:
        trees = [aligned(load(original.MODELS / 'trees' / r / f'fold_{f:02d}' / 'selection.npz'), ids, y)
                 for f in range(5)]
        a, b = [np.mean([d['share'] for d in trees], 0)], [np.mean([d['classifier_prob'] for d in trees], 0)]
        for k in KINDS:
            a.append(np.mean([aligned(load(folder_for(f, r, k) / 'selection_regression.npz'), ids, y)['share']
                              for f in range(5)], 0))
            b.append(np.mean([aligned(load(folder_for(f, r, k) / 'selection_classification.npz'), ids, y)['classifier_prob']
                              for f in range(5)], 0))
        ps[r], cs[r] = np.column_stack(a), np.column_stack(b)
        experts[r] = fit_policy(ps[r], cs[r], y)
    policy = {'format': 'r16_multisupport_neural5_tree5_v1', 'experts': experts,
              'member_order': list(MEMBERS), 'within_architecture_fold_weights': [0.2] * 5,
              'regime_regression_weights': [0.5, 0.5], 'regime_classifier_weights': [0.5, 0.5],
              'score_share_weight': 0., 'fit_role': 'shared local_fit2400/unique400 only',
              'postprocessing': 'clip [0,1] only', 'Test_used_for_policy': False,
              'native12_HF_pilot_RMSE_miss': {'value': .1175534730485378, 'criterion': .11,
                                            'user_authorized_proceed': True}}
    write(FINAL / 'policy.json', policy)
    pins[str((FINAL / 'policy.json').relative_to(ROOT))] = sha(FINAL / 'policy.json')
    registry = FINAL / 'MODEL_REGISTRY.json'
    write(registry, {'status': 'completed', 'artifact_sha256': pins})
    freeze = FINAL / 'GLOBAL_FREEZE.json'
    write(freeze, {'status': 'completed', 'all_models_frozen': True, 'ensemble_policy_frozen': True,
          'data_manifest_sha256': sha(PARENT), 'artifact_sha256': {str(registry.relative_to(ROOT)): sha(registry),
          str((FINAL / 'policy.json').relative_to(ROOT)): sha(FINAL / 'policy.json')}, 'Test_used': False})
    original.verify(pins)
    predictor = FullPredictor()
    rows, ownfold = [], []
    fitted = apply(policy, ps, cs)
    rows.append({'panel': 'local_fit', 'development_only': True,
                 **original.metric(y, fitted['share'], fitted['score'])})

    def one(name, d, weights=None, folds=tuple(range(5)), cv=False):
        s, c = predictor.predict(d['db'], folds=folds)
        e = apply(policy, s, c)
        save(FINAL / 'evaluation' / (name + '.npz'), {'sample_id': d['sample_id'],
             'target_share': d['share'], **e, **{'share_' + r: s[r] for r in REGIMES},
             **{'classifier_' + r: c[r] for r in REGIMES}})
        m = {'panel': name, 'prediction_folds': list(folds), **original.metric(d['share'], e['share'], e['score'], weights)}
        if 'site' in d:
            m['site_metrics'] = {str(site): original.metric(d['share'][d['site'] == site], e['share'][d['site'] == site],
                               e['score'][d['site'] == site]) for site in np.unique(d['site'])}
            m['site_macro_RMSE'] = float(np.mean([v['rmse'] for v in m['site_metrics'].values()]))
        (ownfold if cv else rows).append(m)
        write(MODELS / 'progress.json', {'stage': 'frozen_evaluation', **m})
        print(json.dumps(m), flush=True)
        return e

    # Evaluate each CV example only with that fold's Tree + neural checkpoints.
    # Averaging all five trained folds here would leak own training examples.
    for r in REGIMES:
        ys, pp, cc = [], [], []
        for f in range(5):
            d = role(f, r, 'cv')
            e = one(f'ownfold_{r}_{f}', d, folds=(f,), cv=True)
            ys.append(d['share']); pp.append(e['share']); cc.append(e['score'])
        ownfold.append({'regime': r, 'panel': 'pooled_ownfold_only',
                        **original.metric(np.concatenate(ys), np.concatenate(pp), np.concatenate(cc))})
    for name in ('local_gate', 'local_report', 'domain_holdout'):
        parts = [load_panel(PARENT, name, op, freeze_receipt=freeze) for op in OPERATORS]
        keys = set.intersection(*(set(d) for d in parts))
        one(name, {k: np.concatenate([d[k] for d in parts]) for k in keys})
    parts = [load_long_panel(PARENT, 'native12_positive', operator=op, freeze_receipt=freeze) for op in OPERATORS]
    keys = set.intersection(*(set(d) for d in parts))
    positive = {k: np.concatenate([d[k] for d in parts]) for k in keys}
    for name in ('hf_high_negative', 'st_francis_dense_negative'):
        negative = load_long_panel(PARENT, name, freeze_receipt=freeze)
        keys = set(positive) & set(negative)
        d = {k: np.concatenate([positive[k], negative[k]]) for k in keys}
        weights = np.r_[np.full(len(positive['share']), .5 / len(positive['share'])),
                        np.full(len(negative['share']), .5 / len(negative['share']))]
        one('native12_VS_' + name, d, weights=weights)
    write(FINAL / 'EVALUATION.json', {'status': 'completed', 'rows': rows, 'hybrid_ownfold_CV': ownfold,
          'ownfold_checkpoint_only': True, 'pooled_CV_is_nested_Aprime75_only': True,
          'locked_results_used_to_refit': False, 'unique_backgrounds_per_pooled_primary_panel': 400})
    original.verify(pins)
    query = (ROOT / 'data/raw/Test.csv').resolve(); state['allowed_query'] = query
    qids, db, query_receipt = read_raw_csv(query)
    s, c = predictor.predict(db); e = apply(policy, s, c)
    receipt = write_submission(ROOT / 'submissions/r16_multisupport_fivefold_r1/submission_r16_5fold.csv', qids, e)
    save(FINAL / 'query_predictions.npz', {'sample_id': qids, **e,
         **{'share_' + r: s[r] for r in REGIMES}, **{'classifier_' + r: c[r] for r in REGIMES}})
    stats = {'submission': receipt, 'query_receipt': query_receipt,
             'statistics': {k: _stats(v) for k, v in e.items()},
             'historical_comparison': descriptive_comparison(qids, e),
             'Test_labels_used': False, 'policy_changed_after_query': False, 'official_upload': False}
    write(FINAL / 'TEST_PREDICTION_AUDIT.json', stats); original.verify(pins)
    lines = ['# R16 D0/H0 전체 5-Fold 결과', '',
             'Tree 10쌍(회귀·분류) + 신경망 40회. Fold0 완료 모델을 보존하고 Fold1~4는 독립 초기화, 각100epoch/Adam/noES로 학습했다.',
             '배경은 기존R9 Fold0 TRAIN 안의 nested5fold, 참조는 A′75클립/13그룹이다. 전체509개 음원 CV는 아니다.',
             '회귀 Tree≥50%, 각 NN≤25%, 독립 score에는 share/contrast를 넣지 않는다. D0/H0 각각50:50, 후처리는clip만 적용했다.', '',
             '| 평가 | RMSE | 독립 score AUC |', '|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['panel']} | {row['rmse']:.6f} | {row['classifier_auc']:.6f} |")
    lines += ['', '[전체5Fold CSV](../../submissions/r16_multisupport_fivefold_r1/submission_r16_5fold.csv)',
              '[평가 원자료](../../models/r16_multisupport_fivefold_r1/final/EVALUATION.json)',
              '공식 성능은 미측정이며, Test 분포를 이용한 보정 및 공식 업로드는 하지 않았다.']
    (REPORT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    (REPORT / 'index.html').write_text('<!doctype html><meta charset="utf-8"><pre>' + html.escape('\n'.join(lines)) + '</pre>')
    write(FINAL / 'COMPLETED.json', {'status': 'completed', 'submission': receipt,
          'elapsed_seconds': time.perf_counter() - start, 'official_upload': False})


if __name__ == '__main__':
    finalize()
