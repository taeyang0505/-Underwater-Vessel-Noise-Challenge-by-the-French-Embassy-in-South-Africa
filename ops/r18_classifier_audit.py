#!/usr/bin/env python3
"""R18 CPU classifier input-information ablation, no Test or policy fitting.

Six single-split fits compare temporal statistics, restored Ch4/sqrt(w), and
absolute dB on exactly the same R17 clean train/early-stop sources. Evaluation
panels are diagnostic and do not change the registered protocol.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('DO_NOT_TRACK', '1')
from pathlib import Path
import gc, json, sys, time, warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops.r17_audit_common import sha, load, jsave
from ops.r17_clean_classifier_features import classifier_indices, original_names
from pipeline.relative_grid_r6_v2 import relative_grid_v2
from pipeline.trees_r16_production import array_sha

OUT = ROOT/'reports/r18_fundamental_audit_r1/classifier'
MODELS = ROOT/'models/r18_classifier_audit_r1'
DATA = ROOT/'data/synthetic/r17_clean_v1'
FRESH = ROOT/'data/synthetic/r17_clean_freshfit_v1'
CACHE = ROOT/'reports/r17_clean_pipeline_r1/refit/features'
MODES = ('temporal', 'profile_energy', 'absolute_db')
PANELS = ('local_gate', 'domain_holdout', 'native12_VS_hf_high_negative',
          'native12_VS_st_francis_dense_negative', 'local_report')
SEED = 2026103018
CODE = ('ops/r18_classifier_audit.py', 'pipeline/relative_grid_r6.py',
        'pipeline/relative_grid_r6_v2.py', 'ops/r16_temporal_features.py',
        'pipeline/trees_r16_production.py', 'pipeline/training_r16_multisupport.py')

def no_query(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        name = Path(os.fsdecode(args[0])).name.lower()
        if name in ('test.csv', 'sample_submission.csv', 'query_predictions.npz'):
            raise PermissionError('R18 classifier audit does not access query inputs/outputs')

def event(**record):
    jsave(OUT/'progress.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)

def extra_features(db):
    d = np.round(np.asarray(db, np.float64), 1)
    o = np.isfinite(d)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(d, axis=1)
        whole = np.nanmedian(d, axis=(1, 2))
    x, w = relative_grid_v2(d)
    # The actual Ch4 value is constant along observed time, zero at missing
    # cells. A bandwise representative is used without duplicating 15 times.
    profile = np.nan_to_num((med-whole[:, None])/20., nan=0.)
    restored = np.concatenate([profile, np.sqrt(w).reshape(len(d), -1)], axis=1)
    # Missing cells are filled with the CURRENT WINDOW band median, never a
    # fitted train/Test mean. Empty bands use whole-window median.
    fill = np.where(np.isfinite(med), med, whole[:, None])
    raw = np.where(o, d, fill[:, None])
    absolute = np.concatenate([raw.reshape(len(d), -1)/100., fill/100., whole[:, None]/100.], axis=1)
    return restored.astype(np.float32), absolute.astype(np.float32)

def names(mode):
    result = [original_names()[i] for i in classifier_indices('direct')]
    if mode != 'temporal':
        result += [f'restored_Ch4_band_profile/f{f:02d}' for f in range(25)]
        result += [f'restored_Ch7_sqrtw/t{t:02d}/f{f:02d}' for t in range(15) for f in range(25)]
    if mode == 'absolute_db':
        result += [f'raw_db_filled_div100/t{t:02d}/f{f:02d}' for t in range(15) for f in range(25)]
        result += [f'raw_band_median_div100/f{f:02d}' for f in range(25)] + ['raw_window_median_div100']
    return result

def cached_temporal(name, db):
    m = json.loads((CACHE/(name+'.json')).read_text())
    p = CACHE/(name+'.npy')
    if m['input'] != {'db_sha256': array_sha(db), 'shape': [len(db), 2033]} or sha(p) != m['sha256']:
        raise ValueError('Immutable R17 feature cache binding differs: '+name)
    return np.load(p, mmap_mode='r')

def matrix(record, mode):
    base = np.asarray(record['temporal'][:, classifier_indices('direct')])
    if mode == 'temporal':
        return base
    pieces = [base, record['restored']]
    if mode == 'absolute_db':
        pieces.append(record['absolute'])
    x = np.concatenate(pieces, axis=1)
    assert x.shape == (len(record['data']['db']), len(names(mode))) and np.isfinite(x).all()
    return x

def metrics(y, p, weights=None):
    from sklearn.metrics import roc_auc_score, log_loss
    y = np.asarray(y, int);p = np.asarray(p, float)
    weights = np.ones(len(y), float) if weights is None else np.asarray(weights, float)
    weights = weights/weights.sum()
    z = y == 1;neg = ~z
    auc = float(roc_auc_score(y, p, sample_weight=weights)) if z.any() and neg.any() else None
    ece = 0.
    for lo, hi in zip(np.arange(0, 1, .1), np.arange(.1, 1.01, .1)):
        take = (p >= lo)&(p < hi if hi < .999 else p <= 1.)
        if take.any():
            ece += weights[take].sum()*abs(np.average(p[take]-y[take], weights=weights[take]))
    return {'n': len(y), 'prevalence': float(weights@y), 'auc': auc,
            'brier': float(weights@((p-y)**2)), 'sqrt_brier_not_share_RMSE': float(np.sqrt(weights@((p-y)**2))),
            'logloss': float(log_loss(y, p, sample_weight=weights, labels=[0, 1])),
            'ECE10': float(ece), 'score_mean': float(weights@p),
            'score_ge090': float(weights@(p >= .9)), 'score_ge099': float(weights@(p >= .99)),
            'negative_score_mean': float(np.average(p[neg], weights=weights[neg])) if neg.any() else None,
            'positive_score_mean': float(np.average(p[z], weights=weights[z])) if z.any() else None,
            'fpr_at05': float(np.average(p[neg] >= .5, weights=weights[neg])) if neg.any() else None}

def fixture():
    from pipeline.training_r16_multisupport import tree_features
    rng = np.random.default_rng(SEED)
    d = np.round(rng.normal(80., 6., (8, 15, 25)), 1).astype(np.float32)
    d[0, 0] = np.nan;d[1, 3:5] = np.nan
    x, w = relative_grid_v2(d)
    restored, raw = extra_features(d)
    gain = 20.
    xb, wb = relative_grid_v2(d+gain)
    rr, aa = extra_features(d+gain)
    temporal = tree_features(d, 'classification')[:, classifier_indices('direct')]
    temporal_gain = tree_features(d+gain, 'classification')[:, classifier_indices('direct')]
    # Ch0+Ch4 reconstructs quantized input relative to the window median.
    medwhole = np.nanmedian(d, axis=(1, 2))
    reconstructed = x[:, 0]*10.+x[:, 4]*20.
    valid = np.isfinite(d)
    reconstruction_error = np.max(abs(reconstructed[valid]-(d-medwhole[:, None, None])[valid]))
    pband = np.round(rng.normal(0, 8, (8, 1, 25)), 1).astype(np.float32)
    offset_temporal = tree_features(d+pband, 'classification')[:, classifier_indices('direct')]
    return {'common_gain_db': gain,
            'temporal_common_gain_max_abs': float(abs(temporal-temporal_gain).max()),
            'restored_profile_energy_common_gain_max_abs': float(abs(restored-rr).max()),
            'raw_db_common_gain_max_abs': float(abs(raw-aa).max()),
            'temporal_static_per_band_offset_max_abs': float(abs(temporal-offset_temporal).max()),
            'Ch0_plus_Ch4_reconstruct_db_minus_window_median_max_abs': float(reconstruction_error),
            'interpretation': 'Regression 17-channel grid retains the entire quantized observed dB grid up to ONE common additive gain, not arbitrary per-band loss. Temporal classifier intentionally discards static per-band offsets. Relative representation does not imply a numerical Bayes floor.'}

def main():
    import lightgbm as lgb
    sys.addaudithook(no_query)
    OUT.mkdir(parents=True, exist_ok=True);MODELS.mkdir(parents=True, exist_ok=True)
    registered = {
        'status': 'fixed_before_fit', 'seed': SEED, 'CPU_threads': 2,
        'regimes': ['D0', 'H0'], 'modes': {m: len(names(m)) for m in MODES},
        'maximum_trees': 1000, 'early_stopping_rounds': 40, 'learning_rate': .035,
        'leaves': 31, 'minimum_leaf_rows': 50, 'max_bin': 127,
        'early_stop_role': 'earlystop', 'fit_and_Test_not_used_for_training_or_ES': True,
        'reference_pool': 465, 'train_rows_per_regime': 12000,
        'all_modes_remove_25_explicit_observation_fraction_features': True,
        'selection': 'Report all registered modes. No mode is promoted, no ensemble/Test calibration is fitted.',
        'limitations': ['Static profile/weights can still reveal O indirectly; exact presence/O pairing controls this in new data.',
            'Old fixed panels were repeatedly inspected; fresh-fit source independent of new TRAIN/ES but shared reference pool.',
            'Feature restoration in LightGBM does not establish its effect inside a pretrained neural backbone.'],
        'source_code_sha256': {n: sha(ROOT/n) for n in CODE},
        'dataset_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
            [DATA/f'{role}_{r}.npz' for role in ('train', 'earlystop') for r in ('D0', 'H0')]
            +[FRESH/f'ensemble_fit_fresh_{r}.npz' for r in ('D0', 'H0')]},
    }
    pp = OUT/'PROTOCOL.json'
    if pp.exists():
        if json.loads(pp.read_text()) != registered:raise ValueError('Registered protocol changed')
    else:jsave(pp, registered)
    jsave(OUT/'invariance_fixture.json', fixture())
    for mode in MODES:jsave(OUT/f'features_{mode}.json', names(mode))
    started = time.perf_counter();datasets = {}
    for r in ('D0', 'H0'):
        for role in ('train', 'earlystop', 'fresh_fit'):
            p = FRESH/f'ensemble_fit_fresh_{r}.npz' if role == 'fresh_fit' else DATA/f'{role}_{r}.npz'
            d = load(p);key = role+'_'+r
            restored, absolute = extra_features(d['db'])
            datasets[key] = {'data': d, 'restored': restored, 'absolute': absolute,
                             'temporal': cached_temporal(key, d['db']), 'weights': None}
    for r in ('D0', 'H0'):
        for a, b in (('train', 'earlystop'), ('train', 'fresh_fit'), ('earlystop', 'fresh_fit')):
            if set(datasets[a+'_'+r]['data']['base_window_index'])&set(datasets[b+'_'+r]['data']['base_window_index']):
                raise ValueError('New background role overlap')
    from ops.run_r16_postprocess import panel_data
    freeze = ROOT/'models/r16_pretrained_cuda_bf16_r1/both/final/GLOBAL_FREEZE.json'
    for name in PANELS:
        d, weights = panel_data(name, freeze)
        restored, absolute = extra_features(d['db'])
        datasets[name] = {'data': d, 'restored': restored, 'absolute': absolute,
                          'temporal': cached_temporal(name, d['db']), 'weights': weights}
    outputs = {};fitrows = [];allrows = [];subrows = [];calrows = []
    for r in ('D0', 'H0'):
        tr = datasets['train_'+r];va = datasets['earlystop_'+r]
        for mode in MODES:
            folder = MODELS/r/mode;folder.mkdir(parents=True, exist_ok=True)
            model_path = folder/'classifier.txt';done = folder/'COMPLETED.json'
            if done.exists():
                rec = json.loads(done.read_text())
                if rec['protocol_sha256'] != sha(pp) or rec['model_sha256'] != sha(model_path):
                    raise ValueError('Existing model identity differs')
                model = lgb.Booster(model_file=str(model_path))
            else:
                event(stage='fit', regime=r, mode=mode, features=len(names(mode)))
                x = matrix(tr, mode);v = matrix(va, mode);history = {};begin = time.perf_counter()
                learner = lgb.LGBMClassifier(n_estimators=1000, learning_rate=.035, num_leaves=31,
                    min_child_samples=50, max_bin=127, n_jobs=2, objective='binary', metric='binary_logloss',
                    random_state=SEED, verbosity=-1, colsample_bytree=.85, subsample=.85, subsample_freq=1,
                    deterministic=True, force_col_wise=True)
                learner.fit(x, tr['data']['presence'], eval_set=[(v, va['data']['presence'])],
                    callbacks=[lgb.early_stopping(40, verbose=False), lgb.record_evaluation(history)])
                model = learner.booster_;model.save_model(str(model_path))
                jsave(folder/'history.json', history)
                rec = {'regime': r, 'mode': mode, 'features': len(names(mode)),
                       'best_iteration': int(learner.best_iteration_), 'total_run_trees': len(history['valid_0']['binary_logloss']),
                       'seconds': time.perf_counter()-begin, 'protocol_sha256': sha(pp), 'model_sha256': sha(model_path)}
                jsave(done, rec);del x, v, learner;gc.collect()
            fitrows.append(rec)
            outputs[r+'__'+mode] = {}
            for name, record in datasets.items():
                # TRAIN predictions are diagnostics, not evaluation scores.
                x = matrix(record, mode);p = model.predict(x, num_threads=2);del x
                outputs[r+'__'+mode][name] = p
                allrows.append({'trained_regime': r, 'mode': mode, 'panel': name,
                    **metrics(record['data']['share'] > 0, p, record['weights'])})
            np.savez_compressed(folder/'predictions.npz', **outputs[r+'__'+mode])
            pd.DataFrame(allrows).to_csv(OUT/'metrics_progress.csv', index=False)
            event(stage='completed_fit', **rec)
            del model;gc.collect()
    for mode in MODES:
        for name, record in datasets.items():
            if name.startswith(('train_', 'earlystop_')):continue
            d = record['data'];p = .5*(outputs['D0__'+mode][name]+outputs['H0__'+mode][name]);w = record['weights']
            allrows.append({'trained_regime': 'D0_H0_same_input_50_50', 'mode': mode, 'panel': name,
                            **metrics(d['share'] > 0, p, w)})
            site = d.get('site', np.full(len(p), 'unknown')).astype(str)
            nobs = np.isfinite(d['db']).any(2).sum(1)
            group_defs = [('site', site), ('n_obs', nobs.astype(str))]
            for by, values in group_defs:
                for value in np.unique(values):
                    take = values == value
                    if take.sum() < 20:continue
                    subrows.append({'mode': mode, 'panel': name, 'group_by': by, 'group': str(value),
                        **metrics(d['share'][take] > 0, p[take], None if w is None else w[take])})
            for threshold in (.03, .1):
                take = (d['share'] == 0)|((d['share'] > 0)&(d['share'] <= threshold))
                subrows.append({'mode': mode, 'panel': name, 'group_by': 'weak_positive_vs_zero',
                    'group': f'0<y<={threshold}', **metrics(d['share'][take] > 0, p[take], None if w is None else w[take])})
            for bi in range(10):
                take = (p >= bi/10)&((p < (bi+1)/10) if bi < 9 else p <= 1)
                if take.any():calrows.append({'mode': mode, 'panel': name, 'bin': bi, 'n': int(take.sum()),
                    'mean_score': float(p[take].mean()), 'true_presence_fraction': float((d['share'][take] > 0).mean())})
    pd.DataFrame(allrows).to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(subrows).to_csv(OUT/'subgroup_metrics.csv', index=False)
    pd.DataFrame(calrows).to_csv(OUT/'calibration_bins.csv', index=False)
    pd.DataFrame(fitrows).to_csv(OUT/'training_jobs.csv', index=False)
    averaged = pd.DataFrame(allrows).query("trained_regime == 'D0_H0_same_input_50_50'")
    auc = averaged.pivot(index='panel', columns='mode', values='auc')
    brier = averaged.pivot(index='panel', columns='mode', values='brier')
    saturation = averaged.pivot(index='panel', columns='mode', values='score_ge090')
    lines = ['# R18 독립 분류기 입력 정보 복원 CPU 대조', '',
        '동일한 465개 참조 기반 TRAIN 12,000행/체계와 별도 ES 2,000행/체계에서 LightGBM 6개를 학습했다. '
        '대형 신경망 재학습·추가 K-fold·Test 읽기·Test 기반 보정은 하지 않았다. 각 입력 패널에 D0/H0 모델을 둘 다 통과시킨 뒤 확률을 50:50 평균한다.', '',
        '## 사전 고정한 입력 대조', '',
        '- temporal: 기존 분류 2,033개에서 순수 관측 비율 25개만 제외한 시간/동조 특징 2,008개.',
        '- profile_energy: temporal + Ch4 대역 상대 프로파일 25개 + Ch7 sqrt(w) 375개 = 2,408개.',
        '- absolute_db: 위 특징 + 창 내부 중앙값으로만 결측을 채운 원시 dB 375개, 대역 중앙값 25개, 창 중앙값 1개 = 2,809개.',
        '- 모든 후보의 데이터·seed·최대 트리 수·ES·leaf 크기는 동일하다. 사이트/녹음기 ID는 입력하지 않았다.', '',
        '## AUC', '', auc.to_markdown(floatfmt='.6f'), '',
        '## Brier 오차 (낮을수록 좋음; share 회귀 RMSE와 다름)', '', brier.to_markdown(floatfmt='.6f'), '',
        '## score ≥ 0.90 비율', '', saturation.to_markdown(floatfmt='.6f'), '',
        '## 해석 범위', '',
        'AUC는 양성과 음성의 순위를 측정한다. 높은 확률에 몰려도 순위가 맞으면 AUC는 높을 수 있어, 포화율만으로 실패 원인을 확정하지 않는다. '
        '원시 dB/정적 프로파일 복원 효과도 실제 대조 수치로 판단하며, 로컬 개선을 Public AUC 0.89 달성의 증명으로 해석하지 않는다.', '',
        '새 fresh-fit은 이번 TRAIN/ES와 배경 원천·시간 분리되지만 참조 풀은 공유한다. 기존 고정 패널은 반복 사용된 진단셋이다. '
        '후보 수를 늘려 최적값만 고르지 않았고 제출 정책을 바꾸지 않았다. sqrt(w)의 0값 등은 결측 정보를 간접 포함하며 '
        '새 TRAIN의 전체 O 패턴별 양성 비율 일치가 이를 통제한다. 특징 삭제만으로 결측 편법이 완전히 제거됐다고 주장하지 않는다.', '',
        '회귀의 Ch0와 Ch4는 관측된 반올림 dB를 공통 창 중앙값 차감까지 복원하므로, 17채널이 주파수 스펙트럼 전체를 잃었다는 설명은 틀리다. '
        '절대 공통 gain을 없애는 것은 y의 공통 에너지 스케일 불변성과 양립한다. 반면 분류 temporal 입력은 대역별 정적 오프셋을 제거해 '
        '잠재적 출처 스펙트럼 단서도 배제하며, 이것이 유리한지 불리한지는 위 비교로 한정해 판단한다.', '',
        '공통 E에 대해 B/A 분해는 별도 prior 없이는 원시 dB로도 유일하지 않다. 따라서 현재 실측 RMSE를 Bayes error floor로 명명할 근거는 없다.', '']
    (OUT/'R18_CLASSIFIER_INPUT_AUDIT.md').write_text('\n'.join(lines))
    artifacts = {str(p.relative_to(ROOT)): sha(p) for folder in (OUT, MODELS) for p in folder.rglob('*')
                 if p.is_file() and p.name not in ('SHA256SUMS.json', 'COMPLETED.json', 'progress.json')}
    jsave(OUT/'SHA256SUMS.json', artifacts)
    jsave(OUT/'COMPLETED.json', {'status': 'completed', 'CPU_classifier_fits': 6,
        'elapsed_seconds': time.perf_counter()-started, 'Test_read': False, 'new_neural_training_steps': 0,
        'new_kfold': False, 'protocol_sha256': sha(pp), 'metrics_sha256': sha(OUT/'metrics.csv'),
        'report_sha256': sha(OUT/'R18_CLASSIFIER_INPUT_AUDIT.md')})
    event(stage='all_completed', elapsed_seconds=time.perf_counter()-started)

if __name__ == '__main__':main()
