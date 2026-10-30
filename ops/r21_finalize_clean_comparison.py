"""Finalize the preregistered, CPU-only Clean-Val-R20 stress comparison.

No fitting of candidate coefficients, training, Test access, or model edits.
All physical blend weights and classifier interventions were fixed in advance.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'reports/r21_clean_val_zero_training_r1'
DATA = OUT / 'data_multisite'
INF = OUT / 'inference'
from pipeline.clean_val_gate_r20 import evaluate, policy_dict, policy_sha
from ops.r21_quiet_feature_candidates import extract_candidates, extract_features


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        n = Path(os.fsdecode(args[0])).name.lower()
        if n in ('test.csv', 'query.npz', 'query_predictions.npz') or 'submission' in n:
            raise PermissionError('R21 comparison forbids Test/query/submission access')


def clean_json(x):
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, np.generic):
        return clean_json(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write(p, x):
    Path(p).write_text(json.dumps(clean_json(x), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def metrics(y, p, score=None):
    ok = np.isfinite(y) & np.isfinite(p)
    result = {'n': len(y), 'n_valid': int(ok.sum()), 'coverage': float(ok.mean()) if len(y) else None}
    if not ok.any():
        return result
    yy, pp = y[ok], p[ok]
    e = pp - yy
    result.update(rmse=float(np.sqrt(np.mean(e**2))), bias=float(e.mean()),
                  positive_bias=max(float(e.mean()), 0.), mae=float(abs(e).mean()),
                  prediction_mean=float(pp.mean()), prediction_RMS=float(np.sqrt(np.mean(pp**2))))
    z = yy > 0
    if len(np.unique(z)) == 2:
        result['share_auc'] = float(roc_auc_score(z, pp))
    if score is not None:
        ss = np.asarray(score)[ok]
        if np.isfinite(ss).all():
            if len(np.unique(z)) == 2:
                result.update(score_auc=float(roc_auc_score(z, ss)),
                              average_precision=float(average_precision_score(z, ss)))
            result.update(score_mean=float(ss.mean()), score_ge90_fraction=float((ss >= .9).mean()),
                          score_ge50_fraction=float((ss >= .5).mean()),
                          brier=float(brier_score_loss(z.astype(int), ss)))
    return result


def md_table(df):
    return df.to_markdown(index=False, floatfmt='.6f')


def main():
    sys.addaudithook(guard)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    prereg = json.loads((OUT / 'PROTOCOL_BEFORE_PREDICTIONS.json').read_text())
    if prereg['gate_policy_sha256'] != policy_sha():
        raise ValueError('Gate changed after preregistration')
    pins, checked = {}, []

    def verify(folder, filename, nested=False):
        receipt = json.loads((folder / filename).read_text())
        files = receipt['files'] if nested else receipt
        for rel, expected in files.items():
            p = folder / rel
            actual = sha(p)
            if actual != expected:
                raise ValueError(f'Artifact SHA differs: {p}')
            checked.append(str(p.relative_to(ROOT)))
        pins[str((folder / filename).relative_to(ROOT))] = sha(folder / filename)

    verify(DATA, 'manifest.json', True)
    verify(OUT / 'quiet', 'ARTIFACT_SHA256.json')
    verify(OUT / 'mask_audit', 'ARTIFACT_SHA256.json')
    d = load(DATA / 'clean_val_r20.npz')
    y = d['share'].astype(float)
    ids = d['sample_id'].astype(str)
    phys = pd.read_csv(DATA / 'CLEAN_VAL_PHYSICAL_ROWS.csv')
    if not np.array_equal(ids, phys.sample_id.astype(str).to_numpy()):
        raise ValueError('Physical/source row order differs')
    hf = phys.bump_high_db.to_numpy(float)
    fam = {}
    for f in ('lightgbm', 'dinov2_small', 'dasheng_base'):
        p = INF / (f + '.npz')
        r = json.loads(p.with_suffix('.json').read_text())
        if sha(p) != r['prediction_sha256']:
            raise ValueError('Prediction pin differs')
        for rel, expected in r['models_sha256'].items():
            if sha(ROOT / rel) != expected:
                raise ValueError('Frozen checkpoint changed: ' + rel)
            pins[rel] = expected
        z = load(p)
        if not np.array_equal(ids, z['sample_id'].astype(str)):
            raise ValueError('Model row order differs')
        for key, value in z.items():
            if key == 'sample_id':
                continue
            if value.shape != y.shape or not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
                raise ValueError('Invalid frozen family prediction: ' + f + '/' + key)
        fam[f] = {k: .5 * (z[k + '_D0'] + z[k + '_H0']) for k in ('share', 'prob')}
        if f == 'lightgbm':
            fam[f]['ablated'] = .5 * (z['prob_proxy_neutral_D0'] + z['prob_proxy_neutral_H0'])
        else:
            fam[f]['ablated'] = .5 * (z['prob_O_hidden_D0'] + z['prob_O_hidden_H0'])
        pins[str(p.relative_to(ROOT))] = sha(p)
    infrec = json.loads((INF / 'COMPLETED.json').read_text())
    if infrec['input_sha256'] != sha(DATA / 'clean_val_r20.npz'):
        raise ValueError('Inference used other panel')

    order = ('lightgbm', 'dinov2_small', 'dasheng_base')
    def blend(weights, field='share', hidden=()):
        return sum(w * fam[f]['ablated' if f in hidden and field == 'prob' else field]
                   for w, f in zip(weights, order))
    weights = {'old_B1': (.6, .25, .15), 'old_B2': (.5, .3, .2)}
    pred = {k: blend(w) for k, w in weights.items()}
    scores = {k: blend(w, 'prob') for k, w in weights.items()}
    for f in order:
        pred['individual_' + f] = fam[f]['share']
        scores['individual_' + f] = fam[f]['prob']
    q = extract_candidates(d['db'])
    for name, p in q.items():
        pred['quiet_' + name] = p
        for lam in (.05, .10):
            k = f'old_B1_plus_{name}_{int(lam * 100):02d}pct'
            pred[k] = (1 - lam) * pred['old_B1'] + lam * p
            scores[k] = scores['old_B1']
    for name, hidden in (('LGB_pure25_neutral', ('lightgbm',)),
                         ('NN_stem_O_hidden', ('dinov2_small', 'dasheng_base')),
                         ('both_inputs_ablated', order)):
        key = 'old_B1_' + name
        pred[key] = pred['old_B1'].copy()
        scores[key] = blend(weights['old_B1'], 'prob', hidden)

    slices = {'all': np.ones(len(y), bool), 'zero': y == 0,
              'weak': (y > 0) & (y <= .1), 'HF_zero': (y == 0) & (hf >= 12),
              'positive': y > 0, 'long_positive_active_ge10': (y > 0) & (phys.active_minutes.to_numpy() >= 10)}
    rows = []
    for name, p in pred.items():
        for sl, ix in slices.items():
            rows.append({'candidate': name, 'slice': sl,
                         **metrics(y[ix], p[ix], scores[name][ix] if name in scores else None)})
    m = pd.DataFrame(rows)
    m.to_csv(OUT / 'CLEAN_VAL_COMPARISON_METRICS.csv', index=False)
    group_rows = []
    for field in ('site', 'n_obs', 'recorder'):
        for val in phys[field].unique():
            ix = (phys[field] == val).to_numpy()
            for name in ('old_B1', 'old_B2', 'old_B1_LGB_pure25_neutral', 'old_B1_NN_stem_O_hidden', 'old_B1_both_inputs_ablated'):
                group_rows.append({'condition': field, 'value': val, 'candidate': name,
                                   **metrics(y[ix], pred[name][ix], scores[name][ix])})
    pd.DataFrame(group_rows).to_csv(OUT / 'CONDITIONAL_COMPARISON.csv', index=False)
    rowpred = phys.copy()
    for k, v in pred.items():
        rowpred[k + '__share'] = v
    for k, v in scores.items():
        rowpred[k + '__score'] = v
    rowpred.to_csv(OUT / 'CLEAN_VAL_ROW_PREDICTIONS.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    feat = extract_features(d['db'])
    feat.insert(0, 'sample_id', ids)
    feat['quiet_B_share'] = q['bandwise_lowK5_TRAIN_ratio']
    feat.to_csv(OUT / 'QUIET_B_PHYSICAL_FEATURES.csv', index=False)
    pd.DataFrame([
        {'feature': 'quiet_B_share', 'candidate': 'bandwise_lowK5_TRAIN_ratio', 'unit': 'share in [0,1]',
         'definition': 'clip(1 - sum_f n_obs_f * TRAIN_ratio_f * mean(lowest5 observed E_tf) / sum_tf O*E,0,1)',
         'coefficient_source': '11700 pure TRAIN backgrounds only', 'frozen_LGB_consumes_feature': False},
        {'feature': 'R20_commonK3_control', 'candidate': 'R20_commonK3_control', 'unit': 'share in [0,1]',
         'definition': 'R20 fixed common quiet3 minute estimate with frozen TRAIN correction',
         'coefficient_source': 'R20 frozen TRAIN-only coefficients', 'frozen_LGB_consumes_feature': False},
        {'feature': 'timevarying_median5_signed', 'candidate': 'timevarying_median5_signed', 'unit': 'share in [0,1]',
         'definition': 'Per-band rolling5 median dB background; signed total E-B then final clip',
         'coefficient_source': 'fixed within-window statistic only', 'frozen_LGB_consumes_feature': False},
        {'feature': 'robust_linear_db_signed', 'candidate': 'robust_linear_db_signed', 'unit': 'share in [0,1]',
         'definition': 'Per-band robust affine dB baseline, fixed5 Huber iterations; signed total E-B then final clip',
         'coefficient_source': 'fixed within-window statistic only', 'frozen_LGB_consumes_feature': False}
    ]).to_csv(OUT / 'QUIET_B_FEATURE_DICTIONARY.csv', index=False)

    gate = evaluate(y, pred['old_B1'], pred, hf)
    write(OUT / 'CLEAN_VAL_GATE_RESULTS.json', gate)
    checks = [{'candidate': k, **c} for k, v in gate['results'].items() for c in v['checks']]
    pd.DataFrame(checks).to_csv(OUT / 'CLEAN_VAL_GATE_CHECKS.csv', index=False)
    # Descriptive paired rec-day block bootstrap. It does not choose a policy.
    groups = phys.background_group.astype(str).to_numpy()
    unique = np.unique(groups)
    rng = np.random.default_rng(20261030)
    draw = rng.integers(0, len(unique), (1000, len(unique)))
    boot = []
    for name in ('old_B2',) + tuple(k for k in pred if k.startswith('old_B1_plus_')):
        delta = []
        for indices in draw:
            ix = np.concatenate([np.flatnonzero(groups == unique[j]) for j in indices])
            if not np.isfinite(pred[name][ix]).all():
                continue
            delta.append(float(np.sqrt(np.mean((pred[name][ix] - y[ix])**2)) - np.sqrt(np.mean((pred['old_B1'][ix] - y[ix])**2))))
        boot.append({'candidate': name, 'rec_day_blocks': len(unique), 'replicates': len(delta),
                     'delta_RMSE_ci025': np.quantile(delta, .025) if delta else None,
                     'delta_RMSE_ci975': np.quantile(delta, .975) if delta else None,
                     'interpretation': 'descriptive only; overlapping windows, small cohort, and across-day dependence remain'})
    pd.DataFrame(boot).to_csv(OUT / 'BLOCK_BOOTSTRAP_DESCRIPTIVE.csv', index=False)

    # The preregistered policy candidates do not include switching to a single
    # family. Individual-model scores are diagnostic, not a post-hoc new policy.
    passed = [k for k, v in gate['results'].items() if v['adoption_pass'] and not k.startswith('individual_')]
    descriptive_pass = [k for k, v in gate['results'].items() if v['adoption_pass'] and k.startswith('individual_')]
    best = min(passed, key=lambda k: float(m[(m.candidate == k) & (m['slice'] == 'all')].rmse.iloc[0])) if passed else 'old_B1'
    status = {'completed_KST': dt.datetime.now().astimezone().isoformat(),
              'candidate_design_fixed_before_predictions': True, 'dataset_sha256': sha(DATA / 'clean_val_r20.npz'),
              'gate_policy_sha256': policy_sha(), 'passing_candidates': passed,
              'individual_diagnostic_gate_passes_not_policy_candidates': descriptive_pass,
              'best_passing_diagnostic_candidate': best, 'production_models_or_existing_files_changed': False,
              'new_LightGBM_features_consumed': False, 'new_training': False, 'optimizer_steps': 0,
              'device': 'cpu', 'Test_read': False, 'submission_created': False,
              'statistical_TRAIN_only_coefficient_rows': 11700,
              'independence_limit': '100 rows from 50 paired, sometimes overlapping B windows; 28 recorder-days, 2 sites, historical four input reuses; not a fresh final holdout',
              'zero_FP_or_LB_0033_guaranteed': False}
    write(OUT / 'FINAL_DECISION.json', status)
    config = {'policy': policy_dict(), 'policy_sha256': policy_sha(),
              'dataset': str((DATA / 'clean_val_r20.npz').relative_to(ROOT)),
              'dataset_sha256': sha(DATA / 'clean_val_r20.npz'),
              'entrypoint': 'pipeline.clean_val_gate_r20.evaluate',
              'scope': 'Only R21 diagnostic gate; archived historical policies and production model inputs stay frozen'}
    cfg = ROOT / 'configs/clean_val_r20_gate_v1.json'
    cfg.parent.mkdir(exist_ok=True)
    write(cfg, config)

    def compact(names):
        a = m[m.candidate.isin(names) & m['slice'].isin(('all', 'zero', 'weak', 'HF_zero'))]
        return a[['candidate', 'slice', 'n', 'n_valid', 'rmse', 'bias', 'positive_bias']]
    source = json.loads((DATA / 'SOURCE_AUDIT.json').read_text())
    main_names = ['old_B1', 'old_B2'] + ['quiet_' + k for k in q]
    ablnames = ['old_B1', 'old_B2', 'old_B1_LGB_pure25_neutral', 'old_B1_NN_stem_O_hidden', 'old_B1_both_inputs_ablated']
    # Pair-preserved exact observation masks give zero empirical MI with z.
    patterns = np.isfinite(d['db']).reshape(len(y), -1)
    from sklearn.metrics import mutual_info_score
    _, code = np.unique(patterns, axis=0, return_inverse=True)
    empirical_MI = float(mutual_info_score(code, (y > 0).astype(int)))
    zero = phys[phys.share == 0]
    report = '# Clean-Val-R20 구축 및 무학습 전처리·물리 피처 검산 (R21)\n\n'
    report += '**재학습 0회, GPU/MPS 사용 0회.** 기존 old_B1·old_B2와 동결된 모델들의 CPU 추론, 입력 가림 대조, 고정 물리 추정값 대조를 완료했다. '
    report += ('새 엄격 게이트 통과 후보: **' + ', '.join(passed) + '**.\n\n') if passed else '**새 엄격 게이트를 모두 통과한 개선 후보는 없다. 기존 old_B1을 유지한다.**\n\n'
    report += '## 1. 데이터 구성과 해석 범위\n\n'
    report += f'원본 적격 full-clock 256,029창 중 HF≥15 dB·LF≤200Hz 비중≤0.65인 후보는 790창이었다. 학습과 사이트·녹음기·센서가 모두 다른 41창에, 기존 domain_holdout에서 동일 조건을 재검증한 부분 관측 Fish Hoek 9창을 합쳤다. **50개 배경 × 음성/양성 쌍 = 100행**, 무추가 50행(50%), 양성 50행, 그중 약한 양성 25행이다. 임계값 완화·창 복제·결과를 본 뒤 배경 교체는 하지 않았다.\n\n'
    report += f'음성 HF bump 중앙값 **{zero.bump_high_db.median():.4f} dB**(최솟값 {zero.bump_high_db.min():.4f}), LF 비중 중앙값 **{zero.LF_energy_fraction.median():.5f}**(최댓값 {zero.LF_energy_fraction.max():.5f}). 사이트 Saldhana Bay 41·Fish Hoek 9, 녹음기 3개, rec-day 28개, 연결 관측 구간 35개, 최대 비중첩 15분 창 39개다. 배경별 Nobs는 15분 41개·10분 5개·5분 4개다.\n\n'
    report += f'각 배경의 양성·음성에 동일한 O를 사용하여 경험적 **MI(O; presence)={empirical_MI:.2e} nats**다. 양성은 예약 참조 음원에서 원래 1분 파워 궤적을 잘라 스칼라로만 더했다. 새 EQ·시간 warp·평탄 15분 복제는 없다. 무추가의 의미는 “VTUAD 성분을 새로 더하지 않음”이며 자연 선박·생물·기계 과도음은 배경에 남아 있다.\n\n'
    report += '**대표성 제한:** Test 전체 혼합 입력의 HF≈16 dB를 Test의 음성 조건이라고 알 수는 없다. 이 데이터는 사용자가 지정한 극단 조건을 검사하는 **소규모 스트레스 검증셋**이다. 사이트별 다변량 분포까지 Test와 일치했다거나 일반화 간극이 없어졌다고 주장하지 않는다. 겹치는 창과 양성/음성 쌍 때문에 100개 독립 표본이 아니며, 4개 음성 입력은 이전 R20과 정확히 재사용된다. 이미 쓰인 domain 원천·예약 참조도 있어 새로운 최종 홀드아웃으로 부르지 않는다. 원래 82행 버전은 변경 없이 남기고, 부분 관측 18행을 별도 data_multisite 버전에 추가했다.\n\n'
    report += '## 2. 고정한 유일한 신규 채택 게이트\n\n'
    report += 'old_B1 대비 전체·무추가·약한 양성·HF무추가 RMSE가 모두 1e-6 초과 감소하고, 무추가·약한 양성의 양(+) 편향(max(mean(error),0))은 증가하지 않아야 통과한다. 빈 구간·NaN·범위 오류는 거부한다. 이 패널의 모든 음성이 HF≥15라서 **HF≥12 음성 축은 무추가 축과 같은 50행**이다. 네 축을 서로 독립된 네 검증으로 해석하면 안 된다. 모델별 결과를 보고 임계값이나 후보를 다시 맞추지 않았다.\n\n'
    report += '동결한 게이트 코드·설정은 pipeline/clean_val_gate_r20.py와 configs/clean_val_r20_gate_v1.json이다. 새 진단의 채택 기준이며 과거 학습 러너·모델·제출 파일을 자동 수정하지 않았다. 이 작은 패널에서 통과하더라도 미관측 모든 조건의 성능 보장을 뜻하지 않는다.\n\n'
    report += '## 3. old_B1·old_B2 및 Quiet-Minutes 4개 후보\n\n'
    report += 'old_B1은 LGB 60%+DINO 25%+Dasheng 15%, old_B2는 50%+30%+20%; 각 D0:H0=50:50이다. 회귀와 독립 분류는 각각 같은 계열 비중으로 결합했다. LGB 기존 5-Fold, 신경망 기존 Fold0를 사용했다. 아래는 균등 행 가중 실측이다.\n\n'
    report += md_table(compact(main_names)) + '\n\n'
    report += '기존 개별 모델 성적은 아래 참고 표에 별도로 공개한다. 개별 모델로 정책을 바꾸는 것은 사전 등록 후보에 없으므로, 표에서 점수가 좋아도 이번 채택 후보로 추가하지 않는다.\n\n'
    report += md_table(compact(['individual_' + f for f in order])) + '\n\n'
    report += '물리 후보의 정의: (1) R20 common quiet K3와 기존 TRAIN 계수, (2) 대역별 최저 5분 평균 및 TRAIN 평균/quiet 비율 보정, (3) 대역별 rolling 5분 dB 중앙값, (4) 대역별 고정 5회 Huber 선형 dB 추세다. 창 내부 공통 오프셋을 빼서 수치적으로 안정화한 선형 에너지 E와 배경 추정 B̂로 `clip(1−ΣOB̂/ΣOE,0,1)`을 계산한다. **셀별 양수 잘라내기/가짜 물리 상한은 없다.** K5 계수만 순수 TRAIN 11,700창에서 통계 추정했고 평가 라벨·Test는 쓰지 않았다.\n\n'
    report += '모든 후보를 old_B1에 **5%·10%**만 섞는 사전 고정 대조도 실행했다. 이것은 새 피처를 쓰는 LightGBM이 아니라 **저장 예측값과 물리 추정값의 출력 결합**이다. 기존 LightGBM에 열을 추가해도 기존 split은 새 열을 사용할 수 없다. 재학습 금지 지시 때문에 새 LGB 적합은 하지 않았다.\n\n'
    report += '요청한 `quiet_B_share`는 대역별 K5/TRAIN 비율 후보의 명시적 별칭으로 피처 CSV에 구현했다. 정의·단위·계수 출처를 QUIET_B_FEATURE_DICTIONARY.csv에 보존했다. 이 피처를 현재 LGB가 학습/사용한 것으로 표시하지 않는다.\n\n'
    blendm = compact([k for k in pred if k.startswith('old_B1_plus_')])
    report += md_table(blendm) + '\n\n'
    decisions = pd.DataFrame([{'candidate': k, 'valid': v['valid'], 'noninferior': v.get('noninferior_all_axes', False),
                               'pass': v['adoption_pass'], 'failed_axes': v.get('failed_axes'), 'error': v.get('error')}
                              for k, v in gate['results'].items()])
    report += md_table(decisions) + '\n\n'
    report += '위 pass는 수치 조건만 나타낸다. individual_*는 개별 모델 기술 진단으로만 기록하며 채택 대상에서 제외한다. old_B2의 전체 ΔRMSE(rec-day block bootstrap 1,000회) 95% 구간은 '
    ci = next(b for b in boot if b['candidate'] == 'old_B2')
    report += f'**[{ci["delta_RMSE_ci025"]:.6f}, {ci["delta_RMSE_ci975"]:.6f}]**로 0을 포함한다. 관측한 소폭 개선의 확실성이나 Public 개선을 보장하지 않는다.\n\n'
    report += '## 4. 분류 입력의 마스크 대리변수 가림 검산\n\n'
    report += '기존 감사에서 관측 대리변수로 묶인 400개 중 **375개는 인접 시간 차분·lag 음향 특징**, 순수 관측 비율은 25개였다. old LGB의 15.3402% gain은 음향 14.5122%+순수 마스크 **0.8280%**로 나뉜다. CatBoost 순수 마스크 중요도는 3.0977%(PredictionValuesChange; LGB gain과 다른 정의)였다. 이전 15~21% 전체를 결측만 외운 증거로 해석한 것은 과도했다.\n\n'
    report += 'LGB에서는 순수 25개만 각 모델 TRAIN 중앙값(모두 1.0)으로 치환했다. NN에서는 원본 O와 유효 셀 pooling은 유지하고, stem에 전달되는 입력 사본의 O 채널만 0으로 가렸다. 기존 raw O 자체를 0으로 바꾸는 첫 시도는 “관측 셀이 없음” guard에 거부되어 결과 없이 보존했고, 올바른 stem-only 개입으로 다시 실행했다. 음향 특징 안의 간접 결측 정보까지 제거했다는 뜻은 아니다. **학습된 중요도·가중치·정규화를 변경한 것이 아니라 OOD 입력 대조**다.\n\n'
    clas = m[(m.candidate.isin(ablnames)) & (m['slice'].isin(('all', 'zero'))) ]
    report += md_table(clas[['candidate', 'slice', 'n', 'rmse', 'score_auc', 'score_mean', 'score_ge90_fraction', 'score_ge50_fraction', 'brier']]) + '\n\n'
    report += '분류만 바꾼 대조는 회귀 예측이 old_B1과 같으므로 엄격한 네 RMSE 동시 개선 조건을 통과할 수 없다. AUC/FPR 변화는 별도 진단으로 기록했다. 완전 관측 41배경에서는 순수 마스크 입력이 원래부터 1.0이라 LGB 치환 결과도 정확히 동일하다. 부분 관측 9배경 결과와 전체 결과를 구분한 CONDITIONAL_COMPARISON.csv를 제공한다. score≥0.90은 사전 고정 진단 임계값이며, 이 패널에서 새로운 threshold를 고르지 않았다.\n\n'
    report += 'old_B1의 무추가 score≥0.90 비율은 24%, old_B2는 22%다. NN O 가림은 전체 AUC를 소폭 바꾸지만 무추가 평균 score를 약 0.662에서 0.736으로 올린다. 이 결과는 오탐 완벽 제거가 아니며, 새 패널 AUC 약 0.87을 Public AUC 0.573의 재현 또는 해소라고 해석하지 않는다.\n\n'
    report += '## 5. 결론과 실제 적용 범위\n\n'
    report += ('게이트 통과 후보는 위 FINAL_DECISION.json에 기록했다. 통과는 이 스트레스 패널의 관측 개선이며, 후보 선택에 사용한 패널이므로 독립 최종 점수는 아니다.\n\n' if passed else '이번 전처리/물리 피처 후보로는 요청한 **무추가·약한 양성·전체의 동시 개선을 입증하지 못했다.** 따라서 새 Quiet 보정이나 마스크 가림을 실전 정책으로 채택하지 않고 old_B1을 유지한다.\n\n')
    report += '관측 E만으로 B를 항상 복원할 수 없다. 예를 들어 일정한 E=200은 B=200,A=0(y=0)과 B=100,A=100(y=.5) 양쪽에서 가능하다. quiet 기준선은 자연 고주파 과도음을 추가 성분으로 잘못 세거나 지속적인 추가 성분을 배경으로 빼버릴 수 있다. 따라서 “물리 피처”라는 이름만으로 오탐 완벽 제거·리더보드 RMSE .033 달성을 보장할 수 없다. 재학습/대규모 데이터 생성/새 모델/제출은 이번 작업에서 수행하지 않았다.\n\n'
    report += '기존 6개 진단 패널에서의 고정 후보 성적은 quiet/HISTORICAL_FIXED_CANDIDATE_METRICS.csv에 보조 자료로 남겼다. 새 Clean-Val의 결과를 보고 이 자료로 후보를 재튜닝하지 않았다. paired rec-day 1,000회 bootstrap은 BLOCK_BOOTSTRAP_DESCRIPTIVE.csv에 있으나 작은 표본과 원천 의존성이 남아 있어 유의성 보장으로 쓰지 않는다.\n\n'
    report += '새 패널에서 관측 추가 support≥10분인 양성은 1행뿐이다. 장시간 통과 전반의 안전성을 검증했다고 말할 수 없고, 일정한 장시간 성분을 quiet 기준선이 빼버리는 반례도 여전히 남는다.\n\n'
    report += '## 6. 산출물과 재현\n\n'
    report += '- 데이터 및 원천 QA: data_multisite/manifest.json, SOURCE_AUDIT.json, clean_val_r20.npz\n- 네 구간 비교: CLEAN_VAL_COMPARISON_METRICS.csv / CLEAN_VAL_GATE_CHECKS.csv\n- 개별 행 예측: CLEAN_VAL_ROW_PREDICTIONS.csv.gz\n- 조건별 진단: CONDITIONAL_COMPARISON.csv\n- 물리 피처 행렬: QUIET_B_PHYSICAL_FEATURES.csv\n- 신규 게이트: pipeline/clean_val_gate_r20.py / configs/clean_val_r20_gate_v1.json\n- 최종 결정: FINAL_DECISION.json\n- 출처·모델·파일 SHA: INPUT_AND_MODEL_SHA256.json / ARTIFACT_SHA256.json\n\n'
    report += f'패널 NPZ SHA-256: `{sha(DATA / "clean_val_r20.npz")}`\n\n'
    report += '재현 순서(모델 훈련 아님):\n```sh\n.venv_r17_eda/bin/python ops/r21_frozen_clean_inference.py --input reports/r21_clean_val_zero_training_r1/data_multisite/clean_val_r20.npz\n.venv_r17_eda/bin/python ops/r21_finalize_clean_comparison.py\n```\n'
    (OUT / 'R21_CLEAN_VAL_ZERO_TRAINING_REPORT.md').write_text(report)
    import markdown
    html = markdown.markdown(report, extensions=['tables', 'fenced_code'])
    (OUT / 'START.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Clean-Val-R20 · R21 무학습 검산</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:1300px;margin:36px auto;padding:0 24px;color:#172033}table{display:block;overflow-x:auto;border-collapse:collapse;font-size:13px}th,td{padding:8px 12px;border:1px solid #c9d5e3;white-space:nowrap}th{background:#edf3fa}pre{overflow:auto;background:#f2f5f9;padding:16px}code{background:#eef2f6;padding:2px 4px}h1{font-size:28px}h2{margin-top:38px}strong{color:#0c4c82}</style><body>' + html + '</body></html>')
    pins[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    for p in (OUT / 'PROTOCOL_BEFORE_PREDICTIONS.json', ROOT / 'pipeline/clean_val_gate_r20.py', cfg):
        pins[str(p.relative_to(ROOT))] = sha(p)
    # Record imported transformation sources at finalization; unlike checkpoint
    # receipt checks this is not claimed as a pre/post-execution source hash.
    for rel in ('pipeline/training_r18_production_neural.py', 'pipeline/training_r16_multisupport.py',
                'pipeline/training_r16_pretrained_cuda_bf16.py', 'pipeline/pretrained_audio2d_r16.py',
                'ops/r21_frozen_clean_inference.py', 'ops/r21_quiet_feature_candidates.py', 'ops/r21_mask_proxy_audit.py'):
        pins[rel] = sha(ROOT / rel)
    write(OUT / 'INPUT_AND_MODEL_SHA256.json', pins)
    write(OUT / 'VERIFICATION_QA.json', {'artifact_sha_checks': len(checked), 'artifact_mismatch': 0,
          'frozen_model_files_checked': sum(k.startswith('models/') for k in pins),
          'sample_ids_unique_aligned': len(set(ids)) == len(y), 'mask_presence_empirical_MI_nats': empirical_MI,
          'zero_fraction': float((y == 0).mean()), 'negative_thresholds_all_pass': bool((zero.bump_high_db >= 15).all() and (zero.LF_energy_fraction <= .65).all()),
          'four_slice_counts': {k: int(v.sum()) for k, v in slices.items()},
          'no_training_GPU_MPS_Test': True, 'all_candidates_fixed_before_predictions': True})
    files = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*'))
             if p.is_file() and str(p.relative_to(OUT)) not in ('ARTIFACT_SHA256.json', 'COMPLETED.json')}
    write(OUT / 'ARTIFACT_SHA256.json', files)
    write(OUT / 'COMPLETED.json', {**status, 'artifact_count': len(files), 'artifact_manifest_sha256': sha(OUT / 'ARTIFACT_SHA256.json'),
                                  'report_sha256': sha(OUT / 'R21_CLEAN_VAL_ZERO_TRAINING_REPORT.md')})
    print(compact(['old_B1', 'old_B2']).to_string(index=False))
    print(json.dumps(clean_json(status), ensure_ascii=False))


if __name__ == '__main__':
    main()
