"""R15 descriptive input/output audit: stored predictions, no label inference.

The historical kNN probe is reused as an estimated domain type, never actual
Test site ground truth. Raw Test is read only for a predeclared, unlearned
per-band temporal scalar. No model, feature-state fitting, calibration or
submission creation is performed. Every output is a new diagnostic artifact.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = 'reports/r15_descriptive_audit_r1'
TEST_SHA = '3ef819192c05c4308b66564e6307e6f4f26fe806afe96ec93e2d15f25a3474d4'
CACHE = 'eda_deep/tables/test_window_features.csv'
WIDE = 'reports/r15_member_audit_r1/aligned_saved_predictions_1500.csv'
FEATURES = ('bump_high_db', 'bump_low_db', 'corr_low_high', 'naive_share_q20',
            'tilt_db', 'level_db', 'std_broadband_db', 'Nobs', 'S_phys_safe_mid_db',
            'centroid_hz', 'mean_abs_step_db', 'lag1_acf_broadband',
            'd7943_5012', 'probe_conf', 'probe_nn_rms_db')
MAIN = ('R5_CNN', 'R9_tree5', 'R9_all5', 'R14_step1', 'R14_exact_Fold0',
        'R14_tree5', 'R14.nn_bg_cnn', 'R14.nn_fdy', 'R14.nn_itransformer',
        'R14.tree_baseline', 'R14.tree_highshare')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def write(p, d):
    Path(p).write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def scalar_phys(db):
    """Fixed safe mid/high mean(max_t dB - median_t dB), no fitted weights."""
    x = np.asarray(db, np.float64)
    region = x[:, :, 10:23]
    return np.nanmean(np.nanmax(region, axis=1)-np.nanmedian(region, axis=1), axis=1)

def corr(a, b, rank=False):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b); a = a[ok]; b = b[ok]
    if rank:
        a = pd.Series(a).rank(method='average').to_numpy()
        b = pd.Series(b).rank(method='average').to_numpy()
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0: return None, int(len(a))
    return float(np.corrcoef(a, b)[0, 1]), int(len(a))

def distribution(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a): return {'n': 0}
    q = np.quantile(a, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return {'n': len(a), 'mean': float(a.mean()), 'median': float(q[4]),
            'std_ddof0': float(a.std()), 'rms': float(np.sqrt(np.mean(a*a))),
            **dict(zip(('min', 'p01', 'p05', 'p25', 'p50', 'p75', 'p95', 'p99', 'max'), map(float, q))),
            'fraction_ge090': float((a >= .90).mean()), 'fraction_ge099': float((a >= .99).mean())}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULT)
    args = parser.parse_args(); out = (ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'reports') or out.exists():
        raise FileExistsError('Fresh project reports subdirectory required')
    paths = [CACHE, WIDE, 'reports/r15_member_audit_r1/RESULTS.json',
             'reports/r15_member_audit_r1/distribution_summary.csv',
             'eda_deep/deep_eda.py', 'pipeline/inference_r5.py',
             'data/raw/Test.csv', 'models/r14_production_outputs_r2/hybrid_fast3/policy.json',
             'models/r14_production_outputs_r2/hybrid_fast3/COMPLETED.json',
             'submissions/r14_main_r2/submission_r14_step1.csv',
             'ops/r15_input_output_audit.py']
    pins = {p: sha(ROOT/p) for p in paths}
    if pins['data/raw/Test.csv'] != TEST_SHA: raise ValueError('Original public query bytes differ')
    # The plan is persisted before query numeric cells or predictions are opened.
    out.mkdir(parents=True)
    plan = {'version': 'r15_descriptive_input_output_v1', 'created_unix': time.time(),
            'source_file_sha256': pins, 'hidden_labels_available': False,
            'fit_inference_calibration_submission': False,
            'features': list(FEATURES), 'prediction_source': WIDE,
            'S_phys_formula': 'mean_f(max_observed_t db_tf - median_observed_t db_tf), f=10..22 inclusive, Hz316.2..5011.9; unitsdB',
            'S_phys_policy': 'Unlearned descriptive scalar, not a ship probability or share label; no parameters inferred from Test/LB',
            'group_rules': {'low_temporal': 'bump_high_db<=its q25 AND std_broadband_db<=its q25',
                            'high_static_proxy': 'tilt_db<=its q25 (LOW-minus-HIGH tilt); proxy only, not PCA/LF-anchor maps',
                            'joint': 'intersection of the preceding groups',
                            'historical_low_bump': 'bump_high_db<=5dB'},
            'threshold_scope': 'Input-only descriptive quartile summaries; never used for labels, fitting or changing model policy',
            'top50': 'Largest absolute R14_step1.share-R9_all5.share; retain signed delta. Positive-only top50 separately.',
            'site_semantics': 'Historical kNN domain-probe prediction; not actual Test site GT',
            'outputs': ['joined1500.csv', 'model_distribution.csv', 'site_model_stats.csv',
                        'feature_prediction_correlations.csv', 'model_correlations_pearson.csv',
                        'model_correlations_spearman.csv', 'lowtemporal_highstatic_groups.csv',
                        'top50_absdelta.csv', 'top50_positive_delta.csv', 'RESULTS.json', 'README.md']}
    write(out/'plan.json', plan); started = time.time()
    source = json.loads((ROOT/'reports/r15_member_audit_r1/RESULTS.json').read_text())
    if source['status'] != 'PASS_saved_output_descriptive_audit': raise ValueError('Member audit must have passed')
    for p, h in source['source_sha256'].items():
        if sha(ROOT/p) != h: raise ValueError('Stored prediction provenance changed:'+p)
    f = pd.read_csv(ROOT/CACHE, dtype={'ID': str})
    pred = pd.read_csv(ROOT/WIDE, dtype={'ID': str})
    if len(f) != 1500 or len(pred) != 1500 or f.ID.nunique() != 1500 or pred.ID.nunique() != 1500 or set(f.ID) != set(pred.ID):
        raise ValueError('Exact1500 one-to-one feature/prediction ID join required')
    if not pred.drop(columns='ID').apply(np.isfinite).all().all(): raise ValueError('Finite stored predictions required')
    # Exact public schema; do not call any model/query-inference entry point.
    tags = ('31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5','251.2','316.2','398.1','501.2','631','794.3','1000','1258.9','1584.9','1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3')
    columns = [f'm{m:02d}_TOL_{hz}' for m in range(1, 16) for hz in tags]
    with (ROOT/'data/raw/Test.csv').open(newline='', encoding='utf-8-sig') as stream:
        rd = csv.reader(stream)
        if next(rd) != ['ID', *columns]: raise ValueError('Canonical375 columns differ')
        raw = list(rd)
    ids = [v[0] for v in raw]
    db = np.asarray([[float(v) if v.strip() else np.nan for v in row[1:]] for row in raw]).reshape(-1,15,25)
    phys = pd.DataFrame({'ID': ids, 'S_phys_safe_mid_db': scalar_phys(db)})
    if ids != pred.ID.tolist(): raise ValueError('Saved query order differs from original query ID order')
    fixture = np.arange(3*15*25, dtype=float).reshape(3,15,25)/7
    fixture[0,0] = np.nan; fixture[1,14] = np.nan
    offsets = np.linspace(-30, 45, 25)[None,None,:]
    invariance = float(np.max(abs(scalar_phys(fixture)-scalar_phys(fixture+offsets))))
    if invariance > 1e-12: raise ValueError('Static band-offset invariance failed')
    joined = pred.merge(f, on='ID', how='left', validate='one_to_one', sort=False).merge(phys, on='ID', validate='one_to_one', sort=False)
    joined['Nobs'] = 15-joined['n_missing']
    joined['probe_site_estimate_not_GT'] = joined['probe_site']
    expected_obs = np.isfinite(db).all(2).sum(1)
    if not np.array_equal(expected_obs, joined.Nobs.to_numpy()): raise ValueError('Cached Nobs does not match original query mask')
    del db
    joined['delta_R14minusR9_share'] = joined['R14_step1.share']-joined['R9_all5.share']
    joined['absdelta_R14minusR9_share'] = joined['delta_R14minusR9_share'].abs()
    qh = float(joined.bump_high_db.quantile(.25)); qs = float(joined.std_broadband_db.quantile(.25)); qt = float(joined.tilt_db.quantile(.25))
    groups = {'all': np.ones(len(joined), bool), 'bump_high_le5db': joined.bump_high_db <= 5,
              'low_temporal_q25': (joined.bump_high_db <= qh) & (joined.std_broadband_db <= qs),
              'high_static_proxy_lowtilt_q25': joined.tilt_db <= qt}
    groups['low_temporal_AND_high_static_proxy'] = groups['low_temporal_q25'] & groups['high_static_proxy_lowtilt_q25']
    groups['low_temporal_NOT_high_static_proxy'] = groups['low_temporal_q25'] & ~groups['high_static_proxy_lowtilt_q25']
    groups['not_low_temporal_AND_high_static_proxy'] = ~groups['low_temporal_q25'] & groups['high_static_proxy_lowtilt_q25']
    for name, mask in groups.items(): joined['group_'+name] = np.asarray(mask, bool)
    predcols = list(pred.columns[1:]); dist = []; sites = []; groupstats = []; relations = []
    for col in predcols:
        target = col.rsplit('.', 1)[-1]; model = col.rsplit('.', 1)[0]
        dist.append({'series': col, 'model': model, 'target_kind': target, **distribution(joined[col])})
        for site, s in joined.groupby('probe_site', sort=True):
            sites.append({'probe_site_estimate_not_GT': site, 'series': col, **distribution(s[col]),
                          'probe_conf_mean': float(s.probe_conf.mean()), 'bump_high_median': float(s.bump_high_db.median()),
                          'delta_R14minusR9_share_mean': float(s.delta_R14minusR9_share.mean())})
        for name, mask in groups.items():
            s = joined.loc[mask]
            groupstats.append({'group': name, 'series': col, **distribution(s[col]),
                               'delta_R14minusR9_mean': float(s.delta_R14minusR9_share.mean()) if len(s) else None})
        for feature in FEATURES:
            pe, n = corr(joined[feature], joined[col]); sp, _ = corr(joined[feature], joined[col], True)
            relations.append({'feature': feature, 'series': col, 'model': model, 'target_kind': target,
                              'n': n, 'Pearson': pe, 'Spearman': sp})
    joined.to_csv(out/'joined1500.csv', index=False)
    pd.DataFrame(dist).to_csv(out/'model_distribution.csv', index=False)
    pd.DataFrame(sites).to_csv(out/'site_model_stats.csv', index=False)
    pd.DataFrame(groupstats).to_csv(out/'lowtemporal_highstatic_groups.csv', index=False)
    pd.DataFrame(relations).to_csv(out/'feature_prediction_correlations.csv', index=False)
    for method in ('pearson', 'spearman'):
        joined[predcols].corr(method=method).to_csv(out/f'model_correlations_{method}.csv', index_label='series')
    joined.sort_values(['absdelta_R14minusR9_share','ID'], ascending=[False,True]).head(50).to_csv(out/'top50_absdelta.csv', index=False)
    joined.sort_values(['delta_R14minusR9_share','ID'], ascending=[False,True]).head(50).to_csv(out/'top50_positive_delta.csv', index=False)
    maincols = [n+'.'+k for n in MAIN for k in ('share','score') if n+'.'+k in joined]
    focus = pd.DataFrame(relations); focus = focus[focus.series.isin(maincols)]
    focus.to_csv(out/'main_feature_prediction_correlations.csv', index=False)
    for p,h in pins.items():
        if sha(ROOT/p) != h: raise ValueError('Input changed during descriptive audit:'+p)
    counts = {str(k): int(v) for k,v in joined.probe_site.value_counts().items()}
    result = {'status': 'passed_descriptive_only', 'rows': len(joined), 'prediction_series': len(predcols),
              'elapsed_seconds': time.time()-started, 'script_sha256': sha(__file__),
              'plan_sha256': sha(out/'plan.json'), 'input_files_unchanged': True,
              'source_file_sha256': pins, 'member_audit_input_sha256': source['source_sha256'],
              'probe_site_counts_estimates_not_GT': counts,
              'attachment_SF489_does_not_match_actual_cached_SF': counts.get('07 St Francis'),
              'features': list(FEATURES), 'input_only_group_thresholds': {'bump_high_q25_db':qh, 'std_broadband_q25_db':qs, 'tilt_q25_db':qt},
              'group_counts': {k: int(np.asarray(v).sum()) for k,v in groups.items()},
              'S_phys_numeric_static_band_offset_fixture_max_error': invariance,
              'raw_Test_read_scope': 'Only original public-ID/mask integrity and the predeclared S_phys scalar; no fitting/inference/hidden labels',
              'actual_model_inference_training_calibration_or_submission_calls': 0,
              'unavailable': ['Per-query exact R14 PCA/LF-anchor maps were not cached and were not reconstructed; tilt/centroid are explicitly different proxies.',
                              'R14 raw classifier probabilities are not present in the saved query NPZ. No algebraic reconstruction is presented as an observed probability.'],
              'limits': ['kNN probe_site is an estimated domain label, not true Test site; no new matching was fitted.',
                         'Predictive output mean/RMS are not target moments. Correlation with physical features is neither AUC nor evidence of correct vessel labels.',
                         'Four public submissions changed multiple factors together, so their ordering cannot isolate CNN weight, architecture, PCA or a generator as the cause.',
                         'S_phys is a fixed temporal descriptor, not a predicted share or probability. A static per-band dB offset cancels algebraically; arbitrary changes before0.1dB rounding retain a quantization limitation.',
                         'Cached historical EDA columns were rounded to five decimals. This audit reuses them instead of silently replacing definitions.',
                         'No Test-label inference, public-score target matching, new thresholds/weights/calibration for prediction or submission file generation occurred.'],
              'outputs_sha256': {str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()}}
    write(out/'RESULTS.json', result)
    text = ['# R15 입력·저장 예측 기술통계 감사', '',
            '결과: 저장 예측57개 열과 기존 EDA 캐시1,500개 ID를 일대일 결합했다. 새 모델 추론·학습·보정·제출 파일 생성은0회다.', '',
            '## 추정 사이트와 정의', '',
            'probe_site는 기존TRAIN의 분별25대역 dB를 창 내 시간중앙값으로 요약한 KNN25 도메인 프로브 결과다. 기존 group5fold는 site×ISOweek 기준 보조검증이며 Test의 실제 지점 정답은 아니다.', '',
            '실제 캐시의 추정 건수: '+str(counts)+'. 첨부의 St Francis489개와 달리 현재 캐시는487개다. 보정하거나 재추정하지 않았다.', '',
            'bump_high_db는3162.3–7943.3Hz 다섯 대역의 **선형 에너지합을 dB로 바꾼 시간곡선**의 최대−중앙값이다. bump_low_db는31.6–501.2Hz 열세 대역에 같은 공식을 쓴다. 이 통계는 밴드별 bump평균과 다르다. tilt_db는≤125.9Hz 평균 dB−≥1995.3Hz 평균 dB이며 작은 값이 상대적 중고역 우세를 뜻한다.', '',
            'S_phys는 사전 고정한316.2–5011.9Hz 각 대역의 시간최대−중앙값을 대역평균한 dB 기술량이다. 정답이나 확률이 아니며 static band-offset 합성 수치 검산 오차는'+str(invariance)+'다.', '',
            '## 파일', '',
            '- [전수 ID 결합표](joined1500.csv)', '- [모델별 분포](model_distribution.csv)',
            '- [추정 유형별 분포](site_model_stats.csv)', '- [특징–예측 Pearson/Spearman](main_feature_prediction_correlations.csv)',
            '- [모든57개 열의 특징 상관](feature_prediction_correlations.csv)',
            '- [모델 간 Pearson](model_correlations_pearson.csv)', '- [모델 간 Spearman](model_correlations_spearman.csv)',
            '- [낮은 시간변화·정적 proxy 그룹](lowtemporal_highstatic_groups.csv)',
            '- [절대차 상위50](top50_absdelta.csv)', '- [양의 차이 상위50](top50_positive_delta.csv)',
            '- [선언 계획](plan.json)', '- [입력·출력 SHA와 한계](RESULTS.json)', '',
            '원 R14 PCA/LF-anchor의 Test별 맵 캐시 및 R14 raw classifier 확률은 저장되지 않아 이번 감사에서 관측값처럼 재구성하지 않았다. tilt·centroid는 다른 proxy다. 숫자가 큰 예측이 오탐인지, 낮은 예측이 정답인지 숨겨진 라벨 없이 판단할 수 없다. 상관관계·네 번의 제출 순서만으로 CNN/PCA/정적특징의 인과 원인을 확정하지 않는다.']
    (out/'README.md').write_text('\n'.join(text)+'\n')
    final = {p.name:sha(p) for p in out.iterdir() if p.is_file()}
    write(out/'DELIVERY_MANIFEST.json', {'status':'completed_descriptive_audit', 'files_sha256':final,
                                        'script_sha256':sha(__file__), 'no_submission_created':True})
    print(json.dumps({'status':result['status'], 'output':str(out), 'rows':len(joined),
                      'prediction_series':len(predcols), 'site_counts':counts,
                      'RESULTS_sha256':sha(out/'RESULTS.json')}, ensure_ascii=False))

if __name__ == '__main__': main()
