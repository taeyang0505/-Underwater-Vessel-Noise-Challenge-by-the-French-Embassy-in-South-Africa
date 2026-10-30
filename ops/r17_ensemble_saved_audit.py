#!/usr/bin/env python3
"""Read-only comparison of prescribed R17 mixtures from saved R16 predictions.

No training, model loading, model inference, coefficient fitting, or submissions.
All requested panels are now retrospective diagnostic panels, never untouched tests.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops.r16_postprocess_math import summary_metrics

OUT = ROOT / 'reports/r17_eda_audit_20261030/task1'
BASE = ROOT / 'models/r16_v1_fourway_r2'
MEMBERS = ['lightgbm5', 'catboost5', 'dasheng_base', 'dinov2_small', 'timemoe']
POLICY_PATH = ROOT / 'models/r16_pretrained_cuda_bf16_r1/both/final/policy.json'
PANELS = ['local_gate', 'local_report', 'domain_holdout',
          'native12_VS_hf_high_negative', 'native12_VS_st_francis_dense_negative']
MIXTURES = {
    'A_original': None,
    'B_L60_DINO25_Dash15': [0.60, 0., 0.15, 0.25, 0.],
    'B_L50_DINO30_Dash20': [0.50, 0., 0.20, 0.30, 0.],
    'C_DINO50_Dash50': [0., 0., 0.50, 0.50, 0.],
    'D_balanced4': [0.25, 0.25, 0.25, 0.25, 0.],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def write_csv(path, rows):
    if not rows:
        raise ValueError(path)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def subset(panel, ix):
    n = len(panel['sample_id'])
    return {k: v[ix] if v.ndim and len(v) == n else v for k, v in panel.items()}


def descriptive(v):
    v = np.asarray(v, float)
    q = np.quantile(v, [0., .01, .05, .1, .25, .5, .75, .9, .95, .99, 1.])
    return {'mean': float(v.mean()), 'std': float(v.std()),
            'rms': float(np.sqrt(np.mean(v*v))),
            **{k: float(x) for k, x in zip(
                ['min', 'p01', 'p05', 'p10', 'p25', 'median', 'p75', 'p90', 'p95', 'p99', 'max'], q)}}


def correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return {'pearson': None, 'spearman': None}
    return {'pearson': float(pearsonr(a, b).statistic),
            'spearman': float(spearmanr(a, b).statistic)}


def predict(panel, weights, original):
    if weights is None:
        s = sum(.5 * panel['share_'+r] @ np.array(original['experts'][r]['regression_weights'])
                for r in ['D0', 'H0'])
        c = sum(.5 * panel['classifier_'+r] @ np.array(original['experts'][r]['independent_classifier_weights'])
                for r in ['D0', 'H0'])
    else:
        weights = np.array(weights, float)
        assert abs(weights.sum()-1) < 1e-12 and np.all(weights >= 0)
        s = .5 * (panel['share_D0'] + panel['share_H0']) @ weights
        c = .5 * (panel['classifier_D0'] + panel['classifier_H0']) @ weights
    return np.clip(s, 0, 1), np.clip(c, 0, 1)


def macro_site_rmse(panel, pred):
    if 'site' not in panel:
        return None
    sites = np.unique(panel['site'])
    return float(np.mean([np.sqrt(np.mean((pred[panel['site'] == s]-panel['target_share'][panel['site'] == s])**2)) for s in sites]))


def metrics(panel, s, c):
    y = panel['target_share']
    m = summary_metrics(y, s, c, panel.get('weights'))
    m['site_macro_rmse'] = macro_site_rmse(panel, s)
    m['unweighted_rmse'] = float(np.sqrt(np.mean((s-y)**2)))
    m['n_zero'] = int(np.sum(y == 0))
    return m


def md_table(rows, columns):
    result = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    for row in rows:
        def fmt(v):
            return '미측정' if v is None else f'{v:.6f}' if isinstance(v, (float, np.floating)) else str(v)
        result.append('| ' + ' | '.join(fmt(row.get(k)) for k in columns) + ' |')
    return '\n'.join(result)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    original = json.loads(POLICY_PATH.read_text())
    assert original['member_order'] == MEMBERS
    paths = {'selection': BASE/'predictions/selection.npz',
             **{p: BASE/f'predictions/{p}.npz' for p in PANELS},
             'query': BASE/'query_predictions.npz'}
    pins = {str(p.relative_to(ROOT)): sha(p) for p in [POLICY_PATH, *paths.values()]}
    selected = load(paths['selection'])
    panels = {'checkpoint_selection': subset(selected, selected['checkpoint_selection_row_indices']),
              'ensemble_fit': subset(selected, selected['ensemble_fit_row_indices']),
              **{p: load(paths[p]) for p in PANELS}}
    assert len(panels['checkpoint_selection']['sample_id']) == 1194
    assert len(panels['ensemble_fit']['sample_id']) == 1206
    assert not set(panels['checkpoint_selection']['sample_id']) & set(panels['ensemble_fit']['sample_id'])
    individual, mixtures, site_rows, conditional, row_tables, bootstrap_rows = [], [], [], [], {}, []
    # Every loaded panel has its own target; all ID/shape checks precede metrics.
    for name, panel in panels.items():
        n = len(panel['sample_id'])
        assert len(np.unique(panel['sample_id'])) == n
        for key in ('share_D0', 'share_H0', 'classifier_D0', 'classifier_H0'):
            assert panel[key].shape == (n, 5) and np.isfinite(panel[key]).all()
        table = []
        for i in range(n):
            row = {'panel': name, 'sample_id': str(panel['sample_id'][i]), 'target_share': float(panel['target_share'][i])}
            for k in ['site', 'background_group', 'base_window_index', 'operator', 'support_minutes', 'recorder']:
                if k in panel:
                    row[k] = panel[k][i].item()
            for j, family in enumerate(MEMBERS):
                row['share_'+family] = float(.5*(panel['share_D0'][i,j]+panel['share_H0'][i,j]))
                row['score_'+family] = float(.5*(panel['classifier_D0'][i,j]+panel['classifier_H0'][i,j]))
            table.append(row)
        for regime in ['D0', 'H0', 'D0_H0_average']:
            s = .5*(panel['share_D0']+panel['share_H0']) if regime == 'D0_H0_average' else panel['share_'+regime]
            c = .5*(panel['classifier_D0']+panel['classifier_H0']) if regime == 'D0_H0_average' else panel['classifier_'+regime]
            for j, family in enumerate(MEMBERS):
                individual.append({'panel': name, 'regime': regime, 'family': family, **metrics(panel,s[:,j],c[:,j])})
        predictions = {}
        for mix, w in MIXTURES.items():
            s, c = predict(panel,w,original); predictions[mix] = s
            mixtures.append({'panel': name, 'mixture': mix, 'score_policy': 'original_LGB100' if w is None else 'same_fixed_family_weights_as_regression', **metrics(panel,s,c)})
            for i, row in enumerate(table):
                row['share_'+mix], row['score_'+mix] = float(s[i]), float(c[i])
            if 'site' in panel:
                for site in np.unique(panel['site']):
                    ix = panel['site'] == site
                    sub = subset(panel, ix)
                    site_rows.append({'panel': name, 'site': str(site), 'mixture': mix, **metrics(sub,s[ix],c[ix])})
            for label, mask in [('zero',panel['target_share']==0), ('weak_positive', (panel['target_share']>0)&(panel['target_share']<=.03)),
                                ('share_005_015',(panel['target_share']>=.05)&(panel['target_share']<=.15)),
                                ('high_share',panel['target_share']>.2)]:
                if np.sum(mask):
                    conditional.append({'panel':name,'slice':label,'mixture':mix,**metrics(subset(panel,mask),s[mask],c[mask])})
        # Descriptive source-group bootstrap; does not make reused groups independent.
        if 'background_group' in panel and len(np.unique(panel['background_group'])) >= 2:
            groups, inverse = np.unique(panel['background_group'], return_inverse=True)
            sw = panel.get('weights', np.ones(n))
            denom = np.bincount(inverse, weights=sw)
            se = {k:np.bincount(inverse,weights=sw*(v-panel['target_share'])**2) for k,v in predictions.items()}
            rng = np.random.default_rng(20261030)
            draws = rng.integers(0,len(groups),size=(1000,len(groups)))
            base = np.sqrt(se['A_original'][draws].sum(1)/denom[draws].sum(1))
            for mix in MIXTURES:
                delta = np.sqrt(se[mix][draws].sum(1)/denom[draws].sum(1))-base
                bootstrap_rows.append({'panel':name,'mixture':mix,'n_background_groups':len(groups),'bootstrap_repeats':1000,'rmse_delta_A_p025':float(np.quantile(delta,.025)),'rmse_delta_A_p50':float(np.quantile(delta,.5)),'rmse_delta_A_p975':float(np.quantile(delta,.975)),'caution':'retrospective reused source-group bootstrap, not official uncertainty'})
        row_tables[name] = table

    query = load(paths['query'])
    r9path = ROOT/'submissions/r9_fold0_all5_cpu_r1/submission.csv'
    if not r9path.exists():
        candidates = list((ROOT/'submissions/r9_fold0_all5_cpu_r1').glob('*.csv'))
        if len(candidates) != 1:
            raise ValueError(f'R9 CSV ambiguous: {candidates}')
        r9path = candidates[0]
    with r9path.open(newline='') as f:
        r9 = list(csv.DictReader(f))
    mapping = {x['ID']:x for x in r9}
    assert set(mapping) == set(query['sample_id'].astype(str))
    r9s = np.array([float(mapping[str(i)]['share_added_vessel']) for i in query['sample_id']])
    r9c = np.array([float(mapping[str(i)]['score_added_vessel']) for i in query['sample_id']])
    pins[str(r9path.relative_to(ROOT))] = sha(r9path)
    tests, qrows = [], []
    for i, sid in enumerate(query['sample_id']):
        row={'sample_id':str(sid),'R9_share':float(r9s[i]),'R9_score':float(r9c[i])}
        for j,family in enumerate(MEMBERS):
            row['share_'+family]=float(.5*(query['share_D0'][i,j]+query['share_H0'][i,j]))
            row['score_'+family]=float(.5*(query['classifier_D0'][i,j]+query['classifier_H0'][i,j]))
        qrows.append(row)
    allquery = {f: (.5*(query['share_D0'][:,j]+query['share_H0'][:,j]),.5*(query['classifier_D0'][:,j]+query['classifier_H0'][:,j])) for j,f in enumerate(MEMBERS)}
    allquery.update({m:predict(query,w,original) for m,w in MIXTURES.items()})
    for label,(s,c) in allquery.items():
        for kind,v,ref in [('share',s,r9s),('score',c,r9c)]:
            tests.append({'model_or_mixture':label,'output':kind,**descriptive(v),'ge090_fraction':float(np.mean(v>=.9)),**{'R9_'+k:v for k,v in correlation(v,ref).items()}})
        for i,row in enumerate(qrows):
            row['share_'+label],row['score_'+label]=float(s[i]),float(c[i])
    for name,table in row_tables.items():
        write_csv(OUT/f'rows_{name}.csv',table)
    write_csv(OUT/'rows_Test_descriptive_only.csv',qrows)
    for name,rows in [('individual_metrics',individual),('mixture_metrics',mixtures),('mixture_site_metrics',site_rows),('mixture_target_slice_metrics',conditional),('group_bootstrap_rmse_differences',bootstrap_rows),('Test_descriptive_statistics',tests)]:
        write_csv(OUT/(name+'.csv'),rows)
    # Fit/selection leakage is a provenance limitation, not a causal estimate.
    protocol={'created_KST':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(), 'member_order':MEMBERS,
              'prescribed_weights':MIXTURES,'D0_H0_weights':[.5,.5],
              'classification_rule':'A preserves original LGB100. B/C/D average independent classifier probabilities using their fixed regression-family weights. No share enters score. No AUC optimization.',
              'D_definition':'LGB/Cat/Dash/DINO each25%, Time0. User requested balanced combination without a numeric definition; this fixed equal allocation is the declared interpretation.',
              'no_retraining':True,'no_model_inference':True,'no_weight_fitting':True,'no_submission_csv_created':True,'official_upload':False,
              'Test_labels_used':False,'Test_used_to_select_or_adjust_weights':False,
              'leakage_disclosure':'Legacy LGB early stopping used original2400-row local_fit, including current1206 ensemble_fit. New PEFT checkpoints used separate1194 rows. Cat own-fold CV and weights fit on1206. Prior panels reused across development. Stored outputs cannot undo legacy checkpoint-selection exposure.',
              'stress_weights':'Preserve saved weights: positives50%, negatives50%; also report unweighted RMSE.',
              'source_sha256':pins}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    report=[]
    report += ['# R17 저장 예측 기반 배합 A~D 및 개별 분류기 전수 감사','',
               '재학습·모델 추론·가중치 적합·새 제출 CSV 생성은 모두 0회다. 기존 저장 예측만 읽고 요청된 고정 비율을 계산했다. 원본 정책과 파일은 변경하지 않았다.','',
               '## 1. 비교 정의','',
               '- 모든 배합은 D0:H0=50:50이다. A는 원본의 서로 다른 D0/H0 회귀 비중과 LGB100 독립 분류를 그대로 보존한다.',
               '- B1: LGB60/DINO25/Dash15. B2: LGB50/DINO30/Dash20. C: DINO50/Dash50. D: LGB/Cat/DINO/Dash 각각25. TimeMoE는 개별 성적을 공개하지만 B~D에는 포함하지 않는다.',
               '- B~D의 score는 해당 고정 계열 비율로 독립 분류 확률을 평균한다. 회귀 share를 score에 섞지 않는다. 회귀만 바꾸고 score를 A처럼 유지하면 AUC는 당연히 A와 같으므로, 이 표에서는 독립 분류 배합의 실제 변화도 함께 비교했다.',
               '- native12 스트레스는 저장된 양성50%/음성50% 평가 가중치를 보존했다. CSV에 비가중 RMSE도 병기했다.','',
               '## 2. 개별 독립 분류기 AUC: D0/H0 평균','']
    display=['ensemble_fit',*PANELS]
    clsrows=[]
    for family in MEMBERS:
        row={'family':family}
        for name in display:
            item=next(x for x in individual if x['panel']==name and x['regime']=='D0_H0_average' and x['family']==family)
            row[name]=item['score_auc']
        clsrows.append(row)
    report += [md_table(clsrows,['family',*display]),'',
               'D0·H0 각각의 회귀 RMSE/편향/분류 AUC, 체크포인트 선택셋까지 전수 수치는 `individual_metrics.csv`에 있다.','',
               '## 3. 고정 배합의 RMSE / 독립 score AUC','']
    for name in display:
        report += [f'### {name}','',md_table([x for x in mixtures if x['panel']==name],['mixture','n','rmse','bias','score_auc','share_auc','site_macro_rmse']),'']
    report += ['## 4. Test 1,500행 기술 통계: 정답 없음','',
               md_table([x for x in tests if x['output']=='share'],['model_or_mixture','mean','std','rms','median','p90','R9_pearson','R9_spearman']),'',
               '이 통계는 입력·출력 감사용이다. Test 진짜 정답 평균·양성률·편향을 추정하거나 이를 근거로 비율을 선택하지 않았다. 원본과 B~D의 공식 점수를 예측하지 않는다.','',
               '## 5. 누수·대표성 구분','',
               '신경망 체크포인트 선택셋1,194행과 현재 앙상블 fit1,206행의 sample ID는 겹치지 않는다. 그러나 재사용한 LightGBM의 조기 종료는 두 집합을 나누기 전 local_fit2,400행을 사용했다. 따라서 현재 fit은 모든 계열에 동등하게 미노출인 시험지가 아니며, 이미 선택된 LGB checkpoint를 저장 예측 연산만으로 미노출 상태로 되돌릴 수는 없다.',
               '이 이력 때문에 LGB에 선택상의 유리함이 있을 가능성은 있으나, 그 영향량이나 이번 Public 악화의 직접 원인을 확정할 수 없다. CatBoost는 own-fold CV로 조기 종료했고, 동일 fit에서 우수했던 Cat과 미관측 사이트에서 우수했던 DINO의 순위 역전은 별도 사실이다.',
               'local_gate/local_report 및 스트레스 패널도 반복 개발 이력이 있다. 이번 요청에 따라 모두 사후 진단으로 읽었으므로 새 독립 시험이나 공식 성능의 대체물로 부르지 않는다. `group_bootstrap_rmse_differences.csv`는 저장 background_group 기준 탐색적 재표집으로, 적은 그룹/재사용 배경/그룹 간 의존성을 해결하지 않는다.','',
               '## 6. 파일과 재현','',
               '- 재현: `python ops/r17_ensemble_saved_audit.py` (NumPy/SciPy 필요)',
               '- 개별 성적: `individual_metrics.csv`; 배합 성적: `mixture_metrics.csv`',
               '- 사이트/타깃 구간 오차: `mixture_site_metrics.csv`, `mixture_target_slice_metrics.csv`',
               '- Test 통계: `Test_descriptive_statistics.csv`; 모델별 행 단위 감사: `rows_*.csv`',
               '- 원본 SHA 및 고정 비교 정의: `protocol.json`; 전체 산출물 SHA: `SHA256SUMS.json`','']
    (OUT/'R17_SAVED_PREDICTION_ENSEMBLE_AUDIT.md').write_text('\n'.join(report))
    result={'protocol':protocol,'individual_metrics':individual,'mixture_metrics':mixtures,'Test_descriptive_statistics':tests,'group_bootstrap':bootstrap_rows}
    (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    assert pins=={rel:sha(ROOT/rel) for rel in pins}
    hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS.json'}
    hashes['../../../ops/r17_ensemble_saved_audit.py']=sha(Path(__file__))
    (OUT/'SHA256SUMS.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'completed','output':str(OUT),'individual_rows':len(individual),'mixture_rows':len(mixtures),'files':len(hashes)},ensure_ascii=False))


if __name__ == '__main__':
    run()
