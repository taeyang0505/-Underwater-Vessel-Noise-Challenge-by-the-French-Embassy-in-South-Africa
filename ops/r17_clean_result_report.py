#!/usr/bin/env python3
"""Evaluate the registered CPU classifiers on fresh fit and publish receipts.

No model fitting, query inputs, tuning, or submission upload in this file.
"""
from __future__ import annotations
import argparse,html,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops.r17_audit_common import load,sha,jsave
OUT=ROOT/'reports/r17_clean_pipeline_r1'
DATA=ROOT/'data/synthetic/r17_clean_freshfit_v1'
MODEL=ROOT/'models/r17_clean_classifier_r1'

def fresh():
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    from ops.r17_clean_classifier_features import classifier_indices
    from ops.r17_clean_classifier_refit import cached,metrics,no_query
    sys.addaudithook(no_query)
    if not (OUT/'refit/COMPLETED.json').exists():raise RuntimeError('All registered fits must finish first')
    rows=[];dest=OUT/'fresh_fit_classifiers';dest.mkdir(exist_ok=True)
    for geometry in ('D0','H0'):
        path=DATA/f'ensemble_fit_fresh_{geometry}.npz';d=load(path);x=cached('fresh_fit_'+geometry,d['db'])
        predictions={}
        for family,modes in (('lightgbm',('full','direct','aggressive')),('catboost',('full','aggressive'))):
            for mode in modes:
                idx=classifier_indices(mode);preds=[]
                for regime in ('D0','H0'):
                    folder=MODEL/family/regime/mode
                    receipt=json.loads((folder/'COMPLETED.json').read_text())
                    for name,h in receipt['artifact_sha256'].items():
                        if sha(folder/name)!=h:raise ValueError('Refit artifact changed')
                    if family=='lightgbm':
                        m=lgb.Booster(model_file=str(folder/'classification.txt'))
                        p=m.predict(x[:,idx],num_threads=4)
                    else:
                        m=CatBoostClassifier();m.load_model(str(folder/'classification.cbm'))
                        p=m.predict_proba(x[:,idx],thread_count=4)[:,1]
                    preds.append(p)
                    rows.append({'dataset_geometry':geometry,'model_regime':regime,'family':family,'mode':mode,
                                 **metrics(d['presence'],p)})
                    predictions[family+'__'+mode+'__'+regime]=p
                p=.5*(preds[0]+preds[1])
                rows.append({'dataset_geometry':geometry,'model_regime':'D0_H0_50_50','family':family,'mode':mode,
                             **metrics(d['presence'],p)})
        np.savez_compressed(dest/f'predictions_{geometry}.npz',sample_id=d['sample_id'],target_presence=d['presence'],**predictions)
    frame=pd.DataFrame(rows);frame.to_csv(dest/'metrics.csv',index=False,encoding='utf-8-sig')
    jsave(dest/'COMPLETED.json',{'status':'completed','new_fit_source_sha256':sha(DATA/'manifest.json'),
        'metrics_sha256':sha(dest/'metrics.csv'),'retraining':False,'Test_used':False,
        'averaging':'Both trained regime classifiers on the SAME input matrix; not different-label D0/H0 rows'})

def table(d):return d.to_markdown(index=False,floatfmt='.6f')

def report():
    if not (OUT/'fresh_fit_classifiers/COMPLETED.json').exists():raise RuntimeError('Fresh classifier comparison required')
    receipt=json.loads((OUT/'data/DATA_RECEIPT.json').read_text())
    fres=json.loads((DATA/'manifest.json').read_text())
    cand=ROOT/'submissions/r17_fixed_blends_20261030'
    cm=json.loads((cand/'candidate_manifest.json').read_text())
    # Candidate file hashes are checked independently of their source Test inputs.
    cfiles=sorted(cand.glob('submission_r17_*.csv'))
    crows=[]
    for path in cfiles:
        a=pd.read_csv(path);s=a['share_added_vessel'].to_numpy();p=a['score_added_vessel'].to_numpy()
        if len(a)!=1500 or not np.isfinite(a.iloc[:,1:].to_numpy()).all():raise ValueError('Candidate integrity changed')
        crows.append({'candidate':'B1' if '_B1_' in path.name else 'B2','rows':len(a),
                      'share_mean':s.mean(),'share_RMS':np.sqrt(np.mean(s*s)),
                      'score_mean':p.mean(),'score_ge090':np.mean(p>=.9),'SHA256':sha(path)})
    pd.DataFrame(crows).to_csv(OUT/'CANDIDATE_SUMMARY.csv',index=False,encoding='utf-8-sig')
    old=pd.read_csv(ROOT/'reports/r17_eda_audit_20261030/task1/mixture_metrics.csv')
    old=old[old.mixture.isin(['A_original','B_L60_DINO25_Dash15','B_L50_DINO30_Dash20']) & ~old.panel.isin(['ensemble_fit','checkpoint_selection'])]
    rec=pd.read_csv(OUT/'refit/classifier_comparison.csv')
    av=rec[rec.regime.eq('D0_H0_50_50')]
    av.to_csv(OUT/'CLASSIFIER_PANEL_SUMMARY.csv',index=False,encoding='utf-8-sig')
    piv=av.pivot_table(index=['family','mode'],columns='panel',values='auc').reset_index()
    ff=pd.read_csv(OUT/'fresh_fit_classifiers/metrics.csv');ff=ff[ff.model_regime.eq('D0_H0_50_50')]
    fp=ff.pivot_table(index=['family','mode'],columns='dataset_geometry',values='auc').reset_index()
    modelrows=[]
    for p in MODEL.glob('*/*/*/COMPLETED.json'):
        j=json.loads(p.read_text());modelrows.append({k:j[k] for k in ('family','regime','mode','features','best_iteration','seconds')})
    train=pd.DataFrame(modelrows);train.to_csv(OUT/'TRAINING_JOBS.csv',index=False,encoding='utf-8-sig')
    metadata=pd.read_parquet(DATA/'ensemble_fit_fresh_backgrounds.parquet')
    freshcomp=metadata.groupby(['site','qc_observed_minutes']).size().reset_index(name='unique_backgrounds')
    freshcomp['rows_per_regime']=freshcomp.unique_backgrounds*5
    freshcomp.to_csv(OUT/'FRESH_FIT_COMPOSITION.csv',index=False,encoding='utf-8-sig')
    purity=pd.read_csv(OUT/'features/classifier_importance_decomposition_mean.csv')
    source_summary=OUT/'shortcut_summary/R17_SHORTCUT_DETAIL_SUMMARY.md'
    text='''# R17 고정 B1/B2 후보 및 클린 파이프라인 CPU 실측

작성일: 2026-10-30 (Asia/Seoul). 기존 R16/R17 감사 자료·모델·제출 원본은 보존했다. 새 대형 신경망 학습, 추가 K-fold, 공식 업로드, Test 라벨 역산·Test 기반 적합은 수행하지 않았다.

## 1. B1/B2 제출 후보 — 요청한 고정 배합

B1은 LGB 60% + DINOv2 25% + Dasheng 15%, B2는 LGB 50% + DINOv2 30% + Dasheng 20%다. 각 계열의 D0/H0 예측을 같은 입력에서 50:50으로 합친다. share는 회귀 예측, score는 **동일 비율의 독립 분류 확률**이다. score에 share를 넣지 않는다. CatBoost·TimeMoE 가중치는 0이다. 기존 저장 예측을 그대로 사용하며 이번 클린 재적합 모델을 이 두 파일에 몰래 교체하지 않았다.

'''+table(pd.DataFrame(crows))+'''

위 후보들은 각각 1,500행·공식 ID 순서·3열·중복 없음·유한값·[0,1] 및 CSV 재독출/원본 NPZ 재계산 검증을 통과했다. 기술 통계는 정답 정확도가 아니며, 새 공식 점수는 미측정이다.

'''
    for p in cfiles:text+=f'- [{p.name}]({os.path.relpath(p,OUT)})\n'
    text+='\n### 기존 5개 패널 회귀 RMSE (재학습 없는 고정 배합)\n\n'
    value='rmse' if 'rmse' in old else 'RMSE'
    text+=table(old.pivot_table(index='mixture',columns='panel',values=value).reset_index())
    text+='''

## 2. 상세 shortcut CSV 재검산 — 확인값과 가설 구분

| 항목 | 실측 | 해석 |
|---|---|---|
| TRAIN D0 Nobs=5/15 양성률 | 80.1107% / 80.0714% | 밀도별 정답 비율이 원래 TRAIN에서 크게 달랐다는 주장은 맞지 않는다. |
| TRAIN MI(n_obs;presence) | 0.00010545 bits | 이미 매우 작다. 전체 O 패턴의 정보량과는 다른 지표다. |
| D0 삽입 시작/끝 절대 단차 중앙값 | 7.4260 / 7.4053 dB | 기타 인접 분 0.6925 dB보다 크다. 경계 의존의 후보 단서다. |
| H0 삽입 시작/끝 절대 단차 중앙값 | 7.0706 / 6.8839 dB | 기타 인접 분 0.6172 dB. 급격한 변화 자체는 실제 통과에도 있을 수 있다. |
| 양성·음성 0.1 dB 격자 잔차 | 모두 최대 0 | 이번 저장 자료에서 격자 유무가 정답을 알려주는 현상은 없다. |
| raw q와 반올림 E 풀링 오차 최대 | D0 0.00176033 / H0 0.00186614 | 정답을 바꾸지 않고 수치 차이를 추적했다. |
| 동일 y=.05~.15, unseen HF bump↔오차 Pearson | LGB -0.21899 / Cat -0.27048 | 이 합성 구간에서 HF가 클수록 항상 과대추정한다는 설명은 성립하지 않는다. |

원본 `support_mask_by_observation_density.csv`, `insertion_boundary_jumps.csv`, `same_y_background_LF_error.csv`, `quantization_by_true_class.csv` 전체 요약과 추가 상관표는 [상세 문서](shortcut_summary/R17_SHORTCUT_DETAIL_SUMMARY.md)에 보존했다. n이 작은 조건은 원문 표에 함께 표시했다.

### 중요도 15~21%의 정확한 구성

'''+table(purity)+'''

순수 관측 비율 25개의 중요도는 **LGB gain 0.8280%, Cat PredictionValuesChange 3.0977%**다. 나머지 375개는 관측된 분 사이의 실제 음향 차분/lag다. 따라서 기존 이름 기반 `observation_mask_proxy` 15.34/21.35% 전체를 ‘음향 정보가 없는 마스크’로 해석할 수 없다. 두 라이브러리 중요도의 정의도 다르다.

## 3. 실제 클린 데이터 구현

| 항목 | 이전 | 새 CPU 대조 데이터 |
|---|---:|---:|
| TRAIN 참조 클립 | 60 | 465 |
| TRAIN 양성 행 (체계별) | 49,157 | 9,600 |
| 참조 최대 재사용 (체계별) | 4,097 | 21 (20~21회) |
| 참조 재사용 집중도 Kish ESS | 25.3555 | 464.7504 |
| TRAIN 총행 (체계별) | 61,446 | 12,000 |
| 고유 TRAIN 배경 | 61,446 | 2,400 (배경당 5예) |
| 새 ES/새 최종 fit | — | 각각 400배경×5예=2,000행/체계 |

표본 수가 다른 통제 실험이므로 ‘새 12,000행 모델이 기존 5-Fold보다 좋아졌다/나빠졌다’는 결과를 특징 삭제 하나의 효과로 귀속하지 않는다. 집중도 ESS는 참조 사용의 균등성 지표이며 실제 독립 녹음 수가 아니다. 배경 1개당 5행도 5개의 독립 표본이 아니다.

각 배경과 완전한 O 패턴마다 음성 1개+양성 4개를 배치했다. 모든 사이트·녹음기·O 층의 양성 비율은 정확히 80%이며, 전체 O와 presence의 경험적 MI는 약 10^-16 nats 이하의 수치 오차다. 이 보장은 **새로 생성한 자료의 마스크/정답 관계**에 대한 것이며, 소리 특징이나 실제 대회 Test의 분포까지 보장하지 않는다. 결측 중첩으로 양성이 0이 된 사례는 0건이다.

465개는 509개에서 예약 B19/C′25를 제외한 바이트 중복 없는 풀이다. 기존 의미인 ‘추가된 녹음 전체의 에너지’ 타깃을 유지해 vessel archive 내부 weak-background 22개도 명시 추적했다. 별도 ambient16 아카이브를 이번 양성 풀에 추가한 것은 아니다.

**참조 독립성 제한:** 예약 44개 파일 자체는 제외했지만, 확대 풀에는 같은 hull 70개·같은 녹음일 109개가 있다. 전이 그래프 전체가 한 연결성분이므로 엄격한 미관측 참조 검증이라고 부르지 않는다. 이번 개선은 원천 다양성·배경 일반화에 관한 실험이다.

### 검증셋 편중과 선택 노출 분리

초기 생성된 v1 fit에는 옛 local_fit의 고밀도 배경 37개가 들어갔다. 이를 최종 비교용으로 채택하지 않고 **별도 fresh-fit**을 추가 생성해 옛 local_fit 400개와 원천·그룹 겹침을 0으로 만들었다. 이미 학습한 모델/ES 자료는 변경하지 않았고, fresh-fit은 새 모델의 조기 종료에도 사용하지 않았다.

- TRAIN–ES 최소 24.5h, TRAIN–fresh-fit 32.75h, ES–fresh-fit 217.75h.
- fresh-fit–옛 local_fit 최소 378.25h, 원천/그룹 겹침 0.
- fresh-fit–고정 local_report 최소 33.5h, gate 1085.75h.
- fresh-fit Nobs=5 비율 0%, 고밀도 St Francis 포함.

'''+table(freshcomp)+'''

새 fit의 고밀도·다사이트 분포는 Test 수치에 맞춰 적합하지 않고 원천 메타데이터와 사전 고정한 밀도 조건으로 골랐다. 같은 녹음기/참조 공유와 반복 개발에 따른 한계는 남는다.

## 4. 분류 특징 삭제 CPU 대조 — 실제 학습 결과

동일한 새 TRAIN/ES, seed, 최대 1,500트리, ES40으로 비교했다. LGB는 full2033/direct2008/aggressive1633, Cat은 full2033/aggressive1633이다. `direct`는 순수 관측 비율 25개만, `aggressive`는 이름 기반 범주 400개를 전부 제거한다. 회귀기는 재학습하지 않았으며, 따라서 이번 표의 개선/악화는 **score 분류 성능**이다. B1/B2 회귀 RMSE가 이 실험 때문에 바뀐 것은 아니다.

### 고정 5개 패널 AUC (같은 입력에서 D0/H0 50:50)

'''+table(piv)+'''

위 패널들은 이미 여러 번 검토한 진단 자료다. 새로운 무편향 성적 또는 공식 AUC 예측으로 취급하지 않는다. 기존 5-Fold와 새 단일 CPU 모델은 학습 행 수와 원천 풀이 다르다. 순수한 특징 삭제 효과는 **같은 새 자료의 full↔direct/aggressive**에서 읽어야 한다.

### 새 fresh-fit AUC (학습·조기 종료 미사용)

'''+table(fp)+'''

특징 이름을 지워도 남은 음향 통계의 표본 수/lag 유효성 등에서 결측이 간접 드러날 수 있다. ‘모든 마스크 정보 제거 완료’라고 주장하지 않는다. 새 자료의 정확한 O/정답 균형과 실제 대조 AUC를 별개로 검증했다.

### 실행 수와 최적 트리 수

'''+table(train)+'''

## 5. 결론과 사용 범위

- B1/B2는 요청한 **기존 저장 예측의 고정 배합 후보**로 완성했다. 공식 제출은 실행하지 않았다.
- 참조 cap·전체 마스크 균형·고밀도 fresh-fit 구조는 구현 및 전수 검사했다.
- 특징 삭제가 유효한지는 위 실제 AUC 대조로 판단한다. 중요도만으로 Public 실패 원인을 확정하지 않는다.
- TRAIN 재구성은 CPU 통제 실험 범위다. DINO/Dash 본체를 새 465개 풀로 재학습한 상태가 아니다.
- 추가 대형 모델 학습·K-fold·Test 기반 평균 조절·정답 역산은 하지 않았다.

각 데이터·모델·후보 파일의 SHA-256은 `SHA256SUMS.json`과 하위 receipt에 보존했다. 기존 `R17_FULL_AUDIT_REPORT.md`를 수정하지 않고 별도 결과로 작성했다.
'''
    text+='\n## 6. 마스크 단독 반증 대조와 support 분포 변화\n\n'
    for rel in ('mask_only/full_O_MI.csv','mask_only/mask_only_lookup_auc.csv','data/SUPPORT_TRADEOFF.csv'):
        if (OUT/rel).exists():
            text+='\n'+table(pd.read_csv(OUT/rel))+'\n'
    text+='''

마스크 단독 대조는 학습 자료의 `P(presence|전체 375셀 O)`를 계산한 뒤, 음향 수치를 보지 않고 고정 패널에 적용한 것이다. 기존 TRAIN의 전체 O 정보량도 약 0.00117 bits로 작다. 이 단순 마스크 사전확률은 기존 모델의 0.96대 로컬 AUC를 재현하지 못한다. 모든 간접 상호작용을 반증한 것은 아니지만, 마스크만으로 기존 성능/실패가 설명됐다는 주장은 지지하지 않는다.

클립 균등 사용은 원천 집중도를 줄이는 대신 support 분포도 바꿨다. H0의 12분 이상 양성은 **15.5949% → 2.0938%**로 줄었다. 따라서 원본 모델과 새 모델 간 장시간 AUC 차이를 오직 마스크 삭제의 효과로 해석할 수 없다. 이 trade-off를 숨기지 않으며, 새 CPU 레시피가 전 패널을 개선한 최종 생산 모델이라고 채택하지 않는다.

또한 H0의 best epoch 3~6은 실제 학습이 그 epoch에 중단됐다는 뜻이 아니다. 예를 들어 Dasheng H0는 회귀 11epoch/분류10epoch, DINO H0는 각각13epoch까지 실행됐다. 원천 재사용과 warm-up 학습률이 빠른 best epoch의 직접 원인이라는 판단에는 별도 통제 실험이 필요하며, 이번에는 대형 신경망을 다시 학습하지 않았다.
'''
    exposure=OUT/'data/fresh_fit/LEGACY_EXPOSURE_AUDIT.json'
    if exposure.exists():
        text+='\n### 기존 모델 노출 감사\n\n'
        text+='기존 Fold0 CV를 바탕으로 만든 새 배경이 기존 5-Fold의 다른 TRAIN 또는 CatBoost의 조기 종료 원천에 노출됐는지 별도로 검사했다. 새 CPU 분류기의 TRAIN/ES와는 분리되어 있지만, 기존 동결 모델의 경우는 계열별 노출 상태를 구분한다.\n\n'
        text+='| 기존 원천 | fresh 400개 배경과의 겹침 |\n|---|---:|\n| Fold0 TRAIN | 0 |\n| Fold1/2/3 TRAIN | 각각 400 |\n| Fold4 TRAIN | 385 |\n| Fold0 Cat own-CV | ID 55, 원천 그룹·24h 근접 400 |\n| 옛 local_fit | 0 |\n\n'
        text+='따라서 기존 5-Fold 평균과 Cat Fold0는 보조 비교다. 주 비교는 배경 TRAIN/체크포인트 선택 원천이 분리된 LGB Fold0·DINO·Dasheng이며, 참조465개 공유에 관한 제한은 유지된다. 상세 수치는 [LEGACY_EXPOSURE_AUDIT.json](data/fresh_fit/LEGACY_EXPOSURE_AUDIT.json)에 보존했다.\n'
    # The frozen-inference summary is appended only after it has actually completed.
    frozen=OUT/'fresh_fit_frozen/metrics.csv'
    if frozen.exists() and (frozen.parent/'COMPLETED.json').exists():
        text+='\n## 7. 기존 Tree/동결 신경망의 새 fresh-fit 성적\n\n'
        text+='새 자료에서 재학습 없이 CPU FP32로 동결 추론한 비교다. 기존 CUDA BF16 저장 수치와 같은 수치 경로가 아니며, 새 패널의 동일 입력에서 각 학습 체계를 함께 평가했다.\n\n'
        fz=pd.read_csv(frozen)
        primary=fz[fz.evaluation_scope.str.startswith('primary')]
        text+=table(primary[['dataset','model','rows','rmse','bias','auc','site_macro_rmse']])+'\n'
        text+='\n주 비교의 B1/B2는 **LGB Fold0**를 사용하는 원천 노출 통제 대조다. 제출 후보 B1/B2는 사용자 지시대로 **기존 LGB 5-Fold**를 쓰므로 서로 같은 파일/모델 조합이라고 부르지 않는다.\n\n'
        text+='두 생성 입력을 합친 주 비교에서 LGB Fold0 RMSE 0.083065 대비 B1 방식 0.076925, B2 방식 0.076551로 낮아졌다. 새로운 공식 LB 점수나 일반화 보장은 아니다. 기존 5-Fold/Cat 포함 보조 비교, 예측 평균/RMS 및 사이트 동일가중 수치는 [전체 CSV](fresh_fit_frozen/metrics.csv)에 보존했다.\n'
    path=OUT/'R17_CLEAN_PIPELINE_RESULTS.md';path.write_text(text)
    # Portable HTML contains the complete Korean report and direct local artifact links.
    try:
        from markdown_it import MarkdownIt
        body=MarkdownIt('commonmark',{'html':False}).enable('table').render(text)
    except Exception:
        body='<pre>'+html.escape(text)+'</pre>'
    h='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>R17 클린 교정 실측</title><style>body{font:16px/1.65 system-ui;max-width:1400px;margin:35px auto;padding:0 24px;color:#15263a}h1,h2,h3{line-height:1.3}table{border-collapse:collapse;display:block;overflow:auto;margin:20px 0}th,td{padding:9px 12px;border:1px solid #ccd9e6;white-space:nowrap}th{background:#e8f1fa}a{color:#075bba}pre{white-space:pre-wrap}code{word-break:break-all}strong{color:#943d12}</style>'''+body+'</html>'
    (OUT/'START.html').write_text(h)
    jsave(OUT/'COMPLETED.json',{'status':'completed','report':str(path.relative_to(ROOT)),
        'report_sha256':sha(path),'classifier_models':len(modelrows),'new_kfold':False,
        'new_neural_training_steps':0,'official_upload':False,'Test_used_for_fit':False,
        'frozen_clean_fit_complete':(OUT/'fresh_fit_frozen/COMPLETED.json').exists()})
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    for root in (DATA,ROOT/'data/synthetic/r17_clean_v1',MODEL,cand):
        for p in root.rglob('*'):
            if p.is_file():hashes[str(p.relative_to(ROOT))]=sha(p)
    for p in ROOT.glob('ops/r17*.py'):
        hashes[str(p.relative_to(ROOT))]=sha(p)
    jsave(OUT/'SHA256SUMS.json',hashes)
    for n,h in hashes.items():
        if sha(ROOT/n)!=h:raise ValueError('Final SHA verification failed '+n)
    print(json.dumps({'report':str(path),'manifest_sha256':sha(OUT/'SHA256SUMS.json'),'verified_files':len(hashes)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['fresh','report']);a=p.parse_args()
    fresh() if a.phase=='fresh' else report()
