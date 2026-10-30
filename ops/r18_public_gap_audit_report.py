"""Assemble the R18 post-public-result audit from immutable local evidence.

No model training/inference, raw Test open, leaderboard label inference, or
policy change. The user-provided official scores are observation records only.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
from decimal import Decimal
import numpy as np
import pandas as pd
import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/r18_public_gap_audit_r1'
P = OUT / 'predictions'
B = OUT / 'background'
S = OUT / 'starter'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def table(frame, digits=6):
    return frame.to_markdown(index=False, floatfmt=f'.{digits}f')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec='seconds')
    image = Path('/var/folders/c1/ff1kv08j2fd06b_058sckl9w0000gn/T/codex-clipboard-cecd73dd-a3c9-463f-82f8-7739bf26f40d.png')
    official = {
        'source': 'User-provided submission result and screenshot; not independently downloaded',
        'submission_id': 'sdrhrvuo', 'identified_by_user_as': 'R18 B2',
        'public_score': '0.77872311', 'public_RMSE': '0.221276889', 'public_AUC': '0.573555555',
        'reference': {'submission_id': 'wSNRSQBH', 'candidate': 'R17 B1',
                      'score': '0.789471136', 'RMSE': '0.210528863', 'AUC': '0.572711111'},
        'delta_score': str(Decimal('0.77872311') - Decimal('0.789471136')),
        'delta_RMSE': str(Decimal('0.221276889') - Decimal('0.210528863')),
        'delta_AUC': str(Decimal('0.573555555') - Decimal('0.572711111')),
        'local_B2_path': 'submissions/r18_production_r1/submission_r18_B2.csv',
        'local_B2_sha256': sha(ROOT / 'submissions/r18_production_r1/submission_r18_B2.csv'),
        'screenshot_sha256': sha(image) if image.exists() else None,
        'Pod_stop': 'User explicitly replied already stopped; no agent Pod mutation',
        'additional_model_training': False, 'additional_model_inference': False,
        'Test_label_inference': False, 'Test_based_calibration': False,
        'created_KST': now,
    }
    (OUT / 'OFFICIAL_RESULT.json').write_text(json.dumps(official, ensure_ascii=False, indent=2) + '\n')

    metrics = pd.read_csv(P / 'all_saved_prediction_metrics.csv')
    overall = metrics[(metrics.scope == 'same_common_input') & (metrics.regime == 'D0H0_50_50') &
                      (metrics.target_interval == 'all')]
    individual = overall[overall.candidate.isin(['new_lightgbm', 'new_dinov2_small', 'new_dasheng_base'])][
        ['panel', 'candidate', 'n', 'rmse', 'bias', 'classifier_auc']]
    individual.to_csv(OUT / 'R18_individual_models_all_panels.csv', index=False)
    weak = metrics[(metrics.scope == 'same_common_input') & (metrics.regime == 'D0H0_50_50') &
                   metrics.candidate.isin(['old_B1', 'new_B2']) &
                   metrics.target_interval.isin(['y=0', '0<y<=0.10'])][
        ['panel', 'candidate', 'target_interval', 'n', 'rmse', 'bias', 'prediction_mean', 'target_mean']]
    weak.to_csv(OUT / 'zero_and_weak_comparison.csv', index=False)
    query = pd.read_csv(P / 'query_distribution_and_oldB1_correlations.csv')
    query = query[query.regime.eq('D0H0_50_50')].copy()
    assert not query.candidate.duplicated().any()
    qcols = ['candidate', 'share_mean', 'share_rms', 'share_median', 'share_p90',
             'score_mean', 'score_ge090', 'share_oldB1_pearson', 'share_oldB1_spearman',
             'share_oldB1_mean_difference']
    query[qcols].to_csv(OUT / 'query_technical_statistics.csv', index=False)
    provenance = pd.read_csv(P / 'model_training_provenance.csv')
    new = provenance[(provenance.generation == 'new') & (provenance.selected == True)]
    training = new[['family', 'regime', 'task', 'fold', 'train_rows', 'selection_rows',
                    'epochs_run', 'best_epoch', 'executed_rounds', 'best_iteration', 'input_feature_count']].fillna('—')
    aux = pd.read_csv(P / 'lgb_auxiliary_selection.csv')
    channels = pd.read_csv(P / 'neural_17channel_dictionary.csv')
    prior = pd.read_csv(P / 'train_target_prior_overall.csv')
    rank = pd.read_csv(P / 'zero_vs_weak_classifier_rank_and_fpr.csv')

    # Re-aggregate cached exhaustive SA feature tables; no raw observations/Test reads.
    raw = pd.read_csv(B / 'all_unfiltered_stride5_recorder_clock_join_windows.csv.gz')
    eligible = raw[raw.full_clock15.eq(True) & raw.n_eligible_rows.eq(15)]
    assert len(eligible) == 256029
    similar = eligible[eligible.HF_ge16_LF_le058.eq(True)]
    assert len(similar) == 379
    sizes = eligible.groupby(['site', 'recorder']).size().rename('eligible_windows')
    group = similar.groupby(['site', 'recorder']).agg(
        similar_windows=('LF_energy_fraction', 'size'), HF_bump_p50=('bump_high_db', 'median'),
        LF_fraction_p50=('LF_energy_fraction', 'median'), LF_bump_p50=('bump_low_db', 'median'),
        boundary_candidate_windows=('n_boundary_flagged_rows', lambda s: int((s > 0).sum())),
        source_file_crossing_windows=('crosses_source_file', 'sum'))
    group = group.join(sizes).reset_index()
    group['similar_fraction'] = group.similar_windows / group.eligible_windows
    group = group.sort_values('similar_windows', ascending=False)
    group.to_csv(OUT / 'valid_natural_similar_conditions_by_site_recorder.csv', index=False)
    site = similar.groupby('site').agg(similar_windows=('LF_energy_fraction', 'size'),
        HF_bump_p50=('bump_high_db', 'median'), LF_fraction_p50=('LF_energy_fraction', 'median'),
        LF_bump_p50=('bump_low_db', 'median')).join(eligible.groupby('site').size().rename('eligible_windows')).reset_index()
    site['similar_fraction'] = site.similar_windows / site.eligible_windows
    site = site.sort_values('similar_windows', ascending=False)
    site.to_csv(OUT / 'valid_natural_similar_conditions_by_site.csv', index=False)
    reasons = pd.read_csv(B / 'window_lineage_site_selection_reason.csv').groupby('selection_reason', as_index=False).windows.sum()
    row_reasons = pd.read_csv(B / 'row_disjoint_filter_attrition.csv').query("grouping=='site'").groupby('row_exclusion_reason', as_index=False).rows.sum()
    headline = pd.read_csv(B / 'headline_feature_distributions.csv')
    headline = headline.pivot(index='dataset', columns='feature', values='p50').reset_index()
    shapes = pd.read_csv(B / 'headline_shape_conditions.csv')[['dataset', 'windows', 'count', 'fraction']]
    qc = pd.read_csv(B / 'minute_QC_paired_summary.csv')
    qc = qc[qc.site.isna() & qc.recorder.isna()].drop(columns=['site', 'recorder'])
    transitions = pd.read_csv(B / 'minute_QC_paired_count_transitions.csv')
    weak_report = weak[weak.panel.eq('local_report')]
    effect = pd.read_csv(P / 'query_exact_generation_and_weight_effects.csv')
    fixture = json.loads((S / 'conversion_fixture.json').read_text())

    text = f'''# R18 공식 실패 후 근본 원인 감사

갱신: {now}. 새 모델 학습·추론·정책 변경·공식 제출·Test 라벨 역산·Test 통계 기반 보정 **0회**. Pod은 사용자가 이미 Stop했다고 확인했다. Test 입력 감사는 저장된 관측 특징과 동결 query NPZ만 재사용했다.

## 1. 공식 결과와 결론

| 제출 | Public Score | RMSE | AUC |
|---|---:|---:|---:|
| R17 B1 `wSNRSQBH` | 0.789471136 | 0.210528863 | 0.572711111 |
| R18 B2 `sdrhrvuo` | 0.778723110 | 0.221276889 | 0.573555555 |
| 변화 | −0.010748026 | +0.010748026 | +0.000844444 |

공식 결과는 사용자가 제공한 값이다. 제출 파일 연결도 사용자 확인에 따른다. 서버에서 제출물을 다시 내려받아 비교했다고 주장하지 않는다. 이번 결과로 **로컬 합성 검증 개선을 공식 성능 개선으로 옮기는 데 실패했다는 사실**은 확정됐다. 아래 사실과 기각된 가설을 구분해야 한다.

1. **확인:** 전체 로컬 개선은 저share 오차 악화와 함께 발생했다. report의 `0<y≤0.10` RMSE는 **0.026007→0.040986**, bias는 **+0.008883→+0.019392**다. 세 모델 모두 이 구간에서 악화했다. 전체 RMSE만을 통과 조건으로 둔 선택 게이트는 이 문제를 차단하지 못했다.
2. **기각:** 공통 레벨/조건부 배경 특징이 최종 LGB 예측을 올렸다는 설명. 두 regime 모두 보조 회귀기가 탈락했고 실제 제출에는 상대 **4,667특징**만 들어갔다. 미사용 보조 branch의 직접 기여도는 **0**이다.
3. **기각:** 원본 SA의 95%가 고주파 필터로 폐기됐다는 설명. 171만은 행, 78,869는 Fold0 TRAIN 창이다. 원본 행은 모두 저장돼 있고 **98.9745%가 원래 창에 포함**됐다. 같은 TRAIN 창에서 minute QC는 유사 조건을 **104→110창**으로 바꿨으며 소실은0창이었다.
4. **확인:** Test 전체 입력과 합성 배경의 분포는 다르다. 그러나 Test 전체 통계를 **무추가 음성의 통계로 바꿔 해석할 수는 없다.** 혼합 결과와 순수 배경을 동일 분포로 강제하는 것도 검증이 아니다.
5. **확인:** 공식 스타터는 완전한 Test 생성기가 아니다. 공개 audio helper는 수치 검산을 통과했지만, 우리6연산자의 배경·참조·SNR·support·EQ 법칙이 공식 Test 생성법과 같다는 근거는 없다.
6. **아직 미규명:** Public의 어느 개별 행/음성/양성 조건이 실패했는지, 숨은 생성기에서 어느 law가 다른지. Public RMSE/AUC만으로 이를 역산하거나 특정 원인으로 단정하지 않았다.

제가 약한 양성 악화를 이미 확인하고도 전체 RMSE 게이트를 통과한 후보로 완료 보고한 점은 검증 정책의 허점이었다. 이를 단순히 ‘남은 한계’로 축소해서는 안 된다. 또한 ‘스타터 방식과 일치’라는 이전 설명은 audio 요약 helper 범위에 한정됐어야 했으며 전체 Test 합성법까지 검증됐다는 뜻으로 읽히면 부정확하다.

## 2. 실제 학습 모델·행 수·Fold·특징 전부 공개

최종3계열은 LGB·DINOv2-small·Dasheng-Base다. **이번 R18 새 K-Fold는0회**, D0/H0 각각 단일 source split이다. 각 regime TRAIN60,000행은 **12,000개 고유 배경×5행(무추가1+추가4)**이다. D0/H0는 배경을 공유하며120,000개의 독립 배경이 아니다. 465개 참조를 사용하고 B19/C′25는 예약했지만 동일 선박/날짜 공유 가능성이 남는다. clean_ES/fresh_fit 각 regime2,000행·400배경이며 clean_ES는 조기종료, fresh_fit는 배합 선택용이다.

{table(training)}

신경망은 공개 backbone을 동결하고 Stem/Head 및 stage2 LoRA/LayerNorm을 학습했다. stage1 3epoch, stage2 최대25epoch, ES7이다. R17 비교 LGB는 기존5-Fold 평균이지만 R18은 단일 분할이고, R17 신경망도 단일fold 모델이다. 비교에는 이 학습범위 차이가 포함된다.

### 2.1 실제 입력 및 손실

- LGB 회귀: **relative4,667특징**. 순수 관측/마스크 대리변수를 정리한 시간상대·에너지·물리 통계 기반 직접 회귀.
- LGB 분류: **profile_energy2,408특징**. 관측 비율25개를 제외하고 상대 주파수 프로파일25개 및 sqrt(w)375개를 복원했다. 순수 관측 비율 제외가 모든 간접 missingness 신호 제거의 수학적 보증은 아니다.
- NN 회귀·독립 분류: 둘 다 **17×15×25** 입력. 서로 다른 모델 객체/파라미터/그래디언트를 쓰며 분류에 Ch4·Ch7을 포함한다.
- NN 회귀는 `q=sigmoid(mask_logits)`, `share=sum(w*q)` 및 `share MSE+0.25*energy-weighted cell MSE`. 분류는 독립 BCE. 마스킹은 학습에서만 사용했고 추론에서 껐다.
- B1은60/25/15, B2는50/30/20(LGB/DINO/Dasheng). 각 계열 D0:H0=50:50. score도 같은 비중의 독립 분류 확률이며 share 혼합0. 후처리는clip[0,1]만 사용했다.

17채널 원문 명칭:

{table(channels)}

전체 Tree 특징명·인덱스는 [tree_feature_dictionary.csv](predictions/tree_feature_dictionary.csv)에 **회귀4,667·미사용보조회귀4,679·분류2,408·제거된회귀493·제거된분류25의 스키마별 목록**으로 저장했다. 보조회귀4,679는상대4,667에12개보조를더한것이다. 공개 목록의 중복 이름은 서로 다른 schema다. 코드와 생성 당시 summary SHA도 검증했다.

### 2.2 공통 레벨/배경 특징의 실제 기여

{table(aux)}

추가12특징 branch는 clean_ES에서 더 나빠 두 regime 모두 제외됐다. 따라서 실제 Test 제출 예측에서 이12개 특징이 이동시킨 양은 **0**이다. aux 모델을 별도로 Test에 다시 추론하는 반사실 실험은 하지 않았다. 미사용 branch라서 이번 실패의 직접 원인으로 볼 수 없다는 결론과, 원시 절대 레벨이 미래에 유용할 수 있는지는 별개다.

### 2.3 개별 모델의 검증 성적

아래는 **같은 입력에 D0/H0 모델을 모두 적용한 후50:50**한 지표다. clean_ES의 D0/H0 자체 데이터는 서로 입력·정답이 다르므로 이처럼 평균하지 않았다. clean_ES는 저장된 own-regime 예측만 따로 제공하고, 저장되지 않은 old/cross-regime 성적을 만들어내지 않았다.

{table(individual)}

[전체1,110행 모델×regime×평가구간 지표](predictions/all_saved_prediction_metrics.csv), [모든학습작업42행 provenance](predictions/model_training_provenance.csv), [clean_ES 가능/불가 목록](predictions/clean_ES_saved_prediction_availability.csv).

## 3. 무추가·약한 양성 오차의 정확한 해부

bias=`mean(prediction−truth)`다. 양수는 해당 **라벨이 있는 합성 평가셋에서** 과대예측이다. native12 패널의 원래 양/음성50:50 가중치를 보존했으며 행 수가 곧 평가 가중치는 아니다.

{table(weak)}

report에서 약한 양성은1,070/2,400행이다. 전체 개선이 이 구간 개선을 뜻하지 않는다. `0<y≤.03`, `.03<y≤.10`을 나눈 값과 개별 모델의 같은 구간 성적도 전체CSV에 제공했다. LGB·DINO·Dasheng의 report `0<y≤.10` RMSE는 각각 **0.029904→0.042140**, **0.030497→0.050506**, **0.038610→0.049835**로 모두 악화했다. 새B1→새B2 비중 변화는 이 구간의 손실을 조금 더 늘렸지만 주된 악화는 구세대→신세대 모델 교체에서 생겼다.

### 3.1 TRAIN이 더 높은 타깃으로 바뀐 것인가?

{table(prior)}

새TRAIN의 평균은 D0 **0.153500→0.123017**, H0 **0.206080→0.152056**으로 낮아졌다. 낮은 양성 비중도 증가했다. 따라서 ‘새TRAIN이 고share 위주여서 Test 예측 평균이 상승’이라는 단순 사전분포 설명은 지지되지 않는다. targetSSE에서 y>.20의 비중은 여전히D0약90.30%, H0약93.15%지만 이것은 상수0 예측의 target 에너지이며 실제 학습 loss 기여율과 같다는 뜻은 아니다. 입력-정답 조건관계·장시간/채널 law·모델 근사오차의 변화는 남는다. 원천/레시피를 함께 바꿨으므로 이 감사만으로 각 요인의 인과효과를 분리하지 않았다.

### 3.2 분류 AUC와 확률 포화를 구분

[약한 양성 대 음성 순위·FPR 전수표](predictions/zero_vs_weak_classifier_rank_and_fpr.csv). report의 `y=0 + 0<y≤.10` 분류 AUC도 **0.955140→0.970175**로 개선했다. 이 구간의 로컬 순위가 무너진 것은 아니다. 반면 report 음성에서 score≥.9 비율은0→1.25%, HF 음성에서는3.34→6.41%로 증가했다. 높은 확률 비율은 임계값/보정 문제를 보여줄 수 있지만, 그 자체로 ROC-AUC 저하의 증명은 아니다. 확률이 높아도 순위가 맞으면AUC는 높을 수 있다.

Public AUC0.573555555는 실제 공개부분의 순위 구분이 약하다는 결과다. 로컬 AUC와의 차이는 여전히 **현 합성 검증이 실제 추가/무추가 구분을 대표하지 못함**을 보여준다. Test에 라벨이 없으므로 실제 음성의 score 평균·HF 분포·FPR를 계산했다고 주장하지 않는다.

## 4. Test1,500행 저장 예측의 기술 통계

아래 표는 실제 제출과 동일하게 각 계열 D0/H0 예측을50:50으로 결합한 값만 표시한다. D0/H0 개별 출력도 [전체 기술 통계](predictions/query_distribution_and_oldB1_correlations.csv)에 보존했다.

{table(query[qcols])}

R18 B2−R17 B1 share 평균차는 **+0.02241122**다. 같은B1 비중으로 모델만 바꾼 효과는+0.02267024, 새B1→새B2 비중 효과는−0.00025902다. 즉 평균 상승의 대부분은 모델 세대 교체 효과이고, 신경망 비중을40%에서50%로 올린 효과는 평균을 아주 조금 낮췄다. 세 모델 모두 신세대 평균이 높아졌다.

{table(effect)}

표의 `mean_square_prediction_change_contribution`은 라벨 없는 **예측 제곱평균 변화**이며 Test MSE 기여도가 아니다. Public 오류의 모델별 기여도로 오인하면 안 된다. score 평균0.883053→0.952696, score≥.9 비율71.13%→87.27%로 높아졌다. 전체 입력의 분포 변화와 약한 라벨 평가의 양의 bias는 문제의 단서지만, 실제 Test 평균오차를 계산하거나 예측을 이 비율로 깎지 않았다.

## 5. 원본 SA 전수 계보: 95% 폐기 주장의 검산

원본은 **1,715,779개 관측 행**, 모델 입력은 **15분 창**이다. 원본171만행 모두 `sa_rows_db.npy`에 저장됐고 기존창에 들어간 행은1,698,184개(98.9745%)다. 원본 삭제와 창에 미포함은 다른 작업이다.

{table(row_reasons)}

기본 경로는 `pipeline/prepare_sa.py:38–81`이다. `finite`, sentinel−156.5 제외, duration55..65초, timestamp초≤5, 분중복 우선순위,15분 고정clock 및 관측≥3분을 썼다. **고주파 bump를 기준으로 창을 제거하는 조항은 없다.** 로드된 `qc_25band_spread_ge6` flag도 usable 식에는 쓰지 않는다. `negative_level`은 기록용flag다. 임의로 모든음수값이 삭제됐다고 해석하면 안 된다.

원래15분창은133,641개다. source quality 주별split/24h purge 이후 R9 허용풀100,965개를 구성했고, Fold0 TRAIN은78,869개다. 나머지는 단순폐기만이 아니라 평가/보정/domain/purge 역할로 분리됐다.

{table(reasons)}

R18 실제TRAIN은 이 계보에서12,000개 배경을 고른 뒤 각각5개의 합성 예제를 만들었다. ‘R18은78,869개 배경을 모두 사용했다’고 할 수 없다. 사이트 균형·고밀도/희소 quota 등 뒤단 선택도 남아 있다. [단계별단위·개수](background/stage_counts_units.csv), [사이트·녹음기·제외이유 전수표](background/window_lineage_site_recorder_selection_reason.csv).

### 5.1 같은 창의 minute QC 전후를 비교

기존 boundary flag는 recorder별 run 시작/끝의 **broadband 중앙 초과량>3dB**를 근거로 생성됐다. 이는 원인이 확인된 기계오류 라벨이 아니며 원천파일에 그대로 보존돼 있다. R9에서는 해당 분만NaN 마스킹하고 창 전체를 지우지 않는다. 같은 TRAIN78,869창에서4,453분이 마스킹됐다.

{table(qc)}

{table(transitions)}

위 표는 전체 풀 집계다. 사이트·녹음기별 같은 창의 QC 전후 지표는 [minute_QC_paired_summary.csv](background/minute_QC_paired_summary.csv)에 모두 보존했다.

TRAIN 동일창의 joint조건(HF≥16dB & LF fraction≤.58)은 **104→110**, 소실0·생성6이었다. HF bump p50은0.908729→0.833731, p90은10.497929→10.337696으로 변했다. LF bump p90은21.998814→12.915881로 크게 낮아졌으므로 QC가 완전히 무영향인 것도 아니다. 하지만 해당 joint조건이 대량 삭제됐다는 설명과는 맞지 않는다. 기존 R4의 과거창전체제외와 현재R9 분마스킹을 혼동하지 않았다.

## 6. 필터 없는 원본15분 창의 전수 검색

새 감사는 clock stride5로 모든 nonempty 창을 구성하고, 파일경계 금지와 같은recorder 실제시간 연결을 각각 검산했다. 연결정책410,571창/그중파일교차2,028창, 파일경계금지412,600개이며 중복clock key2,029개를 별도 기록했다. gap은NaN이고 서로 떨어진 행을 억지로 이어붙이지 않았다. 겹치는 stride5 창은 독립 녹음표본이 아니다.

HF bump는3.162..7.943kHz band20..24 선형power합의 분별dB max−median, LF bump는31.6..501.2Hz band0..12의 같은 정의다. **LF energy fraction은31.6..199.5Hz band0..8** 비중이므로 LF bump와 범위가 다르다. 저장 Test1500행에partial-band minute0임도 확인했다.

전체p50 비교:

{table(headline)}

‘Test와 비슷한가’라는 설명에는 정확한 조건과 분모가 필요하다. 다음 절대조건은 모양 비교용이며 라벨/학습선택규칙이 아니다.

{table(shapes)}

무필터전체는 sentinel/짧은분/경계이상을 포함해2,118창이지만, 관측15분 모두기존 eligibility를충족하는256,029창에서는 **379창(0.148%)**이다. R18 실제12,000배경은18창(0.150%)이다. 비슷한 배경이0개인 것은 아니며, 유효전수 분포에서도 희귀하다.

### 6.1 유효15분 유사조건이 존재하는 사이트

{table(site)}

### 6.2 녹음기별 전수표

{table(group)}

조건별 월·시간·관측밀도·파일경계·선택사유 전수는 [absolute_shape_conditions_all_groups.csv](background/absolute_shape_conditions_all_groups.csv), [feature_quantiles_all_groups.csv](background/feature_quantiles_all_groups.csv)에 있다. 유사한 원본창을 실제 Test 생성 원천으로 역추정하거나 매칭하지 않았다.

### 6.3 Test 무추가 음성에 대한 해석 제한

Test423/1500=28.2%는 **전체 혼합입력**의 joint조건 빈도다. Test 음성만의 HF median이나 LF fraction을 뜻하지 않는다. 양성의 추가 성분만으로도 HF bump 증가·LF비중 감소가 가능하다. 그러므로 이 값으로 ‘Test음성도 대부분HF16dB’ 또는 ‘양성비율이몇%’를 알 수 없다. 원본 SA도 자연 선박 부재가 검증된 라벨셋이 아니다. 원본을 **추가 없이** 합성대조군으로 쓰면 added-share=0이라는 정의는 성립하지만, 자연 선박 소리가 없다는 뜻은 아니다.

## 7. 공식 스타터 원문과 현재 합성기의 대조

로컬 공식ipynb는12셀, 정의된 함수는마지막cell11(0-based)의 `minute_band_power`28줄뿐이다. **배경 선택·참조 추출·SNR/스케일링·혼합·label 생성의 완전 합성 함수는 없다.** [공식전셀 원문](starter/OFFICIAL_STARTER_FULL_SOURCE.md), [실제코드 행번호 발췌](starter/LOCAL_CODE_SOURCE_EXCERPTS.md), [1:1 상세대조](starter/R18_PUBLIC_STARTER_FORENSICS.md)에 모든 근거를 제공한다.

| 항목 | 공식 공개 근거 | 실제 R18 구현 | 판정 |
|---|---|---|---|
| 배경 선택 | SA source 자료 | 자체source group/24h purge/site round-robin12,000 | 공식법칙 미공개 |
| 참조 선택 | VTUAD v1.3.2 유래 | 465개·길이층·clip cap·예약44개 | 공식 clip law 미공개 |
| WAV→25band | helper 명시 | 2초Hann PSD **적분합×df**,30개선형평균 | 핵심수식검산 통과 |
| PCM/dB 기준 | helper는 입력을float64로cast,Pa교정안함 | decoderfloat32/channel0,SA는source-export dB | helper일치와압력교정은별개 |
| 디지털 gain 폐기 | 설명에 명시 | 상대배경power와peak SNR alpha | 공식 alpha식 미공개 |
| 시간support | 시간궤적 변형 설명 | minute-first,crop,정수분inside배치,D0/H0 | 공식 law 미공개 |
| EQ/SNR | 채널·에너지 변형 설명 | 아래6law·범위±6/±12등 | 공식 분포·순서 미공개 |
| label | controlled addition의energy fraction | O적분raw A/(A+B),그후db0.1round | 의미와일치,정확순서 미공개 |

앞4law는 native전체/EQ후 peak로alpha를정한뒤 crop, 뒤2law는 삽입crop/O반영/EQ전 peak로alpha를정한뒤 EQ를가한다. 뒤2law는후EQ 재정규화가 없다. **같은SNR숫자가 같은실현share를 뜻하지 않는다.** 순수결정론예시에서requestedSNR0dB인데control share0.001996008,inserted-preEQ share0.166666667이었다. 이는 로컬법칙 차이의 예시이며 Test 정답/분포 추론이 아니다.

공식helper와독립FFT 최대절대차4.90059e−17, 실제6연산자×3조건18건 수식parity 통과, 기존509WAV/1,978완전분 전수검산의11pin도현재SHA와일치했다. 기존최대유의dB차는약5.52e−6dB였다. 따라서 대형PSD평균/적분 버그나변환순서오류를이번 공식실패의확정원인으로볼근거는없다. 숨은Test 합성기 전체와동등함을입증한것은아니다.

## 8. 현재 가장 근거 있는 해석과 남은 경계

**검증 설계의 대표성이 확보되지 않았다.** clean_ES/fresh_fit는배경을분리했지만같은465개참조와우리6law를공유한다. 미관측사이트도같은혼합기조건이며 고정17개패널은누적개발에반복사용됐다. 이패널성능개선은 실제공식문제로전달된다는보증이아니었다.

**회귀 선택의명백한구멍은저share 손실을막지못한것이다.** `PLAN.json`의gate는전체RMSE4축이며 무추가·0<y≤.10의비열화조건이없었다. report는동결후확인만하므로B2를다시고르면또선택누수가된다. 이번감사에서도정책/CSV는수정하지않았다.

**AUC 차이를특정QC/특징하나로설명하기는어렵다.** 원음465복구·마스크균형·profile_energy복구후에도 Public AUC는0.572711→0.573556이었다. 반면로컬약한양성구분AUC는오히려개선됐다. 자연SA과도음과추가VTUAD 성분의조건부스펙트럼/시간/세기/지원분포를우리label문제가실제와다르게구현했을가능성을우선검토해야한다. 하지만 어느차이가주원인인지는현재비공개라벨과recipe없이확정할수없다.

요청대로추가학습을멈췄다. 다음학습을정당화하려면 (1)합성법칙별전력/라벨가정명시,(2)평가법칙까지같은함수를공유하는관문과별개인사전고정검산,(3)전체·음성·저share·장시간오차를함께통과하는채택조건,(4)검증패널반복사용이력을명시하는절차가먼저필요하다. Test 평균/분위수에맞춰생성분포나출력을조절하거나Public점수로타깃모멘트를역산하지않는다.

## 9. 산출물과 재현

- [저장예측·모델·약한양성 상세보고서](predictions/PREDICTION_FORENSICS.md)
- [SA 원본전수 계보·분포보고서](background/SA_PUBLIC_BACKGROUND_FORENSICS.md)
- [동일창 minute QC 전후보고서](background/MINUTE_QC_PAIRED_EFFECTS.md)
- [공식helper·6law 상세대조](starter/R18_PUBLIC_STARTER_FORENSICS.md)
- [공식결과 원문기록](OFFICIAL_RESULT.json)

재현은 `ops/r18_public_prediction_forensics.py`, `ops/r18_public_background_forensics.py --output 새폴더`, `ops/r18_public_starter_forensics.py`, 마지막에 `ops/r18_public_gap_audit_report.py` 순서다. 환경은 `.venv_r17_eda/bin/python`이다. 독립 검산·출처·전체 행표·SHA는각하위폴더에보존했다. 원본데이터·모델·기존제출CSV는변경하지않았다. 전역산출물hash는 `ARTIFACT_SHA256.json`에저장한다.
'''
    target = OUT / 'R18_PUBLIC_GAP_AUDIT.md'
    target.write_text(text)
    css = '''body{font:16px/1.75 system-ui,sans-serif;max-width:1450px;margin:32px auto;padding:0 28px;color:#17324d;background:#f4f7fb}main{padding:30px;background:white;border:1px solid #d8e3ee;border-radius:16px}h1,h2,h3{color:#17466f}h2{margin-top:40px}table{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}td,th{border:1px solid #d5e1ed;padding:7px 10px;white-space:nowrap}th{background:#eaf2fa}code{background:#eef3f8;padding:2px 4px}pre{overflow:auto;background:#eef3f8;padding:16px}a{color:#0865b4}'''
    nav = '<p><b>R18 공식 결과 감사 · 추가 학습 중지</b> · <a href="R18_PUBLIC_GAP_AUDIT.md">전체 MD</a> · <a href="predictions/PREDICTION_FORENSICS.md">저장예측</a> · <a href="background/SA_PUBLIC_BACKGROUND_FORENSICS.md">SA 전수</a> · <a href="starter/R18_PUBLIC_STARTER_FORENSICS.md">스타터 대조</a></p>'
    (OUT / 'START.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>R18 공식 실패 · 전수 감사</title><style>' + css + '</style><main>' + nav + markdown.markdown(text, extensions=['tables', 'fenced_code']) + '</main></html>')
    inputs = [x for x in OUT.rglob('*') if x.is_file() and x.name not in {'ARTIFACT_SHA256.json', 'AUDIT_COMPLETED.json'}]
    hashes = {str(x.relative_to(ROOT)): {'sha256': sha(x), 'bytes': x.stat().st_size} for x in sorted(inputs)}
    (OUT / 'ARTIFACT_SHA256.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + '\n')
    done = {'status': 'completed', 'created_KST': now, 'report': str(target.relative_to(ROOT)),
            'report_sha256': sha(target), 'artifact_manifest_sha256': sha(OUT / 'ARTIFACT_SHA256.json'),
            'files': len(hashes), 'new_model_training': False, 'new_model_inference': False,
            'raw_Test_reopened': False, 'Test_labels_or_LB_inversion': False,
            'Test_based_calibration': False, 'Pod_stop_user_confirmed': True}
    (OUT / 'AUDIT_COMPLETED.json').write_text(json.dumps(done, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(done, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
