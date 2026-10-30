"""Assemble evidence-backed Korean R18 audit; no Test input or model fit."""
from pathlib import Path
import datetime, hashlib, html, json, math, re, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r18_fundamental_audit_r1'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads((OUT/p).read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(df):return df.to_markdown(index=False,floatfmt='.6f')

def main():
    # Existing code, models and candidate CSVs are inputs, never replaced.
    for name in ('identifiability/COMPLETED.json','background/COMPLETED.json','classifier/COMPLETED.json','wav/RESULTS.json'):
        if not (OUT/name).exists():raise FileNotFoundError('Await completed experiment '+name)
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    public=[
        {'model':'R5 CNN','Public':.65199132,'RMSE':.348008679,'AUC':.539533333},
        {'model':'R9 Tree5','Public':.768128675,'RMSE':.231871324,'AUC':.548088888},
        {'model':'R9 all5','Public':.782464193,'RMSE':.217535806,'AUC':.555066666},
        {'model':'R14 step1','Public':.705436197,'RMSE':.294563802,'AUC':.497866666},
        {'model':'R16 Tree5','Public':.776856976,'RMSE':.223143023,'AUC':.568777777},
        {'model':'R16 V1 원본','Public':.775233744,'RMSE':.224766255,'AUC':.568777777},
        {'model':'R17 B1 (wSNRSQBH)','Public':.789471136,'RMSE':.210528863,'AUC':.572711111},
        {'model':'Kem_Astr (사용자 전달 1위)','Public':.966888119,'RMSE':.03311188,'AUC':.8878},
    ]
    pubdf=pd.DataFrame(public)
    save(OUT/'PUBLIC_RESULTS_USER_REPORTED.json',{'source':'User-provided screenshots and message; not live leaderboard revalidated',
        'recorded_KST':now,'B1_candidate_sha256':sha(ROOT/'submissions/r17_fixed_blends_20261030/submission_r17_B1_L60_DINO25_Dash15.csv'),
        'Test_label_recovery_or_aggregate_moment_inversion':False,'results':public})
    pubdf.to_csv(OUT/'PUBLIC_RESULTS_USER_REPORTED.csv',index=False)
    reg=pd.read_csv(OUT/'identifiability/metrics.csv')
    r=reg[reg.trained_regime.eq('D0_H0_50_50')]
    regtab=r.pivot(index='panel',columns='mode',values='rmse').reset_index()
    regbias=r.pivot(index='panel',columns='mode',values='bias').reset_index()
    clf=pd.read_csv(OUT/'classifier/metrics.csv')
    # Schema is defined by the classifier owned script, not assumed silently.
    regimecol=next(c for c in ('trained_regime','regime') if c in clf)
    cl=clf[clf[regimecol].astype(str).isin(['D0_H0_50_50','D0_H0_same_input_50_50','mean_D0_H0','D0H0'])]
    if cl.empty:raise ValueError('Classifier 50:50 rows not identified')
    modecol=next(c for c in ('mode','feature_group') if c in clf)
    ct=cl.pivot(index='panel',columns=modecol,values='auc').reset_index()
    bg=pd.read_csv(OUT/'background/background_reconstruction_metrics.csv')
    bt=bg[bg.pooling.eq('integral_clip') & bg.estimator.isin(['window_q20','learned_totalB_raw_summary','oracle_true_source_B'])]
    bgtable=bt.pivot(index='panel',columns='estimator',values='rmse').reset_index()
    inv=read('background/SA_FULL_INVENTORY.json');wav=read('wav/RESULTS.json');fixture=read('identifiability/mathematical_fixtures.json')
    delta=.789471136-.782464193;delta_v1=.789471136-.775233744
    ratio=.210528863/.03311188
    # Two complementary toys: raw level can be informative with calibrated B,
    # yet neither representation uniquely determines an arbitrary B+A split.
    toy={'calibrated_known_B_100':[
        {'E':100.,'B':100.,'A':0.,'y':0.,'db_relative_to_E100':0.},
        {'E':200.,'B':100.,'A':100.,'y':.5,'db_relative_to_E100':10*math.log10(2)}],
        'normalized_grid_equal_for_constant_across_all_cells':True,
        'same_raw_E_200_latent_possibilities':fixture['same_raw_db_different_y_counterexamples'],
        'equal_probability_y0_y05_toy_Bayes_RMSE':.25,
        'toy_floor_is_not_real_dataset_floor':True,
        'leader_MSE_user_reported':.03311188**2,
        'B_integral_identity':'yhat-y=-(Bhat_total-Btotal)/Etotal before clipping',
        'necessary_condition_for_MSE_target':'E[Var(y|observed deployable inputs)] <= requested MSE; not a measured value'}
    save(OUT/'IDENTIFIABILITY_AND_PRECISION.json',toy)
    sections=[]
    def add(title,body):sections.append((title,body))
    add('결론',
        'B1은 공식 최고점을 경신했지만, 데이터 생성 조건의 실전 전달 문제가 해결됐다는 증거는 아니다. 이번에는 대형 모델을 다시 학습하지 않고 '
        'SA 원본 전수 집계, 509개 WAV 변환 전수 검산, 소형 회귀 6개·분류 6개·배경 적분 회귀 1개를 실제 실행했다. '
        '**절대 dB 복구만으로 로컬 RMSE 0.02~0.03에 도달하는 결과는 얻지 못했다.** 반면 분류기에 상대 주파수 프로파일·에너지 분포를 복원하는 것은 여러 교차 패널에서 유효했다. '
        '현재 오차 0.07을 Bayes floor로 부르거나, 원시 dB와 사전학습 대형 모델을 넣으면 1위 수준이 된다고 판단할 근거는 없다.')
    add('1. 공식 결과와 비교 범위',table(pubdf)+'\n\n'
        f'B1의 Public 향상은 R9 대비 **+{delta:.9f}**, R16 V1 대비 **+{delta_v1:.9f}**다. '
        f'사용자가 전달한 1위 RMSE와의 비율은 **{ratio:.2f}배**, MSE 비율은 **{ratio**2:.2f}배**다. '
        '서로 다른 정답 분포인 자체 합성 검증 RMSE와 비공개 Public RMSE는 난이도가 같지 않으므로 수치만으로 모델의 이론적 한계를 비교할 수 없다. '
        'Kem_Astr의 내부 생성법·배경 복원법·모델은 공개 점수만으로 알 수 없다. 새 1위 값은 사용자 전달 자료로 기록했고 실시간 리더보드를 재검증했다고 표시하지 않는다. '
        '상수 제출과 점수를 연립해 숨겨진 정답 평균·양성률을 역산하지 않았다.')
    add('2. 17채널은 무엇을 잃었고 무엇을 보존했나',
        '`Ch0=(db-median_t(db))/10`, `Ch4=(median_t(db)-median_tf(db))/20`이므로 관측 셀에서 '
        '**`10*Ch0+20*Ch4 = db-median_tf(db)`**다. 또한 `Ch7=sqrt(w)`에 대해 `20log10(Ch7)=db-C`이다. '
        f'실제 구현 대조 최대 복원 오차는 각각 **{fixture["profile_and_temporal_reconstruction_max_db_error"]:.3g} dB**, '
        f'**{fixture["sqrt_w_reconstruction_centered_db_max_error"]:.3g} dB**다. 즉 회귀 입력은 대역 간 에너지 분포를 지우지 않았고, 주로 공통 절대 레벨 한 자유도를 제거했다.\n\n'
        'B와 A를 같은 배율로 바꾸면 y는 불변이다. 하지만 이것이 절대 dB가 항상 쓸모없다는 뜻도 아니다. '
        '**교정된 B=100을 이미 안다면 E=100/y=0과 E=200/y=0.5는 정규화된 평탄 입력은 같아도 절대 레벨이 3.0103dB 달라 구분된다.** '
        '반대로 E=200만 알면 B=200/A=0과 B=100/A=100을 구분할 수 없다. raw dB도 믿을 수 있는 배경 prior 없이는 성분 분해의 정답을 주지 않는다.\n\n'
        '분류 15채널은 Ch4/Ch7을 제거하므로 정적 대역별 오프셋에 불변이다. 일정한 배경에 일정한 대역별 추가 성분이 붙는 반례에서 '
        f'분류 입력 차이는 **{fixture["stationary_band_addition_classifier_feature_collision_max"]:.1f}**지만, 회귀의 프로파일과 가중치는 달라졌다. '
        '이는 표현의 가능한 사각지대를 보인 수학적 반례이며, 실제 Test에서 이런 사례가 얼마나 있는지는 알 수 없다.')
    add('3. 상대 입력 vs 원시 dB 추가: 실제 소형 회귀 대조',
        'D0/H0 각각 12,000행 TRAIN, 2,000행 ES, 고정 seed·leaf·학습률·최대 1,000트리/ES50로 3입력×2체계를 비교했다. '
        '매 평가 입력에 D0/H0 모델을 둘 다 적용해 50:50 평균했다. '
        '`relative17`은 전체 17채널 6,375개 셀 값, `relative17_common_level`은 창 중앙 dB 1개 추가, '
        '`relative17_raw_db`는 원시 dB 375개 추가다. 사이트·녹음기 ID나 정답 q/B는 입력하지 않았다.\n\n'
        +table(regtab)+'\n\n예측 편향:\n\n'+table(regbias)+'\n\n'
        +'TRAIN·ES 대조:\n\n'+table(reg[reg.panel.isin(['train','earlystop'])][['mode','trained_regime','panel','rmse','bias']])+'\n\n'
        '이 대조는 **같은 작은 트리 함수공간에서 정보 추가의 효과**를 측정한다. 기존 사전학습 신경망과 용량·자료 수가 다른 만큼 절대 수치를 '
        '신경망 성능 상한으로 해석하면 안 된다. 수많은 후보를 돌려 최소값을 고르거나 Test로 레시피를 조정하지 않았다. '
        'TRAIN 오차와 purge된 평가 오차를 함께 저장했으므로, 훈련 데이터에서 잘 맞는 현상과 새로운 배경 일반화를 구분할 수 있다. '
        'site_macro·고비율·음성평균은 비가중 보조 지표이고 장시간 전체RMSE는 원래50:50 평가가중치를 사용한다. '
        '원천/24시간 분리는 기존R17의검증자료 및 background/BACKGROUND_SOURCE_PURGE.json에 근거한다.')
    add('4. SA 171만 행으로 배경 B를 복원하면 바로 0.03이 되는가',
        f'**{inv["rows"]:,}행 ×25대역** 전체 집계를 완료했다. 적격 행 {inv["eligible_rows"]:,}개, '
        f'{inv["sites"]}사이트·{inv["recorders"]}녹음기 문자열·{inv["sensor_groups"]}센서 그룹이다. '
        '각 사이트/녹음기/월/시간별 집계 및 대역 p10/p50/p90를 별도 CSV로 제공한다. 원본 export 시계의 UTC/현지 시각은 검증되지 않아 계절·시간 해석에 한계가 있다.\n\n'
        '모든 171만 행으로 prior를 적합한 다음 그 안에서 잘라낸 로컬 창을 평가하면 배경 원천 누수가 된다. '
        '따라서 전체 자료는 전수 기술 통계에 사용하고, 복원기는 Fold0 TRAIN 61,446개 배경 창과 합성 TRAIN만 사용했다. '
        '새 ES/fresh-fit은 최소 24h purge, gate는25.5h, report는25h 분리했으며 미관측 사이트는 센서를 공유하지 않는다. '
        '사이트/녹음기/월/시간 prior는 진짜 원천 메타데이터를 알려준 유리한 진단 대조다. Test에 이런 정답 메타데이터가 제공된다고 가정하지 않는다.\n\n'
        +table(bgtable)+'\n\n'
        '`oracle_true_source_B`는 원래 배경을 정확히 알고 계산한 비현실적 대조이며 실제 추론기가 아니다. '
        '오차가 약0.0003~0.0004인 것은 반올림 자체가0.03 달성을 막는 큰 수치 오차가 아니라는 검산이다. '
        '`learned_totalB_raw_summary`는 합성24,000행의 정답 B 적분을 학습한591특징·최대500트리 대조이고, '
        '배경 셀 자체의 정답은 추론 입력에 쓰지 않았다. 단순 q20·사이트/녹음기 평균·PCA prior도0.03을 달성하지 못했다.\n\n'
        'clip 전에는 `yhat-y=-(Bhat_total-Btotal)/Etotal`이다. 따라서0.033 수준의 정확도를 배경 적분 복원으로 달성하려면 '
        '**배경 총에너지 오차/E총에너지의 RMS가 약0.033 이하**여야 한다. 이번 평가 분포에서 share오차0.03은 '
        '배경 총량 약3.1~3.2%, 약0.13~0.14dB의 오차에 대응한다(개별 샘플·y에 따라 다름). '
        '실제 학습 복원기의 총량 dB RMSE는 약0.90~1.13dB였다. 사이트 기후 평균과 개별15분 창의 counterfactual B는 같은 정보가 아니다. '
        '또 `sum(max(E-Bhat,0))/sum(E)`와 `clip(1-sum(Bhat)/sum(E),0,1)`는 서로 달라 둘 다 별도 표로 측정했다.')
    add('5. WAV→25대역 수식과 합성 순서 전수 검산',
        '공식 스타터 helper를 독립 추출하고 PCM 스케일·실제 fs·채널·프레임 수·SHA를 확인했다. '
        '**2초 Hann periodogram, DC 제거, 단측 PSD density → base-10 deci-decade(중심비10^0.1) 대역 경계 안의 PSD 합×Δf → 완전한60초의 선형 평균** 순서다. '
        '셀 dB를 평균하거나 PSD의 대역 내 평균을 대역파워처럼 쓰는 방식과 다르다.\n\n'
        f'- 원본 WAV **{wav["raw_WAVs_SHA_and_header_verified"]}개** SHA/header 확인, 전체 길이 **{wav["recomputed_complete_minutes"]:,}분** 공식 helper 전수 재계산.\n'
        f'- 공식 helper 대비 저장 캐시 최대 유효 상대 오차 **{wav["maximum_significant_relative_helper_vs_cache_error"]:.3g}**, '
        f'dB차 **{wav["maximum_significant_abs_dB_helper_vs_cache_error"]:.3g}dB**.\n'
        f'- 전체509개 native 2초→1분 캐시 검산 최대 상대 오차 **{wav["all509_native_vs_minute_cache_max_relative"]:.3g}**.\n'
        f'- 독립 FFT/Parseval 수치 오차 **{wav["independent_FFT"]["parseval_error"]:.3g}**.\n'
        f'- 25대역 전체에 필요한 최소 fs 약17,826Hz. VTUAD 원본 전체는 해당조건을 충족했다. SA파일명의12kHz추정이 실제fs라면 상위2대역은지원되지않으며 실제export처리는미검증이다.\n'
        f'- VTUAD LF에너지비중 중앙값: 올바른 적분 **{wav["LF_fraction_p50_integrated"]:.5f}**, '
        f'잘못된 PSD평균 가정 **{wav["LF_fraction_p50_if_wrong_PSDmean"]:.5f}**. '
        f'HF는 각각 **{wav["HF_fraction_p50_integrated"]:.5f} / {wav["HF_fraction_p50_if_wrong_PSDmean"]:.5f}**. '
        '따라서 PSD평균 오류가 있었다면 큰 차이가 났겠지만, 현재 구현에서 그 오류는 발견되지 않았다.\n\n'
        '공식 스타터 자체가 **디지털 음원은 uncalibrated mean-square이며 SA export의 실제 압력교정·주파수 응답 재구성이 아니다**라고 설명한다. '
        'VTUAD PCM과 SA dB를 같은1µPa기준 절대 압력이라고 연결하면 안 된다. 현재 합성기는 참조 절대 디지털 이득을 버리고 별도α로 스케일한다. '
        '주변음을 제거한 순수선박 stem이 아니라 측정녹음 전체가 추가 성분이며, 분리된 ambient 정답을 알고 있지 않다.\n\n'
        '**문제의레이블은“이창에배가있는가”와같지않다.** 원래SA배경에이미존재한선박·생물·기계과도음은추가성분이아니므로y=0일수있다. '
        '외부음원전체를더했을때의기여비율이회귀정답이며, 독립분류정답은그추가성분의유무다. '
        '음향사전학습모델이선박음자체를잘탐지하더라도추가분리타깃과바로일치하지않는다.\n\n'
        'HF_bump는 **max−median**, share는 **추가 에너지의 관측합/전체합**이다. 장시간 통과는 median도 높여 bump를 줄이면서 '
        '총 추가 에너지는 높일 수 있다. 기존R16 TRAIN의H0 share평균0.20608은D0의0.15350보다 높지만 HF bump중앙값은3.5127 대8.8953dB다. '
        'H0의 낮은 bump를 곧바로“선박을 작게 넣었다”로 해석할 수 없다. '
        'local_gate의 LF비중은 양성뿐 아니라 무추가 배경부터0.950216으로 높아 배경 사이트·장비 구성도 영향을 준다. '
        '기존 Test 기술통계와 D0/H0의 차이는 관측분포 차이로 남지만, 그 값을 맞추도록 generator EQ/SNR를 역적합하지 않았다.')
    add('6. 독립 분류 입력 복원 실측: 시간만 vs 주파수·에너지 포함',
        '같은465개 참조·12k TRAIN/2k ES에서 소형 LGB 분류기6개를 학습했다. temporal2008, '
        'temporal+Ch4/sqrt(w)2408, 거기에raw dB를 더한2809특징을 비교했다. '
        'D0/H0는 같은 입력에서50:50 평균했다. 다음은 실제 ROC-AUC다.\n\n'+table(ct)+'\n\n'
        '상대 주파수 프로파일·에너지 분포 복원은 fresh-fit·일반·미관측·장시간/HF에서 유익하지만 native/SF까지 모두 단조 개선한 것은 아니다. '
        'raw dB는 추가 소폭 이득과 미관측 사이트 반례가 함께 있다. '
        '이것은 **15채널 분류기의 지나친 불변성 제약을 완화할 근거**이지, 실전 AUC0.5727의 단일 원인이 증명됐다는 뜻은 아니다.\n\n'
        'native12/HF의raw분류 AUC는0.869672지만 Brier오차는0.251502로, 확률적합성은해결되지않았다. '
        'temporal2008은 순수관측비율25개만 제외한입력이며 lag/차분의유효성과sqrt(w)=0에는 간접마스크정보가남는다. '
        '새자료의완전O패턴별양성률80%일치를별도로확인했으며, 특징이름삭제만으로maskblind를보장하지않는다. '
        'R17에서15~21%로분류됐던광범위범주중375개는실제음향차분/lag였고, 일괄삭제는장시간HF성능을해쳤다.\n\n'
        'AUC는 `P(score양성>score음성)+0.5P(동점)`이다. 70%가0.9이상이더라도 올바른 순위면 AUC는 높을 수 있다. '
        '따라서 saturation비율과 정확도 붕괴를 동일시하지 않고 Brier·확률분포·음성오탐도 같이 저장했다. '
        '사후 공통gain±20dB 대조도 수행했으며, 원시 dB는 장비이득 변화에 반응하는 반면 상대 프로파일/정규화에너지는 불변이다. '
        '예를들어raw모델의freshH0 +20dB에서AUC는0.952352→0.948628,평균score는0.812045→0.841492였다. '
        '자세한 수치는 classifier 하위보고서와 COMMON_GAIN_STRESS.md에 있다.')
    add('7. 핵심 열쇠와 정공법 재설계',
        '**이번에 지지된 핵심은 “더 큰 모델”이나 “절대 dB만 복구”가 아니라, 생성기에서 배운 `P(y,presence|E)`가 실제 평가로 전달되는지 검증하는 것**이다. '
        '최소한 다음 조건을 만족해야 한다.\n\n'
        '1. **배경의 counterfactual 복원 또는 그에 준하는 직접 비율 추정:** 바다가 원래 갖고 있던 선박형 과도음까지 추가 VTUAD로 오인하지 않아야 한다. '
        'E의 크기만으로 B와A가 분리되지 않으므로 참조 변화·지속시간·전파채널·관측조건을 아우르는 prior가 필요하다.\n'
        '2. **에너지 일관성:** 대역파워 단위와 관측마스크를 유지하며 y를 계산한다. 실제 B를 알아야 성립하는 q상한을 창 내 q05/q20으로 대체해 강제로 자르지 않는다.\n'
        '3. **분류는 출처 서명과 시간 변화를 함께 사용:** 상대 스펙트럼·정규화에너지·안전대역1~5kHz·동조/지속성을 독립 분류에 제공한다. '
        '절대dB는 교정 상태/장비전이가 검증된 보조 branch로 두고, 회귀와 분류 gradient분리를 유지한다.\n'
        '4. **원천 단위 검증:** 같은 배경의 재합성 행을 독립표본처럼 세지 않고 site/sensor·날짜purge·참조hull/day관계를 기록한다. '
        '465개 풀의 ESS확대와 장시간support 감소의 trade-off를 같이 통제한다.\n'
        '5. **대회 규정:** 제공훈련자료/외부음원/스타터 공개변환으로 생성·교정하고 own TRAIN/validation에서만 적합한다. '
        'Test 분포에 EQ/SNR/양성률/예측평균을 맞추거나 AV를 학습하거나 Public 점수로 숨겨진 타깃을 복구하지 않는다. '
        '[공식staff답변 보관본](wav/official_rules_archived.json)의35368/35370에근거하며 이번실시간웹재확인은실패해최신규정확인으로표시하지않는다.\n\n'
        '수학적으로 RMSE0.03311이 가능한 필요조건은 배포 가능한 입력X에서 `E[Var(y|X)]≤0.0010964`와 충분히 작은 추정·전이 오차다. '
        '이 조건부 분산을 현재 자료에서 측정한 것은 아니다. AUC0.8878도 실제 추가 유무의 순위를 약88.8%의 양성-음성 쌍에서 맞혀야 한다는 뜻이다. '
        '이쌍해석은동점에0.5를부여한것이다. 1위 점수로 특정 배경 검색/정확한생성기/대형모델 사용을 역추정하지 않는다.\n\n'
        '**권장 다음 본체 후보:** 기존 검증된 상대 회귀와 독립 분류를 기준으로 유지하되, 분류기에 검증된 Ch4/sqrt(w) 정보를 복원한 소형 경로를 우선한다. '
        '별도의 원시 레벨 + TRAIN 조건부 B 경로는 회귀 개선과 장비 전이가 검증될 때 결합한다. 원시 입력만으로 0.03에 도달한다는 전제를 두지 않는다. '
        '기존 모델과 원본 B1을 보존하고, 소수의 사전 고정 레시피를 원천을 분리한 배경·미관측 장비·별도 참조에서 비교한다. '
        '이번 작업만으로 새 생산 모델이나 CSV를 승격하지 않았다.')
    add('8. 산출물·검증·남은 불확실성',
        '- [배경 복원 전수 보고](background/R18_BACKGROUND_RECONSTRUCTION.md)\n'
        '- [배경 적분/셀/사이트별 전체 수치](background/background_reconstruction_metrics.csv)\n'
        '- [17채널/raw 회귀 대조](identifiability/R18_INFORMATION_AND_REGRESSION.md)\n'
        '- [독립 분류 입력 대조](classifier/R18_CLASSIFIER_INPUT_AUDIT.md)\n'
        '- [공식 WAV 변환 상세 보고](wav/R18_WAV_PHYSICS_AUDIT.md)\n'
        '- [공식 WAV 변환 수치·원천 SHA](wav/RESULTS.json)\n'
        '- [사용자 전달 공식 점수 이력](PUBLIC_RESULTS_USER_REPORTED.csv)\n\n'
        '대형 신경망 학습 0회, 추가 K-fold 0회, 공식 업로드 0회, Test 입력·저장 Test 예측 읽기 0회(기존 기술 통계 인용만), '
        'Test 모멘트 역산 0회다. 배경 전체 집계와 모델 적합 범위를 구분하고 평가 원천 분리를 유지했다. '
        '모든 소형 모델의 protocol·seed·특징 정의·source/model SHA와 TRAIN/ES 오차를 보존했다. '
        '기존 고정 패널은 이미 여러 번 사용된 개발 진단이고, fresh-fit도 학습 참조 풀을 공유하므로 완전한 blind 성능이 아니다. '
        '단순 입력 복원과 단위검산으로 0.02~0.03도달은 입증되지 않았으며 Public/Private성능 보장은 없다.')
    report='# R18 근본 데이터·입력·배경 복원 정밀 검산 및 재설계\n\n작성: '+now+' (Asia/Seoul)\n\n'
    report+='\n\n'.join('## '+t+'\n\n'+b for t,b in sections)+'\n'
    (OUT/'R18_FUNDAMENTAL_REDESIGN_AUDIT.md').write_text(report)
    # Markdown renderer if already available; no installations or remote CDN.
    try:
        import markdown_it
        body=markdown_it.MarkdownIt('commonmark',{'html':False}).enable('table').render(report)
    except ImportError:
        body='<pre>'+html.escape(report)+'</pre>'
    doc='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>R18 근본 정밀 검산</title><style>body{font:16px/1.7 -apple-system,BlinkMacSystemFont,sans-serif;margin:32px auto;padding:0 24px;max-width:1250px;color:#172738;background:#f5f7fb}h1,h2{line-height:1.3}h2{margin-top:44px}table{display:block;overflow:auto;border-collapse:collapse;background:white;font-size:14px}td,th{padding:9px 12px;border:1px solid #d6dde9;white-space:nowrap}th{background:#e7edf7}code{background:#e8edf3;padding:2px 4px}a{color:#145bc3}pre{white-space:pre-wrap}strong{color:#143d73}</style>'+body+'</html>'
    (OUT/'START.html').write_text(doc)
    input_pins={}
    for role in ('train','earlystop'):
        for g in ('D0','H0'):
            p=ROOT/f'data/synthetic/r17_clean_v1/{role}_{g}.npz';input_pins[str(p.relative_to(ROOT))]=sha(p)
    for g in ('D0','H0'):
        p=ROOT/f'data/synthetic/r17_clean_freshfit_v1/ensemble_fit_fresh_{g}.npz';input_pins[str(p.relative_to(ROOT))]=sha(p)
    for p in (ROOT/'data/synthetic/r17_clean_v1/manifest.json',ROOT/'data/synthetic/r17_clean_freshfit_v1/manifest.json',
              ROOT/'pipeline/relative_grid_r6.py',ROOT/'pipeline/relative_grid_r6_v2.py',
              ROOT/'pipeline/training_r16_multisupport.py',ROOT/'ops/r16_temporal_features.py'):
        input_pins[str(p.relative_to(ROOT))]=sha(p)
    save(OUT/'FINAL_INPUT_RECEIPT.json',{'sha256':input_pins,'fresh_fit_pins_added_after_execution':True,
        'note':'Final input identity receipt; training protocol is preserved without retroactive modification.'})
    # Collect only this version and its owned small models/scripts.
    pins={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('SHA256SUMS.json','COMPLETED.json')}
    for glob in ('ops/r18*.py','models/r18_regression_audit_r1/**/*','models/r18_classifier_audit_r1/**/*','models/r18_background_audit_r1/**/*'):
        for p in ROOT.glob(glob):
            if p.is_file():pins[str(p.relative_to(ROOT))]=sha(p)
    save(OUT/'SHA256SUMS.json',pins)
    if any(sha(ROOT/p)!=h for p,h in pins.items()):raise ValueError('Artifact SHA changed')
    links=re.findall(r'href="([^"]+)"',doc)
    local=[p.split('#')[0] for p in links if not p.startswith(('http:','https:','#'))]
    broken=[p for p in local if not (OUT/p).exists()]
    if broken:raise ValueError('Broken report links '+repr(broken))
    save(OUT/'COMPLETED.json',{'status':'completed','recorded_KST':now,'new_small_regressors':6,
        'new_small_classifiers':6,'new_background_integral_learner':1,'huge_neural_training_steps':0,
        'new_KFold':False,'Test_read_or_fit':False,'official_upload':False,'production_policy_changed':False,
        'report_sha256':sha(OUT/'R18_FUNDAMENTAL_REDESIGN_AUDIT.md'),
        'manifest_sha256':sha(OUT/'SHA256SUMS.json'),'verified_files':len(pins),'html_local_links_checked':len(local)})
    print(json.dumps(read('COMPLETED.json'),ensure_ascii=False))

if __name__=='__main__':main()
