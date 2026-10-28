# R8: 공개 SA·과거 Test EDA 근거 목록과 정의 감사

이번 산출물은 **기존 EDA의 모든 주요 지표군을 출처·정의·분모·버전과 연결한 evidence catalog**입니다. 원본 Test·cross/OUTER 관측 배열을 열거나 통계를 다시 계산하지 않았습니다. EDA·kNN·PCA·자연 사건 탐지·모델을 실행하거나 학습하지 않았습니다. 기존 원본·코드·보고서도 수정하지 않았습니다.

[정량 JSON](../../R8_SA_EDA_EVIDENCE.json)에는 305개 지표 항목과 26개 기존 집계 CSV의 전체 저장값을 포함했습니다. 지표는 `value, unit, denominator, source_file, source_section, status, dataset_version, caveat`로 기록합니다. CSV 값은 저장된 정밀도 그대로이며, 원 Test 사례별 값·ID를 새로 분석한 것이 아닙니다.

## 1. 어떤 문서를 읽었는가

- [Documents TASK_EDA_REPORT](../../../TASK_EDA_REPORT.md): SHA `127bf693e833e6b3e5f6291806e6dbba72f67e07fbfc2a6836d7c7d3e81966e3`.
- Desktop `French_Embassy_Underwater_Vessel_Noise/TASK_EDA_REPORT.md`: SHA `c43dad3b358ef95b9087a96e0d1f009eb0042dd50849d065b09fa1bba048ff13`.
- 두 파일은 같지 않습니다. Documents 파일에는 2026-10-26의 고대역 bump 해석 정정 인용문이 추가되어 있고 Desktop 파일에는 없습니다. 이번 감사에서 어느 쪽도 고치지 않았습니다.
- [R5 EDA factcheck](../../R5_EDA_INTEGRATION_FACTCHECK.md), [원천 이상값 감사](../../R5_OUTLIER_AUDIT.md), [경계 후보 감사](../../data_quality_boundary_audit.json), [duty-cycle·전처리 코드 검토](../../R6_PUBLIC_TRAINING_PROTOCOL_CODE_REVIEW.md)를 함께 읽었습니다.

`verified`는 명시한 코드·메타데이터 감사로 지지되는 사실, `datasetdependent`는 특정 필터·버전·집단에서 저장된 통계, `inferred`는 가설·조건부 계산, `incorrect`는 정의와 충돌하는 해석입니다. 기존 결과 파일이 존재한다는 것과 원자료를 이번에 독립 재계산했다는 것은 구분합니다. Test 관련 수치는 모두 **과거 결과의 감사**입니다.

## 2. 원 공개 SA: 규모·스키마·단위

| 항목 | 값 | 분모·의미 |
|---|---:|---|
| 원 관측 행 | 1,715,779 | 모두 보존; 독립 음원·선박 수가 아님 |
| 원 열 / TOL 열 | 74 / 62 | 공통 메타데이터와 전체 대역 스키마 |
| 사이트 / site×recorder / 고유 recorder | 10 / 17 / 12 | recorder ID 재사용이 있어 셋이 다름 |
| export-clock 범위 | 2015-04-21 00:00:00–2025-12-06 23:59:00 | UTC 정합 미검증 |
| 전체 62대역 셀 / NaN | 106,378,298 / 27,102,411 | 약25.4774%; 장비별 대역 가용성 차이가 포함됨 |
| 공통 25대역 셀 / NaN | 42,894,475 / 0 | finite sentinel도 이 분모에 포함됨 |
| 공통 대역 | 31.6–7943.3 Hz | 25개 deci-decade 중심; 정확한 경계는 중심×10^±0.05 |
| 명목 sample rate | 12/24/48/96 kHz | 파일명·export metadata 기반; waveform header 재확인 아님 |
| SA ZIP | 94,916,719 bytes | 94.916719 decimal MB =90.519637 MiB; “90.5MB”는 단위 혼동 가능 |

출처: [기초 EDA summary](../../../eda/summary.json), [site별 전체 통계](../../../eda/tables/site_summary.csv), [recorder별 통계](../../../eda/tables/recorder_summary.csv), [파일 크기·SHA](../../../eda/tables/file_inventory.csv).

[공식 제공 패키지 README](../../../data/extracted/README.md)는 source-export band summary이며, 정답·원시 audio·AIS·Test split membership가 들어 있지 않다고 설명합니다. [data dictionary](../../../data/extracted/DATA_DICTIONARY.md)는 finite flag가 주파수 지원·calibration·선박 유무를 보장하지 않는다고 명시합니다. 원 SA가 무라벨이라는 것은 사용 불가라는 뜻이 아니며, 자연 선박이 없는 음원이라는 뜻도 아닙니다.

## 3. 품질 수치: 오류·후보·경고의 분리

아래 분모는 별도 표기가 없으면 원 1,715,779행입니다. 서로 겹치는 수치를 합산하면 안 됩니다.

| 항목 | 행 수 | 판정·처리 |
|---|---:|---|
| 정확한 -156.5 sentinel | 653 | invalid flag; finite이지만 창 입력에서 제외 |
| 공통 대역 중 음수 존재 | 654 | sentinel 외 duration40초인 1행 포함; “sentinel654”는 틀림 |
| duration≠60 | 36,234 | 모든 비60초가 확정 오류라는 뜻은 아님 |
| duration<55 / >65 | 16,896 / 22 | 초기 적격 범위 밖 |
| duration 범위 | 1–360초 | 별도 nonfinite·0이하·1초미만은0행 |
| 1≤duration<5 / 5≤duration<30 / 30≤duration<55 | 10,968 / 2,994 / 2,934 | 서술적 구간; 새 제외 기준 아님 |
| timestamp 초 부분≠0 / >5 | 21,082 / 7,672 | strict EDA와 현 준비 조건이 다름 |
| 25band spread<6dB | 4,974 | 단독 hard rejection이 아닌 선별 지표 |
| 경계 후보 전체 | 12,643 | 기기 과도음으로 확정된 정답이 아님 |
| 그중 고정창에 들어간 후보 | 6,777 | 원 전체 행의0.39498%; 모든 경계 후보의53.60278% |
| URN_8321 nominal12kHz 경고 | 136,518 | 상위 대역 지원·export provenance 미해결; blanket band 삭제 아님 |
| 초기 선별 부적격 합집합 | 17,555 | 1.023150%; 확정된 물리 오류율 아님 |
| 적격 행 / 실제 창 포함 행 | 1,698,224 / 1,698,184 | 중복·최소 관측 창 단계를 구분; 40행 차이 |

교집합·합집합·boolean QC·duration 구간의 전체 값은 JSON에 수록했고, 원근거는 [R5_OUTLIER_AUDIT.json](../../R5_OUTLIER_AUDIT.json)입니다. 높은 dB·높은 합성 share를 자동 outlier로 정의하지 않았습니다.

경계 후보는 recorder별 연속 분 run에서 시작/끝 분의 broadband excess 중앙값이3dB를 넘는지로 정했습니다. RB7005 첫분23.98dB, RB201383949 마지막11.17dB, ST6008210/8230 첫분28.69/30.08dB, URN6509 계열 첫분31.88/28.85dB는 기존 표의 값입니다. “마스킹하면 반드시 성능 향상”이나 “모두 선박과 무관한 장비 오류”는 입증되지 않았습니다. [원 표](../../../eda_deep/tables/run_edge_artifacts.csv).

## 4. 서로 다른 창·행 분모를 혼동하지 않기

| 데이터 버전·집계 | 수량 | 정확한 조건 |
|---|---:|---|
| 기초 EDA raw sliding candidate | 1,269,356 | 15개 행이 모두60초 간격인 overlapping 시작점; 고정 시계 창 아님 |
| 기초 EDA base / strict candidate | 1,259,260 / 1,253,096 | 위 sliding 시작점에 각 선별 조건 적용 |
| deep EDA GOOD 행 | 1,678,051 | 음수 제외·정확한 분 경계·dur≥55·EDGE 제외 |
| deep EDA aligned 창 | 102,414 | 정각15분·관측≥13분·음수/EDGE/초≠0 제외 |
| 원 prepare_sa 창 | 133,641 | 55≤dur≤65·초≤5·같은 분 dedup·관측≥3분 |
| quality_v2 활성 TRAIN+validation 풀 | 100,965 | 91,186+9,779; reserved/purge 역할과 분리 |
| R4/R5 clean TRAIN 역할 합 | 267,837 | 5fold 반복 항목; 독립 원천 수 아님 |
| R4 모든 역할 합 | 297,837 | TRAIN+INNER15,000+OUTER15,000 |
| 후속 R6 pilot | arm별 TRAIN8,192 / INNER1,024 | 기존297,837 전체 학습이 아님 |

**발견한 문서·구현 차이:** deep EDA의 GOOD에는 `dur>=55`가 있지만, 102,414개 aligned 창을 만드는 별도 루프에는 `FULL` 필터가 없습니다. `ONGRID & ~SENT & ~EDGE`로 구성한 뒤 `has_short_dur`를 기록만 합니다. 따라서 102,414개를 “55초 이상 행만으로 만든 창”이라고 설명하면 틀립니다. [deep_eda.py](../../../eda_deep/deep_eda.py), `GOOD` 및 recorder duty/WT loop.

원 prepare_sa는 실제 분 위치만 NaN 창에 넣습니다. 공백을 압축·보간·반복하지 않습니다. duration은 적격·중복 선택에 쓰며 각 행을dur/60으로 재가중하지 않습니다. 후속 R4/R5에서는 경계 후보를 포함한 창 전체를 제외했지만, R6에서는 허용 TRAIN 창을 복구해 실제 후보 분만 추가 NaN 처리합니다. **과거 R5 outlier 보고서의 “Cape Recife/Riy Banks TRAIN0”을 현재 R6에 그대로 적용하면 안 됩니다.**

## 5. 사이트·장비·duty cycle과 시간적 다양성

사이트별 행 수·기간·모든 recorder 조합은 JSON의 `site_summary/recorder_summary/recorder_duty_cycle` 원 표 전체에 있습니다. Richards Bay1,001,174행(58.35099%), St Croix250,717행, St Francis169,488행 등 관측 수가 균형적이지 않습니다. 장비별 offset·대역 범위 차이는 export 값의 기술 통계이지 calibration 차이만으로 원인을 분해한 결과가 아닙니다.

St Francis는 공개 메타데이터에서169,488행, 실제 초기 창 포함169,487행이며, 27,154개 clock 창 중5관측분23,782개·15분3,371개·12분1개입니다. 2016년에도15관측분 창741개가 있어 문헌의 연도 설명을 모든 공개 행에 동일 적용할 수 없습니다. **현재 R6에서는17,380개 St Francis source 창이 cross_sensor metadata이고 TRAIN은0개**입니다. 원V1 준비의 St Francis는 train21,802/validation2,076/calibration3,276이었으므로 처음부터 domain holdout이었다는 설명도 틀립니다. [상세 코드·메타데이터 감사](../../R6_PUBLIC_TRAINING_PROTOCOL_CODE_REVIEW.md).

5분 녹음/20분 주기는25%이며20%가 아닙니다. 보존된 clock 창의5/15는 관측 있는 창에 조건화된 coverage이지 전체 운용 duty ratio가 아닙니다. Cape Recife/Riy Banks의 오래된 EDA12분 run은 경계 후보를 뺀 뒤의 값이며, 원자료13분 관측과 같은 분모가 아닙니다. Fish Hoek도 구 recorder와 신 recorder의 스케줄이 다릅니다.

일주기·월별 통계는 recorder 중앙값을 뺀 GOOD 관측 dB입니다. 저장 표에는 모든 사이트·hour·month가 수록됐습니다. 타임존 미검증과 관측 계절·일수 편향이 남으며 물리적 소음원 원인이나 계절 전체의 평균으로 승격하지 않습니다. ACF는 실제 clock의 gap을 NaN으로 두고 lagged 유효 pair에서 계산했으며, site 곡선은 pair 수로 가중했습니다. 창 안 `lag1_acf_broadband`와 분모가 다릅니다.

## 6. 에너지·스펙트럼·PCA·시간 특징의 정의

| 지표군 | 정의·주의 |
|---|---|
| 대역 에너지 비율 | TRAIN은 각 GOOD minute의 정규화 비율 평균, 과거 Test는 각 window 정규화 비율 평균; 전체 모든 에너지의 한 번의 비율이 아님 |
| ≤200Hz 비율 | TRAIN RB.839/Durban.849/Cape.822/Bird.791/StCroix.804/Riy.352/SF.436/Fish.747/Sea.957/Saldanha.826; 과거 Test.575. 집계 단위 차이 주의 |
| old EDA LOW | ≤520Hz, 실제 중심31.6–501.2Hz; 현재 R6 LOW≤199.5Hz와 다름 |
| old EDA HIGH | 중심≥3000Hz, 실제3162.3–7943.3Hz |
| bump / range | 지역별 minute linear power 합을dB로 바꾼 뒤 max−median / max−min |
| centroid_hz | **기하 중심** `10**(ΣE log10(f)/ΣE)`; 산술 가중 주파수 평균 아님 |
| tilt_db | time-mean band dB에서f≤130 평균−f≥1990 평균 |
| naive_share_q20 | temporal q20를 **선형 power에서 보간**; quiet-floor 휴리스틱이며 실제 added share 아님 |
| peak excess spectrum | LOW peak minute의 혼합 dB−band temporal median; 분리된 reference/선박 스펙트럼 아님 |
| old EDA PCA | recorder 중앙값 제거·최대400,000 GOOD 행 RNG0 표본·band zscore·SVD; 설명분산.743/.119/.067/.022/.014 |
| R5 PCA | TRAIN window log-mean-power spectral shape의 PCA3 residual; 위 row-PCA와 다른 fit/입력/스케일 |
| R6 physics27 | 9개 통계+9개 정의 mask+9개 관측 count; LOW/MID/HIGH/SAFE·완전 지역 minute 조건 사용 |

R6의 nominal MID는indices10:20, HIGH20:25, SAFE15:23입니다. SRC mask·cutoff·집계 순서·고정 분모가 다르면 같은 “bump/correlation/PCA” 이름이라도 직접 풀링하지 않습니다. 모든 old EDA 특징의 정확한 식은 JSON `metric_definition_registry`와 [window_features](../../../eda_deep/deep_eda.py)에 보존했습니다.

## 7. 자연 LF transient 5,725건은 어떤 정답인가

정답으로 인증된 사건 목록이 아닙니다. 과거 코드에서 다음 휴리스틱을 만족한 **candidate**5,725건입니다.

1. EVT_LOW는50.1–501.2Hz, reference는1584.9Hz 이상입니다.
2. 실제 minute clock에 관측을 놓고 61분 centered rolling median, 최소 관측10분으로 baseline을 계산합니다.
3. `exLow>=6dB` 및 `exLow−exRef>=3dB`가3개 이상 연속 분에서 성립해야 합니다.
4. 앞뒤 실제 관측이 없거나 peak excess>40dB인 후보를 제외합니다. 기존 제외 합은1,416건입니다.
5. 지속 중앙값4분·p90 9분·최대27분입니다. rate는 calendar day가 아니라 `candidate / GOOD recorded_minutes ×1440`입니다.

AIS·원시 waveform·자연 선박 유무로 검증하지 않았습니다. 고래·기상·장비·다른 음원이 섞일 수 있고, 추가된 reference를 검출하는 synthetic label도 아닙니다. R6 hard-negative의LOW≤200/q90/safe-HF 대비3dB 조건과도 다릅니다. A=0 hard negative는 “인위적 추가 없음”이며 자연 선박 없음이 아닙니다.

## 8. 과거 Test EDA: 기록은 남기되 사용 경계를 지키기

아래는 기존 [기초 summary](../../../eda/summary.json), [deep summary](../../../eda_deep/summary.json), 기존 CSV를 읽은 값입니다. 이번에 Test 입력으로 재검산하지 않았으며 학습 sampling·augmentation·보정·원천 매칭의 목표로 사용하지 않습니다.

| 지표 | 과거 저장값 | 정확한 의미 |
|---|---:|---|
| 창 / 전체 셀 / finite 셀 | 1,500 /562,500 /544,150 | shape15×25/window |
| missing 셀 / whole missing 분 | 18,350 /734 | partial minute0 |
| complete / any missing 창 | 775 /725 | 1,500창 분모 |
| 오직1분차 / 오직15분차 결측 | 620 /97 | exclusive pattern `{1}` / `{15}` |
| 복합 패턴 포함 첫분 / 마지막분 결측 | 624 /100 | marginal41.6% /6.6667%;620/97과 다름 |
| 추가 exclusive 패턴 | `{1,2}`3, `{14,15}`2, `{13,14}`1, `{8,9}`1, `{1,14,15}`1 | 중복 없이 전체 패턴 집계 |
| finite dB 범위 | 29.2999992371–152.8999938965 | float32 저장값 |
| 0.1dB grid | finite셀 검사True | `abs(v*10-round(v*10))<1e-3`; analog 해상도·SA전체·rounding 시점 증명 아님 |
| ID 중복 / 전체 acoustic row 중복 | 0 /0 | 기초 summary의 정의; 모든 minute vector 중복0으로 확대하면 안 됨 |

historical kNN25의 TRAIN site×ISO-week5fold balanced accuracy.705는 **Test site 정확도가 아닙니다.** 사이트당 최대4,000개 창이며, Cape/Riy처럼 eligible window0인 class도 있어 “10개 클래스 모두4,000개 학습”이라고 부르면 안 됩니다. 예측 Test 구성비 SF.325/RB.282/Saldanha.189/StCroix.157 등은 hidden site 정답이 아닌 classifier 출력입니다. 거리 p99 초과.011도 실제 모든 Test가 동일 source/in-domain임을 인증하지 않습니다.

KS는 입력 특징 CDF의 차이입니다. old EDA는 원 TRAIN window 분포와 Test를 비교했고, site-balanced 열은 feature별50,000개 weighted resample을 사용했습니다. 정답 share 분포·양성 비율·숨은 SNR/길이/EQ를 검증하거나 복원한 것이 아닙니다. “같은 사이트 비교”도 true TRAIN site 대 **추정 Test site** 비교여서 site 효과·하드웨어 효과가 완전히 제거되지 않습니다.

`window_feature_train_vs_test.csv`의 모든 분위수·KS12개, predicted-site별5개 특징 비교, transient spectrum·peak-position·top-band 비율, 고밴드 지원 차이의 기존 숫자를 JSON에 수록했습니다. 이것이 Test의 원천·정답을 확정한 표는 아닙니다.

## 9. 바로잡아야 하는 해석과 아직 모르는 것

- “sentinel654” → 정확한 sentinel653, 음수 포함654.
- “620/97이 전체 첫·마지막 분 결측” → exclusive count이며 marginal624/100.
- “고대역 bump=추가 passage 확정 지문·Test 양성률” → 혼합 관측 통계이며 hidden label 아님.
- “kNN site 비중=실제 Test site 구성·동일 source” → 예측값·가설.
- “같은 site 비교로 차이는 모두 추가 성분 때문” → 원인 분리 실패. Documents의 정정 인용문도 외부 에너지의 존재·출처를 확정한 것으로 확대하지 않음.
- “Benchmark는all0 또는.1임이 확인됨” → 알고리즘 mapping 미확정. 공식 notebook share.1과 sample share0은 별개이며 public score를 label moment·prior로 역산해 fitting하지 않음.
- “Test 기술 통계만 냈으므로 공식 규정상 자동 허용” → 기존 보고서의 주장일 뿐. 이번 factcheck는 inference-only 경계를 적용하며 Test 기반 설계를 채택하지 않음.
- “문헌5/20=20% / SF5분을 이어15분으로 만들기” → 산술25%이며 연결·보간하면 실제 관측을 만들게 됨. 현 코드는 원 결측을 보존.
- “모든 source를 같은 전처리” → SA는 이미 band-dB 요약에 row QC, 외부 audio는 waveform→PSD·band power 집계. 같은25band/output geometry와 동일한 source-level 처리라는 주장은 다름.

R5 factcheck의 모델 검증 상태는 작성 시점 기록입니다. 후속 본학습·고정100epoch 실험의 현재 상태로 다시 인용하지 않습니다. 관측 dB만으로 물리 pressure calibration, 실제 배포 좌표, UTC 정렬, 자연 선박 GT, 숨은 generator의 상세 분포를 확정하지 않았습니다.

## 10. 전체 지표군·원 표 찾기

정량 JSON의 `historical_aggregate_tables`에는 다음 집계 표의 **모든 저장 행/열**과 각 SHA·단위·분모가 있습니다. 원 Test per-window feature 파일이나 새로운 관측값 분석은 포함하지 않았습니다.

- site/recorder 원 행·시간·duration·QC·슬라이딩 candidate; 열별·recorder별 대역 결측; 공통25band 전역·site별 min/p01/p10/p50/p90/p99/max.
- 파일 크기·SHA, duration count, recorder duty/run/gap·aligned coverage, run-edge excess, recorder spectrum p10/p50/p90.
- band energy fractions, 시간대별 band deviation, 월별 broadband deviation, recorder별 모든 lag ACF, band correlation matrix.
- old window12개 특징의 모든 TRAIN·trainbal·Test 분위수와 KS; band별 peak excess/range profiles.
- domain-probe CV confusion, predicted site×missing pattern, predicted-site-matched feature 비교, recorder top-band support 차이.
- Test의 기존 exclusive missing pattern·minute×band marginal missing count와 pattern별 feature summary.

새로운 fitting·탐지·합성·Test 통계 재계산이나 동결 파일 변경은 없었습니다. 사용자 재검토 시 과거 `deep_eda.py`를 무조건 다시 실행하라는 의미도 아닙니다. 그 파일은 Test를 직접 읽으므로 현재 작업 경계와 분리해야 합니다.
