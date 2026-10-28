# Test gap 관련 주장 검토

2026-10-26. **목표는 실제 일반화 오차를 줄이는 것입니다. AV·출력 RMS·상관의 임의 기준을 만족시키는 것만으로 Test 오차가 작아졌다고 판단할 수 없습니다.** 이번 검토에서는 Test 배열·새 예측·새 학습 결과를 읽지 않았고, 파라미터를 변경하지 않았습니다.

## 공식적으로 확인되는 범위

대회는 `share_added_vessel`의 `1 − RMSE`로 순위를 정합니다. 정답은 관측된 동일 minute-band cells에서 **추가한 reference의 선형 에너지 합 / 혼합물의 선형 에너지 합**입니다. 자연 배경에 이미 있던 선박 소음 전체의 비율은 아닙니다. 공식 규칙은 Test를 추론에만 사용하고, Test 기반 학습·미세조정·보정·의사라벨·학습 예제 생성을 금지합니다. 고정된 개별 window 변환은 허용합니다. 따라서 Test-domain 분류기를 학습하거나 Test 통계에 맞춰 생성기·출력 분포를 조정하는 작업은 진행하지 않습니다. [공식 설명·평가·규칙](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge)

공식 Test에는 ID와 375개 음향값만 제공되며, station 위치·관측 timestamp·AIS·추가 전 배경은 제공되지 않습니다. Test 배경 기간은 학습 공개분에서 제외됩니다. **“Test의 St Francis 489 windows”는 공개 정답 metadata로 확인된 사실이 아닙니다.** 과거의 분류·매칭 추정이라면 추정으로 구분해야 하며, 숨겨진 원천 정체성을 복원해 학습·보정에 사용하지 않습니다. [공식 데이터 설명](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/data)

## 주장별 판단

| 주장 | 판단과 이유 | 사용할 수 있는 판단 기준 |
|---|---|---|
| **Test AV AUC ≤ .6이면 gap이 사라진다** | 보장되지 않습니다. 유한한 분류기·특징·표본·CV의 검출력이 부족해도 AUC가 낮을 수 있습니다. 설령 입력 주변분포 `P(X)`가 같아도 정답과 입력의 관계 `P(y｜X)`는 다를 수 있습니다. | 허용 TRAIN/INNER에서 원천 group 분리 AV를 보조 진단으로 사용하고, 실제 회귀 오차·bias·원천별 slice를 함께 봅니다. `.6`은 합격선이 아닙니다. |
| **높은 AV AUC는 곧 높은 회귀 오차다** | 이것도 보장되지 않습니다. 녹음기 응답·missing mask·정적 주파수 형태 등 구별 가능한 nuisance가 정답 예측에는 무관할 수 있습니다. 반대로 AV가 낮아도 추가 성분을 잘못 구분할 수 있습니다. | 어떤 원천·표현·측정 조건을 구별한 것인지 분리해야 합니다. AUC를 낮추려고 유용한 물리 신호를 지우지 않습니다. |
| **예측 RMS를 .22–.27로 맞추면 Benchmark와 정합된다** | 근거가 없습니다. `MSE = E[p²] + E[y²] − 2E[py]`이므로 예측 RMS만으로 오차가 정해지지 않습니다. **모든 window에 `.25`를 내는 모델도 RMS `.25`**지만 사건별 예측 능력은 없습니다. | 허용된 labelled validation에서 RMSE와 target별 bias를 측정합니다. 출력 RMS를 목표값으로 강제하거나 Test에 맞춰 조정하지 않습니다. |
| **Benchmark 점수로 Test의 `E[y²]`를 알 수 있다** | 일반적으로 불가능합니다. 공개 Benchmark의 실제 예측기와 예측값이 확인되지 않았습니다. 그 점수는 해당 예측기의 공개 구간 MSE일 뿐, zero-predictor의 `E[y²]`라고 놓을 수 없습니다. | 저장된 정답이 있는 자체 validation에서만 상수 baseline과 MSE 항등식을 계산합니다. `.24–.27`을 생성기·정답 분포의 목표로 삼지 않습니다. |
| **bump 상관 ≥ .65면 전달 성능이 충분하다** | 대역·집계·대상 간 상관의 정의부터 명확해야 합니다. 어떤 정의라도 `.65`가 회귀 오차나 선박 정답성을 보장하지 않습니다. 강한 동반 상승은 다른 소음에서도 발생하고, 약한 동반 상승이 실제 reference를 배제하지도 않습니다. | 사전 고정한 feature 정의를 쓰고, 정확한 추가 성분 label에 대한 오차·간섭 hard-negative·주파수 ablation으로 유용성을 확인합니다. 상관 하한을 통과시키기 위한 filtering은 하지 않습니다. |
| **St Francis 489개에 맞춰 학습 비중을 정하면 된다** | 공식 Test에는 site label이 없으므로 그 숫자를 정답처럼 사용할 수 없습니다. 추정한 원천 배분을 Test 목표 비율로 학습에 넣으면 원천 추정에 의존하는 보정이 됩니다. | 공개 TRAIN의 metadata로 site 균형 추출과 sensor/source holdout을 정합니다. Test 구성 추정은 사용하지 않습니다. |
| **cross-site RMSE .17–.21이면 실제 LB도 그 수준이다** | 동일성을 입증하지 못했습니다. 합성 reference·추가 SNR·시간 배치·receiver 응답·결측·원천·target 분포가 공식 hidden test와 다를 수 있습니다. | 자체 합성 cross-site 지표로만 보고합니다. 공식 제출 점수와 자체 지표를 별도 표에 기록하고 수치 환산·동일 성능 예고는 하지 않습니다. |

## 이미 확인한 R6 결과의 정확한 의미

이전 ES 실행의 **동일 arm `r6law_relative8_global`**은 21 epochs를 실행했고 INNER 기준 best epoch는 14였습니다. 다음은 이미 평가한 자체 합성 데이터의 결과입니다.

| 자체 평가 구간 | RMSE | 해석 |
|---|---:|---|
| INNER | 0.0641477535 | 해당 합성 law·원천 분할에서 checkpoint 선택에 사용한 값 |
| CROSS_SENSOR | 0.1155697948 | 미학습 sensor 원천을 사용한 자체 합성 진단 |
| CROSS_SENSOR_GENERATOR | 0.1228301499 | 같은 sensor holdout에 다른 합성 family를 적용한 진단 |

근거: [이전 arm summary](../models/r6_cpu_guardrails_v2/r6law_relative8_global/summary.json), [sensor metrics](../models/r6_cpu_guardrails_v2/r6law_relative8_global/cross_sensor_metrics.json), [joint metrics](../models/r6_cpu_guardrails_v2/r6law_relative8_global/cross_sensor_generator_metrics.json). **어느 값도 공식 Test RMSE가 아닙니다.** 이미 본 진단 구간이므로 이후 변경 결과와 비교할 때도 개발 비교라고 명시해야 합니다.

별도 **100epoch·ES 미사용** 실행의 결과는 이 검토에 포함하지 않았습니다. 그 실행의 고정 설정·체크포인트 정책은 유지하고, 이번 주장 검토를 이유로 중간 파라미터를 바꾸지 않습니다.

## 규정 안에서 다음에 할 수 있는 일

아래는 실행 가능한 후속 설계이며, **이 문서에서 새로 구현·학습했다는 뜻은 아닙니다.**

1. 공개 TRAIN metadata 기준 site 균형과 sensor·원천 group 분리를 유지하고, sample-weighted/사이트 macro/최악 사이트/target 구간 오차를 함께 기록합니다. 원천 반복 횟수를 독립 표본 수로 계산하지 않습니다.
2. 공개·고지된 reference의 독립 녹음·선종·시간 길이·원천 다양성을 늘리고, 같은 원천의 반복 합성 증가와 구별합니다. reference group은 TRAIN과 validation 사이에 공유하지 않습니다.
3. 허용 TRAIN만 근거로 사전 고정한 smooth receiver 응답, 주파수 기울기, 간섭·결측 등의 domain randomization을 비교합니다. 변환 후 실제 알려진 두 성분에서 `q`, share, presence를 다시 계산합니다. 특히 주파수 응답·crop·mask가 바뀌면 기존 share를 그대로 복사하지 않습니다.
4. 간섭과 LF transient는 **추가 성분 A=0**인 control로 검사합니다. 자연 배경의 선박 부재 label로 해석하지 않습니다.
5. 후속 실험의 조건과 선택 기준은 결과를 보기 전에 고정합니다. 원천·generator holdout으로 전달 성능을 검사하되, 이미 본 패널은 개발용으로 표시하고 별도 독립 검증과 구분합니다. Test 통계·Benchmark 역산·숨은 site matching을 목적함수로 사용하지 않습니다.

## 수학적 반례

`X ~ Uniform(0,1)`인 두 domain을 생각합니다. 학습 domain에서는 `y=X`, 다른 domain에서는 `y=1−X`이고 모델은 `p=X`라고 하겠습니다. 두 입력 분포가 완전히 같으므로 이상적인 domain classifier도 AUC `.5`입니다. 하지만 새 domain의 모델 RMSE는 `sqrt(E[(2X−1)²]) = 1/sqrt(3) ≈ .57735`입니다. **입력 분포 일치만으로 회귀 관계의 전달을 보장할 수 없습니다.** 이 반례는 실제 대회에 이 반전이 있다는 주장이 아니라, AV 임계값만으로 성능을 확정할 수 없다는 증명입니다.
