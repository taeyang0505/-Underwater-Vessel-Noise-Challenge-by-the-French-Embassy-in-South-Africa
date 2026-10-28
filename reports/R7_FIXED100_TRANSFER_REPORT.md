# R7 fixed100 체크포인트 전이 평가

5개 CNN의 **이미 own-INNER에서 동결한 best-window** checkpoint를 기존 동일 cross 패널3개에서 CPU FP32/threads2로 평가했다. Protocol은 패널 array/새 예측을 읽기 전에 고정했고 checkpoint·계수·loss·seed·데이터를 결과에 따라 바꾸지 않았다. 이 패널은 이전 R6에서 이미 관찰한 **개발용 전이 비교**이며 새 blind holdout, OOF, Test 성능이 아니다.

선택 epoch: r4law_legacy8_global=24, r4law_relative8_global=34, r6law_legacy8_global=29, r6law_relative8_global=31, r6law_relative8_local=45. 정답·site·family는 metric에만 사용하고 신경망 입력은 db-only transform 및 해당 db의 물리 w다. legacy8의 prior는 기존 자기 TRAIN에서 fit한 동결 값만 사용했다. 모든 model state는 exact schema/strict=True로 복구했고 resize/partial load는 하지 않았다.

## cross_generator

| Arm/control | RMSE | Macro RMSE | MAE | Bias | Score AUC | Share AUC |
|---|---:|---:|---:|---:|---:|---:|
| r4law_legacy8_global | 0.074687 | 0.070745 | 0.037073 | 0.015106 | 0.885318 | 0.824939 |
| r4law_relative8_global | 0.074590 | 0.069445 | 0.034757 | 0.007638 | 0.865461 | 0.832410 |
| r6law_legacy8_global | 0.056932 | 0.059556 | 0.030631 | 0.010108 | 0.889997 | 0.857332 |
| r6law_relative8_global | 0.060657 | 0.057498 | 0.032439 | 0.010278 | 0.889812 | 0.850606 |
| r6law_relative8_local | 0.063506 | 0.062774 | 0.034703 | 0.011347 | 0.887072 | 0.822707 |
| constant0 | 0.189820 | 0.205052 | 0.109099 | -0.109099 | 0.500000 | 0.500000 |
| constant0_1 | 0.155601 | 0.165346 | 0.114858 | -0.009099 | 0.500000 | 0.500000 |
| TRAIN_r4_mean | 0.314033 | 0.304347 | 0.294243 | 0.272925 | 0.500000 | 0.500000 |
| TRAIN_r6_mean | 0.156678 | 0.163797 | 0.126716 | 0.020471 | 0.500000 | 0.500000 |
## cross_sensor

| Arm/control | RMSE | Macro RMSE | MAE | Bias | Score AUC | Share AUC |
|---|---:|---:|---:|---:|---:|---:|
| r4law_legacy8_global | 0.448952 | 0.448952 | 0.374535 | 0.366373 | 0.620383 | 0.532370 |
| r4law_relative8_global | 0.511655 | 0.511655 | 0.416501 | 0.410126 | 0.634529 | 0.435687 |
| r6law_legacy8_global | 0.183585 | 0.183585 | 0.116073 | 0.103031 | 0.854174 | 0.733073 |
| r6law_relative8_global | 0.095061 | 0.095061 | 0.047728 | 0.028549 | 0.919626 | 0.891348 |
| r6law_relative8_local | 0.106678 | 0.106678 | 0.066098 | 0.033122 | 0.896404 | 0.794132 |
| constant0 | 0.200983 | 0.200983 | 0.093201 | -0.093201 | 0.500000 | 0.500000 |
| constant0_1 | 0.178197 | 0.178197 | 0.129151 | 0.006799 | 0.500000 | 0.500000 |
| TRAIN_r4_mean | 0.339303 | 0.339303 | 0.324901 | 0.288823 | 0.500000 | 0.500000 |
| TRAIN_r6_mean | 0.181744 | 0.181744 | 0.145571 | 0.036370 | 0.500000 | 0.500000 |
## cross_sensor_generator

| Arm/control | RMSE | Macro RMSE | MAE | Bias | Score AUC | Share AUC |
|---|---:|---:|---:|---:|---:|---:|
| r4law_legacy8_global | 0.449931 | 0.449931 | 0.375616 | 0.363265 | 0.562776 | 0.493700 |
| r4law_relative8_global | 0.515098 | 0.515098 | 0.422588 | 0.411122 | 0.582281 | 0.431146 |
| r6law_legacy8_global | 0.184142 | 0.184142 | 0.117351 | 0.087121 | 0.808088 | 0.707204 |
| r6law_relative8_global | 0.110407 | 0.110407 | 0.056546 | 0.017366 | 0.878407 | 0.848672 |
| r6law_relative8_local | 0.115731 | 0.115731 | 0.071462 | 0.021588 | 0.863024 | 0.766395 |
| constant0 | 0.199626 | 0.199626 | 0.093060 | -0.093060 | 0.500000 | 0.500000 |
| constant0_1 | 0.176745 | 0.176745 | 0.127777 | 0.006940 | 0.500000 | 0.500000 |
| TRAIN_r4_mean | 0.338661 | 0.338661 | 0.324099 | 0.288964 | 0.500000 | 0.500000 |
| TRAIN_r6_mean | 0.180343 | 0.180343 | 0.143751 | 0.036511 | 0.500000 | 0.500000 |

## 이전 ES checkpoint와의 기술적 차이

아래 Δ는 fixed100 best-window−이전 ES macro-selected다. 중단 여부 외에 checkpoint 기준·학습 예산·threads도 달라서 ES 제거 단독의 인과 효과가 아니다. 모든 악화도 보고하며 이 결과를 근거로 새 모델/epoch/ensemble을 선택하지 않았다.

| Panel | Arm | fixed100 epoch | Δ RMSE |
|---|---|---:|---:|
| cross_generator | r4law_legacy8_global | 24 | -0.000541 |
| cross_generator | r4law_relative8_global | 34 | -0.005041 |
| cross_generator | r6law_legacy8_global | 29 | -0.006157 |
| cross_generator | r6law_relative8_global | 31 | -0.007786 |
| cross_generator | r6law_relative8_local | 45 | +0.000275 |
| cross_sensor | r4law_legacy8_global | 24 | +0.000768 |
| cross_sensor | r4law_relative8_global | 34 | -0.004002 |
| cross_sensor | r6law_legacy8_global | 29 | -0.018924 |
| cross_sensor | r6law_relative8_global | 31 | -0.020509 |
| cross_sensor | r6law_relative8_local | 45 | +0.000813 |
| cross_sensor_generator | r4law_legacy8_global | 24 | +0.001073 |
| cross_sensor_generator | r4law_relative8_global | 34 | -0.003910 |
| cross_sensor_generator | r6law_legacy8_global | 29 | -0.016252 |
| cross_sensor_generator | r6law_relative8_global | 31 | -0.012423 |
| cross_sensor_generator | r6law_relative8_local | 45 | +0.000075 |

site/target-bin/family별 n·RMSE·site-macro·MAE·bias·score/share AUC는 **모든5모델과 상수4개**에 대해 [전체 JSON](R7_FIXED100_TRANSFER_RESULTS.json)의 evaluations.metrics에 보존했다. 단일 class/빈 slice AUC는 null이다. 상수 .0/.1과 두 law의 TRAINmean은 protocol에 미리 고정했고 상수 presence score는 .5다.

전체 경과 27.45초. 원본 NPZ·모델·core source SHA는 전후 불변이며, 원천 data·모델 weight를 수정하거나 fit하지 않았다. 15개 prediction NPZ와 원래 row 순서의 SHA/target SHA를 새 run 폴더에 저장했다. cross-sensor와 joint는 같은 배경 draw를 공유하고, cross-generator는 원천 참조와 generator가 함께 바뀌므로 독립 패널 세트나 단일 원인 대조로 과장하지 않는다. 실제 대회 일반화·공식 LB 점수를 보장하지 않으며, 이 결과에 맞춘 재학습·calibration·패널 반복 재선택은 하지 않았다.

[사전 protocol](../configs/r7_transfer_audit_v1.json) · [row bindings](../models/r7_fixed100_transfer_v1/row_bindings.json) · [모든 예측/metric](../models/r7_fixed100_transfer_v1/results.json)
