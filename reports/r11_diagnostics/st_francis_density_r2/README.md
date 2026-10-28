# R11 St Francis 관측밀도 진단

**실측 배경 원천은 sparse1,000개와 dense312개이며, 새 모델을 학습한 실험이 아니다.** 동결된 Fold0의4개 신경망·LightGBM 및 기존 가중치 앙상블을 그대로 사용했다. Test site 추정값, Test 입력, report 데이터, 공식 점수로 조건을 맞추지 않았다.

## 1. 실제 원천 지원과 기록 시각

| 범위 | SF 창 수 | 관측5분 | 관측15분 | 기타 |
|---|---:|---:|---:|---:|
|original|27,154|23,782|3,371|1|
|R9_allowed_pool|23,172|19,827|3,344|1|
|selection|2,660|2,348|312|0|

Fold0 selection의>=13분 군은 실제로 모두15분이다. 추가적인 TRAIN·calibration·domain·report 원천으로 확장하지 않았다.

| 실제군 | 선택 가능 창 | 사용 unique B | 기록시각 범위 | 일수 / source group |
|---|---:|---:|---|---:|
|sparse|2,348|1,000|2016-10-11 00:00:00 ~ 2018-02-28 14:15:00|33 / 6|
|dense|312|312|2015-04-27 00:00:00 ~ 2015-11-08 23:00:00|13 / 2|

두 군의 recorder 문자열은 `SM2M_Cape_StFrancis_SM2M.1.48000`, sensor_group은4로 같다. 다만 동일 문자열만으로 실제 장비 보정·gain·운용 조건까지 같았다고 보장하지 않는다. 기록시각은 export clock이며 독립 검증된 UTC로 재명명하지 않았다. sparse는2016–2018년, dense는2015년으로 시기·음향 환경이 함께 다르다.

Sparse 관측은 띄엄띄엄 떨어진5분이 아니라 연속된5분 블록이다. 전체 selection의 mask는[0..4]783개,[5..9]783개,[10..14]782개다. 관측분 시작 간격의 최빈값1분/16분은5분 블록이20분 간격으로 반복되는 패턴과 일치한다. Dense의1분/46분 간격은15분 블록이60분 간격으로 반복되는 패턴과 일치한다. 이는 데이터에 나타난 패턴이며 recorder 설정 파일을 확인한 진술은 아니다. 따라서Nobs5를곧바로“5분 간격 sampling”으로 표현하면 부정확하다.

TRAIN과 source group·base window 중복은0이며 같은 sensor의 TRAIN지원구간(900초)과 가장 가까운 간격을 별도로 재계산한 최솟값은24.250000시간이다. 원래24h purge를 유지했다.

## 2. 사전 고정한 진단 구성

312개를 모두 사용하면서 정수 예제에서 정확히20% zero를 만들기 위해 각 unique B에1개A=0+4개양성 view를 만들었다. 같은 B의5views는 독립 녹음5개가 아니다.

| Panel | 실제 unique B | 생성 예제 | A=0 | 양성 |
|---|---:|---:|---:|---:|
| 실제 sparse Nobs5 |1,000|5,000|1,000|4,000|
| 실제 dense Nobs15 |312|1,560|312|1,248|
| 같은 dense를5분 mask로 삭제 |동일한312|1,560|312|1,248|

전체 실제 배경은1,312개이며 paired312개를 다시 더해 독립 원천 수를 부풀리지 않았다. Sparse1,000개 선택은 seed2026102614의 source key SHA 순위로 고정했다. Dense312개는 전수다.

공통32개 measured reference/treatment bank를 사용했다. Family16inside/8short/8edge,32개SNR LHS strata[-10,+10]dB, smooth EQ±6dB를 고정했다. Sparse의 각 treatment는125회, dense와paired는각39회여서 **reference ID·family·SNR·EQ의 경험적 분포가 정확히 같다.** 전체509개로의 일반화 실험은 아니다.

### 모델 결과를 보기 전 발견한 지원 제약과 R2 수정

초기R1은 모델 추론 전 중단됐다. Native2..6분의 진짜15분-window-edge passage는 관측중간블록[5..9]에 기하학적으로 닿을 수 없어 placement를4096번 재시도해도 양성 관측 지원이 생기지 않는다. 이를 양성으로 라벨 조작하거나 source를 늘리지 않았다. R1계획·코드·실패 로그는 그대로 보존했다.

R2에서는8개edge reference만 native>=7분인43개 후보 중 metadata/sourceSHA로 선택했다. 다른24reference와 모든family/SNR/EQ draw는R1과 동일하게 유지했다. 이 수정은 원래509균등참조R9법과다른 **조건화된 reference/family 결합분포**다. 같은 수정 bank를 sparse/dense 양쪽에 적용했으며 모델 오차로 reference를 고르지 않았다.96개treatment×실제mask 지원 가능성을 예측 전 검사했다.

양성 요청은 같은reference/family/SNR/EQ/full-native-peak를 유지하고 placement만 관측A>0까지 재추첨했다. 이로 인해 실제 위치·유효 겹침 분포는 군 간 같지 않다. Paired는 dense에서 생성한B/A/alpha를 그대로 두고 실제5분block3종 중 하나로만 관측을 삭제한다. 양성support가없으면 다른mask를최대3개까지검사했다. 삭제 뒤 원시성분 에너지로y/q를 다시 계산했고 db는0.1dB로round했다. Model pooling weight는그rounded db에서계산한다.

## 3. 목표 분포와 조건부 선택

| Panel | 목표 share 평균 | 표준편차 | 중앙값 | q90 |
|---|---:|---:|---:|---:|
|sparse|0.200597|0.235894|0.086954|0.649710|
|dense|0.108113|0.146719|0.038306|0.390446|
|dense_mask5|0.184529|0.220912|0.084315|0.573506|

정확한zero비율과 reference/family/SNR/EQ 분포를 맞춰도, 관측 지원과 B의 시간 변화 때문에 목표share분포는 같지 않다. Paired의목표평균도dense0.108113에서0.184529로변한다. 따라서RMSE차이는순수한“관측개수만의정보손실”과동일하지않다.

| Panel | 양성 placement 평균 시도 / 최대 | placement 재시도 양성 수 | paired mask 재시도 양성 수 |
|---|---:|---:|---:|
|sparse|2.0987 / 9|2018|0|
|dense|1.0000 / 1|0|0|
|dense_mask5|1.0000 / 1|0|654|

Support조건을 만족시켜 양성으로 유지하는 절차는 명시적인 조건부 선택 편향이다. 특히 paired 양성의mask선택은A지원에 조건부이고zero의mask는그조건이없다. 이차이도AUC해석에포함해야한다.

## 4. 동결 모델 결과

| 모델 | 실제 sparse RMSE | 실제 dense RMSE | paired mask5 RMSE | paired−dense |
|---|---:|---:|---:|---:|
|bg_cnn|0.167593|0.146680|0.200134|+0.053454|
|tfgrid_fdy_conformer|0.142883|0.101174|0.161497|+0.060323|
|fdy|0.160825|0.135588|0.193873|+0.058285|
|itransformer|0.170089|0.148164|0.209415|+0.061251|
|lgbm_physics|0.093889|0.076718|0.126558|+0.049840|
|all5_frozen|0.105665|0.083171|0.134205|+0.051034|

각 셀의 순서는 sparse / dense / paired다.

| 모델 | 평균 bias | AUC share | AUC score | zero RMSE |
|---|---:|---:|---:|---:|
|bg_cnn|-0.058560 / 0.065987 / 0.004198|0.831575 / 0.690215 / 0.743514|0.820712 / 0.702147 / 0.745548|0.052450 / 0.160678 / 0.147159|
|tfgrid_fdy_conformer|-0.037420 / -0.019498 / -0.028674|0.880659 / 0.756510 / 0.801589|0.890892 / 0.772349 / 0.818976|0.048331 / 0.060565 / 0.080414|
|fdy|-0.052100 / 0.023457 / -0.011342|0.828575 / 0.694632 / 0.731791|0.843283 / 0.694922 / 0.771638|0.055074 / 0.126254 / 0.150117|
|itransformer|-0.055877 / 0.053388 / 0.037282|0.856784 / 0.738381 / 0.773719|0.848613 / 0.758878 / 0.779739|0.053461 / 0.144085 / 0.172595|
|lgbm_physics|-0.010200 / 0.000210 / 0.003960|0.886906 / 0.783089 / 0.819493|0.915022 / 0.860271 / 0.885908|0.046101 / 0.048180 / 0.083309|
|all5_frozen|-0.026516 / 0.012459 / 0.002522|0.893451 / 0.787157 / 0.825024|0.910675 / 0.855975 / 0.874421|0.042820 / 0.066232 / 0.092047|

모든 모델에서 이번 paired mask5 RMSE가 증가했다. 그러나 all5 score AUC는dense0.855975에서paired0.874421로증가했고, 실제sparse의AUC도0.910676으로dense보다높다. 관측이적으면모든지표가일관되게악화한다는해석은맞지않는다. 실측두군간차이는연도·음향환경과함께변하며paired도목표에너지지원과양성mask선택을변경한다.

### 반복 view를 고려한 조건부 불확실성

같은base_window_index의5views를하나의cluster로200회bootstrap했다. Paired차이는같은resampled dense B IDs로계산한다. 아래CI는고정된32treatment조건부이며station·reference509개모집단불확실성이나6/2week-group내시간상관까지반영하지않는다.

| 모델 | paired ΔRMSE | source-ID cluster SE | 조건부95% percentile CI |
|---|---:|---:|---|
|bg_cnn|+0.053454|0.006953|[+0.039909, +0.067127]|
|tfgrid_fdy_conformer|+0.060323|0.005156|[+0.049911, +0.070271]|
|fdy|+0.058285|0.007178|[+0.045446, +0.071362]|
|itransformer|+0.061251|0.007968|[+0.044046, +0.074114]|
|lgbm_physics|+0.049840|0.003233|[+0.042471, +0.055335]|
|all5_frozen|+0.051034|0.003367|[+0.044445, +0.057777]|

## 5. 원래 R9의 mask-induced zero는 독립적인 무작위 사건이 아니다

원래R9는placement를mask와독립적으로draw하고, 관측된성분에너지로share/presence를계산한다. 그러므로positive reference요청도관측A=0이될수있다. 이것은그정의에서라벨오류가아니며, 이번조건부진단과다른법이다. 아래는모든native분의broadbandpower가양수라는실제bank검사를전제로한placement의정확한기하확률이다.509개reference균등평균이며실제학습빈도나공식숨은generator를측정한값이아니다.

| 관측5분block | inside: P(관측A0\|positive요청) | short | edge | edge가구조적으로지원불가한ref |
|---|---:|---:|---:|---:|
|[0, 1, 2, 3, 4]|0.570952|0.645786|0.496213|0 /509|
|[5, 6, 7, 8, 9]|0.385173|0.583142|0.962970|465 /509|
|[10, 11, 12, 13, 14]|0.570952|0.645786|0.496213|0 /509|

중간블록[5..9]에는465/509(91.36%) reference가true edge placement로닿을수없다(1분reference는별도inside fallback). 실제SF3mask빈도와원family비율을가중한uniform-reference예시에서positive요청이관측A0가될확률은0.573690; full15분에서는0이다. 독립20%explicitzeros를가정하면totalzero예시는0.658952이지만, 실제TRAINhard-negative할당조건과유한schedule이있으므로이를실제SF학습zero비율이라고쓰면안된다.

즉 family·native길이·관측block의결합으로labelprior가달라질수있다는사실은수학적으로확인된다. 이것만으로공식Test의site·targetprior나leaderboard격차원인을확정하지않는다.

## 6. 물리 정답·파일 검증과 범위

Scalar oracle는B=1과3분A=1에서fullshare1/6, 첫5분maskshare3/8을검사했다. Rawcomponent pool identity최대오차1.11e-16; 저장된q와rounded modelweight간gap최대0.001628는사전에쓴quantizationbound이내다. 독립검증은192개case를recipe+원B+reference로재구성해db/q/share를정확히재현했고, paired의관측db/q와alpha보존도확인했다. 저장예측으로RMSE/bias/직접쌍별AUC/clusterSE·CI를다시계산한최대차이는0이다.

NN추론은CPUFP32·2threads이며tree는기존저장API의4threads를유지했다. 모델·source·sampler·config·가중치는수정하지않았고새학습·Test·unusedreport·네트워크·외부송신은없다. 이보고서는selection-source진단이며OOF·독립report·자연선박존재GT가아니다.

[계획](plan.json) · [전체결과](results.json) · [전체지표CSV](metrics.csv) · [원천지원](source_feasibility.json) · [물리생성검증](generation_gate.json) · [원R9geometry계산](original_R9_mask_geometry.json) · [독립수치검증](independent_numeric_verification.json) · [R1구조적실패보존](../st_francis_density_r1/FAILED.json)
