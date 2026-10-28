# R16 학습 전 독립 설계 검토

이 검토는 사용자 첨부 `1fc448b5-90b6-4ff8-b790-a20fe8ecf1ee/붙여넣은 텍스트.txt`, 기존 생성·모델 소스와 순수 수치 fixture만 사용했다. **Test 입력 및 Test 유래 입력 통계 파일은 읽지 않았다.** 새 학습·모델 추론·이전 생산 소스 변경은 하지 않았다.

## 1. EQ 이후 α 재정규화는 유효한 조건부 분포이며, 채널 형상을 없애는 수학 버그는 아님

기존 R15의 정의는 다음과 같다. R은 실제 reference minute power, g_f는 고정 채널 gain, S는 미리 추출한 broadband SNR, B_med는 관측 배경의 분별 broadband power 중앙값이다.

\[
\alpha=\frac{10^{S/10} B_{\rm med}}{\max_t\sum_f g_fR_{tf}},\qquad A_{tf}=\alpha g_fR_{tf}.
\]

이 식은 **EQ 이후 전체 native broadband peak SNR을 S로 고정**하는 조건을 정확히 만족한다. scalar α가 채널의 전체 크기 일부를 정규화하지만,

\[
\frac{A_{t,H}}{A_{t,L}}=\frac{g_H}{g_L}\frac{R_{t,H}}{R_{t,L}}
\]

이므로 상대 주파수 응답은 남는다. [독립 수치 검증](PRE_FIT_MATH_VERIFICATION.json)에서 HF+12dB/LF−6dB의 상대 추가 power 비율은 정확히18dB, 약63.0957배 변했다. 동시에 R_tf=a_t s_f 같은 시간·주파수 분리형 reference에서는 summed A의 시간 궤적이 고정돼 control/EQ의 최종 y가 같을 수 있음을 직접 확인했다. **비슷한 y 평균이 나온다는 사실만으로 버그나 라벨 강제 고정이라고 결론 낼 수 없다.**

삽입 전의 α와 channel gain을 독립 추출하거나 대역별 SNR을 독립 추출하는 방식도 유효한 새 가설이다. 다만 기존 SNR 조건을 다른 사전분포로 바꾸는 실험이며, 공식 hidden 생성법의 유일한 올바른 구현으로 주장하지 않는다. mixture의 max−median 변화가 channel gain dB와 정확히 같아야 할 이유도 없다.

근거: [R15 생성 소스](../../ops/r15_forward_prototype.py)의 `synthesize_pair`.

## 2. ‘원래 peak가 빠진 crop으로 스케일했다’는 일괄 주장은 잘못됨

`choose_trajectory`는 EQ 이전 native broadband peak를 계산하고 그 peak가 포함되는 crop 시작점만 추출한다. 9개 peak 위치×100개 seed의900회 독립 fixture가 모두 이를 통과했다.

다만 시간에 따라 주파수 형상이 달라지면 **EQ 이후 peak 위치가 이동**할 수 있다. fixture에서는 EQ 전 peak1을 포함한1분 crop인데 EQ 후 peak가0으로 이동했다. 이때 full-native SNR은 요청값을 유지하지만 실제 inserted-crop peak SNR은 더 작다. 원 정의는 full-native 기준이므로 내부 모순이나 q/y 오류가 아니다. 새 inserted-observed-crop 기준은 이 조건을 명시적으로 바꾸는 대조법이다.

새 law가 관측 O 이후 crop peak를 분모로 쓰면 α가 observation mask에 의존한다. 따라서 기존 law와 비교할 때 이 mask conditioning의 변화도 함께 기록해야 한다.

## 3. score–share 상관은 가중치나 AUC 상한이 아님

Spearman은 순위의 Pearson 상관이다. 예를 들어 순위 [1,2,3,4]와 [1,2,4,3]의 Spearman은0.8이지만 pairwise 순위 일치 비율은5/6이다. 따라서 0.9679를 ‘score의96.79%가share’, ‘순위96.79% 일치’, ‘독립 classifier 비중3.21%’라고 해석할 수 없다. 양의 단조 변환은 계수가 달라도 Spearman1이 될 수 있다.

정확한 target은 무추가일 때 y=0, 관측된 추가 성분이 양수면 y>0이다. 정답 y를 완벽히 예측하면 양성 y가 매우 작고 배경 분모가 커도 모든 음성0보다 크므로 **share 자체로 AUC1**이 가능하다. 수치 fixture에서 positive share1e−8~0.8, negatives0의 AUC가1임을 확인했다. ‘분모가 달라 share를 점수로 쓰면 AUC가0.55를 절대 넘을 수 없다’는 일반적 주장은 성립하지 않는다.

실제 모델은 오차 때문에 tiny-positive와 false-positive를 잘못 정렬할 수 있다. 독립 classifier는 유용한 대조이지만 성능 개선은 로컬 검증으로 확인해야 한다. 실제 추가 에너지가0인 음성의 참 share는 배경이 조용하거나 자연 HF 변동이 있어도0이며, 높은 값이 나온다면 추정 오차와 구분해야 한다.

## 4. 8채널 회귀와 독립 temporal classifier 계약

새 회귀 입력은 db-only relative8이며 PCA/anchor 없이 같은 rounded energy weights를 사용해 q_hat∈[0,1], y_hat=Σwq_hat로 pooling해야 한다. raw q/y와0.1dB 입력 weights의 작은 차이는 별도 quantization bound로 검사하고 라벨을 rounded pooling으로 덮어쓰지 않는다.

기존 relative8의 모든 채널이 static per-band gain에 불변인 것은 아니다. 시간 상대 채널0/1/2/3/6에서는 상쇄되지만 profile 채널4, w, sqrt(w) 채널7은 남는다. 따라서 분류기의 정확한 offset 불변을 주장하려면 이들을 포함한 회귀 encoder를 공유하는 head가 아니라 **독립 temporal-only 입력·encoder**가 필요하다. 관측 mask 채널5는 사용 가능하다. temporal Pearson 등은 상수 채널·결측이 안전하게 처리돼야 한다.

불변 fixture는 반올림 후 offset 또는0.1dB grid offset처럼 rounding과 교환되는 조건으로 검사한다. 임의 offset을 반올림 전에 주면 quantization 경계가 바뀔 수 있어 무조건 bitwise 불변을 주장하지 않는다. 수신기 gain을 전체 mixture에 곱하는 경우와 added reference만 변형하는 경우의 물리 라벨도 구분한다.

## 5. 가중치 cap은 family와 개별 모델을 구분해야 함

- 개별 NN≤0.4, Tree family≥0.4, 합1이라는 계약은 가능하다.
- **Tree family0.5 + 단일 CNN0.5는 개별 NN cap 위반**이다.
- 새2NN 대조라면 Tree family0.5 + CNN_A0.25 + CNN_B0.25가 가능하다.
- 기존4NN 고정 대조는 Tree family0.5 + 각NN0.125가 가능하다.
- Tree family0.5는5fold Tree를 합친 총비중이다. 각 tree 또는 각 fold의 cap0.4와 같은 뜻이 아니다.

가중치·후보 설정은 predeclared fit 역할의 cross-operator 패널에서만 정하고, gate는 이미 동결된 후보를 검사한다. gate 결과를 보고 새 cap/weight를 반복 탐색하지 않는다. report는 모든 정책을 먼저 동결한 뒤 연다. 이 역할들은 모두 개발용 검증이며 새로운 untouched official holdout이라고 부르지 않는다.

## 6. 예정 source/data QA와 해석 경계

생성자와 합의한 예정 계약은119개/22개 reference component를 A/B/C로 나누고, 새로운 mini TRAIN A와 fit/gate B·report/domain C를 분리하는 것이다. 구체적인 manifest·hull/day/time/bytes 원천 분리와900초 support+24h purge는 생성 완료 뒤 독립 검사한다. 여러 operator 패널은 동일 배경·reference·위치·mask·zero schedule을 공유하며, label은 각 operator의 실제 B/A/E에서 다시 계산해야 한다.

**기존 R9/R14는 전체509 reference를 이미 사용했다.** 따라서 B/C는 새 mini 모델에 대해서만 reference holdout이며, 기존 모델 비교를 unseen-reference 성능이라고 표현할 수 없다. 기존 graph 전체가1component라는 사실과 새119개 제한 pool의 독립 분할 가능성도 구분한다.

독립 QA는 q/y·support·source·feature 계약의 내부 정합성을 검증한다. 이전 Public 실패 원인이100% 증명됐거나, 새 생성법이 공식 Test의 물리 law와 동일하거나, 큰 모델 없이/큰 모델이면 성능이 개선된다고 보장하지 않는다.
