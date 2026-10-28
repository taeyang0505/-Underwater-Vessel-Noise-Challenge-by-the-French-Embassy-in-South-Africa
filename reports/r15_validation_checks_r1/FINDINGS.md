# R15 독립 확인: 정지·score 정책·규정·식별 한계

학습이나 모델 추론을 실행하지 않았다. 기존 production 코드·checkpoint·데이터·정책은 수정하지 않았다. [독립 검증 JSON](POLICY_PAUSE_RULES_VERIFICATION.json)과 [실제 process 관측](PROCESS_OBSERVATION.json)에 출처 SHA를 기록했다.

## 1. 학습 정지 및 checkpoint 보존

PID 80906은 별도 읽기 전용 `ps`에서 **Ts**였다. TFGrid 마지막 완료 epoch와 best epoch는 모두 **37**, best selection RMSE는 **0.058705338189848084**다. 중지 시 epoch38은 미완료 상태이므로 ‘38epoch 학습 완료’라고 표시하지 않는다.

`best.pt`와 `latest.pt`는 각각 31,442,721bytes이며, 모두 기존 pause receipt의 SHA `172b81179f9fc7f0355b5ccc40bc4a6d43d627e01908652dfd61e593caf759c9`와 같다. 새로운 signal, 자동 재개, 학습을 수행하지 않았다.

## 2. ‘R14 최종 score에 classifier가 없었다’는 주장은 틀림

R14 **개별 NN**은 별도 classifier head 없이 `score=share`를 반환한다. 그러나 실제 제출한 **최종 R14 step1**은 동결 정책의 `lambda=0.25`, `classifier=highshare`를 적용했다.

\[
s=\operatorname{clip}(0.75\,\hat y+0.25\,p_{\text{highshare}},0,1).
\]

여기서 main step1의 classifier는 다섯 Tree-highshare classifier 확률의 산술평균이다. 따라서 ‘NN 회귀 배합의 CNN 비중이81%’라는 사실과 ‘최종 score에 classifier가 포함됐는가’는 서로 다른 질문이다. raw classifier 배열을 최종 query NPZ에 **저장하지 않은 것**도 실제 계산에 사용하지 않았다는 뜻이 아니다.

고정된 실제 `apply_policy` 함수에 두 개의 순수 수치 fixture를 넣어 수식과 bitwise 일치를 확인했고, score가 share와 실제로 다름도 검증했다. 모델 예측·fit·Test 입력은 사용하지 않았다.

근거: [실제 policy](../../models/r14_production_outputs_r2/hybrid_fast3/policy.json), [score 혼합 코드](../../pipeline/ensemble_r14_production.py) 93–102행, [Tree5 classifier 평균과 적용](../../ops/run_r14_production_outputs_v2.py) 294–301행, [개별 NN 반환값](../../pipeline/models_r14_production.py) 95–97행.

## 3. 로컬 보관된 대회 규정 근거

로컬 [thread archive](../../scratch/discussions/thread_bodies.json)의 **35368**에는 Zindi Staff `daniil_zindi`가 2026-10-20 22:05에 답한 내용이 저장돼 있다. [공식 토론 원문](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/35368):

> This excludes tuning generators against test distributions and training adversarial-validation classifiers on test rows, even for diagnostics.

즉 Test의 p50/p90·사이트 구성·missing 패턴 등에 맞춰 생성기 파라미터를 선택하거나, Test 행으로 domain classifier를 학습하는 작업은 금지된다고 명시돼 있다. 모델의 최종 loss에 Test를 직접 넣지 않는다는 이유로 허용되는 것이 아니다.

[35370의 운영진 답변](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/35370)은 상수 scale 자체는 가능하되 자기 TRAIN/validation으로 보정하고 Test는 추론 전용으로 두도록 설명한다. 이 둘은 참가자 개인의 추측성 댓글과 구분했다. 적용되는 경로의 `AGENTS.md`는 찾지 못했으며, 이 제한의 출처를 AGENTS라고 설명하지 않는다.

**기술통계만 보는 모든 사후 EDA가 포괄적으로 허용된다는 적극적 문구는 이 로컬 운영진 답변에서 확인되지 않았다.** 기술통계를 새 생성기의 목표함수나 후보 선택 기준으로 사용하는 것은 명시된 generator fitting 금지와 구분할 수 없다. 따라서 R15 prototype은 TRAIN·공개 외부 원천만 사용한 사전 고정 물리 민감도 분석으로 제한하고, Test 분위수에 가까운 후보를 골라 채택하는 절차를 진행하지 않는다. 웹의 최신 원문 확인은 부모 에이전트의 별도 작업이다.

## 4. 입력 돌출만으로 ‘실제 Test share≈0.08’를 알 수 없는 이유

관측한 mixture 에너지를 E, 알려지지 않은 배경을 B, 추가 성분을 A라 하자. 임의의 c∈[0,1]에 대해

\[
A=cE,\qquad B=(1-c)E
\]

로 두면 관측 E=B+A와 모든 입력 통계는 동일하지만

\[
y=\frac{\sum OA}{\sum OE}=c
\]

이다. 본 검증은 같은 순수 수치 E에 c=0,0.08,0.4,0.8을 대입해 에너지 재구성과 서로 다른 정답을 직접 확인했다. 따라서 band별 max−median·HF bump·예측 평균이 같다는 정보만으로 실제 정답 평균·중앙값·무추가 비율을 식별할 수 없다.

이 예시는 **일반적인 비음수 에너지 분해에서의 불식별성**을 보이는 반례다. 임의의 모든 분해가 공식 reference 코퍼스·공식 생성기 제약에 실제로 속한다는 주장은 아니다. 공식 데이터 생성법에 대한 추가 가정과 독립 검증이 있어야 mapping을 좁힐 수 있다.

TRAIN의 특정 HF 구간에서 알려진 share 중앙값이0.066–0.093이었다는 사실 역시 Test에서 같은 값이라는 증거가 아니다. 두 집단의 conditional relation이 같다는 가정이 추가로 필요하다. 기존 Public 점수·출력 모멘트로 숨은 정답을 맞추거나, 반례를 이용해 새로운 정답 추정치를 만들지 않았다.

## 5. 확정 사실과 미확정 원인

확정된 것은 입력 기술통계 차이, 실제 동결 회귀 배합, 최종 classifier 혼합식, Public 결과 악화다. R9→R14에서는 생성법·입력·모델·epoch·배합·Fold 범위가 함께 달라졌다. 이 자료만으로 ‘실패 원인100% 규명’, ‘highshare/PCA/LF 하나가 유일한 원인’, ‘Test 정답의 중앙값0.08’, ‘장시간 양성은 Test에 없다’를 확정하지 않는다.

## 6. TRAIN 전용 prototype 전수 독립 검산 완료

[독립 물리 QA](PROTOTYPE_INDEPENDENT_QA.json)는 **PASS**다. 4,000개의 서로 다른 TRAIN 배경에 같은 원천·crop·배치·SNR을 공유한 고정4후보, 총16,000창을 검산했다. 후보마다 무추가800개·관측 양성3,200개다.

- 실제 원 TRAIN bank와 경계 minute QC를 독립 적용해 저장 B/O와 bitwise 일치를 확인했다. E=B+A도 bitwise 일치하며 q/y는 raw 성분에서 다시 계산했다.
- 원 background/reference bank의 전체 SHA와509개 허용 reference slice SHA를 별도로 확인했다. 30개의2초 power를 평균한 native minute부터 연속 crop·reference EQ·전체 native peak 기반 α·A_full까지 독립 재생성한 최대 절대오차는 약1.78×10⁻¹⁵다.
- 결측 마스크·raw/rounded dB·quantization pooling bound·관측 양성 지원·정확한0 target·4후보 무추가 입력 동일성을 확인했다. PCA/anchor/model feature는 없고 label을 목표 범위로 다시 쓰지 않았다.
- TRAIN 대4개 heldout 역할의 metadata ID/group 교집합은0이며, 같은 sensor의900초 support 간격은 최소24시간이다. heldout 수치 배열·정답은 열지 않았다.

첫 생성의 B dtype 불일치는 생성자가 보존한 실패 산출물과 수정한 새 실행으로 구분된다. 이번 독립 검산은 최종 동결 generator와 완료 manifest만 대상으로 하며 원 production 소스는 수정하지 않았다.

이것은 내부 물리·원천 정합성 검사다. Test 근접도·후보 채택·성능을 검증한 것이 아니다. 관측 O와 겹치는 placement를 조건부 선택하는 정책은 원 R9의 mask-independent geometry와 다르며,509개 reference는 공유되므로 reference-independent validation도 아니다. 이전 Test 기술통계를 본 뒤의 개발이므로 진정한 blind 실험이라고 부르지 않는다.

## 7. 저장된 기간·점수 기술통계의 독립 재집계

[기간·점수표 QA](SCORE_DURATION_SAVED_TABLE_QA.json)는 원 Test를 재열지 않고 `r15_score_duration_r3`의 기존 derived NPZ/CSV만으로6080개 분위수, 그룹 분모, 추정 사이트별 점수 분포·명시적 pairwise 순위확률·KS·상관을 재계산해 PASS했다. 모든 NPZ field를 `allow_pickle=False`로 읽었으며 ID는 object가 아닌 Unicode다. r1/r2는 이전 보존 산출물이며 최종 검산 대상은r3다.

HF>15dB801창 중 LF≤200Hz 돌출≤3dB는120창으로 전체의8.0%, HF그룹의14.9813%다. 동시 상승 조건364창은 전체의24.2667%, HF그룹의45.4432%다. 단순히 LF≤3dB인 전체279창을 HF조건의 분자로 쓰지 않았다. 원 SA 비교 대상은 Fold0TRAIN78,869창이며 공개171만행 전수가 아니다.

[R9 score–자신의 share 상관 검산](R9_OWN_SCORE_CORRELATION_QA.json)도 PASS다. Pearson0.937497, Spearman0.967879는 두 예측 사이의 관계이며, 숨은 양성/음성에 대한 AUC를 뜻하지 않는다. ‘활동 minute 수’와 연속 돌출 run도 실제 선박 통과 길이 정답이 아니다. 이 기술통계는 생성기의 acceptance·파라미터 fitting·새 후보 선택에 사용하지 않았다.
