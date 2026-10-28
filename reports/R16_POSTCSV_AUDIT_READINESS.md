# R16 첫 앙상블 완료 후 감사 실행 준비

작성: 2026-10-28 16:18 KST. **최종 감사 결과가 아니라 실행 준비·복구 확인서다.**

## 현재 실행과 순서

1. D0/H0 각각 Tree 5-Fold와 신경망 4종 Fold0를 끝내 첫 후보 `submissions/r16_multisupport_full_r1/submission_r16.csv`를 생성한다.
2. 첫 실행의 완료 영수증, 모델/정책/소스 해시, CSV의 SHA·1,500개 고유 ID·열·유한 범위 검증을 통과한 뒤 네 감사를 별도 CPU 프로세스에서 한 번 실행한다.
3. 이미 승인된 신경망 Fold1~4 후속 학습도 첫 후보 검증 후 시작한다. 감사는 CPU 1스레드/낮은 우선순위로 수행한다. 원본 정책·체크포인트·CSV를 덮어쓰지 않는다.

현재 Tree 10쌍은 완료. D0 CNN은 62epoch까지 복구 후 완료했고 63epoch를 진행 중이다. 첫 앙상블 CSV와 네 감사의 최종 수치는 **아직 미산출**이다. 이 문서의 상태는 스냅샷이며 실제 진행은 아래 state를 확인한다.

- 학습: `../ops/r16_multisupport_full_r1/state.json`
- 사후 감사: `../ops/r16_first_ensemble_audit_r1/state.json`
- 후속 Fold1~4: `../ops/r16_multisupport_fivefold_r1/state.json`
- 실행 프로토콜: [PROTOCOL.json](../ops/r16_first_ensemble_audit_r1/PROTOCOL.json)
- 사전 검사: [PREFLIGHT_VERIFICATION.json](../ops/r16_first_ensemble_audit_r1/PREFLIGHT_VERIFICATION.json)

## 1. score 순위·포화·보정

Tree 10개 개별 확률과 최종 정책의 Tree/NN 확률에 대해 확률 평균, 로짓 평균, 공통 ln4 차감 로짓 평균, 패널 내부 순위 평균, 고정 local_fit ECDF 순위 평균을 비교한다.

ROC-AUC, 관측 prior/클래스50:50 Brier·Logloss, ECE10·신뢰도 곡선, ≥0.9·>0.98 비율, float64/.17g CSV roundtrip 동률을 보고한다. 조건부 AUC는 모든 방법에 **동일한 확률 평균 상위25% 부분집합**을 사용한다. 한 클래스만 남으면 미정의로 표시한다.

- 공통 ln4 차감은 순위를 바꾸지 않는다. 확률 보정 효과와 AUC 효과를 혼동하지 않는다.
- 50:50 가중은 평가 prior 정의이며 Test prior를 추정한 것이 아니다.
- 패널 내부 rank는 코호트 의존 진단에 한정한다. 이 결과를 Test 집단 순위 후처리에 그대로 적용하지 않는다. 무모수 후보에도 사후 선택 편향은 있다.

## 2. 연산자별 조건부 회귀 개선

R9 Fold0 Tree, R9 Tree5 평균을 R16 D0/H0 Tree10 평균과 같은 고정 패널에서 비교한다. ALL 및 6개 연산자별 RMSE·bias·기술적 calibration slope, 연산자 사이 bias 분산, local_fit에서만 적합한 new~old 선형 투영의 잔차-기존 오차 상관, 예측 불일치 행의 개선 승률을 보고한다. 장시간/HF·SF 전체 RMSE는 기존 패널의 클래스 균형 가중치를 유지한다.

투영 계수는 평가 패널에서 재적합하지 않는다. R9는 509개 참조를 사용했으므로 참조 독립성 조건은 R16과 같지 않다. 여러 학습 요소가 함께 달라진 결과만으로 원인 하나의 인과효과를 확정하지 않는다.

## 4. 정적 채널 치환

local_report의 고유 무추가 배경창만 사용한다. 다른 사이트의 **관측 마스크가 정확히 같은** donor를 우선 선택하며, 없으면 제외 수를 공개한다. 별도 +0~12dB 정적 기울기 변형도 비교한다.

Ch4만, Ch7만, 둘 다 바꾸는 입력효과와 원래 q를 유지한 pooling-only 효과, input+pooling 동시효과를 분리한다. Tree는 해당 채널의 펼침·집계 특징만 교체하고 나머지 물리 특징은 유지한다.

입력만 Ch7을 바꾸면 정상 입력의 `Ch7=sqrt(w)` 계약에서 벗어난다. 따라서 별도 감사의 mask-only 계산을 사용하며 **production forward의 검사는 그대로 둔다**. 결과는 OOD 특징 민감도이며 장비 암기의 인과 증명이 아니다. 15채널 모델 재학습 대조는 이번 즉시 감사에 포함하지 않는다.

## 5. 그룹 부트스트랩

local_fit은 400개 고유 배경×6연산자=2,400행이나 배경 원천 그룹은 **10개**다. 1,000회 background_group 블록 부트스트랩에서 그룹에 속한 모든 행을 함께 뽑고 D0/H0의 추출을 짝짓는다. 가중치는 in-bag에서 적합하고 최적 가중치와 중심점의 위험 차이는 out-of-bag 그룹에서 계산한다.

모델별 가중치 표준오차/95% percentile 구간, OOB RMSE 차이의 bootstrap SD, 중심점 `(0.5,0.125,0.125,0.125,0.125)`의 1-SE 권고를 보고한다. SD를 √1000으로 나누지 않는다. reference_component 단위 민감도도 별도로 계산하되 교차 의존성이 완전히 제거됐다고 주장하지 않는다. 10개 그룹에서 얻은 구간은 넓고 불안정할 수 있다.

1-SE 중심점이 권고되더라도 첫 동결 정책/CSV를 몰래 변경하지 않는다. 별도 권고 파일로 남긴다.

## 사전 검증 실측

- 실제 local_fit 16개에 대한 회귀 Tree 5,160특징 재구성: **bitwise 일치**.
- R9 5쌍+R16 10쌍: 모델 로드 및 개발 예측 연결 통과.
- CNN/FDY/iTransformer/TFGrid 네 구조: mask-only와 정상 forward 물리 풀링 최대 오차 **0**. 분류기 Ch4/Ch7 불변성 확인. 정상 forward는 잘못된 sqrt(w) 치환을 계속 거부한다.
- 로그 오프셋 순위 불변성, 고정 ECDF의 쿼리 집단 독립성, 동률/단일 클래스 처리, 가중치 cap 및 부트스트랩 재현성 통과.
- Test/query 배열 읽기와 production 경로 쓰기 차단, 별도 경로 그림 저장 통과.
- 저장된 Tree10의 **개발 local_fit 재검산** AUC 0.960339, Brier 0.065816(관측 prior), 0.086730(50:50), ≥0.90 비율 57.17%. 이 수치는 새 고정 패널 감사나 공식 LB 결과가 아니다.

## 학습 프로세스 중단 발견과 검증된 복구

저장 state는 62epoch 진행으로 남아 있었으나 기존 PID 12773/21489/23279는 실제로 존재하지 않았다. 마지막 완료 체크포인트/epoch 로그는 **61epoch**였다. Python 예외 기록이 없어 종료 원인은 미확정이다.

원본 trainer를 변경하지 않고 별도 복구 진입점을 추가했다. 원본과 재개 코드의 **epoch 학습 본문 AST가 동일**하며, 원본 소스·데이터·Tree·특징 캐시 SHA를 확인했다. 마지막 완료 epoch의 Adam·cosine scheduler·best 기록·CPU/MPS/Python/NumPy RNG를 복구하고 62epoch부터 진행한다. 미완료62epoch의 중간 스텝은 버린다. 61epoch 시점 파일들은 별도 SHA 백업했다.

기존 Python3.14 가상환경이 Documents 아래 클라우드 관리 패키지 파일 읽기에서 정체되어, 이미 설치된 로컬 Python3.12/PyTorch2.14.1 환경으로 실행 경로를 옮겼다. **Mac MPS GPU를 계속 사용**한다. 새 라이브러리를 설치하거나 기존 가상환경을 수정하지 않았다.

- CPU/MPS Adam·scheduler·RNG 복원 fixture: 파라미터/손실 오차 **0**.
- 기존 최적 회귀/분류 checkpoint의 MPS local_fit 예측 재현: 최대 절대 차이 **0**.
- NumPy2.5.4/2.3.5의 epoch62(61,446행) permutation SHA 동일: `4226f43c5c99256ed8b29bf8ee20f5c150eab6d792997c9b182f7c583bc760fe`.
- 62epoch 완료를 실제 로그에서 확인: TRAIN RMSE 0.046119, local_fit RMSE 0.113640, best local_fit RMSE 0.100030(epoch46), best 분류 AUC 0.943053(epoch50).
- MPS 수치연산 전체가 다른 프로세스에서도 항상 bitwise 동일하다는 보장은 하지 않는다.

복구 영수증: [VERIFIED.json](../ops/r16_multisupport_resume_20261028_r1/VERIFIED.json). 원래 감독 로그와 checkpoint snapshot을 보존했다. Fold1~4 대기만 하던 종료 프로세스도 영수증을 보관한 후 중복 없이 다시 대기시켰다. 자동 재시도·공식 업로드는 하지 않는다.

## 실행 파일과 완료 후 산출물

- `ops/audit_r16_first_ensemble.py`: 완료 영수증 기반 일회성 감사/대기.
- `ops/r16_postaudit_metrics.py`: 순위·보정·가중 위험·그룹 부트스트랩.
- `ops/verify_r16_postcsv_audit.py`: fixture 및 실제 local_fit 연결 검사.
- 완료 후 `reports/r16_first_ensemble_audit_r1/R16_FIRST_ENSEMBLE_POSTCSV_AUDIT.md`, `RESULTS.json`, `reliability.png`, `ONE_SE_RECOMMENDATION.json` 생성 예정.

이번 감사는 원음 풀 확대나 새 후처리 채택을 실행하지 않는다. 509개 전체가 강한 hull/day/time 연결 규칙에서 연결된 기존 제약은 남는다. B/C 잠금을 유지하면서 400+개를 복구할 수 있다고 미리 보장하지 않는다. 일부 E=B+A 분해의 비식별성은 인정하지만 정량적 Bayes error 하한이나 시간 특징만이 유일한 해법이라는 결론으로 확대하지 않는다.
