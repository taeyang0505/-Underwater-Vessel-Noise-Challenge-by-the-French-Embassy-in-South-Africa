# R6 guardrails v2: 사전 고정 실행 계획

이 계획은 실제 데이터 모델 학습 전에 고정했다. GPU 본학습·Test 이용·공식 제출은 실행하지 않는다. 기존 R4/R5 데이터·모델·코드는 덮어쓰지 않는다.

## 구현 변경

| 위치 | 구현 |
|---|---|
| `pipeline/source_split_r6.py` | 기존 허용 TRAIN/INNER 원천에서 새 역할을 분리. 기존 OUTER·예약 원천 제외. 동일 센서의 900초 support 간 24시간 purge. 실제 원본 행 ID로 경계 분만 결측 처리. |
| `pipeline/robust_sampler_r6.py` | minute-first/짧은 native passage/경계 통과/2초-first 무신축 연산자. 전체 native 참조의 peak-minute 대비 SNR U[-10,+10]dB. 관측 결측과 독립인 배치. EQ 최대 ±6dB, 추가 lowpass 없음. |
| `pipeline/hard_negatives_r6_v2.py` | TRAIN만 사용한 LF bump 90분위수 및 LF > max(안전 중고역, 고역)+3dB 후보. 명시적 무추가 TRAIN 20% 중 절반에 후보를 배정. 자연 선박 부재 라벨을 추정하지 않음. |
| `pipeline/relative_grid_r6_v2.py` | 대역별 median/q20/q05 차감, 인접 관측 분 차분, 저·중고역 동조 특징, 상대 대역 프로파일, 관측 마스크, sqrt(w) 총8채널. db/100·q0·Bhat/E 없음. |
| `pipeline/relative_models_r6_v2.py` | 셀 sigmoid 마스크→에너지 풀링. 분별 중역·고역·안전역 문맥을 모든 주파수로 전달. 자유 presence head 대신 고정 share/마스크 시간 대비 점수. |
| `pipeline/physics_features_r6_v2.py` | 저/중/고/안전1~5kHz bump, 상관계수, broadband std, tilt 및 유효 마스크·관측 수 총27개. |
| `pipeline/cpu_ablation_r6_v2.py` | Mac CPU 전용 CNN/LightGBM. 학습 전용 prior는 과거 특징 비교군에만 유지. 임의 출력 보정 없음. |
| `ops/run_r6_cpu_guardrails_v2.py` | 새 immutable NPZ·코드/설정/metadata SHA, 독립 프로세스 실행. 7개 arm을 모두 동결한 뒤 3개 공통 진단 패널 평가. |

## 고정 비교 실험

- CNN: R4법칙/과거8채널/global, R4법칙/상대8채널/global, R6법칙/과거8채널/global, R6법칙/상대8채널/global, R6법칙/상대8채널/local 총5개.
- LightGBM: R6법칙에서 과거8채널과 상대8채널 각각1개. 모두 동일27개 물리 요약을 추가한다.
- TRAIN 8,192개, 각 법칙의 INNER 1,024개. TRAIN 배경 draw와 명시적 무추가 위치를 R4/R6 비교군에서 동일하게 사용한다. 과거 R4 양성 LHS target/duration 쌍은 유지하고 위치만 고정 스케줄로 재배열한다.
- CNN 최대100epoch, early stopping7, Adam, dropout0.1, batch128, 4 CPU threads. 손실=share MSE+0.25 에너지 가중 cell MSE+0.05 고정 파생 score BCE.
- LightGBM 최대1,500trees, ES40. 회귀 선택은 INNER site-macro RMSE, 분류 선택은 INNER binary logloss.
- CNN checkpoint도 각 생성법칙의 INNER site-macro RMSE로 선택. 다른 target분포의 INNER RMSE 절댓값으로 개선을 주장하지 않는다.
- cross-generator / cross-sensor / 결합 패널 각각1,024개. 전 모델 동결 후 1회 평가하며 재학습·튜닝·앙상블 선택에 사용하지 않는다. 공통 패널에서 상수0/.1/각TRAIN평균도 평가한다.
- 센서 단독/결합 패널은 같은 원천 draw를 공유하므로 독립 표본으로 합산하지 않는다. 이는 전체5fold나 기존 Wide CNN 완전 재현이 아닌 소규모 기전 실험이다.

## 수식·물리 조건

알려진 추가 성분 A, 배경 B, 원래 관측 E=B+A에서 q=A/E 및 y=sum(O*A)/sum(O*E)를 보존한다. 입력은 .1dB 반올림하며, 모델 w는 반올림 관측에서 계산한다. 따라서 원래 y와 sum(w*q)는 완전히 같지 않을 수 있다. 양자화 최대 오차 .05dB에서 이 차이는 tanh(ln(10)*.05/20)=0.0057564 이내이며 float32 허용오차와 별도로 검증한다. 정답을 교사 점수나 반올림 값에 맞춰 다시 정의하지 않는다.

20%는 명시적 무추가 요청 비율이다. 실제 추가가 관측되지 않은 분에만 배치되면 관측 정답은0이므로 전체 observed-zero 비율은20%보다 클 수 있다. 이는 관측 위치에 맞춰 배치를 재추첨하지 않은 결과이며 별도 보고한다. q05/min도 순수 배경 정답이라고 가정하지 않는다.

sqrt(w)는 공통 dB 오프셋에 정규화 불변이다. 임의의 오프셋을 반올림 전에 적용하면 격자 경계 때문에 작은 차이가 생기며, 대역별 EQ에는 불변이 아니다. 상대 스펙트럼 프로파일 역시 정적 대역 차이를 완전히 제거하지 않는다.

## 채택하지 않은 수치 맞추기

공개 Benchmark의 알고리즘이 확인되지 않아 RMSE 제곱으로 Test target 모멘트를 추정할 수 없다. 상수 c의 MSE는 E[y²]-2cE[y]+c²이다. 제시된 전체 RMS .246/양성 비율 .75~.8/양성 평균 .20~.23에서는 상수.1 RMSE가 약 .184~.201로 계산되므로 두 상수 baseline을 모두 .24~.26으로 만들라는 조건도 서로 양립하지 않는다.

Test 입력 집단·리더보드 점수에서 추정한 분포에 생성기·target·사이트 비율을 맞추지 않는다. SNR/EQ/연산자 범위는 사전 설정한 스트레스 조건이며, 공식 비공개 생성기와 일치한다고 주장하지 않는다. 자체 원천별 관측 요약과 target 모멘트는 결과로만 보고한다.

## 아직 확인되지 않은 것

- 새 검증의 공식 LB 예측력, 원인별 인과관계, 개선 AUC/RMSE는 실측 전 미확정이다.
- 현재 비교 CNN 약64k 파라미터는 R5 Wide CNN1.488M 재현이 아니다.
- 고정 진단 score는 확률 보정값이 아니고 AUC 향상을 보장하지 않는다.
- 경계 flag는 이상 원인의 확정 진단이 아니다. 후보 분 마스킹의 민감도는 별도 후속 실험이 필요하다.
- 센서 그룹2와 일부 소수 사이트에는 새 INNER 원천이 없어 site-macro도 TRAIN 모든 사이트를 대표하지 못한다. cross-sensor St Francis는 SHA 규칙으로 정했고 Test 비율을 쓰지 않았다.

구체적 설정은 `../configs/r6_cpu_guardrails_v2.json`, 원천 검증은 `R6V2_SOURCE_SPLIT_AUDIT.md`를 참조한다.
