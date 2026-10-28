# R16 실행 현황

확인 시각: 2026-10-28T05:18:25+09:00 (Asia/Seoul)

R11 고정 스트레스 평가와 본체 Tree 회귀·독립 분류 5-Fold는 완료했다. 알려진 일반화 실패를 검토하기 위해 완전 epoch 경계에서 일시 정지; 자동 재개 없음.

## 주 교차 검증과 남은 스트레스

| 평가 | 모델 | RMSE | 독립 분류 AUC / 동결 score AUC |
|---|---|---:|---:|
| 별도 배경 local_gate | 본체 Tree5 | 0.076535 | 0.970384 (독립 분류) |
| 각 Fold의 자체 CV 5,000건 결합 | 본체 Tree | 0.077040 | 0.968760 (독립 분류) |
| R11 native 15분 vs 자연 HF 변동 | 본체 Tree5 | 0.139986 | 0.259200 (독립 분류) |
| R11 native 15분 vs St Francis 고밀도 | 본체 Tree5 | 0.146637 | 0.337596 (독립 분류) |
| R11 평탄 15분 vs 자연 HF 변동 | R16 소형 60% Tree + 40% CNN | 0.134024 | 0.184856 (score) |
| R11 native 15분 vs 자연 HF 변동 | R16 소형 60% Tree + 40% CNN | 0.134767 | 0.177716 (score) |

위 행들은 서로 다른 평가 자료다. 같은 의미의 점수로 직접 비교하거나 공식 LB로 환산하지 않는다. R11의 실패는 남아 있으며, 주 기준은 사전 지정한 별도 참조·크롭, 미관측 사이트 및 독립 LF/HF 연산자 전이다. R11 결과로 레시피를 다시 튜닝하지 않았다.

추가 확인: 본체 Fold0 TRAIN 양성 49,157건의 실제 추가 활성 시간 중앙값은 2분, 최대 10분이며 12분 이상은 0건이다. 15분 이상 native 참조 3개는 모두 C 홀드아웃의 같은 연결 컴포넌트에 있다. 이를 TRAIN으로 옮기면 기존 C 평가는 독립 검증으로 재사용할 수 없다. 장시간 coverage와 자연 HF 음성 구분을 보강한 뒤 새 미관측 참조 평가에서 확인해야 한다. 이 결함을 Public 실패의 유일한 원인으로 단정하지 않는다.

## 완료 Tree5의 공유 selection 지표

| Fold | TRAIN 행 | selection RMSE | 독립 분류 AUC |
|---:|---:|---:|---:|
| 0 | 61,446 | 0.096288 | 0.957520 |
| 1 | 61,368 | 0.095507 | 0.959401 |
| 2 | 61,259 | 0.098385 | 0.960026 |
| 3 | 61,546 | 0.092519 | 0.964544 |
| 4 | 61,685 | 0.093925 | 0.959421 |

selection은 2,400건의 공유 개발셋이며 위 수치를 전체 OOF라고 부르지 않는다.

## 신경망 Fold 0

| 모델 | 완료 epoch / 100 | 최신 TRAIN RMSE | 최적 selection RMSE | 최적 독립 분류 AUC |
|---|---:|---:|---:|---:|
| bg_cnn | 27 / 100 | 0.053343 | 0.075496 | 0.930007 |
| fdy | 대기 | — | — | — |
| itransformer | 대기 | — | — | — |
| tfgrid_fdy_conformer | 대기 | — | — | — |

4개 모델 모두 100epoch, Adam, early stopping 없음. 회귀/분류 최적 체크포인트를 각각 보관한다. 현재 단계에서는 신경망 Fold 1~4를 시작하지 않는다.

## 고정 안전 계약

- relative8 유지; PCA·LF anchor·q0·절대 dB·highshare 배제.
- 회귀는 Share MSE + 0.25 × 에너지 가중 cell MSE. 분류는 파라미터/gradient가 분리된 독립 BCE.
- 최종 회귀 앙상블 Tree5 ≥50%, 신경망 아키텍처별 ≤25%. score의 share 비중은 0.
- TRAIN 원천/그룹 분리 및 24시간 purge, 정밀 정답 보존, 0.1dB 입력·가중치 유지.
- 최종 모델·정책 동결 전 Test를 읽지 않는다. 공식 업로드하지 않는다.

## 파일

- [R11 전체 결과](r16_r11_stress_r1/SUMMARY.md)
- [본체 Tree5 장시간/HF 실패 감사](r16_problem_gate_r1/tree_stress/SUMMARY.md)
- [장시간 자료 공백 및 특징 검토](r16_problem_gate_r1/FEATURE_FAILURE_REVIEW.md)
- [본체 Tree 평가](r16_production_tree_evaluation_r1/SUMMARY.md)
- [신경망 연결 검증](R16_NEURAL_PRODUCTION_READINESS.md)
- [신경망 실제 진행 JSON](../models/r16_production_nn_fold0_r1/progress.json)
- [감독 실행 상태](../ops/r16_production_neural_r1/state.json)

`submission_r16.csv`: 4개 신경망 완료 → 제한 앙상블 적합/정책 동결 → 교차 검증 → 동결 Test 추론 후 생성. 현재는 미생성.

## 실행 보류

사용자 지시에 따라 알려진 실패를 먼저 검토한다. 체크포인트를 보존했으며 신경망 후속 모델, 자동 재개 및 최종화 실행을 보류한다. 앞서 제시한 연속 실행 12~15시간 ETA는 현재 적용하지 않는다.

## 승인된 소형 보완 실험

네 조건 각100epoch와 독립 검산을 완료했다. 시간 특징 단독은 주 교차 RMSE를 개선했으나 데이터 결합안은 사전 기준 미달이다. 수정 데이터 본체 재개와 후보 CSV 생성은 진행하지 않았다. [소형 비교 결과](r16_support_temporal_r1/index.html) · [사전 기준](r16_support_temporal_r1/PROTOCOL.md)

## 최신 다중 support·타깃 보존 검산

[방법 A/B 실측 및 F1 본체 연결 검사](r16_multisupport_r1/index.html). 저장 예측 50:50 결합은 장시간/HF RMSE 한 항목 미달, 타깃 보존 재학습도 동시 통과하지 못했다. 본체 추가 학습·CSV 생성은 실행하지 않았고, 지시된 동시 통과 조건과 본체 완주 조건의 우선순위 답변을 기다린다.
