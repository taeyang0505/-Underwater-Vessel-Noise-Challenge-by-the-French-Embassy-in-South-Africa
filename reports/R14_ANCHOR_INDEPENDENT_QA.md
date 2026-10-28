# R14 anchor 완료·평가 독립 QA

**계산·무결성 검사는 통과했지만 동시 성능 목표는 충족하지 못했다.** fit에서 고정된 정책은 `Blend50_c0` 회귀와 독립 classifier score(λ=1)다. 평가 패널의 결과를 보고 다른 조건으로 바꾸지 않았다.

## 실제 완료 검증

[완료 모델 QA](R14_ANCHOR_COMPLETED_MODEL_QA.json)는 두 CNN 각각 연속100 epoch·Adam·dropout.1·early stopping 없음·27,601 parameters를 확인했다. 최초 validation-fit 최솟값의 checkpoint는 c0 epoch64, c2 epoch89이며, 저장 SHA·엄격한 state shape 복원·모든 tensor finite를 확인했다. optimizer state는 저장되지 않았으므로 그 step counter까지 독립 인증한 것은 아니다.

Tree는880특징이며 저장된 회귀 trees는 c0 1,496개, c2 1,499개, 독립 분류는302개로 동결 기록과 일치한다. source/data/basis 핀도 변하지 않았다. [전체 모델 동결 파일](../models/r14_anchor_cpu_r1/ALL_MODELS_FROZEN.json)의 SHA는 `6611e81558b8fdb4d8e4bde676efcbfc559e233c1aa8a484b605b9089f6d1b28`이다.

## 저장 예측 재검산

[최종 평가 QA](R14_ANCHOR_EVALUATION_INDEPENDENT_QA.json)는 17패널153행, site macro RMSE를 포함한2,788개 scalar를 재계산했으며 최대 차이는0이다. AUC는 producer와 다른 positive-negative 승률·동률0.5 방식으로 계산했다. 메타데이터의 site 열만 추가로 읽었고 원 acoustic/query dB 배열이나 Test는 열지 않았다.

저장 q와 w는 유한하고 [0,1]이며, w=0 위치의 q=0이다. `sum(wq)`와 CNN share의 최대 차이는 float32 재계산1.788e−7, float64 재계산1.189e−7로 허용3e−6 안이다. 원래 R13 동일패널의 ID·target·관측 w와 exact 일치했고 두 CNN/Tree의50:50 결합도 exact 재현했다. 이 검산은 예측 pooling의 일관성이며 q 정확도나 자연 선박 정답을 증명하는 것은 아니다.

fit1,000의 저장 예측에서 회귀 argmin과 score λ argmax를 독립 재선택해 고정 정책과 같음을 확인했다. 새 모델 fit·실제자료 추론·네트워크 작업은 하지 않았다.

## 사전 목표 확인

| 선택 정책 패널 | RMSE | share AUC | score AUC | share bias |
|---|---:|---:|---:|---:|
| R12_high_snr_long | 0.147058 | 0.996400 | 0.995300 | -0.102723 |
| R10_unseen__r9_complete | 0.049969 | 0.850625 | 0.948456 | +0.006996 |
| R11_full_smooth15_VS_st_francis_dense | 0.072777 | 0.802071 | 0.766365 | +0.008828 |
| R11_native_long15_VS_st_francis_dense | 0.068816 | 0.791571 | 0.792135 | +0.001723 |

unseen RMSE≤.0527/score AUC≥.92는 충족했다. highSNR RMSE<.145, full15/SF score AUC≥.82, native15/SF score AUC≥.80은 실패했다. local highSNR 오차가 기본0.15634보다 낮아진 사실과 사전 목표 통과 여부를 구분한다.

[전체153행 CSV](r14_anchor_cpu_r1/results.csv)와 [동결 정책](r14_anchor_cpu_r1/selected_policy.json)에 원값을 보존한다. CSV의 `classifier_ge099`는 모든 행에서 **공통 R14 classifier의 포화비율**이다. baseline comparator 행의 자체 classifier 포화로 해석하면 안 된다.

이17패널은 이미 개발 과정에서 노출됐고 reference 녹음도 공유한다. 따라서 blind OOF나 공식 대회 성능으로 부르지 않는다. 본 결과를 근거로 원 R13 정책을 소급 변경하거나 추가 후보를 선택하지 않았다.
