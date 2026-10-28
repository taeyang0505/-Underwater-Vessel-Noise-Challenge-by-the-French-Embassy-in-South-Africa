# R7 소형 모델·단순 대조군 공통 패널 평가

새 학습은 TRAIN-only27개 물리 특징 Ridge(alpha=10)1회입니다. 상수, 기존 LightGBM, fixed100CNN을 동일 INNER와3개 cross panel에서 비교했습니다. 모든 패널은 이미 평가한 개발 자료이며 독립 blind 평가가 아닙니다. Test·기존 OUTER·GPU·추가 CNN학습은 사용하지 않았습니다.

| 고정 대조군 | INNER RMSE | CROSS_GENERATOR | CROSS_SENSOR | CROSS_SENSOR_GENERATOR |
|---|---:|---:|---:|---:|
| constant0 | 0.199072 | 0.189820 | 0.200983 | 0.199626 |
| constant0_1 | 0.160120 | 0.155601 | 0.178197 | 0.176745 |
| TRAIN_mean | 0.159162 | 0.156678 | 0.181744 | 0.180343 |
| Ridge27_alpha10 | 0.115238 | 0.106870 | 0.157309 | 0.164634 |
| frozen_R6_relative_LightGBM | 0.073374 | 0.071427 | 0.224560 | 0.225680 |
| frozen_fixed100_R6_relative_global_CNN | 0.059377 | 0.060657 | 0.095061 | 0.110407 |

Ridge imputer/scaler는 TRAIN에서만 fit했고,예측은사전고정[0,1]clip입니다. 정답·site·reference ID는입력특징이아닙니다. 기존CNN은ownINNER로선택된epoch31 best_window를그대로사용했고,다른agent의고정추론을재검증하여중복CNN추론을하지않았습니다.

사이트·target 구간별 MAE/bias/RMSE·presence AUC와 모든 파일 바인딩은 [JSON](R7_SMALL_MODEL_CONTROLS_RESULTS.json)에 포함됩니다. 이 결과를 공식 LB 개선량이나 모델 크기 변화의 단독 인과효과로 해석하지 않습니다.

## R5 실패와 R6 0.059377 사이에서 실제로 달라진 것

아래 R5 수치는 보관된 Fold0 raw checkpoint summary의 자체 INNER 값입니다. 실제 제출 경로·후처리 재현은 별도 감사 대상이며, 이 표는 공개 LB와 같은 평가분포를 가정하지 않습니다. R5도 이미 direct mask를 사용했습니다. R6에서 새로 prior-logit anchor를 제거했다거나 cell/presence loss 가중치를 처음 추가했다는 설명은 맞지 않습니다.

| 항목 | R5 raw Wide CNN | R6 relative global tinyCNN | 근거 |
|---|---|---|---|
| 파라미터/구조 | 1,488,002, width128/5blocks | 64,449, width32/3blocks+주파수 global context | [Wide](../pipeline/hybrid_models_r5.py#L70), [tiny](../pipeline/relative_models_r6_v2.py#L57) |
| 원천·분할 | 기존 Fold0 TRAIN69 reference clips/12groups, INNER25/5; TRAIN primary53,244 | 새 sensor/source role 분리, TRAIN51/9·INNER18/3; fixed8192draws(unique5639)/INNER1024 | [R6 split](../pipeline/source_split_r6.py#L127), [감사](R6_GUARDRAILS_INDEPENDENT_AUDIT.json) |
| 생성·정답 분포 | 목표 share를 .005–.95로 선추출, observed energy에서 alpha 역산; power timewarp/fade; 매epoch 새합성 | peak-SNR 기반 alpha, native minute/short passage; mask와 독립 geometry; observed y=0도 허용; 고정 합성배열 반복 | [R4](../pipeline/passage_sampler_r4.py#L198), [R6](../pipeline/robust_sampler_r6.py#L115), [고정배열 루프](../pipeline/fixed100_r6.py#L27) |
| 입력·pool | raw legacy8, TRAIN-fitted background prior q0/ratio가 추가입력; raw energy weights | 0.1dB relative8, q05/sqrt(w) 포함; prior 입력 없음, rounded energy weights·unrounded label | [R5 grid](../pipeline/grid_inputs_r5.py#L117), [R6 grid](../pipeline/relative_grid_r6_v2.py#L1) |
| 출력·loss | direct sigmoid q→에너지 pool; 별도 presence head; shareMSE+.25cellMSE+.05BCE | direct sigmoid q→에너지 pool 유지; mask로 계산한 fixed score에 BCE; 가중치는 동일 | [R5 loss](../pipeline/neural_r5.py#L153), [R6 score](../pipeline/relative_models_r6_v2.py#L20), [R6 loss](../pipeline/fixed100_r6.py#L31) |
| optimizer/dropout | Adam lr.001, wd.0003, dropout.1 | 동일 Adam lr.001, wd.0003, dropout.1 | [R5 config](../configs/r5_training_gpu_r1.json), [R6 config](../configs/r6_fixed100_v1.json) |
| epoch·선택 | 26epochs, ES7; best19 | 실제100epochs/noES, ownINNER window best31 | [R5 summary](../ops/server_backups/r5_gpu_r1/restored_first_cnn_v1/fold_00/preprocessing_pilot/raw/bg_cnn/summary.json), [R6 summary](../models/r6_fixed100_v1/r6law_relative8_global/summary.json) |
| 자체 INNER RMSE | 0.090849261, 기존합성label | 0.059377308, 새합성label | 서로다른문제분포, 수치만직접우열판정불가 |

**해석:** 원천·SNR/target law·representation·간섭control·평가split·학습기간이 함께 바뀌었습니다. 소형모델이 더 맞는 표현에서 유용한 패턴을 학습했는지는 위 공통패널 대조군으로 평가할 수 있지만, 모델 축소 하나가 R5의 공식 제출 실패를 해결했다고 증명한 실험은 아닙니다. 두 CNN 모두 from-scratch task model이며 대형 pretrained backbone의 성능을 검증한 결과도 아닙니다.

이번 공통 패널에서 CNN은 상수·27특징 Ridge보다 낮은 오차를 보였습니다. 반면 기존 LightGBM은 INNER가 좋아도 sensor 패널에서 상수0.1보다 오차가 컸습니다. 따라서 입력으로부터 학습할 신호는 있지만 INNER 성능만으로 source 전달을 보장할 수 없다는 점을 동시에 보여 줍니다.

재현성 기록: Ridge는 최초1회만 fit했습니다. 독립 JSON 복원 검사가 설치 sklearn의 float32 mean/scale casting 순서와 달라 중단됐고, 원래 runner/config·저장계수를 보존한 replay-only patch ledger로 산술 순서만 수정했습니다. 계수·alpha·데이터·모델 선택은 바꾸지 않았으며 추가fit은 없습니다. 최종 복원객체와 독립 JSON 예측 차이는0이고,4개 패널104개 지표를 독립 재계산했습니다. [독립 검증](R7_SMALL_MODEL_CONTROLS_VERIFICATION.json)
