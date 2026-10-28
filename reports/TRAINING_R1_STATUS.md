# R1 GPU 학습 상태 및 성능 진단

실행 `group_cv_physics_v2_r1_adam`은 2026-10-25T03:08:15.989754+09:00에 중단되었다. 완료된 CV fold는 **0/5**이다. 첫 fold의 7개 task model 학습과 inner 예측, 두 foundation의 inner 예측, inner ensemble policy 동결까지 완료되었다. TabICL outer 단계의 query batch 재현성 검사에서 실패했다. OOM이나 NaN에 의한 학습 실패는 아니다.

## 첫 fold의 내부 검증 성능

Train 평균값으로 고정 예측한 baseline RMSE는 **0.317663**이다. 아래 지표는 리더보드 점수나 전체 OOF 결과가 아니다.

| 모델 | Inner RMSE | 예측 − 정답 평균 편향 | 마지막 train RMSE | Best / 종료 epoch |
|---|---:|---:|---:|---:|
| lgbm | 0.305524 | -0.083156 | — | — |
| lgbm_physics | 0.283701 | -0.086436 | — | — |
| lgbm_retrieval | 0.283760 | -0.082117 | — | — |
| tabm | 0.289361 | -0.094595 | 0.174117 | 2 / 9 |
| cnn | 0.321155 | -0.180753 | 0.060159 | 6 / 13 |
| fdy | 0.326165 | -0.179504 | 0.075229 | 9 / 16 |
| itransformer | 0.344797 | -0.191482 | 0.095474 | 7 / 14 |

CNN·FDY·iTransformer는 train RMSE에 비해 inner RMSE가 크고, 현재 inner baseline보다도 나쁘다. 이는 성공적인 일반화로 보고할 결과가 아니다. Train loss는 share MSE 외에 presence BCE 및 cell loss를 포함하므로 제곱근이 그대로 train RMSE가 되지는 않는다. Early stopping으로 최적 inner checkpoint는 보존되었으나 이 사실이 최적 성능이나 충분한 학습을 증명하지는 않는다.

## 확인된 원인과 해석의 한계

- Native가 포함된 inner 예제에서 큰 과소예측이 나타났다. 세부 route별 근거는 별도 inner 전용 진단 보고서에 기록한다.

- Train에만 적용한 support/SNR 제한, receiver EQ, blending 및 관측 coverage 차이가 존재한다. 분포 차이의 존재와 전체 오차의 원인을 구분해야 한다. 초기 분포 차이만으로 짧은 참조가 주된 실패 원인이라고 단정하지 않는다.

- Source 격리와 24시간 purge는 유지한다. Outer 점수나 Test로 설정을 조정하지 않는다. 기존 inner/outer 자료와 실패 run을 덮어쓰지 않는다.

- 수치 안정성은 별도 OPS helper로 train context와 inner query만 사용해 auto/FP32 및 batch256/2를 검증한다. 기존 허용 오차를 완화하지 않는다.

## 보존한 증거

- [완료 산출물 ZIP](../ops/server_backups/snapshot_20261024T180856_204764Z_56e064fd.zip): ZIP SHA/CRC와 37개 entry SHA 검증 완료.

- [수치화된 실행 상태](../ops/r1_failure_status.json)

- [서버 실패 상태](../ops/server_preflight/r1_failure/training_state.json)

- [TabICL 오류](../ops/server_preflight/r1_failure/tabicl_seed_42_outer.log)

- [Epoch 로그](../ops/server_preflight/r1_failure/seed_42_training.log)

- [로그 해시 목록](../ops/server_preflight/r1_failure/local_log_manifest.json)

현재 수정 실험·CV 완료·추가 제출은 완료되지 않았다. Pod는 중지하거나 삭제하지 않았다.

## 후속 inner 전용 진단

- [Route별 오차 분석](R1_INNER_ERROR_DIAGNOSIS.md)
- [TabICL 수치 안정성 진단](../ops/tabicl_inner_precision_external.json): auto의 batch256/2 차이는 0.00065619로 실패했고, FP32는 0.000001371로 기존 허용 오차를 통과했다. 7개 위치 쌍·같은 batch 반복·fresh context 재구성과 query 순서 역전까지 확인했다. Frozen code/config와 실패 run은 수정하지 않았으며 전체 학습 재시작은 아직 하지 않았다.
