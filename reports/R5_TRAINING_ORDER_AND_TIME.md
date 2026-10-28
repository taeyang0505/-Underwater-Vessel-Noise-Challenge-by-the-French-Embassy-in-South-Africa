# R5 학습 순서와 시간 실측

현재 실행: `group_cv_r5_gpu_r1`, RTX 4090 24GB 1장.
설정: [r5_training_gpu_r1.json](../configs/r5_training_gpu_r1.json). 기존 integration 설정은 보존했다.

## 실행 순서

Fold 0을 아래 순서대로 완료하고, 같은 절차를 Fold 1~4에 반복한다.

| 순서 | 모델 / 설정 | 장치 | Fold당 시간 계획 범위 |
|---|---|---|---|
| 1 | Wide BG Mask CNN / raw | GPU | 약 30~80분 |
| 2 | Wide BG Mask CNN / 0.1dB 반올림 | GPU | 약 30~80분 |
| 3 | Wide BG Mask CNN / 반올림 + TRAIN 결측 증강 | GPU | 약 30~80분 |
| 4 | TFGrid + FDY + Conformer | GPU | 약 40~100분 |
| 5 | Physics / Retrieval LightGBM | CPU | 약 5~20분; 실제 boosting 시간 미측정 |
| 6 | TimeMoE-200M / adapter + LoRA | GPU | 약 3~8시간 |
| 7 | Dasheng-1.2B / adapter + LoRA | GPU | 약 5~12시간 |
| 8 | FDY-CRNN / 비교군 | GPU | 약 30~80분 |
| 9 | iTransformer / 비교군 | GPU | 약 30~80분 |

위 시간은 신경망이 대략 8~20 epoch 안에 종료된다는 조건의 초기 예산 범위다.
CNN 첫 실제 epoch, 다른 모델의 2-step 합성 CUDA probe, TRAIN 512개 CPU 입력 profile을 토대로 잡았다.
전체 epoch 실측이 없는 모델과 CPU 트리는 확정 ETA가 아니다. 다른 Fold, 관측 증강, 검증, 저장·백업에 따라 달라진다.
100 epoch까지 개선이 이어지면 이 범위를 크게 초과할 수 있다.

5-Fold 전체는 약 **60~150시간 규모의 임시 계획 범위**로 보아야 한다. 이전 5~6시간 예상은 현재 실행 설정으로 검증되지 않았으므로 사용하지 않는다.
시간을 줄이기 위해 승인된 최대 epoch나 early stopping을 임의로 낮추지 않는다.

각 Fold에서 CNN 3회 중 INNER가 가장 좋은 관측 정책을 고르고, 나머지 모델에 동일 정책을 사용한다.
CNN 3개는 최종 앙상블의 독립 3가지 모델이 아니라 전처리 비교다. Fold당 최종 후보는 7종, 학습 호출은 9회다.
전체는 **신경망 40회 + LightGBM 5회 = 45회**다.
INNER에서만 후보/앙상블을 선택한 뒤 정책을 고정하고 OUTER를 평가한다. OUTER/Test로 설정을 수정하지 않는다.

## 측정값과 한계

- Fold 0 첫 raw CNN: 53,244개, 416 optimizer updates, epoch 1 **224.93초**.
- epoch 1 TRAIN RMSE **0.180163**, INNER RMSE **0.131866**. 초기 한 epoch이며 최종 OOF·리더보드 결과가 아니다.
- TRAIN-only 512개 CPU profile: 2.093초. 창 합성 + 관측 처리 + 공통 8채널 변환 약 0.004089초/개. 53,244개로 단순 환산하면 약 3.63분/epoch 입력 처리다.
- 그 profile의 TRAIN aggregate prior 최초 fit: 63.21초. 이는 Fold별 초기화이며 각 batch에서 다시 fit하지 않는다.
- TimeMoE microbatch 8: warm probe 0.13352초/8개. 53,244개 GPU 연산 단순 환산 약 14.8분/epoch.
- Dasheng microbatch 8: warm probe 0.24861초/8개. 같은 환산 약 27.6분/epoch.
- 위 foundation probe는 resident batch의 두 번 forward/backward/Adam update다. 실제 effective batch 32 누적, CPU 전처리, INNER 평가, 저장을 포함한 전체 epoch 실측이 아니다.
- 경량 모델은 CPU 입력 처리의 비중이 크다. 생산 실행 중 코드·데이터·설정을 변경하지 않는다.

## 고정한 학습 설정

- 신경망: 최대 100 epoch, early stopping patience 7, Adam, dropout 0.1, gradient clip 1.0.
- 경량 신경망: effective/micro batch 128.
- 사전학습 두 백본: effective batch 32, microbatch 8 (4회 누적), evaluation batch 2.
- 사전학습 base는 동결하고 stem/head/LoRA만 학습한다. 역전파를 통과하는 백본 연산 비용은 남는다.
- LightGBM: 최대 1,500 trees, early stopping 40.
- 반복 예약 작업은 만들지 않았다. 학습 감독 프로세스가 30초마다 상태와 GPU/RAM을 기록하고 완료 산출물을 SHA/CRC 확인 후 백업한다.

증거: [첫 epoch와 건강 상태](../ops/gpu_preflight_r5_r1/r5_initial_training_observation.json), [CPU 입력 profile](../ops/gpu_preflight_r5_r1/dynamic_cpu_b128_512_fixed.json), [GPU 검증 백업](../ops/gpu_preflight_r5_r1/r5_preflight_evidence_r1.zip), [로컬 SHA/CRC 검증](../ops/gpu_preflight_r5_r1/LOCAL_VERIFICATION.json).
