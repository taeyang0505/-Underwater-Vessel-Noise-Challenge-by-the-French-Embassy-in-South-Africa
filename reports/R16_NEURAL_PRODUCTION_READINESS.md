# R16 본체 신경망 준비·실자료 연결 검증

최종 체크: `R16_NEURAL_ACTUAL_PREFLIGHT_r3.json` **PASS**. 실제 Fold 0 TRAIN 61,446건 중 128건을 연결 검증에만 사용했고, 이 검증의 optimizer 가중치는 폐기했다. 본학습은 별도 명령으로 시작한다. Test 및 고정 스트레스 평가셋은 읽지 않았다.

## 고정 계약

- 개선 혼합 레시피, immutable Fold 0 source manifest SHA: `7ea98290fd08204e4966554b19ab50520d5262acf0bc2f9c152d44b6540e3728`.
- 기존 R16 소형과 동일한 relative8. PCA, LF anchor, q0, absolute dB, highshare 배제. 상대 대역 프로파일 및 sqrt(w)는 회귀에만 허용한다.
- q_hat→관측 에너지 가중 풀링으로 share 계산, Share MSE + 0.25 × energy-weighted cell MSE.
- 진단 분류기는 3,745개 별도 파라미터이며 시간 상대 채널만 받는다. 회귀와 파라미터·gradient·gradient clipping을 공유하지 않는다. 회귀 및 분류의 최적 epoch를 각각 저장한다.
- 모든 신경망 최대/실행 100 epoch, Adam lr=0.001, weight decay=0.0003, dropout=0.1, cosine100 eta_min=0.00001, early stopping 없음. Validation 최솟값의 발견이나 Global Minima를 보장하는 것은 아니다.
- selection 2,400개는 기존 R16 local_fit 6×400을 그대로 사용. 별도 참조·크롭/채널 전이 선택축이며, 이후 고정 stress/report는 정책 고정 후 평가한다.
- 8채널 변환은 CPU FP64 기준 구현으로 **한 번만** 계산하여 float32 mmap에 저장한다. batch마다 CPU quantile/PCA를 수행하지 않는다. GPU 미지원 연산의 조용한 CPU fallback을 사용하지 않는다.
- 기존 중단 R14 TFGrid epoch37은 별도 보존되고 R16 초기 가중치로 사용하지 않는다. RAM 해제 내역: `../ops/r16_old_paused_process_release.json`.

## 연결·수치 검증

CPU/device 추론 최대 차이 7.16e-7 이하, 물리 풀링 최대 차이 1.79e-7 이하. 반대 branch gradient 유입은 없으며 classifier의 상대 프로파일·sqrt(w) 변경 불변 차이도 0이다.

CPU 및 MPS에서 model, Adam states, scheduler, CPU torch/MPS/numpy/Python RNG를 저장·복원한 뒤 다음 optimizer step이 완전히 일치했다(파라미터 및 loss 차이 0). CPU checkpoint 왕복 최대 차이는 4.48e-7 이하다.

## 실제 Mac 짧은 step 측정

이 표는 실제 TRAIN 128건을 이용한 warmup 1회 + 3 step 중앙값이다. 전 학습 시간은 측정하지 않았다. driver 값은 해당 시점의 할당량으로 실제 peak 측정값은 아니다.

| 모델 | 회귀 파라미터 | 독립 분류 | 장치 | Batch | samples/sec | sampled MPS GiB | 1 epoch 연산 하한 | 100 epoch 연산 하한 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| bg_cnn | 1,487,873 | 3,745 | mps | 128 | 1173.2 | 1.13 | 52.4초 | 1.45시간 |
| fdy | 129,901 | 3,745 | mps | 128 | 3142.6 | 1.15 | 19.6초 | 0.54시간 |
| itransformer | 295,599 | 3,745 | cpu | 128 | 2232.7 | 0.00 | 27.5초 | 0.76시간 |
| tfgrid_fdy_conformer | 2,562,085 | 3,745 | mps | 64 | 214.0 | 5.32 | 287.2초 | 7.98시간 |

TFGrid batch128은 sampled MPS driver가 8GiB 제한을 초과하여 채택하지 않았다. batch64는 같은 수치 검증을 통과했다. 4종 100epoch의 합계 연산 하한은 약 10.7시간이며, validation·체크포인트·데이터 전송·온도/전원 변화·CPU Tree 병행 영향을 포함하면 더 걸린다. 본학습 1~3 epoch를 관측한 뒤 ETA를 갱신한다.

## 실행 파일 및 상태

- 모델: `../pipeline/models_r16_production.py`
- 엔진: `../pipeline/training_r16_production.py`
- CLI: `../ops/train_r16_production_nn.py`
- 설정: `../configs/r16_neural_mac_v1.json`
- 실제 연결 검증: `R16_NEURAL_ACTUAL_PREFLIGHT_r3.json`

```bash
# 실제 MPS 사용이 확인된 Python3.14 환경. parent runner가 QA 완료 후 1회만 실행.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:ops/r14_production_runtime_py \
PYTORCH_ENABLE_MPS_FALLBACK=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/Users/hongtaeyang/Documents/Codex/2026-10-27/new-chat/.venv/bin/python \
-m ops.train_r16_production_nn train \
--config configs/r16_neural_mac_v1.json \
--manifest data/synthetic/r16_production_v1/fold_00/manifest.json \
--fold 0 --preflight reports/R16_NEURAL_ACTUAL_PREFLIGHT_r3.json \
--output models/r16_production_nn_fold0_r1
```

별도 feature cache를 생성하면 `--cache-root <프로젝트 내부 cache 경로>`로 동일 데이터를 재사용할 수 있다. cache manifest/source/input SHA를 전수 확인하며, source closure 또는 데이터 manifest 변경 시 실행을 거절한다.

SIGUSR1 또는 run 폴더의 PAUSE_AFTER_EPOCH 파일로 완성 epoch 경계에만 중지할 수 있다. 재개는 PAUSED receipt가 있는 경우에 `--resume`으로 명시한다. PAUSE_AFTER_EPOCH 파일은 명시적 재개 전에 제거해야 한다. 실패 partial run은 재개하지 않는다.


## K-fold 인자와 최종 캐시

최종 preflight r3는 `--fold 0`을 데이터 manifest/실행과 함께 결박한다. 추후 Fold 1~4도 각각의 immutable manifest로 별도 preflight를 통과한 뒤 실행할 수 있다. 전체 5-fold 신경망이 완료되었다는 뜻은 아니며 현재 미시작이다.

새 source closure에 결박된 최종 shared cache는 `../models/r16_production_feature_cache_r2`다. 초기 cache_r1은 fold-aware 수정 전 closure이므로 기록용으로 보존하며 본학습에 재사용하지 않는다.

`../ops/predict_r16_production_nn.py`는 selection/local_fit 개발 단계와 global model+policy freeze가 필요한 local_gate/local_report/domain_holdout/query 단계를 구분한다. CPU Python3.12 및3.14에서 synthetic raw-array/schema gate 검사를 통과했으나, 아직 완료된 R16 본체 checkpoint가 없으므로 실제 trained-model frozen inference는 미측정이다. 공식 Test 파일은 읽지 않았다.
