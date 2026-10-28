# R4 학습 준비 계획

상태: **준비 중, 실행하지 않음**. 새 R4 데이터가 아직 완성되지 않았으며, 서버 변경·업로드·학습 실행은 부모 에이전트의 명시적 실행 신호 이후에만 진행한다. R2와 R3의 데이터·모델·정책은 수정하지 않는다.

## 고정 실행 범위

설정은 `configs/group_cv_passage_training_r4.json`이다. 데이터 경로는 `data/synthetic/group_cv_passage_r4_v1`, protocol은 `source_group_nested5_vtuad_passage_r4`, fold gate version은 `quality_v4_vtuad_passage_group_cv_prebuilt`다. 실제 행 수·원천 수는 완성된 manifest에서 확인하며 현재 예상 숫자를 사실로 쓰지 않는다.

5개 fold와 seed 42를 사용한다. 후보는 `lgbm_retrieval`, CNN, FDY, iTransformer의 네 개다. TFM, TabM, base/physics 단독 LightGBM과 자동 ablation은 실행하지 않는다.

신경망은 Adam, 최대 100 epoch, patience 7, dropout 0.1, ReduceLROnPlateau를 사용하고 inner RMSE가 가장 낮은 체크포인트를 복원한다. 손실은 `share MSE + 0.25 × observed-energy-weighted cell MSE + 0.05 × presence BCE`다. tree는 자체 boosting early stopping을 사용하며 신경망 epoch를 적용하지 않는다.

CNN은 architecture 5의 8채널 입력을 유지한다. 기존 6채널에 TRAIN-only background prior의 `q0`, `log1p(clip(Bhat/E,0,10))/log(11)`을 추가한다. 출력은 직접 mask logits의 sigmoid이며 prior logits을 더하지 않는다. 이전 residual-logit penalty는 0이다. FDY와 iTransformer는 6채널 입력이다. 같은 관측 에너지 가중치로 cell mask를 share로 변환한다. prior와 retrieval prototypes는 현재 fold TRAIN에서 새로 적합한다.

## 내부 검증과 안전 정책

CPU 상수 baseline은 각 fold TRAIN의 share 평균으로 고정한다. 세 mask 신경망의 균등 평균은 사전 고정 fallback이다. CNN 단독 성능도 별도로 보고하며 fallback보다 좋을 수 있음을 숨기지 않는다.

네 후보의 nonnegative/sum-to-one 가중치는 INNER에서만 적합한다. ridge λ=0.05이며 기준 가중치는 `[0, 1/3, 1/3, 1/3]`이다. INNER reference source group의 leave-one-group-out crossfit와 INNER 전체 적합 양쪽에서, CNN 단독 및 세 신경망 균등 평균보다 RMSE가 각각 최소 `1e-4` 개선되어야 후보를 허용한다. 부족한 group·비유한값·최적화 실패·개선 미달이면 fallback을 유지하며 임계값을 완화하지 않는다. 정확한 안전 정책 구현은 부모 에이전트 소유 `pipeline/ensemble_safety_r4.py`에서 제공한다.

모든 fold에서 checkpoint 선택과 앙상블 정책 저장·SHA 기록 후 해당 fold outer 평가를 진행한다. affine은 `[1,0]`이다. Outer gate는 사후 pass/fail 보고만 한다. 해당 결과로 가중치를 다시 적합하거나 이미 계산한 OOF 예측을 CNN/fallback으로 바꾸지 않는다. 비교 이후 OOF는 개발 증거이며 새로운 미사용 holdout으로 표기하지 않는다.

## 실행 전 필수 gate

- 5개 fold와 별도 INNER/OUTER, 완료된 `ready_to_train` manifest, R4 version/protocol 및 source/package/파일 SHA 일치.
- 원천·중복·시간 purge·마스크·cell/window target 재구성 검사 통과. 구체적 품질 항목은 완성된 데이터 gate 및 부모의 독립 검증으로 확정한다.
- R4 core CLI가 `--cnn-head-mode direct_mask --cnn-prior-residual-weight 0 --cell-loss-weight .25 --presence-loss-weight .05`를 실제 반영하고 저장·복원/outer 평가 fixture를 통과.
- 새 R4 project/config/code identity 고정, 이전 실행 state·출력 부재, 동일 GPU에서 다른 훈련이 실행 중이지 않음을 확인.
- 기존 R2/R3, calibration/sealed/Test는 훈련 또는 정책 fit에서 사용하지 않음. 외부 자문·공식 제출·pod 정지는 실행 범위에 없음.

## 실행과 백업

서버는 기존 승인된 GPU 호스트를 사용한다. 새 release base는 `/workspace/underwater_vessel_noise_r4`, project는 그 아래 `project`, OPS는 그 아래 `ops`다. Python은 기존 `/workspace/underwater_vessel_noise_r1/venv/bin/python`을 재사용한다. RUN은 `models/group_cv_passage_r4_adam`으로 분리한다.

`ops/runpod_train_r4.py`는 기본적으로 preflight만 수행한다. `--execute`와 명시적 config SHA가 있어야 한 번의 새 학습을 시작한다. 건강 상태·훈련 로그·수렴을 기록하지만 설정을 자동 변경하거나 재시작하지 않는다. NaN, OOM, 예외, gate 실패는 기록하고 실패 실행을 보존한다. 자동 scheduler/앱 automation은 만들지 않는다.

`ops/collect_r4_training_backups.py`는 수동 1회 실행 수집기다. 완료된 ledger만 기존 검증 백업 helper로 ZIP화하고, 내려받은 archive CRC·개별 SHA·완료 ledger를 검증한다. 미완료 파일·원천 오디오·vendor 가중치는 복사하지 않는다. 모델 history·정책·outer 보고서·최종 OOF를 새 `ops/r4_progress`에 보관한다. 수집은 학습을 시작하지 않는다.

로컬 준비 검증은 통과했다. 두 helper의 Python 구문·CLI, supervisor의 잘못된 optimizer/epoch/head/loss/모델/outer-selection 설정 7건 거부, collector가 생성하는 원격 probe 코드의 구문 및 R4 세 완료 단계·protocol binding을 확인했다. 이 검증은 SSH·GPU·실제 학습을 실행하지 않았다.

현재 이 계획은 실제 학습 완료 또는 데이터 준비 완료를 뜻하지 않는다. 데이터 및 core gate가 완료되고 명시적인 실행 신호를 받을 때까지 launch를 보류한다.
