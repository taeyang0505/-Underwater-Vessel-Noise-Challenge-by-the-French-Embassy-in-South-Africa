# R14 Mac 본학습 실행 기록

2026-10-27, 사용자 추가 승인에 따라 새 생산 실행을 시작했다. 과거 진단의 숫자와 `production_recipe_approved=false` 기록은 보존하고, 별도 승인 파일에 `true`를 기록했다. 기존 장시간/SF AUC 0.819699는 그대로이며 0.82로 바꾸지 않았다.

## 실행 중인 학습

| 구성 | 데이터와 장치 | 고정 예산 | 실행 경로 |
|---|---|---|---|
| Full Tree baseline/highshare 회귀·독립 분류기 | Fold별 TRAIN 78,069~78,869행, 5,252특징, Mac CPU 2스레드 | 최대 1,500 trees, selection 조기 종료 40 | `models/r14_production_trees_r1` |
| iTransformer 304,239 parameters | Fold 0, 14채널, CPU, batch 128 | 100 epoch, Adam, early stopping 없음 | `models/r14_production_nn_fold0_r1/itransformer` |
| FDY 131,629 parameters | Fold 0, 14채널, MPS, batch 128 | 100 epoch, Adam, early stopping 없음 | `models/r14_production_nn_fold0_r1/fdy` |
| Wide CNN 1,494,785 parameters | Fold 0, 14채널, MPS, batch 128 | 100 epoch, Adam, early stopping 없음 | `models/r14_production_nn_fold0_r1/bg_cnn` |
| TFGrid–FDY–Conformer 2,565,541 parameters | Fold 0, 14채널, MPS, batch 64 | 시간 예산 예외 50 epoch, Adam, early stopping 없음 | `models/r14_production_nn_fold0_r1/tfgrid_fdy_conformer` |

Fold 0의 기존 Full Tree 두 learner는 정확한 데이터·코드·PCA·산출물 해시를 검증한 뒤 재사용한다. 나머지 4개 Fold의 두 learner씩을 새로 학습한다. 총 10 learners, 각 learner의 회귀·분류기를 합쳐 20 boosters다.

빠른 신경망 3종은 epoch마다 같은 동적 합성 자료를 공유한 뒤 차례로 학습한다. 이 3종을 완료하면 TFGrid를 학습한다. raw epoch 캐시는 최대 두 개만 남기며, 재현성 영수증은 모두 보존한다. 신경망 손실은 `share MSE + 0.25 × energy-weighted cell MSE`이고, 타깃 가중치와 contrast/presence BCE는 없다. 각 예산 안에서 selection RMSE가 가장 낮은 체크포인트를 따로 보관한다. 이는 탐색한 epoch 중 최저값이며 전역 최적점의 증명은 아니다.

## 완료 뒤 자동으로 이어지는 로컬 산출물

아래 연속 배치 감독기는 이미 실행 중인 학습을 감시하고, 필요한 완료 표식이 생기면 해당 작업을 한 번 실행한다. 훈련을 재시작하거나 설정을 바꾸지 않는다. 실패하면 로그와 상태를 남기고 중단한다. 예약 작업이나 공식 제출 기능은 사용하지 않는다.

1. Tree 5-Fold 완료 → 각 Fold의 공통 baseline fit 500·gate 500으로 정책 고정 → 미사용 report 1,000개씩의 Tree OOF → Tree-only CSV.
2. 빠른 신경망 3종 완료 → Fold 0 NN와 Tree 0의 공통 fit·gate에서 앙상블 정책 고정 → 17개 개발 진단 패널 비교 → Fold 0 CSV와 Tree 5-Fold bagging 확장 CSV.
3. TFGrid 완료 → 4개 본체 신경망을 포함한 별도 정책·17패널·CSV 생성. 앞선 파일을 덮어쓰지 않는다.

| 파일 | 의미 |
|---|---|
| `submissions/R14/submission_r14_fulltree_5fold.csv` | Tree-only 5-Fold 대조군 |
| `submissions/R14/submission_r14_fold0.csv` | 빠른 NN 3종 + Tree의 정확한 Fold 0 정책 |
| `submissions/r14_main/submission_r14_step1.csv` | Fold 0 정책에 Test 추론 시 같은 Tree 계열의 5-Fold 평균을 적용한 확장 |
| `submissions/R14/submission_r14_fold0_with_tfgrid.csv` | TFGrid까지 포함한 Fold 0 정책 |
| `submissions/r14_main/submission_r14_step1_with_tfgrid.csv` | TFGrid 포함 정책 + Tree 5-Fold 추론 확장 |

각 CSV는 1,500개 ID·열 순서·[0,1] 범위·숫자 재읽기 일치·SHA를 검증하고 manifest를 함께 저장한다. R9 로컬 기준 CSV와 평균·RMS·분위수·포화도·모델 간 상관관계를 비교한다. 사용자의 R9 공식 결과는 RMSE 0.217535806, AUC 0.555066666이다. 새 R14는 숨겨진 Test 정답이 없어 공식 RMSE·AUC가 아직 없다.

Tree 5-Fold는 배경 원천 Fold의 OOF이고 509개 참조 전체를 공유한다. 참조 독립 CV라고 부르지 않는다. 신경망은 우선 Fold 0이므로, Tree 5-Fold와 결합한 결과를 전체 신경망 5-Fold OOF라고 부르지 않는다. 다른 Tree Fold가 Fold 0 fit 자료를 학습했을 가능성이 있으므로, Tree 5-Fold 평균을 Fold 0 fit에 넣어 가중치를 학습하지 않는다. 해당 평균은 정책 고정 뒤 Test 추론에서만 적용한다.

## 상태와 검증 자료

- 실시간 상태: [R14_PRODUCTION_START.html](R14_PRODUCTION_START.html)
- 기계 판독 상태: `../ops/r14_production_state.json`
- 승인: `../ops/r14_production_approval.json`
- 고정 설정: `../configs/r14_production_mac_v1.json`
- 정책 규약: `../configs/r14_production_ensemble_protocol_v1.json`
- 독립 검사: [R14_PRODUCTION_PREFLIGHT_INDEPENDENT_QA.md](R14_PRODUCTION_PREFLIGHT_INDEPENDENT_QA.md)
- 출력 경로 검사: `R14_PRODUCTION_OUTPUTS_INDEPENDENT_QA.json`
- 첫 epoch 실측: `R14_PRODUCTION_NN_FIRST_EPOCH.json`

첫 epoch에서 공유 raw 합성 10.72초, iTransformer 29.15초, FDY 32.84초, Wide CNN 77.88초를 측정했다. 이 속도 유지 시 빠른 3종 100 epoch에 약 4.2시간이며, 장시간 실행의 발열·동시 CPU 사용에 따라 달라진다. TFGrid 50 epoch는 첫 실측 전까지 별도 미산정이다. 과거 16.03초는 합성과 특징 변환 시간으로, 신경망 학습 전체 epoch 시간이 아니다.
