# R16 본체 PhysicsLightGBM 5-Fold 결과

회귀 3,827특징·독립 분류 700특징. 모든 가중치와 원천 SHA를 확인한 뒤 동결 평가했다. 새 학습·Test 접근·정책 튜닝은 없다.

## 학습 완료 현황

| Fold | TRAIN | selection | 회귀/분류 round | 학습 분 | selection RMSE | 독립 분류 AUC |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 61,446 | 2,400 | 1306/422 | 6.09 | 0.096288 | 0.957520 |
| 1 | 61,368 | 2,400 | 1046/514 | 5.53 | 0.095507 | 0.959401 |
| 2 | 61,259 | 2,400 | 1443/671 | 6.66 | 0.098385 | 0.960026 |
| 3 | 61,546 | 2,400 | 1376/709 | 11.67 | 0.092519 | 0.964544 |
| 4 | 61,685 | 2,400 | 1491/436 | 10.79 | 0.093925 | 0.959421 |

selection은 모든 Fold가 공유한 개발용 validation이다. 이 수치를 전체 OOF 또는 공식 LB로 부르지 않는다.

## 동결 평가

| 패널 | 모델 | N | RMSE | share AUC | 독립 분류 AUC | bias |
|---|---|---:|---:|---:|---:|---:|
| shared_selection | Tree5_equal | 2,400 | 0.093256 | 0.894915 | 0.962533 | -0.014005 |
| local_gate/control_eq6 | Tree5_equal | 400 | 0.058269 | 0.816641 | 0.963945 | -0.008871 |
| local_gate/mfhf_shelf6 | Tree5_equal | 400 | 0.050956 | 0.825234 | 0.979492 | -0.002817 |
| local_gate/mfhf_shelf12 | Tree5_equal | 400 | 0.047956 | 0.828633 | 0.987227 | -0.000243 |
| local_gate/mfhf_shelf12_lfminus6 | Tree5_equal | 400 | 0.047401 | 0.830742 | 0.989531 | +0.001656 |
| local_gate/alpha_preEQ_inserted | Tree5_equal | 400 | 0.094213 | 0.819531 | 0.966211 | -0.024366 |
| local_gate/independent_subband_snr | Tree5_equal | 400 | 0.125424 | 0.814727 | 0.935898 | -0.035284 |
| local_gate/pooled6 | Tree5_equal | 2,400 | 0.076535 | 0.822585 | 0.970384 | -0.011654 |
| own_fold_cv/fold0 | tree_fold0 | 1,000 | 0.095717 | 0.871406 | 0.970194 | -0.021731 |
| own_fold_cv/fold1 | tree_fold1 | 1,000 | 0.078557 | 0.875106 | 0.976488 | +0.002882 |
| own_fold_cv/fold2 | tree_fold2 | 1,000 | 0.080865 | 0.878969 | 0.958163 | -0.021311 |
| own_fold_cv/fold3 | tree_fold3 | 1,000 | 0.065893 | 0.854506 | 0.977437 | -0.003968 |
| own_fold_cv/fold4 | tree_fold4 | 1,000 | 0.058838 | 0.831175 | 0.966294 | -0.001398 |
| own_fold_cv/pooled5000 | each_own_fold_only | 5,000 | 0.077040 | 0.860604 | 0.968760 | -0.009105 |

각 Fold의 cv_validation에는 해당 Fold 모델만 적용했다. 다른 Fold 모델은 이 원천을 학습했을 수 있으므로 이 패널에서 Tree5 평균을 내지 않았다.

local_report 및 domain_holdout의 새 본체 평가는 신경망 4종과 최종 앙상블 정책이 모두 고정된 뒤 수행한다. 기존 접근 보호를 우회하지 않았다.

전체 수치·검산 근거: [RESULTS.json](RESULTS.json).
