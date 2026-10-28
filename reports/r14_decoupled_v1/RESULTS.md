# R14 저장 예측 분리 결합 결과

**추가 추론·모델 학습 없이 실행했으며 동시 목표는 실패했다.** validation_fit 1,000행에서 정한 정책은 baseline 회귀값과 baseline 분류기 확률이다. 고SNR RMSE는 0.156340으로 목표 <0.145를 충족하지 못하고, full15/SF score AUC도 0.770705로 목표 ≥0.82보다 낮다. 나머지 패널의 결과를 보고 정책을 바꾸지 않았다.

## 실행·선택 규칙

[사전 계획](predeclared_plan.json)은 입력·스크립트 SHA, 후보와 목표를 저장했다. 그 전에 파일 schema/dtype/model_order만 확인했으며 후보 설정에 패널 수치 요약이나 라벨을 사용하지 않았다. 회귀 base/high는 각각 해당 CNN과 Tree의 50:50 예측이다. 선형 high 비중 0/.25/.5/.75/1과 아래 smooth gate 18개, 총 23개 회귀 후보를 만들었다.

`a = low + (high-low) × sigmoid((y_base−0.20)/width)`

`y_mix = (1−a)y_base + a y_high`

low∈{0,.25}, high∈{.5,.75,1}, width∈{.03,.05,.10}은 평가 전에 고정했다. a가 y_base에 대해 단조이며 부드럽다는 뜻이고, 두 회귀기가 서로 다르게 변할 때 최종 y_mix가 모든 입력에서 단조라는 보장은 없다. 출력은 두 예측 사이의 convex 결합으로 물리 정답이나 cell q를 다시 정의하지 않는다.

분류 확률은 base/high/두 확률 평균 3개, score는 `(1−λ)y_mix+λp_classifier`, λ∈{0,.25,.5,1}이다. 회귀는 fit RMSE 최소, 그 회귀값의 score는 fit AUC 최대로 정하며 동률은 사전 순서를 쓴다. 총 276개 조합을 보고하되 평가 패널로 후보를 고르지 않는다.

선택된 회귀는 `linear_high0.0`이다(fit RMSE 0.07004945). 가장 좋은 smooth 후보 `smooth_low0.0_high0.5_width0.03`도 fit RMSE 0.07030670으로 기본보다 낮지 않았다. score는 baseline 분류기 단독(λ=1)이 선택됐다. [정책 파일](selected_policy.json)을 저장한 다음 17개 수치 패널을 읽었다. 기존 validation_gate는 새 파라미터 선택에 쓰지 않았다. 기존 노출된 development 자료라 완전히 독립적인 최종 검증이라는 뜻은 아니다.

## 분류기 교체의 결과

| 독립 분류 확률 | unseen 일반 AUC | full15/SF AUC | native15/SF AUC |
|---|---:|---:|---:|
| base | 0.950581 | 0.770705 | 0.789269 |
| high | 0.945800 | 0.760763 | 0.777237 |
| avg | 0.948925 | 0.766385 | 0.784885 |

분류기를 baseline으로 바꾸면 highshare보다 나아지지만 full15/SF 목표에는 부족하다. baseline의 **회귀 share AUC 0.820224**와 **독립 classifier AUC 0.770705**는 같은 출력 지표가 아니다. baseline share와 classifier를 λ=.25로 혼합하면 unseen score AUC는 0.931775지만 full15/SF는 0.773596이다. 한쪽의 좋은 AUC가 다른 쪽에서도 유지된다고 가정하면 안 된다.

## 회귀 분리 결과

아래 행들은 사전 선형 후보 전체와 fit에서만 정한 최선 smooth 후보다. 패널을 보고 고른 행이 아니다.

| 회귀 후보 | highSNR RMSE | unseen RMSE | full15/SF share AUC |
|---|---:|---:|---:|
| linear_high0.0 | 0.156340 | 0.052679 | 0.820224 |
| linear_high0.25 | 0.149741 | 0.052385 | 0.810750 |
| linear_high0.5 | 0.143878 | 0.052747 | 0.801186 |
| linear_high0.75 | 0.138845 | 0.053752 | 0.791686 |
| linear_high1.0 | 0.134734 | 0.055365 | 0.782135 |
| smooth_low0.0_high0.5_width0.03 | 0.144827 | 0.052001 | 0.819968 |

23×3×4=276개 후보 중 고SNR RMSE<.145, unseen RMSE≤.0527·score AUC≥.92, full15/SF score AUC≥.82, native15/SF score AUC≥.80을 모두 충족한 후보는 **0개**였다. 이는 이 유한한 사전 후보군에 대한 결과이며 다른 모든 방법의 불가능성을 증명하지 않는다.

## fit에서 선택한 정책의 17개 패널 결과

| 패널 | RMSE | share AUC | score AUC | share bias |
|---|---:|---:|---:|---:|
| R10_seen__combined_stress | 0.200386 | 0.976231 | 0.974762 | -0.104098 |
| R10_seen__mean_scale_only | 0.160046 | 0.984525 | 0.987537 | -0.074884 |
| R10_unseen__combined_stress | 0.189233 | 0.984281 | 0.986962 | -0.095227 |
| R10_unseen__r9_complete | 0.052679 | 0.844269 | 0.950581 | +0.007800 |
| R11_full_smooth12_VS_hf_high | 0.072166 | 0.874928 | 0.903420 | +0.025104 |
| R11_full_smooth12_VS_st_francis_dense | 0.066016 | 0.847763 | 0.878994 | +0.024626 |
| R11_full_smooth13_VS_hf_high | 0.072658 | 0.872744 | 0.897624 | +0.023368 |
| R11_full_smooth13_VS_st_francis_dense | 0.066677 | 0.845173 | 0.870436 | +0.022488 |
| R11_full_smooth15_VS_hf_high | 0.078235 | 0.855400 | 0.827236 | +0.015554 |
| R11_full_smooth15_VS_st_francis_dense | 0.074042 | 0.820224 | 0.770705 | +0.012865 |
| R11_native_long15_VS_hf_high | 0.074552 | 0.845136 | 0.838884 | +0.011053 |
| R11_native_long15_VS_st_francis_dense | 0.069205 | 0.806051 | 0.789269 | +0.007321 |
| R12_eval | 0.062234 | 0.938300 | 0.966675 | -0.002600 |
| R12_high_snr_long | 0.156340 | 0.996825 | 0.997050 | -0.111366 |
| R12_validation_gate | 0.090328 | 0.888125 | 0.905250 | +0.014500 |
| R9_ensemble_gate | 0.033557 | 0.837354 | 0.963591 | +0.002256 |
| R9_selection | 0.055490 | 0.814431 | 0.921593 | +0.011175 |

모든 후보의 [전체 수치 CSV](results.csv), [선택 정책 CSV](selected_results.csv), [fit 회귀 후보](validation_fit_regression_candidates.csv), [fit score 후보](validation_fit_score_candidates.csv)를 보존했다. score의 y>0 정답은 알려진 추가 녹음 성분의 에너지 존재이며 자연 선박 부재/존재 정답과 같지 않다.

## 독립 수치 검산 및 한계

[독립 QA](INDEPENDENT_QA.json)는 sklearn과 별개인 tie-half pairwise 승률 방식으로 AUC를 계산했다. 32,844개 scalar 검사의 최대 차이는 2.22e−16이며 17패널 source ID·target 정렬, fit-only 정책 재선택, 모든 입력 SHA 불변을 확인했다. 새 checkpoint 로드·추론·fit·Test 접근은 없었다. CNN의 fit 예측 NPZ에는 ID가 없어, tree가 저장한 ID/target과 공통의 동결 validation 입력/생성 순서 바인딩을 사용했다. 새 모델 replay로 그 순서를 재인증한 것은 아니다.

R12의 높은 classifier 수치가 어떤 데이터 조건인지에 대한 별도 [출처 검토](../R14_CLASSIFIER_PROVENANCE.md)를 먼저 읽는 것이 좋다. 여기서 local 개선·실패를 대회 Test 오차의 원인이나 리더보드 개선으로 단정하지 않는다. 모든 원래 R12/R13 정책·모델·자료는 유지됐다.
