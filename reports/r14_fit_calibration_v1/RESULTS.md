# R14 fit-only 단조 scalar 보정

**두 사전 선택 정책 모두 동시 목표에 실패했다.** 일반 fit MSE가 고른 `share^1.05`는 unseen RMSE를 조금 낮췄지만 고SNR 장시간 오차를 악화시켰다. `1+2y` 가중 fit MSE는 identity를 골랐다. 원래 decoupled 276개 후보의 결과나 정책은 변경하지 않았다.

## 절차

[사전 계획](predeclared_plan.json)을 저장한 뒤 strict baseline CNN/Tree50:50의 저장 validation_fit1,000 예측에만 scalar 후보를 적합했다. identity·power6·logit temperature4·logit affine8·양의 기울기 piecewise15, 총34개다. piecewise의 꺾임점은 사전 고정0.20이며 모든 함수는 [0,1]에서 비감소이고 bounded다. synthetic1,001점 [수치 검사](NUMERIC_FIXTURE.json)를 통과했다.

일반 MSE와 normalized `sum((1+2y)×MSE)/sum(1+2y)` 두 목적별 최소 후보를 선택했다. 각 후보의 score는 원 baseline classifier와 λ∈{0,.25,.5,.75,1}로 결합하며 λ도 fit AUC에서만 고정했다. [정책 파일](selected_policies.json)을 저장한 다음 17개 패널을 읽었다. 두 선택 모두 λ=1, 즉 score는 baseline classifier다. 평가 패널 라벨로 gamma·기울기·temperature·λ를 바꾸지 않았다.

일반 fit MSE: identity 0.004906926 → power1.05 0.004902750. 가중 fit MSE에서는 identity 0.005980751이 power1.05 0.006212972보다 낮았다. 아주 작은 fit 개선을 새로운 일반화 성능으로 해석하면 안 된다.

## 선택된 두 정책

| fit 정책 | 변환 | 패널 | RMSE | bias | share AUC | score AUC |
|---|---|---|---:|---:|---:|---:|
| primary_fit_MSE | power1.05 | R10_unseen__r9_complete | 0.052512 | -0.001049 | 0.844269 | 0.950581 |
| secondary_fit_weighted_MSE | identity | R10_unseen__r9_complete | 0.052679 | +0.007800 | 0.844269 | 0.950581 |
| primary_fit_MSE | power1.05 | R11_full_smooth15_VS_st_francis_dense | 0.072474 | +0.005464 | 0.820224 | 0.770705 |
| secondary_fit_weighted_MSE | identity | R11_full_smooth15_VS_st_francis_dense | 0.074042 | +0.012865 | 0.820224 | 0.770705 |
| primary_fit_MSE | power1.05 | R11_native_long15_VS_st_francis_dense | 0.068833 | +0.000019 | 0.806051 | 0.789269 |
| secondary_fit_weighted_MSE | identity | R11_native_long15_VS_st_francis_dense | 0.069205 | +0.007321 | 0.806051 | 0.789269 |
| primary_fit_MSE | power1.05 | R12_high_snr_long | 0.164706 | -0.119810 | 0.996825 | 0.997050 |
| secondary_fit_weighted_MSE | identity | R12_high_snr_long | 0.156340 | -0.111366 | 0.996825 | 0.997050 |

고SNR 목표<0.145, unseen RMSE≤0.0527·score AUC≥0.92, full15/SF score AUC≥0.82, native15/SF score AUC≥0.80을 모두 통과한 사전 후보는 **0/34**다. 결과를 보고 다른 후보를 채택하지 않았다. 모든17패널·34후보는 [전체 CSV](results.csv), 선택 두 정책은 [선택 CSV](selected_results.csv), fit 목적함수는 [fit CSV](validation_fit_candidates.csv)에 있다.

## 검증 및 해석 범위

[독립 QA](INDEPENDENT_QA.json)는 대체 수식으로 변환·fit 목적·λ 선택을 재계산하고, sklearn과 별개인 tie-half pairwise AUC로 2,312개 지표를 검산했다. 최대 차이는1.11e−16이며 ID/target 정렬과 입력 SHA 불변도 통과했다.

이것은 **scalar 회귀 출력의 사후 보정**이다. 원래 cell q를 바꾸지 않으므로 보정 share가 원래 `sum(wq)`와 같다는 보장은 없다. 정답 y/q나 에너지 가중치를 재정의한 것이 아니며, 보정값을 새 물리 마스크의 보존된 출력이라고 부르면 안 된다. 단조 변환은 원 share 순위를 대체로 보존하므로 새로운 구분 정보를 만들지 않는다(클리핑과 float32 동률은 생길 수 있다).

17개 패널은 이전 development에서 이미 노출됐다. parameter 선택은 이번에도 fit에만 제한했지만 pristine blind 검증이나 대회 Test 개선의 증거는 아니다. model fit0·추론0·Test0·네트워크0이며 원래 모델·자료·R13 정책과 [원276개 분리결합 결과](../r14_decoupled_v1/RESULTS.md)는 그대로다.
