# R15 저장 예측 본체 감사

새 학습·추론·실제 Test 입력·숨은 정답 접근 없이, 기존 NPZ/제출 CSV만 비교했다. 모든 1,500개 ID를 정렬해 57개 예측 series를 결박했다. R14 출력은 fast3이며 아직 완료되지 않은 TFGrid 예측은 넣지 않았다.

| 저장 출력 | share 평균 | share RMS | score 평균 | score RMS | score ≥0.99 |
|---|---:|---:|---:|---:|---:|
| R5_CNN | 0.354658 | 0.445229 | 0.920091 | 0.943444 | 67.07% |
| R9_tree5 | 0.264153 | 0.325630 | 0.592117 | 0.605513 | 0.00% |
| R9_all5 | 0.264744 | 0.326982 | 0.607677 | 0.619639 | 0.00% |
| R14_step1 | 0.303326 | 0.386897 | 0.459316 | 0.500404 | 0.00% |
| R14_exact_Fold0 | 0.303241 | 0.386787 | 0.461230 | 0.501746 | 0.00% |
| R14_tree5 | 0.284263 | 0.349420 | 0.871772 | 0.882443 | 0.00% |
| R14.nn_bg_cnn | 0.309823 | 0.400326 | 0.309823 | 0.400326 | 0.00% |
| R14.nn_fdy | 0.262512 | 0.346125 | 0.262512 | 0.346125 | 0.00% |
| R14.nn_itransformer | 0.251867 | 0.328171 | 0.251867 | 0.328171 | 0.00% |

R5 CNN의 저장 score는 67.07%가 0.99 이상이다. 반면 R14 NN의 score는 별도 classifier가 아니라 share와 정확히 같다. R14 최종 혼합 score, R14 Tree-only 최종 score, 과거 R5/R9의 score는 정의가 달라 평균 크기를 바로 확률 보정의 우열로 비교할 수 없다. 이 표에는 실제 정답에 대한 calibration 지표가 없다.

R14 step1 − R9 all5의 share 평균 차이는 +0.0385828, 예측 차이의 RMS는 0.105271, Pearson은 0.921199이다. score 평균 차이는 −0.148361, 예측 차이의 RMS는 0.184904, Pearson은 0.871456이다. 높은 출력 상관이 비슷한 오차나 높은 AUC를 보장하지 않는다.

R14 frozen 회귀 배합은 CNN 0.8104741, Tree baseline 0.1671633, Tree highshare 0.0223626이며 iTransformer/FDY는 수치적으로 0이다. main step1은 Fold0 NN + Tree5 평균을 사용한 출력으로, 완전한 NN5fold OOF ensemble과 구분한다.

## classifier 저장 여부

- R9 Tree: 다섯 Fold의 원 `score_classifier` 배열이 실제 저장돼 있다. wide table의 `R9_tree_cache.fold_00_score_classifier` 등으로 포함했다.
- R14 Tree: 각 Fold baseline/highshare의 회귀 share 10개와 최종 혼합 score는 저장돼 있다. 각 Fold 원 classifier 확률과 가족별 평균 classifier 확률은 두 query NPZ에 저장돼 있지 않다. 미저장 확률을 임의 추정하거나 역산해 raw 값이라고 표시하지 않았다.
- R14 NN: 독립 classifier head 자체가 없다. wide table의 NN score는 소스 계약 `score=share`를 명시해 복제한 저장 share이며, 새로운 추론 결과가 아니다.
- R9 all5: 과거 모델별 share/score 5쌍을 실제 저장 NPZ에서 읽었다. R14와 같은 score 정의라고 가정하지 않는다.

## 정합성 및 산출물

18개 입력 소스의 전후 SHA가 동일하다. NPZ↔CSV, Tree5 평균↔R14 Tree-family share, R14 정책에 따른 최종 share 재결합이 모두 exact 일치했다. 새 모델·역전파·Test 맞춤·가중치 선택은 없다.

- [모든 통계·출처 SHA·검증 결과](RESULTS.json)
- [ID + 57개 series의 정합 wide table](aligned_saved_predictions_1500.csv)
- [57개 series 전체 분포](distribution_summary.csv)
- [Pearson 전체 행렬](pearson.csv), [Spearman 전체 행렬](spearman.csv)
- [원인 주장과 물리 guardrail 검토](CLAIM_CHECK.md)
- [재현 스크립트](audit_saved_predictions.py), [독립 stdlib 통계 검산](INDEPENDENT_NUMERIC_VERIFICATION.json)

관측된 예측 분포는 숨은 target 모멘트·Test 원천·숨은 시간·개별 모델의 공식 성능을 알려주지 않는다. 이 결과로 생산 정책을 변경하지 않았다.
