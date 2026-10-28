# R11-3 저장 예측 감사: selection과 query

기존 CPU FP32 예측만 읽었습니다. 모델 로드·추론·학습·후처리 재적합은 **0회**입니다. selection 1,000개와 query 1,500개를 비교했으며 원본 Test, 모델, 정책, 제출 CSV의 SHA가 전후 일치했습니다. query의 실제 정답·정답 평균·정답 RMS는 알 수 없습니다. 아래 query 수치는 모두 **저장 예측의 분포**입니다.

## 1. 감사 범위와 재현성

- [사전 계획](predeclared_plan.json)은 Test의 가용성 메타데이터와 통계를 읽기 전에 저장했습니다. 고정 임계값은 `0, .01, .05, .5, .95, .99, 1`, 양쪽 누적비율 `≤, ≥`와 정확 일치율을 모두 출력했습니다. 임계값을 선택하거나 모델을 변경하지 않았습니다.
- [실행 결과](RESULTS.json), [출력 수치 검산](diagnostic_sanity_verification.json), [행별 query 예측·결측 메타데이터](query_ID_availability_and_saved_prediction_components.csv), [selection 행별 자료](selection_ID_availability_and_saved_prediction_components.csv)를 보존했습니다.
- 원본 Test SHA는 `3ef819192c05c4308b66564e6307e6f4f26fe806afe96ec93e2d15f25a3474d4`, ID 순서 SHA는 `09407eb36b4c327f87f902c2fccf176ce54b0ad0ee4f793fa70b9a47376ba771`입니다. 375열 minute-major/25-band 스키마와 1,500개 고유 ID를 확인했습니다. query db는 **finite-mask만** 읽어 Nobs·부분결측·처음/마지막 결측·결측분 패턴을 추출했고 신호분포/특징을 새로 계산하지 않았습니다.
- 원본 query 예측 NPZ SHA는 `87f47dee43e02dff265a0f26d79f8a4aff2e45e00489a56dfbcaa3d85b5d66a8`, 원본 로컬 제출 CSV SHA는 `f016265ae9a1702dbb7121bfb0c852f5900f85a17194881056316768300fb39d`입니다. 고정 가중치 `[.1,.1,.1,.1,.6]`의 계산과 CSV 행별 값이 저장 예측과 정확히 일치합니다. 이는 **로컬 산출물 일치 확인**이며 실제 웹 업로드 바이트의 확인이 아닙니다.

## 2. 예측 share 분포

| 모델 | selection 예측평균 | query 예측평균 | selection 예측 RMS | query 예측 RMS |
|---|---:|---:|---:|---:|
| CNN | 0.116242 | 0.286026 | 0.180968 | 0.356107 |
| TFGrid | 0.110983 | 0.260752 | 0.179789 | 0.340305 |
| FDY | 0.108474 | 0.261929 | 0.171702 | 0.343577 |
| iTransformer | 0.114936 | 0.272464 | 0.177286 | 0.337628 |
| LightGBM | 0.102821 | 0.261044 | 0.173387 | 0.322998 |
| 고정 all5 | 0.106756 | 0.264744 | 0.171654 | 0.326982 |

selection의 알려진 합성 정답 평균은 **0.098866750**, 정답 RMS는 **0.183935105**입니다. query의 정답 모멘트는 미공개이므로 query 예측 평균/RMS를 정답의 대리값으로 해석하거나 public 점수에서 역산하지 않았습니다. 예측 크기 차이는 관측 사실이며, 이것만으로 과대추정·정답 prior·실패 원인을 확정할 수 없습니다.

## 3. score의 구성요소와 포화

신경망의 고정 식은 `score=.5*share+.5*contrast`이며, `contrast=max_{관측 t, f=10..22}(q[t,f]−min_{관측 t}q[:,f])_+`입니다. LightGBM은 `score=.5*share+.5*classifier`입니다. 따라서 `2*score−share`로 복원했습니다. all5의 복원값은 이 구성요소들의 **가중 혼합값**이며 단일 신경망의 q contrast가 아닙니다.

| 구성요소 | selection 평균 | query 평균 | selection ≥.99 | query ≥.99 |
|---|---:|---:|---:|---:|
| CNN contrast | 0.740538 | 0.970127 | 40.60% | 71.13% |
| TFGrid contrast | 0.714583 | 0.983388 | 54.60% | 90.53% |
| FDY contrast | 0.776484 | 0.967692 | 43.80% | 72.93% |
| iTransformer contrast | 0.732467 | 0.950813 | 10.60% | 17.60% |
| LightGBM classifier | 0.660918 | 0.939013 | 20.30% | 44.67% |

query에서 contrast/classifier가 큰 값에 더 몰립니다. 다만 이는 **정답 없는 출력 포화 진단**입니다. 잘못된 예측인지, 실제 양성 비율인지, 특정 주파수/사이트 원인인지는 이 감사로 판정하지 않습니다. score는 판별 점수이고 확률 보정이 검증된 값이 아닙니다.

복원 공식은 저장 selection GPU BF16 실행의 FP32 q/pool과 대조했으며 최대 오차 `1.788139343e−7`, tree 직접 classifier 대조는 `5.960464478e−8`입니다. 주 비교는 **CPU FP32 selection 대 CPU FP32 query**입니다. CPU selection q는 저장돼 있지 않아 GPU q와 동일하다고 주장하지 않습니다. CPU/GPU selection share 최대 차이는 모델별 약 `.00173–.00397`입니다. 점수 logit clamp `1e−6`와 FP32 반올림 때문에 극단값/tie 판정은 수치 민감도가 있습니다. 본 데이터 복원값은 모두 [0,1] 이내여서 범위 clipping은 0개였습니다. 사전 허용 오차는 `3e−6`였습니다.

## 4. 모델 사이 상관

다음은 서로 다른 5개 모델의 10개 쌍 범위입니다. Pearson과 평균 순위 tie를 쓰는 Spearman을 각각 산출했습니다.

| 비교 | selection Pearson | query Pearson | selection Spearman | query Spearman |
|---|---:|---:|---:|---:|
| share | 0.868952–0.949242 | 0.833936–0.951597 | 0.796355–0.907953 | 0.842433–0.947248 |
| contrast / classifier | 0.759073–0.899111 | 0.390945–0.734274 | 0.721283–0.925710 | 0.110859–0.814314 |

share 예측끼리는 query에서도 높은 상관이 남지만 contrast/classifier 사이의 상관은 selection보다 낮아집니다. 상관은 정확도나 인과의 증거가 아닙니다. 동시 오류·같은 특징 의존성 여부도 이 수치만으로 확정할 수 없습니다. 상수 또는 작은 슬라이스에서 상관이 정의되지 않으면 빈 값으로 남겼습니다.

- [selection share Pearson 5×5](selection_pearson_share5x5.csv), [query share Pearson 5×5](query_pearson_share5x5.csv)
- [selection share Spearman 5×5](selection_spearman_share5x5.csv), [query share Spearman 5×5](query_spearman_share5x5.csv)
- [query 구성요소 Pearson 5×5](query_pearson_component5x5.csv), [query 구성요소 Spearman 5×5](query_spearman_component5x5.csv)
- [query share→구성요소 상관](query_pearson_share_to_component5x5.csv), [전체 15변수 Pearson](query_pearson_all15x15.csv), [메타데이터별 상관 전체](correlations_by_metadata_slice.csv)

## 5. 관측 지원과 기존 추정 유형

현재 원본 Test의 완전 관측분 수는 Nobs12 **1개**, 13 **7개**, 14 **717개**, 15 **775개**입니다. 첫 분 전체 결측 **624개**, 마지막 분 전체 결측 **100개**, 일부 band만 결측인 분은 **0개**입니다. 이는 이번에 해시를 확인한 파일을 대상으로 한 finite-mask 집계입니다. Nobs14를 St Francis라고 확정하지 않았습니다.

고정 all5의 query 예측 평균/RMS는 Nobs14에서 `.273840/.326081`, Nobs15에서 `.256545/.328207`입니다. 관측지원 패턴만으로 실제 출처나 정답 크기를 알 수 없습니다.

기존 `eda_deep/tables/test_window_features.csv`의 `probe_site/probe_conf`를 재사용했습니다. `eda_deep/deep_eda.py` 700–725행의 TRAIN-fit kNN25 스펙트럼 domain probe 출력입니다. 현재 ID·Nobs·결측분 패턴은 정확히 일치하지만, 당시 실행시 원본 Test SHA를 별도로 시각 인증한 영수증은 없습니다. **실제 사이트 정답이 아니라 과거의 추정 유형**이고 `probe_conf`도 정답 기반 확률보정 confidence가 아닙니다. 새 site 매칭은 실행하지 않았습니다.

예를 들어 과거 `07 St Francis` 추정 유형은 487개이며 Nobs14 484개·Nobs15 2개·Nobs12 1개입니다. 반대로 Nobs14 717개 모두가 이 유형은 아닙니다. 이 유형의 all5 예측 평균/RMS는 `.277588/.323889`입니다. 이 수치를 실제 St Francis 결과라고 부르지 않습니다. [기존 추정 유형 × Nobs](query_Nobs_by_HISTORICAL_INFERRED_TYPE_NOT_GT.csv)와 [슬라이스 분포 전체](distribution_summary.csv)에 다른 유형·결측 패턴도 함께 보존했습니다.

## 6. 해석과 금지된 활용

본 감사는 저장 출력의 관계·포화·관측지원과의 연관성을 보여 줍니다. query label, site 정답, RMSE/AUC는 계산하지 않았습니다. 쿼리 통계로 generator·calibration·ensemble weight·threshold·source 선택을 변경하지 않았고 이후 그러한 피팅의 근거로 사용하지 않도록 결과에 명시했습니다. 공식 점수/RMS 맞춤, 숨은 정답 복원, Test 재추론, 외부 송신, 서버 접근, 미사용 report 역할 열람은 하지 않았습니다.

[분위수](quantiles.csv), [사전 임계값별 누적 포화율](fixed_threshold_saturation.csv), [구성요소 복원 오차](component_recovery_roundoff.csv), [가용성·알려진 합성 정답 범위](availability_and_known_target_scope.csv)를 함께 제공합니다. 공개하지 않은 정답에 대한 성능 결론과는 구분해서 보아야 합니다.
