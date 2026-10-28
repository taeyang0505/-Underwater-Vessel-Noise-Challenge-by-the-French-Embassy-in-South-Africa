# R10 저장 예측 분해 진단

**실제 실행 PASS.** Fold0 selection1,000·ensemble_fit500·ensemble_gate500만 사용했다. Test·미사용 report·서버 접근, 새 모델 추론·학습·정책 변경은 없다. 각 모델의 저장 예측을 ID로 정렬하고 원본 정답·metadata·recipe·파일SHA와 대조했다. tree selection은 저장ID가 없으므로 원래의 ordered db/y/presence SHA 및 binding·artifact SHA를 사용했다.

선언된 frozen all5 가중치는 NN각0.1+tree0.6 그대로다. selection NN은 저장된 GPU BF16 연산/FP32 pooling 결과이고, fit/gate NN은 CPU FP32다. 따라서 역할 간 성능 차이를 precision 효과로 직접 해석하지 않는다. selection all5 값은 이 혼합 정밀도 저장 예측에 같은 고정 가중치를 적용한 진단값이다.

| 역할 | 모델 | RMSE | AUC score | AUC share | score AUC: 명시 zero vs 관측 positive |
|---|---|---:|---:|---:|---:|
| selection | tfgrid_fdy_conformer | 0.079415 | 0.941968 | 0.865195 | 0.924488 |
| selection | lgbm_physics | 0.057347 | 0.942705 | 0.853063 | 0.933754 |
| selection | all5_frozen | 0.061950 | 0.955967 | 0.868326 | 0.942114 |
| ensemble_fit | tfgrid_fdy_conformer | 0.055293 | 0.959050 | 0.927650 | 0.940133 |
| ensemble_fit | lgbm_physics | 0.049454 | 0.973267 | 0.866550 | 0.964733 |
| ensemble_fit | all5_frozen | 0.049384 | 0.972933 | 0.906033 | 0.960500 |
| ensemble_gate | tfgrid_fdy_conformer | 0.044267 | 0.939864 | 0.865070 | 0.930864 |
| ensemble_gate | lgbm_physics | 0.036603 | 0.886466 | 0.863381 | 0.878886 |
| ensemble_gate | all5_frozen | 0.038135 | 0.944843 | 0.873012 | 0.931170 |

명시적 no-source는200/100/100개다. 양성 source 요청이 관측 결측과 만난 뒤 실제 관측 추가에너지가0이 된 예제는126/100/41개다. 이들은 물리 라벨0을 보존한 mask-induced zero이며, 별도 explicit-negative 대비에서는 제외했다. 양성의 의미는 관측된 추가 recording 에너지가0보다 큼이지 자연 선박 존재가 아니다.

| 역할 | Nobs 단독 AUC | −Nobs AUC | Nobs AUC: explicit zero 대비 | HF bump 단독 AUC | Safe bump 단독 AUC |
|---|---:|---:|---:|---:|---:|
| selection | 0.689777 | 0.310223 | 0.545512 | 0.863392 | 0.853057 |
| ensemble_fit | 0.764417 | 0.235583 | 0.630767 | 0.924650 | 0.911967 |
| ensemble_gate | 0.630238 | 0.369762 | 0.502591 | 0.865367 | 0.850017 |

Nobs±는 방향을 선택하지 않고 둘 다 보고했다. Nobs를5 또는15로 고정하면 이 단일 feature의AUC는0.5이며, 모델 all5 score AUC는각각 selection0.9426/0.9420, fit0.9367/0.9744, gate0.9457/0.9090이다. 따라서 Nobs 한 변수만으로 전체 모델 결과를 설명했다고 볼 수 없다. 다만 다른 결측 pattern·site/recorder confounding을 이 검사가 배제한 것은 아니다. Nobs5 표본은 세 역할 모두 StFrancis에 속해 관측량 효과와 사이트 효과를 분리해 확정할 수 없다. Nobs>=13과15 slice는 서로 겹치므로SSE 점유율을 더하면 안 된다.

HF bump는3.162–7.943kHz(indices20..24), safe는1.000–5.012kHz(indices15..22)의 관측 minute별 합산 선형power를dB로 바꾼 뒤 max−median한 값이다. db를 더하지 않는다. 양·음 방향AUC와 Nobs 조건부AUC를 모두 보존했다. 높은 단일featureAUC는 이 generator에서 해당 특징과 라벨이 관련됨을 뜻하며 shortcut·과적합·공식 성능 실패의 인과 증명은 아니다.

| 역할 | (0,.03] positive수 / explicit-negative수 | all5 score AUC | all5 share AUC |
|---|---:|---:|---:|
| selection | 220/200 | 0.874909 | 0.688273 |
| ensemble_fit | 75/100 | 0.924000 | 0.752667 |
| ensemble_gate | 139/100 | 0.846619 | 0.692302 |

매우 작은 positive에서 continuous share와 판별 score의 순위 성능이 다르다. 현재score는 NN에서0.5share+0.5temporal-q-contrast, tree에서는0.5share+0.5classifier다. 저장된 score/share로 component=2score−share를 복원했고, selection의 직접q contrast와 최대1.79e−7, tree의 저장classifier와5.97e−8 이내로 일치했다. 동결 소스의 clamp epsilon1e−6을 고려해3e−6 허용을 사전 선언했다. 미세한 반올림은AUC tie에 영향을 줄 수 있어 selection의 직접q/raw복원/범위복원 AUC도 별도 CSV로 보존했다. component가 자연 선박의 확률이라는 보장은 없다.

vessel archive 내 weak-class background22개에서 나온 요청 예제는selection33/fit19/gate16개이고 관측positive는27/14/15개다. all5의 해당 slice가 차지한 전체SSE는각3.01%/6.22%/1.31%다. 이 slice만으로 전체 오차를 설명할 증거는 없지만 표본이 작고 역할별 차이가 있어 무해함을 입증한 것도 아니다. 이 recording을 추가한 총에너지 fraction은 알려져도 vessel-only energy나 자연 선박 부재가 검증된 것은 아니다. class별 positive와 공통 explicit-negative의AUC를 따로 보고했다.

site·recorder·sensor connected group·actual route·requested family·weak class·native/cropped/기하 overlap/실제 추가관측 support·target bin별 n/RMSE/MAE/bias/AUC/SSE 비중을 CSV에 모두 보존했다. SSE 분모는같은 역할/모델의 전체SSE이며 겹치는 slice를 합하지 않는다. 200회 background_group block bootstrap은 selection25/fit10/gate8개 group을 재표집했다. 이는 같은 generator와 공유509reference 조건부구간이고 reference 독립성이나 새로운 사이트 일반화를 평가한 것이 아니다.

세 역할에는 정확한 hard-negative flag가 없다. TRAIN-only flag를 heldout에 억지로 붙이지 않았고, 임의 고분산 음원을 자연 hard-negative라고 명명하지 않았다. 별도 자연 hard-negative500 panel은 이 작업에서 생성하지 않았다. fit/gate 및 tree의 cell-q가 저장되어 있지 않아새추론으로 채우지 않았고, selection4NN의 minute/band q-energy 기여만 저장했다.

최초 `saved_predictions` 시도는 진단 CSV를 만들기 전 중복 role column 삽입이라는 자체 출력 오류로 종료했으며 FAILED와계획을보존했다. 입력SHA는그때도그대로였다. 수정한 새 `saved_predictions_r1` 실행은전체검사를통과했다. 모델·원천 구현 결함으로 해석하지 않는다.

[전체 수치·정렬·source SHA](summary.json) · [원래 선언 계획](analysis_plan.json) · [전체 모델지표](overall_metrics.csv) · [조건부 slice/SSE](slice_metrics.csv) · [low-positive 대비](positive_vs_explicit_zero_auc.csv) · [단일 feature 양방향AUC](feature_only_auc.csv) · [weak-class 대비](weak_class_positive_vs_explicit_zero_auc.csv) · [background-group CI](background_group_bootstrap_CI.csv) · [저장 예측 행별 분해](aligned_rows_and_saved_prediction_decomposition.csv) · [Nobs×site 표](Nobs_by_site_counts.csv) · [contrast 반올림AUC 민감도](selection_contrast_AUC_roundoff_sensitivity.csv)


## 요청한 핵심 조건부 표

아래는 모두 고정 all5 예측이다. low-positive는 각 구간의 관측 positive와 **명시적 no-source만** 비교하며 mask-induced zero는 제외했다.

| 역할 | 양성 구간 | positive / explicit zero | AUC score | AUC share | AUC component |
|---|---|---:|---:|---:|---:|
| selection | (0,.03] | 220/200 | 0.874909 | 0.688273 | 0.881432 |
| selection | (.03,.08] | 131/200 | 0.961107 | 0.894122 | 0.964542 |
| ensemble_fit | (0,.03] | 75/100 | 0.924000 | 0.752667 | 0.927333 |
| ensemble_fit | (.03,.08] | 69/100 | 0.953623 | 0.893478 | 0.949710 |
| ensemble_gate | (0,.03] | 139/100 | 0.846619 | 0.692302 | 0.847626 |
| ensemble_gate | (.03,.08] | 74/100 | 0.967703 | 0.978378 | 0.962568 |

| 역할: Nobs=15만 | n / positive | RMSE | AUC score | AUC share |
|---|---:|---:|---:|---:|
| selection | 568/452 | 0.054138 | 0.942020 | 0.843264 |
| ensemble_fit | 251/206 | 0.043237 | 0.974434 | 0.929881 |
| ensemble_gate | 323/251 | 0.031210 | 0.908975 | 0.833610 |

| weak-class background 요청 예제 | n / 관측 positive | RMSE | MAE | bias | 전체 SSE 비중 |
|---|---:|---:|---:|---:|---:|
| selection | 33/27 | 0.059197 | 0.035696 | -0.004317 | 3.01% |
| ensemble_fit | 19/14 | 0.063178 | 0.042805 | 0.004713 | 6.22% |
| ensemble_gate | 16/15 | 0.024397 | 0.015816 | -0.000813 | 1.31% |

background22는 원본 reference pool의 weak-class 레코드 수다. 위 n은 그 레코드가 실제 역할에서 요청된 합성 예제 수이며 독립적인 새 녹음 수가 아니다. 자연 선박 여부 정답과 동일시하지 않는다. 이 표는 보고 있는 development 자료의 오차 분해이며, report·공식Test를 평가하지 않았다.

최종 산술 QA도 통과했다. disjoint site/recorder/route/weak-class/Nobs/target-bin의 SSE 비중 합은1이며, 각 단일feature의 양·음 AUC 합은1, Nobs5/15 고정 시 Nobs단일AUC는0.5다. raw db·q·y 에너지 항등식과 recipe 관측지원/reference index도다시대조했다. [산술·입력 불변 검증](diagnostic_sanity_verification.json)
