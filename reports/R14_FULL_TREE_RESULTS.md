# R14 본체5,252특징 Tree: Fold0 실제 CPU 결과

**baselineP1과 highshareP1 모두 실제78,869행 전수 학습·동결·17개 개발 패널 평가를 완료했습니다.** 소형64행 fixture나 12k/673특징 모델로 대체하지 않았습니다. 본체 Tree는 소형 Tree보다 회귀 성능이 좋아졌지만, 모든 패널의 모든 목표를 동시에 만족했다는 판정이나 생산 승격은 하지 않았습니다.

## 실제 학습과 자원

|조건|TRAIN / selection|특징|own selection RMSE|회귀 best / 분류 best|fit시간|최대RSS|
|---|---:|---:|---:|---:|---:|---:|
|baselineP1|78,869 / 1,000|5,252|0.07352335|1497 / 890|13.07분|6.11GiB|
|highshareP1|78,869 / 1,000|5,252|0.07998436|1496 / 887|12.64분|5.72GiB|

- 두 조건은 별도 process에서 순차 실행했습니다. CPU2threads, 최대1,500/ES40, 원본 PhysicsLGBM의 L2 회귀기와 binary-logloss classifier를 각각 학습했습니다. 합계4개 booster입니다. 두 regression best round가 최대값에 가까우므로 완전한 수렴을 증명한 결과는 아닙니다.
- 두 fit 시간 합계는25.70분, 고정17패널 평가는28.25초였습니다. 한 조건의 float32 TRAIN 특징 행렬은1,656,879,952bytes(약1.54GiB)입니다. runtime RSS와 논리 배열 할당량은 다른 값이며, 최종 최대RSS를 위 표에 사용했습니다.
- 원 R13 epoch1 primary78,869개를 빠짐없이 한 번씩 생성했습니다. 각 조건 zero15,774개/positive63,095개이며, 20% 정수 오차는 원5epoch주기 법칙을 유지합니다. highshare는 양성 중15,774개의 alpha만 바꿉니다. 두 variant의 원천·reference·support·geometry metadata가 일치하는지 검산했습니다.
- 원 production과 같은 각 variant의 자기 selection1,000개만 ES에 사용했습니다. 배경·참조·기하는 paired지만 일부 alpha와 정답이 다르므로 selection RMSE와 best round를 같은 관측 조건의 직접 대결로 해석하면 안 됩니다.
- 원 full Fold0 TRAIN-only PCA SHA `8cc263b3a43f0715daf755bed142b2e45b4373030a1e89fa7c499d77283d2d8d`를 그대로 썼습니다. 피처는11채널 grid4,125 + 시간 통계1,100 + physics27 =5,252입니다. 실제 booster feature dimension도5,252입니다.

## 공통17개 패널: 본체 두 조건

회귀 share와 독립 presence classifier는 다른 출력입니다. 기본 score는 share이고, classifier AUC는 분류 확률의 진단 지표로 따로 보고했습니다. classifier를 바꿔도 RMSE는 바뀌지 않습니다.

|패널|baseline RMSE|baseline classifier AUC|highshare RMSE|highshare classifier AUC|
|---|---:|---:|---:|---:|
|R12_validation_gate|0.088924|0.941700|0.095527|0.942075|
|R12_eval|0.061536|0.974600|0.062795|0.975000|
|R12_high_snr_long|0.134075|0.998525|0.113444|0.998850|
|R9_selection|0.058601|0.943516|0.063907|0.943729|
|R9_ensemble_gate|0.036866|0.944882|0.039796|0.948399|
|R10_unseen__r9_complete|0.055690|0.968662|0.059615|0.967862|
|R10_seen__mean_scale_only|0.192996|0.994094|0.085257|0.996125|
|R10_unseen__combined_stress|0.212105|0.984644|0.133024|0.988350|
|R10_seen__combined_stress|0.218216|0.985281|0.134457|0.990050|
|R11_full_smooth12_VS_st_francis_dense|0.062388|0.912686|0.070931|0.914109|
|R11_full_smooth12_VS_hf_high|0.068058|0.946388|0.078640|0.943892|
|R11_full_smooth13_VS_st_francis_dense|0.063189|0.904269|0.071715|0.907391|
|R11_full_smooth13_VS_hf_high|0.068655|0.941964|0.079216|0.939836|
|R11_full_smooth15_VS_st_francis_dense|0.069917|0.819699|0.076674|0.825256|
|R11_full_smooth15_VS_hf_high|0.073761|0.892364|0.082901|0.889068|
|R11_native_long15_VS_st_francis_dense|0.060407|0.818006|0.064999|0.816641|
|R11_native_long15_VS_hf_high|0.066591|0.887100|0.074361|0.881096|

baseline 단독은 R10 unseen RMSE0.055690/classifier AUC0.968663, highSNR RMSE0.134075/bias−0.092999, full15+SF classifier AUC0.819699/native15+SF0.818006입니다. classifier AUC≥0.8은 두 SF에서 만족하지만 unseen RMSE≤0.0527 기준은 만족하지 못했습니다. full15의 share AUC는0.817519, native의 share AUC는0.799051로 classifier AUC와 혼동하지 않았습니다.

highshare는 highSNR RMSE0.113444/bias−0.078877, scale-only0.085257, unseen combined-stress0.133024로 회귀 고비율 스트레스를 더 줄였습니다. 대신 일반 unseen RMSE는0.059615로 baseline보다 높습니다. 고비율 학습이 일반 오차와 스트레스 오차 사이의 차이를 완전히 없앴다고 할 수 없습니다.

## 소형673특징 Tree와 같은 패널 비교

아래는 Tree끼리 비교한 RMSE입니다. 기존 R12/R13의 CNN+Tree50:50 결과와는 모델 구성이 다릅니다. 예를 들어 원 R12 matched blend의 unseen RMSE0.052679는 새 full baseline Tree0.055690보다 여전히 낮습니다.

|패널|소형 baseline|본체 baseline|소형 highshare|본체 highshare|
|---|---:|---:|---:|---:|
|R12_validation_gate|0.105360|0.088924|0.106258|0.095527|
|R12_eval|0.079101|0.061536|0.072526|0.062795|
|R12_high_snr_long|0.201611|0.134075|0.169629|0.113444|
|R9_selection|0.069357|0.058601|0.071773|0.063907|
|R9_ensemble_gate|0.045834|0.036866|0.047608|0.039796|
|R10_unseen__r9_complete|0.067281|0.055690|0.073730|0.059615|
|R10_seen__mean_scale_only|0.200986|0.192996|0.119580|0.085257|
|R10_unseen__combined_stress|0.219969|0.212105|0.158005|0.133024|
|R10_seen__combined_stress|0.230870|0.218216|0.163581|0.134457|
|R11_full_smooth12_VS_st_francis_dense|0.075403|0.062388|0.090166|0.070931|
|R11_full_smooth12_VS_hf_high|0.080623|0.068058|0.091418|0.078640|
|R11_full_smooth13_VS_st_francis_dense|0.076185|0.063189|0.090536|0.071715|
|R11_full_smooth13_VS_hf_high|0.081218|0.068655|0.091715|0.079216|
|R11_full_smooth15_VS_st_francis_dense|0.082272|0.069917|0.095048|0.076674|
|R11_full_smooth15_VS_hf_high|0.085904|0.073761|0.095350|0.082901|
|R11_native_long15_VS_st_francis_dense|0.081380|0.060407|0.088910|0.064999|
|R11_native_long15_VS_hf_high|0.085212|0.066591|0.090414|0.074361|

이 비교에서는 행 수12k→78,869, 피처673→5,252, 합성 source/HF 법칙, fold-PCA, ES selection, seed2026102712→2026102713이 함께 달라집니다. 따라서 개선을 단순히 특징 개수 증가나 모델 크기 하나의 인과 효과로 단정할 수 없습니다.

## 경계·실행 증거와 해석

- 원 R12/R13 코드·데이터·가중치·정책은 수정하지 않았습니다. R14 source-only policy의 production_recipe_approved=false를 유지했고, 생산 runner나 GPU/MPS·SSH·Test 경로를 실행하지 않았습니다. 소스 guard를 우회하거나 핫패치하지 않았습니다.
- 새 actual fit 경로는 `models/r14_tree_full_r1/{baselineP1,highshareP1}/model`입니다. 두 모델 완료 후 `ALL_TREES_FROZEN.json`을 먼저 저장한 뒤17개 평가를 열었습니다. 원 R13 ensemble_fit/gate/report는 사용하지 않았고, report 역할의 원장 gate를 우회하지 않았습니다.
- TRAIN과17개 패널 구성 원천 메타데이터의 base ID·background_group 교집합은0입니다. 추가 fit-only 결합용 R12validation_fit1,000개와의 ID/group교집합도0이고, 같은 센서900초support 최소 간격은24.25시간입니다. 원 export-clock 차이이며 independently verified UTC라고 주장하지 않습니다.
- VTUAD509개 reference는 공유하고, 모든17패널은 과거에 노출된 개발 평가입니다. source 배경 분리만으로 맹검 reference 일반화나 공식 Test loss를 검증한 것은 아닙니다. local 상승과 LB 상승의 동일성을 약속하지 않습니다.
- 원본 classifier·공유 sample ID·target 순서·회귀 출력을 저장했습니다. 새 ensemble/조건별 gate/score λ를 이 보고서에서 선택하지 않았습니다. 추가 finite 결합은 별도 사전 계획과 fit-only 선택 경로의 독립 작업입니다.
- 독립 saved-array QA는2,682개 scalar를 재계산했고 최대 차이는0.0입니다. 전수 N/primary유일성/25%변경/실제5,252차원/모델·PCA·코드 SHA/저장복원1↔7/원 comparator ID·출력불변을 확인했습니다. site macro와 bootstrap CI endpoint 전체의 별도 재계산은 이번 QA 범위 밖입니다.
- paired CI는 고정 개발 패널·공유 reference에 조건부입니다. R10 sensor×week 일부 인접 block 간격이15분이므로24h 독립 bootstrap block이라고 주장하지 않습니다.

[기계 판독 요약](R14_FULL_TREE_RESULTS.json) · [전체118행](r14_tree_full_r1/results.csv) · [전체 결과·SHA](r14_tree_full_r1/RESULTS.json) · [독립 검산](r14_tree_full_r1/INDEPENDENT_QA.json) · [paired CI](r14_tree_full_r1/paired_cluster_CI.csv)

전체 RESULT SHA256: `6a06a05c88e6050531e4620117f2b1167bd0c547aa86242b8ff019542dff75d3`
