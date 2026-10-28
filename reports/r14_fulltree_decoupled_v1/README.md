# R14 본체 Tree와 독립 분류기의 fit-only 결합

**사전에 정한 fit-only 정책은 5조건 중4개를 통과했고, 전체 동시개선에는 미달했다.** 선택된 회귀는 anchor_c0 CNN25%+full5252 baseline Tree75%, score는 full5252 독립 classifier100%(λ1)다. full15/SF AUC가 **0.819698718**로 사전기준0.82에 못 미쳤다. 결과를 본 뒤 λ나 기준을 바꾸지 않았고 생산 승인false를 유지한다.

동시에, 사전 지정75개 유한 조합을 정책 동결 후 기술적으로 펼쳐본 결과 **3개가5조건을 통과했다.** 0개라고 쓰면 틀린다. 이3개는17패널을 보고 알게 된 사후 feasibility로, 새 정책으로 재선택하거나 독립 검증 성공이라고 부르지 않는다. 현재 `selected_policy.json`은 처음 fit 선택 그대로다.

## 사전 고정과 원천 조건

이 결합의 CNN은 **27,601개 파라미터·6,000 TRAIN·100epoch의 소형 anchor_c0**다. 본체 Wide/TFGrid의 GPU 5-Fold 학습 결과가 아니다. full Tree는78,869 TRAIN/5,252특징이고 strict673 classifier는 별도12,000 TRAIN 모델이므로, 크기·학습량이 같은 모델끼리의 대조로 해석하지 않는다.

- 모델 새학습0, 새17패널 추론0이다. 이미 완료된 baseline5252 Tree, anchor_c0 CNN, R12 strict673 classifier를 사용했다. fit1000의 fullTree 예측만 새로 계산했다. highshare fullTree는 이 결합에 쓰지 않지만, 원 계획대로 full 두 모델이 모두 동결될 때까지 기다렸다.
- fullTRAIN78,869와 R12 validation_fit1000은 primary ID·background group 교집합0이며, 같은sensor에서900초 support를 고려한 간격 최소24.25시간이다. 이는 동일export-clock의 차이이고 확인된 UTC라고 표현하지 않는다.
- 독립 평가와는 다르다. anchor CNN은 같은fit으로 최저checkpoint를 골랐고, fullTree의 own R13 ES selection과 R12 fit에는 배경ID105개·group12개가 겹친다. 수치 합성은 달라도 같은 배경 원천이다. fit 점수의 선택 낙관이 남는다.
- 모든17패널은 기존 개발 과정에 노출됐고 참조 음원도 공유한다. 새 맹검·reference-independent OOF·공식 Test 성능이 아니다. [사전 계획](plan.json) / [원천 분리 근거](../../models/r14_tree_full_r1/R12_VALIDATION_FIT_SOURCE_SEPARATION.json)

## 선택 순서

회귀는 `a×CNN+(1−a)×Tree`, a={0,.25,.5,.75,1} 중fit RMSE 최소·첫 canonical tie를 골랐다. 분류는 full5252/strict673/둘 평균3개와 λ={0,.25,.5,.75,1}의15개 중 같은fit AUC 최대를 골랐다. score는 `(1−λ)×선택된 회귀share+λ×독립classifier`다. 새 affine/threshold/calibration은 없다.

|fit CNN 비중|fit RMSE|fit share AUC|
|---:|---:|---:|
|0|0.06629582|0.89424687|
|0.25|0.06386728|0.90065625|
|0.5|0.06509780|0.90368125|
|0.75|0.06979411|0.90672500|
|1|0.07732731|0.90768125|

선택 결과fit RMSE **0.06386728**, score AUC **0.97400000**는 fitting 수치이며 일반화 추정값이 아니다. 이 정책을 file SHA `cb8c5918a639d35d0b0798abb0e3552a6bec93b727748ec9e8d233228d894d4b`로 저장한 뒤에만17패널의 수치 예측을 열었다. [15개 score 후보](fit_score_candidates.csv) / [정책](selected_policy.json)

anchor fit 예측파일에는 per-row ID가 없다. 같은SHA의validation NPZ와 원 코드가 보존한 순서·exact target vector를 근거로 연결했다. 반복 zero 정답의 일치만으로 unique row 순서를 증명했다고 과장하지 않는다.17패널은 세 원본의 실제 sample_id/target 벡터를 행별로 완전히 대조했다.

## 선택된 정책의17패널 결과

|패널|RMSE|bias|share AUC|score AUC|
|---|---:|---:|---:|---:|
|R10_seen__combined_stress|0.194103|-0.107930|0.986888|0.985281|
|R10_seen__mean_scale_only|0.161072|-0.082759|0.990894|0.994094|
|R10_unseen__combined_stress|0.189837|-0.103334|0.986138|0.984644|
|R10_unseen__r9_complete|0.049560|+0.006159|0.828788|0.968662|
|R11_full_smooth12_VS_hf_high|0.065256|+0.021487|0.874572|0.946388|
|R11_full_smooth12_VS_st_francis_dense|0.058185|+0.020771|0.840071|0.912686|
|R11_full_smooth13_VS_hf_high|0.066065|+0.019465|0.872108|0.941964|
|R11_full_smooth13_VS_st_francis_dense|0.059300|+0.018280|0.836654|0.904269|
|R11_full_smooth15_VS_hf_high|0.071601|+0.012448|0.858024|0.892364|
|R11_full_smooth15_VS_st_francis_dense|0.066746|+0.009638|0.813340|0.819699|
|R11_native_long15_VS_hf_high|0.064630|+0.009504|0.844004|0.887100|
|R11_native_long15_VS_st_francis_dense|0.057318|+0.006013|0.791750|0.818006|
|R12_eval|0.058540|-0.000637|0.934200|0.974600|
|R12_high_snr_long|0.122765|-0.085707|0.997050|0.998525|
|R12_validation_gate|0.084154|+0.011366|0.875650|0.941700|
|R9_ensemble_gate|0.033678|+0.002420|0.846836|0.944882|
|R9_selection|0.053055|+0.009923|0.815314|0.943516|

주요 결과는 highSNR RMSE0.1227653·bias−0.0857066, normal unseen RMSE0.0495601·score AUC0.9686625, native15/SF score AUC0.8180064다. 고SNR 과소추정은 여전히 남고, full15/SF는 기준0.82를 근소하게 못 넘었다. 작은 차이가 통계적으로 유의하다거나 주원인을 완전히 제거했다는 뜻도 아니다.

## 사후75조합의 가능 범위

아래3개는 사전에 고정한75개 중 같은5조건을 모두 만족했다. 조건은 highSNR RMSE<.145, unseen RMSE≤.0527/AUC≥.92, full15/SF AUC≥.82, native15/SF AUC≥.80이다. 전부 같은노출패널의 기술적 탐색 결과여서, 이 정보를 사용해 새정책을 채택하려면 별도사전선택·검증 경계를 정해야 한다. 여기서는 자동채택/재학습하지 않는다.

|CNN 비중|독립classifier|λ|
|---:|---|---:|
|0.25|full5252|0.75|
|0.5|full5252|0.75|
|0.75|full5252|0.75|

[75조합 전체 수치](all75_descriptive_results.csv) / [통과·실패 전수](all75_feasibility.csv). 75개를 보고 가장 좋아 보이는 결과로 `selected_policy`를 덮어쓰지 않았다.

## 검산과 보존

별도코드로 fit5개/15개 승자,17×75개 및선택정책의지표 **10,336개**를 독립재계산했다. 최대차 **2.22e−16**, 세 원본의 ID·target·member vector는 exact일치했다. 원 모델·이전 정책·물리 데이터·source SHA를 변경하지 않았다. MPS/GPU/새 모델 학습·Test/네트워크/공식 제출은0이다. [독립 QA](INDEPENDENT_QA.json) / [기계 판독 결과](RESULTS.json)

실행 소스: `ops/evaluate_r14_fulltree_decoupled.py` SHA `2406da7431a8c1b15fee3d4b680df8b418083c53a6e32a7cfcf3fe9f38b41f1f`. 계획file SHA `e78c8cdd10ee07d0afbfff185886dab79e4623557d1a52971ed72f25d1bc838b`, 결과file SHA `18fc5095a1ff3bf493325fbad7ab9331d40f6ef0efcfd04704b3d9501e9da0ea`. full과compact 비교에는TRAIN수·물리법칙·PCA·ES관측까지 달라지는 요소가 있으므로, 개선을5,252특징 하나의 인과효과라고 주장하지 않는다.
