# R13 단일 paired-epoch mixture 조건 결과

사전 선언한 추가 조건 1개를 완료했습니다. **일반 패널의 score AUC는 개선됐지만, SF 장시간 음향 패널의 AUC≥0.8과 고비율 회귀 개선을 동시에 확보하지 못했습니다.** 원 R13 정책을 승격·교체하지 않았으며, 이 결과 뒤에 새 조건을 반복하지 않습니다.

## 실제 학습 조건

- CNN: 원 primary TRAIN 6,000개를 매 epoch 정확히 한 번씩 사용했습니다. baseline/high 두 버전이 다른 양성 1,200개 중 매 epoch 600개만 high를 선택하고, 다음 epoch에는 보완 600개를 선택했습니다. 50개의 epoch쌍 각각에서 해당 행은 baseline/high를 한 번씩 경험했습니다.
- 매 epoch zero 1,200개·long 1,440개, site/source/recorder/sensor에 해당하는 원천 인덱스와 관측 마스크·passage geometry를 유지했습니다. variant에 맞춰 x/w/q/y를 함께 선택했습니다. 원본 미변경 행은 모든 epoch baseline 값을 그대로 씁니다.
- 같은 26,953파라미터 CNN, PCA P1의 11채널, Adam, dropout0.1, batch128, ShareMSE+0.25×energy-weighted cell MSE, cosine LR0.001→0.00001, 실제100epoch·early stopping 없음입니다. validation_fit만으로 첫 최소 checkpoint를 선택했습니다.
- 트리: baseline12,000+highshare12,000을 연결한 augmentation24,000행, 673피처, 원 TRAIN-only PCA 상태, 최대1,500/ES40, CPU2threads입니다. 회귀기와 독립 presence 분류기를 각각 한 번 학습했습니다. CNN은 CPU4threads로 병행했습니다.
- 24,000행은 독립원천24,000개를 뜻하지 않습니다. 12,000개 primary의 두 augmentation이며, 9,600개 primary는 동일한 값으로 중복되고 2,400개는 서로 다른 비율의 두 버전입니다. sample_id suffix를 추가하고 원 ID를 별도로 보존했습니다.
- 두 모델 동결 후 회귀는 CNN/tree50:50 고정, score는 classifier λ∈{0,.25,.5,1} 중 validation_gate AUC 첫 최대를 선택했습니다. 이 한 조건 내 선택은 λ=1입니다. 이는 score만 독립 classifier를 쓰는 것이며 회귀 share는 계속50:50입니다.

## 실제 완료와 오류 보존

- 원 CNN 경로: `models/r13_dynamic_mix_r1/cnn`. 실제100epoch, best epoch59, validation_fit RMSE 0.0811570752, 학습 522.75초.
- 실제 트리 경로: `models/r13_dynamic_mix_r2/tree`. validation_fit RMSE 0.0843084031, regressor best round1493, classifier best round316, learner fit 95.80초.
- 최초24k copied fit에서 Mapping import가 빠져 부스터 학습 전에 실패했습니다. 이후 동적 주입 방식의 helper는 정적 구현을 요구한 지시에 따라 자료 검증 중 SIGINT130으로 중단했습니다. 두 시점 모두 tree 모델 폴더·부스터fit이 없었습니다. 오류 로그와 중단 사전 receipt를 보존했습니다.
- 정적 helper에 collections.abc.Mapping을 명시 import하고 원 fit 연산을 독립적으로 정의했습니다. 원 fit AST와 train row requirement12,000→24,000 외 동일함을 검증했습니다. namespace monkeypatch를 사용한 실제 학습은 없습니다.
- 원 CNN은 source/plan/진행 상태를 고치거나 재개하지 않고100epoch를 마쳤습니다. 최초 병렬 driver의 실패 상태는 r1/FAILED.json에 보존했습니다. 완료CNN의 summary/history/checkpoint를 검증한 뒤 r2/cnn에 byte-exact staging했고, 원 실제학습 경로를 completion receipt에 명시했습니다.
- 최종 aggregate는 `models/r13_dynamic_mix_r2/{ALL_MODELS_FROZEN.json,CORRECTED_EXECUTOR_COMPLETION.json}`입니다. receipt의 `actual_tree_booster_fits:1`은 한 tree learner.fit 호출을 뜻하는 부정확한 필드명입니다. 실제 부스터는 regressor1+classifier1=2개임을 이 보고서와 독립 QA에서 구분합니다.

## 같은17개 개발 패널 비교

각 칸은 RMSE / score AUC입니다. 원 R12 matched의 score=share, 원 R13 selected의 score는 λ=.5, 새 조건의 score는 λ=1이므로, score AUC 변화 전체를 회귀 학습 개선으로 해석하면 안 됩니다. share AUC와 raw classifier AUC·Brier는 전체 CSV에 따로 보존했습니다.

|패널|R12 strict matched|원 R13 selected|새 paired mixture|
|---|---:|---:|---:|
|R12_validation_gate|0.090328 / 0.888125|0.089586 / 0.907388|0.091634 / 0.915550|
|R12_eval|0.062234 / 0.938300|0.060119 / 0.964300|0.062917 / 0.968000|
|R12_high_snr_long|0.156340 / 0.996825|0.134734 / 0.998375|0.148884 / 0.997525|
|R9_selection|0.055490 / 0.814431|0.059630 / 0.913787|0.056259 / 0.924191|
|R9_ensemble_gate|0.033557 / 0.837354|0.034120 / 0.954128|0.034850 / 0.964065|
|R10_unseen__r9_complete|0.052679 / 0.844269|0.055365 / 0.938044|0.055879 / 0.952894|
|R10_seen__mean_scale_only|0.160046 / 0.984525|0.085086 / 0.987319|0.092445 / 0.988181|
|R10_unseen__combined_stress|0.189233 / 0.984281|0.129641 / 0.987419|0.143796 / 0.988038|
|R10_seen__combined_stress|0.200386 / 0.976231|0.135844 / 0.979962|0.144811 / 0.974800|
|R11_full_smooth12_VS_st_francis_dense|0.066016 / 0.847763|0.075926 / 0.863833|0.072780 / 0.875372|
|R11_full_smooth12_VS_hf_high|0.072166 / 0.874928|0.080018 / 0.881740|0.079373 / 0.899868|
|R11_full_smooth13_VS_st_francis_dense|0.066677 / 0.845173|0.076217 / 0.854365|0.073658 / 0.867462|
|R11_full_smooth13_VS_hf_high|0.072658 / 0.872744|0.080242 / 0.875916|0.080027 / 0.894400|
|R11_full_smooth15_VS_st_francis_dense|0.074042 / 0.820224|0.081356 / 0.759756|0.080890 / 0.765013|
|R11_full_smooth15_VS_hf_high|0.078235 / 0.855400|0.084239 / 0.819604|0.085510 / 0.822472|
|R11_native_long15_VS_st_francis_dense|0.069205 / 0.806051|0.073330 / 0.774827|0.073018 / 0.781974|
|R11_native_long15_VS_hf_high|0.074552 / 0.845136|0.078027 / 0.830032|0.079550 / 0.833180|

일반3패널 모두 score AUC>0.9 조건은 충족했지만, full15+SF 0.765013/native15+SF 0.781974로 SF 두 조건은 충족하지 못했습니다. highSNR RMSE0.148884는 원 R12 matched0.156340보다 낮지만 원 R13 selected0.134734보다 높습니다. validation_gate와 eval RMSE도 원 R13보다 높았습니다.

고비율로 바뀐 행을 매 epoch 전면 교체하지 않으면 일부 SF 성능이 약간 회복되기는 했지만 원 R12 수준으로 돌아오지 않았습니다. 따라서 약한 case를 없앤 것만이 문제였다고 확정하거나, 비율 분포 혼합만으로 안전하게 해결됐다고 말할 수 없습니다. CNN의 epoch 노출과 트리의 두-variant concatenation을 함께 바꾼 한 조건이라 개별 원인의 인과 효과도 분리하지 못했습니다.

## 독립 검산 및 한계

- 저장된169행의 지표 3,876개를 독립적으로 재계산했고 최대 차이는0.0입니다. 고정 blend/score·비교모델 identity의 exact 검사236건을 통과했습니다.
- 저장 cell mask와 관측 energy weight의 physical pooling 최대 차이는2.38e-07이며, 결측 위치의 q=0 및 범위·원 sample ID/target 순서를 확인했습니다.
- source/epoch-schedule/24k binding/복구 경로는 별도의 독립 agent QA가 확인합니다. 이번 지표 QA는 point metric·수식·ID·q pooling을 검산했으며 site macro와 bootstrap CI endpoint 전체를 다시 계산하지 않았습니다.
- 모든 패널은 이미 노출된 개발 평가이며 공유 reference가 있습니다. 새 맹검, 공식 LB 개선, 숨은 Test 생성 법칙 검증이 아닙니다. R10 sensor×week bootstrap block 일부는 인접주간15분 간격이므로24h 독립 block이라고 주장하지 않습니다.
- 새 Test 추론·GPU·외부전송·원 생산정책 수정은 없습니다. 추가 학습 조건은 더 실행하지 않았습니다.

[전체169행](results.csv) · [조건부 paired CI](paired_cluster_CI.csv) · [결과·출력 SHA](RESULTS.json) · [독립 지표 검산](INDEPENDENT_QA.json)

## 고정 식별자

- 결과 SHA256: `0251eab6f7ab8adbe32fcbbe49fea1a4c9b8bd257f303cb0b54dbc58e02f3be7`
- 사전 plan content SHA256: `6f8075ac16c0ce38ead813b9d5e92dadb9ce388dace5286e9bf10de994d05b8c`
- 전체2모델 freeze file SHA256: `d6fd0a39817c210c3a0cd6833d2a2e1be9fef3996ae0a5db3cf72b089d065e05`
