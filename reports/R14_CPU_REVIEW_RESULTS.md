# R14 CPU 개선·전수 Tree 실측 및 Mac GPU 검토

## 1. 실행 범위와 판정 원칙

GPU 본학습을 시작하지 않고 요청한 CPU 진단을 실행했다. 별도 최신 요청에 따라 Mac MPS에서는 실제 본체 신경망의 짧은 호환성·속도 검사만 수행했다. Test.csv와 새 제출 CSV를 읽거나 생성하지 않았고, 공식 제출·외부 자문 전송도 하지 않았다.

수치 검산 통과, 학습 실행 완료, 성능 동시 목표 통과는 서로 다른 판정이다. 아래 17패널은 이미 여러 차례 사용한 개발용 스트레스 패널이며, 새로운 블라인드 OOF 또는 공식 LB가 아니다. 같은 VTUAD 참조가 공유되므로 참조 독립 검증이라고 부르지 않는다.

## 2. 0.9445 분류기 출처 정정

원 R12의 full15/SF 분류 AUC **0.944494**와 native15/SF **0.950814**는 원래 L1H1P1 Tree 조건에서 나왔다. 이는 엄격히 마스크·사이트를 맞춘 HM1 대조군과 다르다. 엄격 매칭 baseline Tree 분류기는 이미 full15/SF **0.770705**, native15/SF **0.789269**였다. R13 highshare 분류기의 full15/SF **0.760763**과 공정하게 비교할 차이는 약 −0.00994다. **0.820224는 CNN/Tree 회귀 결합값의 share AUC**이므로 분류기 AUC와 같은 출력처럼 비교하지 않는다.

원 H1의 St Francis Nobs15 무추가 비율은 20.68%에서 66.97%로 바뀌었다. −Nobs AUC 0.794710의 분모는 St Francis 내부이며 전체 TRAIN 지표가 아니다. 이 교란 때문에 0.9445를 깨끗한 분류기 성능으로 복원하는 접근은 채택하지 않았다. [출처·동일 출력 대조](R14_CLASSIFIER_PROVENANCE.md)

## 3. 저장 예측의 교차 결합과 fit-only 단조 보정

23개 회귀 게이팅 × 3개 분류기 × 4개 score 비중, 총 276개 사전 고정 후보를 저장 예측으로 계산했다. 파라미터 선택에는 validation_fit 1,000개만 사용했다. 동시 목표를 충족한 후보는 0개였다. 선택된 baseline 회귀 + baseline classifier는 미관측 일반 AUC 0.950581을 보였지만 full15/SF 0.770705, native/SF 0.789269와 고SNR RMSE 0.156340이 남았다.

별도 34개 단조 보정 후보 역시 통과 0개였다. fit MSE가 선택한 p^1.05는 미관측 일반 RMSE를 0.052512로 줄였지만 고SNR RMSE를 0.164706으로 악화시켰다. 가중 MSE 선택은 identity였다. scalar share만 보정하면 원 마스크와 sum(w*q)의 일치가 유지된다는 보장도 없으므로 생산 물리 헤드에 적용하지 않았다.

[교차 결합 전체 결과](r14_decoupled_v1/RESULTS.md) · [단조 보정 결과](r14_fit_calibration_v1/RESULTS.md)

## 4. 깨끗한 baseline의 상대 크기 특징·가중 손실 실험

TRAIN 12,000개와 CNN 공통 부분집합 6,000개를 그대로 사용했다. 합성 정답이나 무추가·관측 마스크 분포를 바꾸지 않았다. 원 11채널에 (1) LF 31.6–158.5Hz 대비 251.2–3162.3Hz 상대 레벨, (2) TRAIN 평균 주파수 형태를 LF로 고정한 잔차, (3) 해당 추정 기준 대비 양의 초과 비율 3맵을 추가했다. 새 맵은 5,011.9Hz 이하 안전 23대역만 사용한다.

이들은 기존 상대 주파수 프로파일에 이미 있는 정보를 명시적으로 제공하는 표현이다. LF가 순수 배경이라는 보증이나 실제 추가 에너지의 새로운 관측값이 아니다. 공통 +20dB gain 시험 오차는 0이었지만 사이트별 대역 응답은 상쇄되지 않는다.

14채널 CNN 두 조건은 **27,601파라미터**, 각 **100 epoch·Adam·dropout 0.1·early stopping 없음**, validation_fit의 최저 RMSE 체크포인트를 보관했다. 순수 물리 손실 share MSE + 0.25*에너지 가중 cell MSE를 기본으로 두고, c=0 및 행 가중치 1+2*y를 비교했다. Tree는 880특징·12,000행으로 회귀기 2개와 독립 비가중 분류기 1개를 학습했다(max 1500, ES40).

fit-only 선택은 비가중 CNN/Tree 50:50 회귀 + classifier score였다. 미관측 RMSE와 고SNR RMSE는 개선됐지만 고SNR <0.145 및 장시간/SF AUC 조건은 통과하지 못했다. 이 실험만으로 크기 정보 부족이 유일 원인이었다고 결론 내리지 않는다.

|패널|모델|RMSE|Bias|Share AUC|Score AUC|독립 분류 AUC|
|---|---|---:|---:|---:|---:|---:|
|R12_eval|R14_fit_selected|0.066015|-0.002913|0.936225|0.969425|0.969425|
|R12_eval|R12_matched_share_only|0.062234|-0.002600|0.938300|0.938300|—|
|R12_high_snr_long|R14_fit_selected|0.147058|-0.102723|0.996400|0.995300|0.995300|
|R12_high_snr_long|R12_matched_share_only|0.156340|-0.111366|0.996825|0.996825|—|
|R9_selection|R14_fit_selected|0.054368|0.010384|0.823251|0.925193|0.925193|
|R9_selection|R12_matched_share_only|0.055490|0.011175|0.814431|0.814431|—|
|R10_unseen__r9_complete|R14_fit_selected|0.049969|0.006996|0.850625|0.948456|0.948456|
|R10_unseen__r9_complete|R12_matched_share_only|0.052679|0.007800|0.844269|0.844269|—|
|R10_seen__mean_scale_only|R14_fit_selected|0.131432|-0.060952|0.988256|0.983738|0.983738|
|R10_seen__mean_scale_only|R12_matched_share_only|0.160046|-0.074884|0.984525|0.984525|—|
|R10_unseen__combined_stress|R14_fit_selected|0.170919|-0.085723|0.987594|0.981237|0.981237|
|R10_unseen__combined_stress|R12_matched_share_only|0.189233|-0.095227|0.984281|0.984281|—|
|R11_full_smooth15_VS_st_francis_dense|R14_fit_selected|0.072777|0.008828|0.802071|0.766365|0.766365|
|R11_full_smooth15_VS_st_francis_dense|R12_matched_share_only|0.074042|0.012865|0.820224|0.820224|—|
|R11_native_long15_VS_st_francis_dense|R14_fit_selected|0.068816|0.001723|0.791571|0.792135|0.792135|
|R11_native_long15_VS_st_francis_dense|R12_matched_share_only|0.069205|0.007321|0.806051|0.806051|—|

[전체 17패널 원수치](r14_anchor_cpu_r1/results.csv) · [153개 결과 행·pooling 독립 QA](R14_ANCHOR_INDEPENDENT_QA.md)

## 5. 본체 5,252특징 Tree Fold 0 전수 실측

baselineP1과 highshareP1 각각 **78,869행 전수**를 실제 적합했다. 64행 fixture 결과가 아니다. 각 조건의 own selection 1,000개만 ES에 사용했고 모든 모델 동결 후 공통 17패널을 평가했다. compact 673특징/12,000행과는 피처 수뿐 아니라 자료량·합성 source·PCA·seed·ES validation이 달라, 피처 수만의 독립 효과로 해석하지 않는다.

Full Tree의 기본 score는 share이며 아래 분류 AUC는 독립 classifier 확률이다. 두 출력을 구분한다. 17패널 라벨을 보고 새로운 앙상블 비중을 고르지 않았다.

|패널|모델|RMSE|Bias|Share AUC|Score AUC|독립 분류 AUC|
|---|---|---:|---:|---:|---:|---:|
|R12_eval|FullTree_baselineP1|0.061536|0.000423|0.922975|0.922975|0.974600|
|R12_eval|FullTree_highshareP1|0.062795|0.003036|0.923575|0.923575|0.975000|
|R12_eval|CompactTree_baselineP1|0.079101|-0.001436|0.924200|0.964700|0.966675|
|R12_eval|CompactTree_highshareP1|0.072526|0.002214|0.915575|0.963100|0.966400|
|R12_high_snr_long|FullTree_baselineP1|0.134075|-0.092999|0.996850|0.996850|0.998525|
|R12_high_snr_long|FullTree_highshareP1|0.113444|-0.078877|0.997375|0.997375|0.998850|
|R12_high_snr_long|CompactTree_baselineP1|0.201611|-0.145219|0.996275|0.997825|0.997050|
|R12_high_snr_long|CompactTree_highshareP1|0.169629|-0.122632|0.996138|0.998225|0.997400|
|R9_selection|FullTree_baselineP1|0.058601|0.010918|0.795990|0.795990|0.943516|
|R9_selection|FullTree_highshareP1|0.063907|0.016214|0.791962|0.791962|0.943729|
|R9_selection|CompactTree_baselineP1|0.069357|0.014612|0.777630|0.904790|0.921593|
|R9_selection|CompactTree_highshareP1|0.071773|0.022978|0.782441|0.910747|0.920682|
|R10_unseen__r9_complete|FullTree_baselineP1|0.055690|0.006768|0.804850|0.804850|0.968662|
|R10_unseen__r9_complete|FullTree_highshareP1|0.059615|0.013714|0.804744|0.804744|0.967862|
|R10_unseen__r9_complete|CompactTree_baselineP1|0.067281|0.010168|0.812600|0.937781|0.950581|
|R10_unseen__r9_complete|CompactTree_highshareP1|0.073730|0.018997|0.810856|0.931412|0.945800|
|R10_seen__mean_scale_only|FullTree_baselineP1|0.192996|-0.102330|0.988278|0.988278|0.994094|
|R10_seen__mean_scale_only|FullTree_highshareP1|0.085257|-0.016342|0.990291|0.990291|0.996125|
|R10_seen__mean_scale_only|CompactTree_baselineP1|0.200986|-0.102550|0.974900|0.977812|0.987537|
|R10_seen__mean_scale_only|CompactTree_highshareP1|0.119580|-0.020876|0.980744|0.983744|0.986744|
|R10_unseen__combined_stress|FullTree_baselineP1|0.212105|-0.118417|0.979556|0.979556|0.984644|
|R10_unseen__combined_stress|FullTree_highshareP1|0.133024|-0.049473|0.979425|0.979425|0.988350|
|R10_unseen__combined_stress|CompactTree_baselineP1|0.219969|-0.119552|0.975247|0.985575|0.986962|
|R10_unseen__combined_stress|CompactTree_highshareP1|0.158005|-0.055403|0.972672|0.983844|0.986206|
|R11_full_smooth15_VS_st_francis_dense|FullTree_baselineP1|0.069917|0.013378|0.817519|0.817519|0.819699|
|R11_full_smooth15_VS_st_francis_dense|FullTree_highshareP1|0.076674|0.017894|0.804744|0.804744|0.825256|
|R11_full_smooth15_VS_st_francis_dense|CompactTree_baselineP1|0.082272|0.021115|0.813276|0.770635|0.770705|
|R11_full_smooth15_VS_st_francis_dense|CompactTree_highshareP1|0.095048|0.029082|0.776769|0.754769|0.760763|
|R11_native_long15_VS_st_francis_dense|FullTree_baselineP1|0.060407|0.010955|0.799051|0.799051|0.818006|
|R11_native_long15_VS_st_francis_dense|FullTree_highshareP1|0.064999|0.015238|0.783865|0.783865|0.816641|
|R11_native_long15_VS_st_francis_dense|CompactTree_baselineP1|0.081380|0.012948|0.795878|0.786615|0.789269|
|R11_native_long15_VS_st_francis_dense|CompactTree_highshareP1|0.088910|0.025092|0.769346|0.770141|0.777237|

[전수 Tree 결과와 학습 시간·메모리](R14_FULL_TREE_RESULTS.md) · [전체 17패널 CSV](r14_tree_full_r1/results.csv)

### 5.1 전수 Tree와 소형 CNN의 fit-only 회귀·분류 분리 결합

full baseline Tree와 새 비가중 anchor CNN의 회귀 비중 5개, 독립 분류기 3종 × score 비중 5개를 사전 고정했다. TRAIN과 validation_fit의 배경 source 교집합은 0이고 900초 support 기준 최소 간격은 24.25시간이다. 다만 full Tree ES용 selection과 이 fit의 배경 105개는 중복되며 CNN checkpoint도 이 fit으로 선택됐으므로 독립 성능 시험은 아니다.

fit-only 선택은 **CNN 25% + full Tree 75% 회귀**, **full Tree 독립 분류 확률 100% score**다. 정책 SHA를 먼저 고정하고 17패널을 평가했다.

|사전 고정 조건|실측|판정|
|---|---:|---|
|고SNR RMSE <0.145|0.122765|통과|
|미관측 일반 RMSE ≤0.0527|0.049560|통과|
|미관측 일반 score AUC ≥0.92|0.968663|통과|
|full15/SF score AUC ≥0.82|0.819699|미달|
|native15/SF score AUC ≥0.80|0.818006|통과|

**4/5 조건을 통과했다.** full15의 0.819699를 반올림해 0.82 통과로 바꾸지 않았다. 모든 75개 가능한 조합을 서술적으로 검산하면 실제 통과 조합은 3개 존재한다. 그러나 17패널 결과를 본 뒤 그것을 선택 정책으로 바꾸지 않았고, 생산 승인은 false로 유지했다. 이 미세한 차이의 실전적 중요성이나 공식 LB 개선을 개발 점수만으로 확정할 수 없다.

[결합 원수치·사후 75조합 서술](r14_fulltree_decoupled_v1/README.md) · [선택 정책 17패널](r14_fulltree_decoupled_v1/selected_results.csv)

## 6. 전처리 병목과 Mac GPU 실행 호환성

Torch 벡터화 전처리는 기존 채널 최대 오차 4.55e−13, 가중치 차이 0으로 일치했다. 실제 78,869개 전수 실행은 CPU 2스레드에서 합성/collation 13.047초 + 특징 변환 2.436초, SHA 기록 포함 총 16.030초였다. 이전 115.44초는 1스레드 측정이라 엄격한 동일 조건 속도비가 아니다. 합성 비용 자체는 남아 있다. [전수·부분집합 벤치마크](R14_TORCH_FRONTEND.md)

M4 Pro 20코어 GPU·통합 메모리 24GB에서 실제 4본체 × batch8/16/32, 총 12조건의 FP32 forward/backward/Adam이 통과했다. CPU 자동 fallback은 사용하지 않았다. CPU FP64 전처리 후 MPS FP32 본체에 연결했다. batch32 학습 1step 중앙값은 다음과 같다(준비1회+측정3회, 동시 CPU 부하 미통제).

|본체|파라미터|CPU 2thread|MPS|
|---|---:|---:|---:|
|Wide CNN|1,491,329|0.36044초|0.04095초|
|TFGrid–FDY–Conformer|2,563,813|1.55572초|0.19513초|
|FDY–CRNN|130,765|0.06532초|0.03138초|
|iTransformer|299,919|0.01637초|0.02336초|

전체 epoch/100epoch 시간과 장시간 수렴은 미측정이며, 기존 CUDA 전용 생산 러너를 MPS로 자동 변경하지 않았다. TimeMoE·Dasheng·이번 14채널 mini CNN은 MPS 검사 대상이 아니다. [MPS 상세 검토](R14_MPS_BENCHMARK.md) · [브라우저 검토 화면](R14_MPS_REVIEW_START.html)

## 7. 최종 해석 범위

Mac GPU의 실행 가능성은 확인됐지만 이는 데이터·score의 성능 동시 목표나 공식 LB 일반화 확인과 별개다. 새 official score는 없다. 원 R13 코드·자료·정책을 보존했고 기존 production 승인 false 및 외부 GPU 본학습 보류 상태를 유지했다. 실패한 partial run을 학습 재개에 사용하지 않았다.
