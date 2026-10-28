# R11 고정 스트레스 최종 검산

R16 교차 연산자 검증을 주 기준으로 유지한다. 아래는 노출된 R11의 사후 스트레스 진단이며, 이 결과로 원래 레시피·모델·정책을 변경하지 않았다.

| 조합 | 모델 | RMSE | 독립 classifier AUC | 동결 score AUC |
|---|---|---:|---:|---:|
| R11_full_smooth12_VS_st_francis_dense | control_NN | 0.091170 | 0.610891 | 0.622192 |
| R11_full_smooth12_VS_st_francis_dense | control_Tree | 0.084268 | 0.800083 | 0.797276 |
| R11_full_smooth12_VS_st_francis_dense | control_60T40N | 0.081321 | 0.723058 | 0.723058 |
| R11_full_smooth12_VS_st_francis_dense | mixed_NN | 0.080889 | 0.628147 | 0.644038 |
| R11_full_smooth12_VS_st_francis_dense | mixed_Tree | 0.088433 | 0.792769 | 0.793942 |
| R11_full_smooth12_VS_st_francis_dense | mixed_60T40N | 0.080105 | 0.724096 | 0.732321 |
| R11_full_smooth12_VS_st_francis_dense | R9_all5_frozen_Fold0 | 0.075888 | 미저장 | 0.737494 |
| R11_full_smooth12_VS_st_francis_dense | R14_fast3_exactFold0 | 0.048355 | 미저장 | 0.903974 |
| R11_full_smooth12_VS_hf_high | control_NN | 0.086261 | 0.431224 | 0.454112 |
| R11_full_smooth12_VS_hf_high | control_Tree | 0.089886 | 0.644264 | 0.649956 |
| R11_full_smooth12_VS_hf_high | control_60T40N | 0.081910 | 0.515652 | 0.515652 |
| R11_full_smooth12_VS_hf_high | mixed_NN | 0.094180 | 0.494244 | 0.508996 |
| R11_full_smooth12_VS_hf_high | mixed_Tree | 0.098066 | 0.628888 | 0.633716 |
| R11_full_smooth12_VS_hf_high | mixed_60T40N | 0.090428 | 0.548548 | 0.560576 |
| R11_full_smooth12_VS_hf_high | R9_all5_frozen_Fold0 | 0.084569 | 미저장 | 0.769172 |
| R11_full_smooth12_VS_hf_high | R14_fast3_exactFold0 | 0.045776 | 미저장 | 0.931168 |
| R11_full_smooth13_VS_st_francis_dense | control_NN | 0.102365 | 0.555141 | 0.566929 |
| R11_full_smooth13_VS_st_francis_dense | control_Tree | 0.087824 | 0.771679 | 0.769603 |
| R11_full_smooth13_VS_st_francis_dense | control_60T40N | 0.088119 | 0.682263 | 0.682263 |
| R11_full_smooth13_VS_st_francis_dense | mixed_NN | 0.086809 | 0.564763 | 0.585288 |
| R11_full_smooth13_VS_st_francis_dense | mixed_Tree | 0.092311 | 0.764237 | 0.765660 |
| R11_full_smooth13_VS_st_francis_dense | mixed_60T40N | 0.085019 | 0.680192 | 0.691346 |
| R11_full_smooth13_VS_st_francis_dense | R9_all5_frozen_Fold0 | 0.078373 | 미저장 | 0.709949 |
| R11_full_smooth13_VS_st_francis_dense | R14_fast3_exactFold0 | 0.049471 | 미저장 | 0.897256 |
| R11_full_smooth13_VS_hf_high | control_NN | 0.095917 | 0.363260 | 0.385776 |
| R11_full_smooth13_VS_hf_high | control_Tree | 0.092609 | 0.604128 | 0.612392 |
| R11_full_smooth13_VS_hf_high | control_60T40N | 0.087433 | 0.454292 | 0.454292 |
| R11_full_smooth13_VS_hf_high | mixed_NN | 0.098366 | 0.421588 | 0.437736 |
| R11_full_smooth13_VS_hf_high | mixed_Tree | 0.100926 | 0.586812 | 0.593972 |
| R11_full_smooth13_VS_hf_high | mixed_60T40N | 0.094000 | 0.484924 | 0.500056 |
| R11_full_smooth13_VS_hf_high | R9_all5_frozen_Fold0 | 0.086390 | 미저장 | 0.750828 |
| R11_full_smooth13_VS_hf_high | R14_fast3_exactFold0 | 0.046734 | 미저장 | 0.927084 |
| R11_full_smooth15_VS_st_francis_dense | control_NN | 0.153913 | 0.362038 | 0.362417 |
| R11_full_smooth15_VS_st_francis_dense | control_Tree | 0.138654 | 0.490090 | 0.492160 |
| R11_full_smooth15_VS_st_francis_dense | control_60T40N | 0.143024 | 0.400436 | 0.400436 |
| R11_full_smooth15_VS_st_francis_dense | mixed_NN | 0.145051 | 0.318628 | 0.330955 |
| R11_full_smooth15_VS_st_francis_dense | mixed_Tree | 0.133882 | 0.501564 | 0.509615 |
| R11_full_smooth15_VS_st_francis_dense | mixed_60T40N | 0.135895 | 0.384058 | 0.395987 |
| R11_full_smooth15_VS_st_francis_dense | R9_all5_frozen_Fold0 | 0.116927 | 미저장 | 0.467513 |
| R11_full_smooth15_VS_st_francis_dense | R14_fast3_exactFold0 | 0.062626 | 미저장 | 0.820538 |
| R11_full_smooth15_VS_hf_high | control_NN | 0.141164 | 0.133648 | 0.137684 |
| R11_full_smooth15_VS_hf_high | control_Tree | 0.133881 | 0.267216 | 0.275068 |
| R11_full_smooth15_VS_hf_high | control_60T40N | 0.133976 | 0.159492 | 0.159492 |
| R11_full_smooth15_VS_hf_high | mixed_NN | 0.143671 | 0.148412 | 0.154092 |
| R11_full_smooth15_VS_hf_high | mixed_Tree | 0.133497 | 0.269108 | 0.278972 |
| R11_full_smooth15_VS_hf_high | mixed_60T40N | 0.134024 | 0.175364 | 0.184856 |
| R11_full_smooth15_VS_hf_high | R9_all5_frozen_Fold0 | 0.116521 | 미저장 | 0.589224 |
| R11_full_smooth15_VS_hf_high | R14_fast3_exactFold0 | 0.058151 | 미저장 | 0.885972 |
| R11_native_long15_VS_st_francis_dense | control_NN | 0.159309 | 0.383160 | 0.379788 |
| R11_native_long15_VS_st_francis_dense | control_Tree | 0.129707 | 0.392179 | 0.397788 |
| R11_native_long15_VS_st_francis_dense | control_60T40N | 0.139653 | 0.359699 | 0.359699 |
| R11_native_long15_VS_st_francis_dense | mixed_NN | 0.152315 | 0.362878 | 0.368494 |
| R11_native_long15_VS_st_francis_dense | mixed_Tree | 0.130500 | 0.394128 | 0.403981 |
| R11_native_long15_VS_st_francis_dense | mixed_60T40N | 0.136797 | 0.350192 | 0.359968 |
| R11_native_long15_VS_st_francis_dense | R9_all5_frozen_Fold0 | 0.095733 | 미저장 | 0.520109 |
| R11_native_long15_VS_st_francis_dense | R14_fast3_exactFold0 | 0.048319 | 미저장 | 0.813904 |
| R11_native_long15_VS_hf_high | control_NN | 0.145944 | 0.167784 | 0.169808 |
| R11_native_long15_VS_hf_high | control_Tree | 0.126390 | 0.194612 | 0.203932 |
| R11_native_long15_VS_hf_high | control_60T40N | 0.131057 | 0.152124 | 0.152124 |
| R11_native_long15_VS_hf_high | mixed_NN | 0.149650 | 0.195784 | 0.197804 |
| R11_native_long15_VS_hf_high | mixed_Tree | 0.130749 | 0.192340 | 0.201480 |
| R11_native_long15_VS_hf_high | mixed_60T40N | 0.134767 | 0.169720 | 0.177716 |
| R11_native_long15_VS_hf_high | R9_all5_frozen_Fold0 | 0.099587 | 미저장 | 0.625856 |
| R11_native_long15_VS_hf_high | R14_fast3_exactFold0 | 0.045745 | 미저장 | 0.881768 |

31개 원본 음향 패널은 최초 R11 생성 동결 manifest의 SHA와 모두 일치했다. 새 small 모델 예측 39파일에서 RMSE/MAE/bias 및 세 종류 AUC를 별도 sklearn 식으로 재계산했다.

원천 메타 보강: 첫 source_overlap.csv의 r9_original에는 원래 recipe에 reference_index가 없어서 0이 기재됐다. source_reference_lineage_verified.csv는 당시 저장된 positive_schedule로 연결을 복구한다. 이 0을 참조 미사용이나 미관측 참조로 해석하면 안 된다. 또한 모든 패널에서 할당된 참조와 실제 관측 양성에 기여한 참조를 구분했다. 원본 음향/예측/수치지표는 수정하지 않았다.

안전한 재사용 파일은 predictions_numeric/ 아래39개다. 첫 predictions/의 site 메타는 pandas object dtype이므로 pickle로 열지 않고, 동결 원천 메타에서 Unicode로 다시 기록했다. 잘못 누락된 r9_original 참조 인덱스도 저장 schedule로 채웠다. share/classifier/score/정답 배열은 첫 출력과 비트 단위로 동일하며 원형 파일은 보존했다.

R11 원천은 실제 R16 TRAIN4,000 배경과 ID/group 중복0, 같은 sensor의900초 지원 구간 최소25.25시간 이상 분리다. 참조는 대부분 R16 A와 겹친다. native_long15도 clip 자체는 A와 안 겹치지만 hull/day 중복이 있어 별도 미관측 원천 일반화로 단정하지 않는다.

평탄·늘림 full_smooth, 자연 native_long15, 비대칭 EQ, SF 관측밀도는 서로 다른 스트레스다. 시간 불변 입력이나 이 평가의 실패가 실제 시계열 복원의 불가능성을 증명하지 않는다. 대회 generator의 원인이나 공식 점수를 추론하지 않는다. SF5view는 배경 source 반복이며 sparse/dense의 시기 차이가 남는다.

[전체532지표](metrics.csv) · [정밀 참조 계보](source_reference_lineage_verified.csv) · [기존23모델의 측정 가용성](old_model_comparison_availability.csv) · [원 보고서](README.md)
