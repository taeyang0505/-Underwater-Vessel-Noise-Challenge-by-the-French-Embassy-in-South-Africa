# R16 동결 소형 모델의 고정 R11 스트레스 평가

R16 원래 교차 연산자 검증을 제1기준으로 유지하고, 이 평가는 이미 노출된 고정 R11 스트레스의 사후 진단으로만 수행했다. Test·Test 파생 파일·기존모델 재학습·생성기 수정·정책 재선택은 없다. 원본 패널을 새로 합성하지 않았다.

저장 음향31개(총29,932행)에 control/mixed NN·Tree·고정60T40N을 추론했다. 표준 R11 양성4×음성2 조합8개는 기존 예측을 이어 붙여 평가했다. 조합/반복을 독립 원천으로 세지 않는다.

## 표준 장시간·SF/HF 조합

| 패널 | control NN RMSE | mixed NN RMSE | mixed NN 독립분류 AUC | mixed NN 동결score AUC | R9 기존all5 AUC | R14 기존Fast3 AUC |
|---|---:|---:|---:|---:|---:|---:|
| R11_full_smooth12_VS_st_francis_dense | 0.091170 | 0.080889 | 0.628147 | 0.644038 | 0.737494 | 0.903974 |
| R11_full_smooth12_VS_hf_high | 0.086261 | 0.094180 | 0.494244 | 0.508996 | 0.769172 | 0.931168 |
| R11_full_smooth13_VS_st_francis_dense | 0.102365 | 0.086809 | 0.564763 | 0.585288 | 0.709949 | 0.897256 |
| R11_full_smooth13_VS_hf_high | 0.095917 | 0.098366 | 0.421588 | 0.437736 | 0.750828 | 0.927084 |
| R11_full_smooth15_VS_st_francis_dense | 0.153913 | 0.145051 | 0.318628 | 0.330955 | 0.467513 | 0.820538 |
| R11_full_smooth15_VS_hf_high | 0.141164 | 0.143671 | 0.148412 | 0.154092 | 0.589224 | 0.885972 |
| R11_native_long15_VS_st_francis_dense | 0.159309 | 0.152315 | 0.362878 | 0.368494 | 0.520109 | 0.813904 |
| R11_native_long15_VS_hf_high | 0.145944 | 0.149650 | 0.195784 | 0.197804 | 0.625856 | 0.881768 |

## 정확한 비교와 미측정

R9 과거4NN·Tree Fold0·동결 all5 예측은 같은 원본 samples 파일 SHA와 원래 저장 순서로 재사용했다. 일부 옛 NPZ에는 per-row ID가 없어 SHA-결박 positional comparison이라는 한계를 기록한다. R14는 표준8개 조합에만 정확한 sample_id와 정답이 일치한 캐시를 사용했다. 비교 파일이 없는 기존23모델의 항목은 미측정으로 표기했고 새로 추론하지 않았다. 독립 classifier가 저장되지 않은 옛 출력은 AUC를 만들거나 복원해 채우지 않았다. R9 derived score와 R14 share-only NN score를 독립분류 확률이라고 부르지 않는다.

## 원천·정답·해석 한계

source_overlap.csv는 실제 R16 TRAIN4,000개 배경 ID/group/900초 지원 간격과 R11 각 패널을 비교한다. 참조는 정확한 clip index뿐 아니라 R16 A와 hull·날짜 중복도 보고한다. 배경이 분리돼도 참조가 겹치면 미관측 참조 일반화라고 부르지 않는다. 전체509 참조 그래프가 연결돼 있다는 기존 한계도 유지한다.

R11 full_smooth 계열은 통계적으로 늘린/평탄한 장시간 성분 등 사전 고정된 가정 밖 challenge를 포함한다. 이것이 대회 생성기라는 증거는 없다. edge 복원과 QC대조, 비대칭 EQ의 원래 불가능/확장 조건, 자연SF sparse/dense의2016–18/2015 시기 차이 및 5view 반복도 그대로 유지한다. density의paired dense→5분 삭제는 다른 실제 sparse 녹음과 구분한다. 추가0 정답은 자연 선박 부재가 아니다.

단일 class만 있는 원시 패널은 AUC 미정의로 저장한다. λ는 R16 local_fit에서 동결된0/.25 그대로 사용하며, 독립분류 AUC와 share-only AUC도 별도로 보고한다. 스트레스 결과로 체크포인트·가중치·λ·threshold를 다시 선택하지 않았다. 새 full5fold OOF, 공식 LB 향상 또는 모든 스트레스 해결을 주장하지 않는다.

[모든 지표](metrics.csv) · [원천 중복](source_overlap.csv) · [기존모델 미측정표](old_model_comparison_availability.csv) · [사전 계획](PLAN.json)
