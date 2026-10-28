# R13 평가 독립 검산

PASS: 17개 패널·458개 지표 행, 12366개 scalar와99개 CI 행을 저장 배열에서 검산했습니다.
Gate RMSE 첫 argmin은 highshareP1, 해당 recipe의 presence AUC 첫 argmax λ는 0.5로 정책과 일치합니다.
반올림 dB에서 독립 계산한 물리 가중치 오차 최대 0, Σwq와 CNN share 오차 1.34e-07, contrast 오차 0입니다. q 범위와 결측 q=0도 확인했습니다.
Raw classifier·contrast가 저장되어 해당 AUC·포화·Brier까지 검산했습니다. CNN4개는3개 신규 fit과1개 완료 control 재사용이며 tree4개는 새 fit입니다. checkpoint·100epochs·첫 best epoch·PCA·source hash도 확인했습니다.
zero_prediction_mean의 원래 float32 mean은 정확 재현하고 float64 oracle를 별도 기록했습니다. 최대 차이 7.11e-09, 이 필드만 절대허용치2×10⁻⁷입니다. 다른 점추정과 AUC는 기준을 바꾸지 않았습니다.
CI는 expanded float64 row와 searchsorted tie AUC로 독립 재계산했습니다. 수치 구간 99행, 미계산 0행, 최대 차이 5.55e-17입니다.

## Unseen CI와 해석 한계

R13 unseen CI는 sensor×recording_week_start를 명시 연결한 새 결과입니다. 주간 cluster 사이가 자동으로24시간 이상 떨어지거나 독립이라는 뜻은 아닙니다. 실제 export-clock support gap과24시간 미만인 pair 수를 JSON에 기록했고 CI를 수정하지 않았습니다. 원 R12 null CI도 그대로입니다.
같은 참조와 이미 노출된 개발 패널을 사용했습니다. 새 blind OOF, 공식 LB 예측, 인과 원인의 완전한 증명이 아닙니다. Gate 반복 사용과 다중 비교의 낙관은 남으며 presence Brier/AUC와 에너지 RMSE를 구분해야 합니다.

[전체 수치·군집 gap·모델 SHA](R13_EVALUATION_INDEPENDENT_QA.json)

구체적으로 두 R10 unseen 패널은 각각20개 sensor×week 군집이며, 같은 sensor의 서로 다른 week 군집 사이 실제15분 window support gap의 최솟값은900초(15분)였습니다. 24시간 미만인 군집 pair는 각각12개입니다. 이는 TRAIN/heldout 역할 사이의24시간 purge가 깨졌다는 뜻이 아니라, bootstrap week-block 사이의 통계적 독립이 자동 보장되지 않는다는 한계입니다. 원 CI와 모델 선택은 변경하지 않았습니다.
