# R12 평가 독립 검산

상태: PASS. 저장된 17개 패널·441개 모델별 행을 독립 재계산했습니다. Gate는 25모델, 나머지 16패널은 각각26모델입니다.
RMSE·MAE·bias·평균·RMS·share/score AUC·target-tail SSE·site macro를 7497회 검산했습니다. 최대 차이 0.
직접 positive/negative 쌍과 tie 0.5를 세는 AUC oracle을 썼습니다. CI CSV 99행 중 87행의 실제 구간을 별도 weighted-cluster 알고리즘으로 검산했고, 12행은 원래 구간이 없어 null 상태만 대조했습니다.
원 평가 bootstrap RMSE는 float32에서 차·제곱·평균·제곱근을 계산했고 독립 weighted oracle는 float64였습니다. dtype과 percentile 보간의 최대 구간 차이는 1.29e-08입니다. RMSE CI 끝점에만 절대허용치2×10⁻⁷을 적용하고, 모든 끝점은 별도로 원 float32 expanded 계산과1×10⁻¹² 이내 일치도 요구했습니다. 실제 producer 재현 차이는 0입니다. 점추정과 AUC 허용치는 바꾸지 않았습니다.
선택 정책은 gate의 사전 고정 8개 50:50 recipe 중 첫 argmin(L1H1P1)과 일치합니다. 정책·17개 summary·체크포인트·PCA·데이터·예측 파일 SHA도 확인했습니다.
CNN 9개는100epochs·Adam·dropout0.1·엄격한 첫 최저 epoch·26,953개 저장 tensor 원소를 확인했습니다. Tree8개는466/673특징·TRAIN12,000·1500/ES40·TRAIN-only basis binding을 확인했습니다.

## 검증하지 못한 항목

분류기 AUC 136개와 contrast saturation 153개는 원배열이 NPZ에 저장되지 않아 exact 재계산하지 못했습니다. 값의 범위와 원천 hash만 확인했습니다. 이 수치를 검증했다고 주장하지 않습니다.
R10 unseen 두 패널의 CI 12행은 원 evaluator가 cluster ID를 연결하지 않아 미계산 상태입니다. 원천 metadata에는 고유 base_window_index와 sensor/week 정보가 있으므로 원천 ID 자체가 없다는 뜻이 아닙니다. 새 group 정의나 구간을 추가하지 않았고 원 CSV·정책·모델도 수정하지 않았습니다.
q/x/w가 평가 NPZ에 없으므로 cell-mask pooling을 다시 실행하지 않았습니다. 모델 추론·학습·Test 접근도 하지 않았습니다.

## 해석 한계

패널과 참조는 개발 과정에서 이미 노출됐습니다. 독립 OOF나 blind Test가 아닙니다. CI는 gate 선택 낙관·여러 비교·참조 공유 의존성을 제거하지 않습니다. 점수 정의 변경과 실제 학습 개선을 구분해야 하며 공식 LB나 GPU 성능 향상을 보장하지 않습니다.

[전체 검산·checkpoint SHA](R12_EVALUATION_INDEPENDENT_QA_v2.json)
