# R10 시간 순서·정밀도 진단

실행 완료: 73.68초. 동일한 Fold0 selection1000, 세 사전 고정 seed, NN각.1/트리.6 가중치를 유지했습니다.

|모델|원본 CPU RMSE|shuffle 평균 RMSE|RMSE 변화 범위|원본 AUC|shuffle 평균 AUC|
|---|---:|---:|---:|---:|---:|
|bg_cnn|0.090372|0.100601|+0.006093 .. +0.012870|0.930090|0.919503|
|tfgrid_fdy_conformer|0.079414|0.089101|+0.007481 .. +0.011006|0.942159|0.938174|
|fdy|0.081086|0.115412|+0.030571 .. +0.038937|0.932638|0.900091|
|itransformer|0.096515|0.112648|+0.012247 .. +0.019071|0.907748|0.898289|
|lgbm_physics|0.057347|0.065852|+0.007754 .. +0.008887|0.942705|0.945030|
|frozen_all5|0.061953|0.072781|+0.009734 .. +0.011528|0.955967|0.952739|

원본 CPU FP32와 저장 GPU BF16은 별도로 비교했습니다. ID·target 순서는 일치하며 수치 차이를 CSV 오류로 단정하지 않습니다.
세 permutation 모두 원 관측 NaN 위치·band marginal·에너지 합·raw 정답 비율을 유지합니다. db/rawdb/q에 같은 행 순서를 적용하고 모델 feature와w를 다시 계산했습니다. 원본은 변경하지 않았습니다.
점수가 유지돼도 이 설정의 시간 순서 변화에 민감하지 않다는 의미에 한정합니다. 장치 암기를 입증하지 않습니다. 반대로 악화는 비현실적인 abrupt transition에 대한 민감도일 수 있어 선박의 시간 문맥을 이해했다는 증명도 아닙니다.
Selection은 이미 checkpoint 선택에 사용됐고509 reference를 공유합니다. Test·미사용report·GPU·SSH·외부 전송·학습·가중치 재조정은 없었습니다.

[전체 수치·정밀도·각seed·SHA](RESULTS.json)

## CPU FP32와 저장 GPU BF16 비교

동일 checkpoint와 selection1000을 서로 다른 장치·정밀도·커널·Torch버전에서 비교한 값입니다. GPU참고는2.8.0+cu128,CPU는2.14.1입니다. 따라서 차이를 BF16만의 인과효과로 분리하거나, 차이가 있다는 이유만으로 CSV mapping 오류로 단정하지 않습니다.

|모델|CPU RMSE|저장 GPU RMSE|최대share차|최대score차|
|---|---:|---:|---:|---:|
|bg_cnn|0.090372316|0.090360932|0.001732528|0.003717989|
|tfgrid_fdy_conformer|0.079414114|0.079415079|0.003968716|0.017613128|
|fdy|0.081085812|0.081033664|0.001851618|0.006718084|
|itransformer|0.096514643|0.096523305|0.001985133|0.004701704|

## 해석 범위

시간 shuffle 후 feature와w를 다시 계산했으므로 결과는 network만이 아니라 전체 고정 전처리+모델 파이프라인의 시간 순서 민감도입니다. 특히 adjacent delta/co-rise 채널이 변합니다. LGBM에도 변화가 생기므로 RNN 구조 자체의 이해 능력으로 바로 해석할 수 없습니다.

독립 저장-output 검증은96개 RMSE/MAE/bias/AUC 계산을 다시 확인했고 최대 차이는1.11e-16입니다. 모든4,000 scenario-row의 ID 순서,3개 permutation bijection과NaN/band marginal, 동결 source/model/policy SHA가 일치했습니다. [독립 검증](INDEPENDENT_VERIFICATION.json),[정밀도 환경](PRECISION_ENVIRONMENT.json).
