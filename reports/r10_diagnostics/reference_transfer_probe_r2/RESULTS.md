# R10 참조 A학습 → A/B 비교 CPU probe

메타데이터로 사전에 고정한 A211/B197 분할을 한 번만 사용했습니다. 생산 R9 코드·데이터·모델은 바꾸지 않았습니다.
TRAIN 16,000개 및 early stopping 1,000개는 A만 사용했습니다. 모델을 저장·동결한 뒤, 동일한 완전관측 배경 1,000개에 A/B를 각각 추가해 비교했습니다.

|패널|N|RMSE|known-addition AUC|평균 정답|평균 예측|
|---|---:|---:|---:|---:|---:|
|early_stopping_A|1000|0.057587|0.901006|0.113752|0.122721|
|A|1000|0.061923|0.923206|0.111656|0.119946|
|B|1000|0.069529|0.911800|0.111313|0.114658|

B−A RMSE 차이: +0.007606. AUC 차이: -0.011406.

A/B는 hull·녹음일·UTC 구간·파일 bytes가 겹치지 않습니다. 다만 같은 제공자/수신기 코퍼스이며, 길이와 약한 클래스 분포도 다릅니다. 이 차이를 특정 hull 암기 또는 실제 대회 성능의 단일 원인으로 단정할 수 없습니다.
동일한 배경·no-source 20%·family·SNR·EQ 난수를 썼지만 원참조 길이가 달라 시간 점유율과 정답 share 분포가 같지는 않습니다. 이 분포를 결과에 맞춰 보정하지 않았습니다.
모든 표적은 합성 시 실제로 추가한 전체 녹음의 에너지 비율입니다. 자연 상태 선박의 절대 정답 라벨이 아닙니다.
CPU 실행 71.9초. 회귀 best boosting round 1153, 분류 best round 85. 100-epoch 신경망 정책은 tree에 적용되지 않습니다.

[수치·출처·설정 전체](RESULTS.json) / [사전 계획](predeclared_plan.json) / [참조 분포](reference_metadata_audit.json) / [배경 분리](background_metadata_audit.json)

## 독립 검산과 해석

독립 재계산한 RMSE·MAE·bias·AUC는 저장값과 최대 1.12×10⁻¹⁶ 차이로 일치했습니다. 모델·분할·원천·코드 SHA, A만 학습한 reference index, A/B별 배경 순서와 family/SNR/EQ, zero 200건의 입력 일치도 검증했습니다.

이 한 번의 소형 tree 대조에서는 미학습 참조 B의 RMSE가 0.007606(약 12.3%) 상승했으며, 두 배 수준의 붕괴는 관측되지 않았습니다. 신경망이나 실제 대회 데이터에서도 참조 일반화 문제가 없다고 증명하는 결과는 아닙니다.

Early stopping과 비교는 원래 selection role에서 서로 다른 창을 골랐지만 19개 background group을 공유합니다. 둘 다 TRAIN과는 group이 분리되고 최소 24시간 support purge를 통과했습니다. 따라서 비교 패널을 새로운 독립 최종 holdout이라고 부르지 않습니다. A/B 대조에서는 배경 자체가 완전히 같으므로 이 공유는 두 참조의 차이를 비교하는 공통 조건입니다.

[독립 검증과 모든 산출물 SHA](INDEPENDENT_VERIFICATION.json)
