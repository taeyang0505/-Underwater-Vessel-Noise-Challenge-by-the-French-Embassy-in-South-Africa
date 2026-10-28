> 사용자 요청으로 외부 자문을 중단함. 이 초안은 전송되지 않았으며 추가 전송도 승인되지 않음.

# Gemini Argon 자문 전달 초안

전달 대상: 사용자가 Whale에 열어 둔 Gemini Argon 대화.
전달 범위: 아래의 집계된 RMSE, 익명 범주별 오류, 증강/손실 설정. 원본 데이터, 파일, 소스 코드, 계정/서버 접속 정보는 포함하지 않음.

---

수중 선박 소음의 추가 에너지 비율 회귀에서 native 참조에 대한 일반화가 실패합니다. Source-disjoint Nested 5-fold, 24시간 purge를 유지하고 outer/Test로 설정을 선택하지 않습니다. 다음은 첫 fold train/inner 결과입니다.

Train 53,244개, inner 3,000개. 상수 baseline RMSE .317663. 최적 inner RMSE: LightGBM physics .283701, retrieval .283760, TabM .289361, CNN .321155, FDY .326165, iTransformer .344797, frozen TabICLv2 .344146, frozen Kumo .343103. Inner에서 정한 앙상블은 .281359. CNN best epoch6에서 train .076204 / inner .321155입니다. Adam, dropout .1, early stopping 7이며 energy-weighted cell mask와 window/presence loss를 사용합니다. 라벨 누락·에너지 가중합·최적 체크포인트 복원 오류는 발견하지 못했습니다.

Native 포함 inner 44.8%가 CNN 제곱오차의 93.5%를 차지합니다. Native-only 997개 정답 평균 .484155 / 예측 평균 .079616 / RMSE .480476. Single-short 973개 RMSE .134741, multi-short 92개 .100480, mixed 347개 .412794입니다. Native-only train은 14,809개, 평균 정답 .470819로 양성 감독이 있습니다. Native 독립 원천은 train 3그룹, inner 1그룹이라 첫 fold에서의 한계가 있습니다. Native에는 선박 주석이 있지만 측정된 주변 소음도 포함되어 vessel-only 에너지는 인증되지 않았습니다. Inner native는 train보다 저주파 에너지 비중이 크고 배경과 스펙트럼이 더 비슷합니다.

현재 train만 support cap, smooth reference EQ ±4dB, 배경 blending .25, minute SNR 20dB 제한을 사용합니다. Inner는 과거 NPZ 그대로입니다. 짧은 support/관측 coverage 분포 차이도 있으나 짧은 참조보다 native 오차가 지배적입니다.

수정 중인 두 비교 실험: 매 epoch 저장된 관측 E와 q로 A=qE, B=(1-q)E를 만들고, (A) common smooth response ±4dB를 A/B에 같이 적용해 q를 유지하고 y를 재계산, (B) reference smooth tilt/curve 최대±12dB를 A에 적용해 원래 share를 목표로 alpha 재해결 후 support/SNR cap과 함께 E,q,y를 재계산합니다. 모든 train 행을 순회하고 100max epoch/ES7/Adam, 고정 inner로 평가합니다. 개선된 정책을 전체 모델에 적용할 계획입니다.

질문: 위 수치와 맞는 native 실패 가설과 지금 가장 효과적인 구현 수정/통제 실험 3개를 구체적으로 제안해 주세요. 지속적이고 배경과 비슷한 추가 성분의 비율 식별가능성, absolute/gain 특징 암기, domain augmentation의 타당성과 실패 가능성, 셀 loss 편향을 검토해 주세요. 검증셋을 바꾸거나 held-native를 train으로 이동하는 방법은 금지입니다. 큰 모델/epoch 증가라는 일반론 대신 각 가설의 반증 가능한 예상 관측값을 써 주세요. 성능 수치를 보장하지 마세요.
