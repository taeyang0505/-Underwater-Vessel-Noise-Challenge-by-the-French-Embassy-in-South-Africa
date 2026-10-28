# R15 TRAIN 전용 채널 민감도 프로토타입

이 결과는 고정된 물리 가설 4개를 한 번 비교한 진단용 프로토타입이다. 대회용으로 보정되거나 검증된 학습 자료가 아니다. Test 파일·기존 Test EDA·분포 비교표를 이 스크립트에서 읽지 않았고, Test와의 거리·근접성 통과 기준·최적 후보 선택을 계산하지 않았다. 이미 이전 Test 기술 통계를 보았으므로 사전 선언은 이번 생성 전 고정이라는 뜻이며 진정한 blind 실험을 뜻하지 않는다. 공식 규정상 Test 분포에 맞춘 생성기 조정은 하지 않는다.

## 원천·정답·변경 범위

기존 R9Sources의 Fold0 TRAIN 배경 4,000개를 무복원 추출했다. 기존 24시간 지원 구간 purge와 분리된 원천 ID를 검산했으며 heldout 자료는 메타데이터만 읽었다. VTUAD vessel-archive의 검증된 native minute 파워 509개를 모두 순환 사용했고, 전용 ambient16개는 사용하지 않았다. 참조는 전체 측정 녹음이며 자연 선박 유무나 고립 선박 에너지 정답이 아니다.

양성3,200개는 실제 연속1–6분을 native broadband peak를 포함하도록 잘라 사용했다. 길이 변환·반복·bell·강제 fade·15분 장시간 성분·highshare 강제·PCA/anchor 특징을 추가하지 않았다. 짧은 원천은 실제 사용 가능한 완전 분만 썼다. 원래 R9와 달리 알려진 TRAIN 관측 마스크와 겹치는 합법적인 창 내 시작점을 한 번 열거해 추출했다. 이는 양성의 관측 지원을 보장하기 위한 조건부 배치이며 mask-independent 원본 R9의 재현이라고 주장하지 않는다.

## 고정 후보와 물리식

같은 배경·원천·native crop·배치·SNR·random smooth EQ±6dB를 4개 후보에 공유했다. 추가 채널은 reference에만 적용한다. 200→500Hz log-frequency raised cosine shelf를 사용하며, 500Hz 이상 +6/+12dB, 마지막 후보는 200Hz 이하 −6dB를 더한다. 이 값은 감도 확인용 가설이며 Test의 숨은 채널 응답 추정치가 아니다. 총 채널 범위는 공통 ±6dB와 추가 곡선의 합이다.

각 후보마다 EQ 이후 전체 native trajectory의 분 broadband peak로 alpha=10^(SNR/10)×Bmedian/Rpeak를 다시 풀었다. SNR은 공통 U[−10,+10]dB다. crop/O 이전의 전체 궤적 peak를 썼으므로 tail boost 재정규화는 없다. y=ΣOA/ΣOE, q=OA/E를 raw continuous 성분에서 계산했고 특정 share 범위를 만들지 않았다.

파워는 창의 공통 dB offset을 뺀 상대 단위다. 관측하지 못한 실제 B는 알 수 없으며 B_observed=0은 관측 적분 규약이다. A_full은 결측 분에도 보존하고, A_observed만 마스크 적분에 쓴다. 각 NPZ는 unrounded dB, 0.1dB dB, float64 q/y/A/E를 저장한다. common_backgrounds.npz는 동일 B/O/원천 ID를 담는다.

## 전체 및 양성 정답·HF 변화

| 후보 | share 전체 p50 | share 양성 p50 | HF bump 전체 p50 | HF bump 양성 p50 | 최대 raw 항등식 오차 | 최대 rounded pooling gap |
|---|---:|---:|---:|---:|---:|---:|
| control_eq6 | 0.044104 | 0.079190 | 8.5861 | 11.2663 | 4.44e-16 | 0.001490 |
| mfhf_shelf6 | 0.044397 | 0.080021 | 9.6873 | 12.5515 | 4.44e-16 | 0.001529 |
| mfhf_shelf12 | 0.044204 | 0.079186 | 10.3324 | 13.1874 | 5.55e-16 | 0.001579 |
| mfhf_shelf12_lfminus6 | 0.043917 | 0.079026 | 10.7006 | 13.5168 | 4.44e-16 | 0.001670 |

모든 후보에서 no-source800개/양성3200개이며 zero 입력은 후보 사이 bitwise 동일하다. 수치 변화는 저장한 TRAIN 물리 가설의 효과이며 실대회 정답 분포·일반화 성능·validation gap 해결을 증명하지 않는다. 모델을 학습하거나 추론하지 않았고 원래 생산 코드·자료·정책·학습 상태는 수정하지 않았다.

[사전 선언](plan.json) · [원천 분리](SOURCE_SEPARATION.json) · [25대역 통계](band25_TRAIN_sensitivity.csv) · [정확한 재현 recipe](recipes.json)
