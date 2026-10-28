# R15 입력·저장 예측 기술통계 감사

결과: 저장 예측57개 열과 기존 EDA 캐시1,500개 ID를 일대일 결합했다. 새 모델 추론·학습·보정·제출 파일 생성은0회다.

## 추정 사이트와 정의

probe_site는 기존TRAIN의 분별25대역 dB를 창 내 시간중앙값으로 요약한 KNN25 도메인 프로브 결과다. 기존 group5fold는 site×ISOweek 기준 보조검증이며 Test의 실제 지점 정답은 아니다.

실제 캐시의 추정 건수: {'07 St Francis': 487, '01 Richards Bay': 423, '13 Saldanha Bay': 284, '05 St Croix': 236, '02 Durban': 32, '04 Bird Island': 16, '09 Fish Hoek': 15, '10 Sea Point': 7}. 첨부의 St Francis489개와 달리 현재 캐시는487개다. 보정하거나 재추정하지 않았다.

bump_high_db는3162.3–7943.3Hz 다섯 대역의 **선형 에너지합을 dB로 바꾼 시간곡선**의 최대−중앙값이다. bump_low_db는31.6–501.2Hz 열세 대역에 같은 공식을 쓴다. 이 통계는 밴드별 bump평균과 다르다. tilt_db는≤125.9Hz 평균 dB−≥1995.3Hz 평균 dB이며 작은 값이 상대적 중고역 우세를 뜻한다.

S_phys는 사전 고정한316.2–5011.9Hz 각 대역의 시간최대−중앙값을 대역평균한 dB 기술량이다. 정답이나 확률이 아니며 static band-offset 합성 수치 검산 오차는3.552713678800501e-15다.

## 파일

- [전수 ID 결합표](joined1500.csv)
- [모델별 분포](model_distribution.csv)
- [추정 유형별 분포](site_model_stats.csv)
- [특징–예측 Pearson/Spearman](main_feature_prediction_correlations.csv)
- [모든57개 열의 특징 상관](feature_prediction_correlations.csv)
- [모델 간 Pearson](model_correlations_pearson.csv)
- [모델 간 Spearman](model_correlations_spearman.csv)
- [낮은 시간변화·정적 proxy 그룹](lowtemporal_highstatic_groups.csv)
- [절대차 상위50](top50_absdelta.csv)
- [양의 차이 상위50](top50_positive_delta.csv)
- [선언 계획](plan.json)
- [입력·출력 SHA와 한계](RESULTS.json)

원 R14 PCA/LF-anchor의 Test별 맵 캐시 및 R14 raw classifier 확률은 저장되지 않아 이번 감사에서 관측값처럼 재구성하지 않았다. tilt·centroid는 다른 proxy다. 숫자가 큰 예측이 오탐인지, 낮은 예측이 정답인지 숨겨진 라벨 없이 판단할 수 없다. 상관관계·네 번의 제출 순서만으로 CNN/PCA/정적특징의 인과 원인을 확정하지 않는다.
