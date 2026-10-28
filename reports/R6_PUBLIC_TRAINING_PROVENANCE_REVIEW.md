# 공개 원본 데이터의 출처와 합성에 사용할 수 있는 단서

확인일: 2026-10-26. 공개 학습자료의 문서·metadata와 연구 원문을 대조했다. 이번 조사에서 데이터 생성 설정, 모델 가중치, 평가 대상은 변경하지 않았다.

## 먼저 구별해야 할 두 문제

**배경 원천은 이미 대회가 제공한 바로 그 South African 실측자료를 사용하고 있다.** 비슷한 외부 배경을 찾아 교체하는 문제가 아니다. 공식 공개분은 10개 site의 무라벨 대역 요약 1,715,779행이고, 비공개 평가 배경의 기간은 공개 학습분에서 제외되어 있다. Test는 실측 배경에 녹음된 passage reference를 제어하여 더한 문제다. [공식 데이터 설명](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/data)

그러므로 남은 불확실성은 **어느 공개 배경을 얼마나 사용했는지**, **추가 참조의 장비·주파수·길이·통과 궤적·세기**, **요약·배치·마스킹의 순서**, **검증에 없는 환경으로의 전달**이다. 같은 background archive를 사용한다고 이 조건들이 자동으로 같아지지는 않는다. 공식 starter가 공개된다고 비공개 생성기의 모든 설정이 공개되었다는 뜻도 아니다. 이전의 “Test 추가 음원이 VTUAD라는 확정”은 현재 공식 문서로 입증되지 않으며, VTUAD는 우리 합성의 선택이다.

## 실제로 무엇을 찾았나

| 확인 항목 | 확인 결과 | 확실성 |
|---|---|---|
| 대회 자료 제공 기관 | SAEON, SANBI, SA Underwater Radiated Noise Consortium(SA URN) | 공식 소개에 명시 |
| 공개 학습 패키지 | site별 Parquet, station·recorder metadata, data dictionary | 공식 설명과 로컬 파일 일치 |
| 로컬 metadata 규모 | 10개 site, recorder coverage 17행, 합계 1,715,779행 | 제공 CSV를 직접 집계. 17행을 물리 녹음기 17대라고 해석하지 않음 |
| 공개 export 기간 | 2015-04-21~2025-12-06 | recorder metadata 기준. UTC 여부 미검증 |
| 같은 전체 자료의 기존 외부 dataset DOI·다운로드 | 확인하지 못함 | 검색 실패이지, 공개 자료가 없다는 증명은 아님 |
| 관련성이 높은 부분 원천 연구 | Schoeman·Erbe·Plön, 2022, St Francis/Algoa Bay soundscape 연구 | 관련 문헌 확인. 동일 file/export 연결은 미확인 |

제공 기관은 [공식 소개](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge)에 명시되어 있다. 로컬 근거는 [README](../data/extracted/README.md), [data dictionary](../data/extracted/DATA_DICTIONARY.md), [station metadata](../data/extracted/stations.csv), [recorder metadata](../data/extracted/recorders.csv)다.

SAEON catalogue·Observations Database, MIMS, SANBI 관련 페이지와 관련 연구/DOI를 조회했다. 이름이 비슷한 남아공 육상 음향자료와 아남극 해양자료도 실제로는 다른 dataset이었다. 검색 경로·접근 제한·자료별 licence는 [조사 catalogue](R6_PUBLIC_TRAINING_PROVENANCE_CATALOG.md)와 [기계 판독 기록](R6_PUBLIC_TRAINING_PROVENANCE_CATALOG.json)에 보존했다. 논문 DOI를 dataset DOI로 바꾸어 부르지 않는다.

## St Francis와 연결될 가능성이 높은 논문

**Underwater Chatter for the Win: A First Assessment of Underwater Soundscapes in Two Bays along the Eastern Cape Coast of South Africa** (2022), DOI **10.3390/jmse10060746**. [Publisher](https://www.mdpi.com/2077-1312/10/6/746), [연구 원문](https://www.researchgate.net/publication/360935137_Underwater_Chatter_for_the_Win_A_First_Assessment_of_Underwater_Soundscapes_in_Two_Bays_along_the_Eastern_Cape_Coast_of_South_Africa).

논문은 2015~2018년 St Francis/Algoa Bay에서 SM2M+·48kHz 녹음, 60초 spectral summary, 계획된 간헐 녹음과 자연 생물·바람·선박의 혼재를 설명한다. 로컬 St Francis metadata는 **SM2M·48kHz 표기, 2015-04-21~2018-02-28, 169,488행**이다. 장소·시기·기기 계열이 겹치므로 부분 lineage의 유력한 관련 문헌이다. 전체 공개 archive에는 2025년 자료도 있으므로 이 논문 하나가 전체 dataset의 원본이라는 결론은 성립하지 않는다.

원문 §2.2.1은 2015년 `15분/60분`, 2016~2018년 `5분/20분`을 쓰면서 duty cycle을 20%라고 적는다. 앞 두 비율의 산술값은 모두 **25%**다. 이 내부 불일치를 그대로 기록하며 실제 관측률·특정 배포의 schedule로 확정하지 않는다. §2.3.1의 60초/1Hz LTSA도 대회의 정확한 export 코드가 같다는 증거는 아니다. Data Availability는 보충 그림·표와 요청 가능한 모델링 shapefile을 설명하며, 공개 원시 녹음 archive를 연결하지 않는다.

## 합성·검증에 활용할 단서와 적용 상태

아래 권고는 문헌과 공개 TRAIN에 근거한 모델링 판단이다. 숨겨진 Test의 정답·site 구성·생성 설정을 알아냈다는 뜻이 아니다.

공개 TRAIN에서 만들어 둔 window metadata도 대조했다. St Francis의 **27,154개 clock window 중 23,782개(약 87.6%)는 관측 분이 5개**이고, 15개 관측 분인 창은 3,371개, 12개인 창은 1개였다. 이는 논문과 정확히 같은 녹음이라는 인증은 아니지만, 실측에 간헐 관측이 실제로 많다는 독립적인 단서다. **현재 R6의 St Francis는 교차 sensor 진단용으로 분리되어 TRAIN에는 없다.** 이 수량을 이번 R6 TRAIN 수 또는 Test site 비율로 해석하면 안 된다. [코드·metadata 점검](R6_PUBLIC_TRAINING_PROTOCOL_CODE_REVIEW.md)

| 단서 | 합성·검증에 주는 의미 | 현재 상태 / 남은 확인 |
|---|---|---|
| 녹음 schedule 때문에 생기는 빈 분 | 빈 분은 무음이 아니다. 5분 관측을 반복하여 15분의 실측처럼 만들면 안 된다. | R6 공통 준비 코드는 clock 격자에 실제 분만 놓고 NaN을 보존한다. 별도 code review 참조. |
| 여러 장비·배포·시기의 집합 | 전체를 동일 receiver·절대 calibration으로 가정할 수 없다. | 상대 채널·원천 분리 검증을 사용한다. band-relative 입력도 상대 주파수 응답에 완전 불변인 것은 아니다. |
| 자연 생물·바람·선박이 혼재 | 높은 고주파나 큰 LF bump가 추가 성분의 정답 자체는 아니다. 자연 배경 대조군이 필요하다. | 무추가 control과 TRAIN 내 LF transient hard-negative를 유지한다. `y=0`은 자연 선박 부재가 아니라 추가 성분 부재다. |
| 대역 요약의 생성 순서·단위 | PSD density의 대역 적분과 band energy를 혼동하면 주파수 기울기가 달라진다. log 공간 평균도 에너지 평균과 다르다. | 공개 dictionary는 export dB convention·절대 calibration 미검증을 명시한다. 새로운 문헌의 calibration을 자동 이식하지 않는다. |
| 관측 분 길이 dur와 실제 시간 support | 부분 분을 완전한 60초로 오해하거나 떨어진 시간을 붙이면 궤적을 왜곡한다. | dur/clock/QC 및 최소 관측 support의 구현을 별도 점검한다. 논문 schedule만으로 관측 시간을 채우지 않는다. |
| 참조 녹음 자체의 주변 소리 | 학습은 vessel-only 에너지 분리가 아니라 알려진 추가 reference 전체의 비율을 맞춘다. | 원래 알려진 두 성분과 관측 mask에서 정답을 계산한다. 참조의 자연 배경을 임의로 빼서 같은 정답이라고 부르지 않는다. |
| 주파수 coverage 모순 | 명목 sample rate와 상위 band의 불일치를 문헌의 장비 정보로 덮을 수 없다. | URN_8321의 12kHz 표기와 상위 band 모순은 미해결로 남긴다. safe-band 진단과 해당 recorder 별도 slice가 필요하다. |

## 실제 Test와 비슷해졌는지에 대한 결론

이번 조사는 **원본 배경의 provenance와 관측 프로토콜을 좁힌 것**이다. 비공개 reference·SNR prior·연산 순서를 복원하거나 공식 RMSE가 개선됐다고 검증한 작업은 아니다. 새로운 모델의 자체 INNER 최소값을 실제 LB 성능으로 바꾸어 말하지 않는다. AV·KS·상수 RMSE에 대한 보장되지 않는 해석은 [별도 검토](R6_TEST_GAP_CLAIMS_REVIEW.md)에 정리했다.

현재 100epoch 실험은 GPU 없이 기존의 작은 CNN 5개 비교군으로 완료했다. 새 provenance 정보를 보고 중간 설정을 바꾸지 않았다. [실제 학습 결과와 전체 곡선](R6_FIXED100_TRAINING_REPORT.md)을 별도로 제공한다. 대형 백본의 본학습 재개는 이번 조사에서 실행하지 않았다.
