# R11-1 full-support × HF/경계 패널 독립 QA

**PASS.** 기존 진단 산출물과 원천의 selection 메타데이터·해당 원천 행만 읽어 검산했습니다. 추가 모델 추론·학습·Test 접근·미사용 report 역할 접근은 **0회**입니다. TRAIN은 분리 확인용 **메타데이터만** 읽었으며 TRAIN 수치 배열은 읽지 않았습니다. 원 R9 소스·모델·데이터·정책·CSV를 수정하지 않았습니다.

- [독립 QA JSON](full_support_independent_qa_r1.json)
- [독립 QA 사전 검산 계획](full_support_independent_qa_r1_PLAN.json)
- 실행 코드: `ops/r11_full_support_independent_qa.py`
- 검토 대상 코드: `ops/r11_full_support_hf_audit.py`의 `mixture`, `long_reference`, `run`, `crossing`
- 원 결과: [RESULTS](full_support_hf_r1/RESULTS.json), [35개 대조 × 6개 출력 전체 표](full_support_hf_r1/crossing_metrics.csv), [추론 전 패널 동결 SHA](full_support_hf_r1/FROZEN_BEFORE_PREDICTIONS.json)

## 1. 무결성과 사전 고정

원 source/model/policy **102개** SHA, 추론 전 snapshot의 **29개** 패널·레시피·계획 SHA가 현재 파일과 일치합니다. 독립 검산에서 원천 bank·메타데이터·진단 산출물을 포함한 **156개 입력**의 전후 SHA를 확인했습니다. 원 계획의 script SHA와 현재 코드는 일치합니다. 코드의 실행 순서도 계획 저장 → 패널 생성·동결 → 원 모델 pin 저장 → 모델 예측 → 모든 crossing 출력 순서입니다. 사후 성능에 따라 패널을 버리거나 임계값 `.55` 이상만 고른 기록은 없습니다. 모든 **7 양성 모드 × 5 음성 패널 × 6 출력 = 210행**이 보존돼 있습니다.

총 저장 질의는 **5,812개**입니다. 양성 7×500=3,500개, 음성 500+312+500+500+500=2,312개입니다. 이는 고유·독립 원천 5,812개라는 의미가 아닙니다. 같은 배경·참조가 모드 사이에 공유되고 음성 패널끼리의 교집합도 허용됩니다.

## 2. 물리식 독립 검산

생성기의 함수를 호출하지 않고 source 2초 band-power를 완전한 30개 단위로 평균하여 native minute를 다시 만들었습니다. 저장 순서의 사전 고정 행 `0,1,2,3,4,5,499`를 각 양성 모드에서 독립 복원했습니다. 총 **49개**에서 rounded db·float32 cell_share·float32 global share가 저장 배열과 **정확히 일치**했습니다. R9 원형의 inside/short/edge 배치와 EQ seed도 별도 구현으로 복원했습니다.

관측 mask를 O라 할 때 선형 에너지로 `B=10^((db−peak)/10)`, `A=alpha·O·R`, `E=B+A`, `q=A/E`, `y=sum(A)/sum(E)`입니다. 장지원 모드의 alpha는 목표 u에 대해 `alpha=u·sum(B)/((1−u)·sum(O·R))`입니다. 원래 자연 선박의 분리 정답이나 외부 참조 안의 순수 선박 에너지를 뜻하지 않습니다.

- 독립 원성분 식의 최대 `|sum(E/sum(E)·q)−y|`: **5.551115123×10⁻¹⁷**.
- 저장 양성 **3,500개 전체**의 rounded-weight pooling과 원정밀 label 차이 최대: **0.001944615**. 0.1dB 반올림의 사전 수치 bound 약 **0.005757** 이내입니다. 원정밀 label을 rounded energy로 재정의하지 않았습니다.
- 기본 원형과 full12/full13/full15/stationary/native-long 모드에서 per-row 저장 y가 정확히 같습니다. 평균 **0.141006872**, RMS **0.213753143**입니다. `full_smooth15_share030`만 사전 선언된 별도 고정 u=.30 개입입니다.
- 모든 음성의 추가 성분 label은 정확히 **A=0/y=0**입니다. 이는 자연 선박 부재의 정답이 아닙니다. 원천에 자연 선박이나 경계 후보가 포함돼도 controlled addition 정답은 0입니다.
- 원 `r9_original`의 스케일은 background **median-minute**와 full-reference **peak-minute**입니다. 나머지 장지원 모드는 그 원형의 per-row y를 보존하도록 alpha를 다시 풀었습니다. 따라서 단순 mean-scale로 목표값을 바꾼 비교가 아닙니다.

## 3. 지원 길이와 실제 원천

기본 양성 배경은 전부 실제 **15분 관측**이고, 음성 배경의 합집합과 원천 window ID 교집합이 **0**입니다. selection과 TRAIN도 원천 window/group 교집합 0, 최소 support-gap **24시간**을 확인했습니다. 음성 패널끼리의 중복은 허용되며, 특히 edge-restored와 edge-QC는 의도적으로 같은 500개 원천입니다.

장지원 양성의 **추가 성분**이 활성인 분은 full12=12, full13=13, full15=15, stationary=15, native-long=15입니다. 전체 **관측분**은 모든 양성 모드에서 15이므로 추가 성분 지원과 혼동하지 않아야 합니다. full12에는 순수 B분이 3개, full13에는 2개가 남습니다. 원형은 추가 활성분 1–15가 섞여 있습니다.

현 509개 참조 manifest에서 실제 완전 native minute가 15개 이상인 참조는 **7개**입니다. reference row index는 `1,21,94,208,360,430,463`, 길이는 각각 **16,23,18,20,17,23,19분**입니다. 과거 제한된 참조 풀의 3개와 구분해야 합니다. native-long500행은 이 7개를 순환하여 쓰므로 **500개의 독립 장시간 녹음**이 아닙니다.

full-smooth는 기존 minute-power를 선형 재표본화하고 완만한 `.6+.4sin²` envelope를 곱한 **진단용 통계 궤적**입니다. 새로운 실측 장시간 WAV가 아니며, 원래 R9 production law와 동일하다는 주장도 아닙니다. stationary는 native 평균 스펙트럼을 15분 유지한 이상화 대조군입니다. native-long은 실제 긴 참조에서 연속 15분을 잘라 사용하지만, 짧은 원형과 **참조 identity도 달라져** 지원 길이 하나만의 효과로 분리할 수 없습니다.

## 4. 경계 복원·HF 패널의 정확한 의미

HF 정의는 이번 코드에서 band index **18..24**(약 **1.995–7.943kHz**)의 선형 합에 대해 시간 max−median(dB)입니다. 예전 R10의 index20..24(3.162–7.943kHz) 또는 safe1–5kHz 지표와 같은 지표가 아닙니다. Nobs≥13 selection 원천 **6,920개**에서 고정 q90/q50를 계산하여 HF-high와 quiet를 선택했으며 모델 오류나 Test로 고르지 않았습니다.

| 패널 | 실제 생성 후 관측분 histogram |
|---|---|
| HF-high | Nobs13:18, 14:64, 15:418 |
| 실제 source St Francis dense | Nobs15:312 |
| edge-restored | **Nobs13:132, 15:368** |
| paired edge-QC | **Nobs12:132, 14:368** |
| HF-quiet | Nobs13:5, 14:98, 15:397 |

edge-restored manifest의 `qc_observed_minutes`에는 복원 **이전** 12/14가 남습니다. 이를 복원 후 실제 Nobs라고 해석하면 잘못입니다. 수치 배열에서 13/15임을 별도로 검산했습니다. 원 코드가 복원한 것은 `boundary_mask_bits`에 표시된 **첫 분/마지막 분**뿐입니다. 내부 flagged minute 전체를 복원하거나 원래 NaN을 채운 것이 아닙니다. 원 bank와 비교해 이 범위 외 값·mask 변경이 없음을 500행 전수 확인했습니다.

양성 배경의 site는 Richards Bay329/St Croix92/Durban50/Bird29이고 **St Francis는 0개**입니다. SF 음성312개는 기존 source metadata의 실제 St Francis이며 전부15분 관측입니다. 따라서 길이를 맞춘 대조이지만 **site-matched 대조는 아닙니다**. HF-high도 Richards Bay429/500 등 site구성이 치우쳐 있어 음성 종류 간 차이를 HF 또는 지원 길이만의 효과로 확정하면 안 됩니다.

## 5. 저장 성능 산식의 독립 검산

모든 **210행**의 RMSE·class-balanced RMSE 및 **420개 AUC**를 저장 예측으로 다시 계산했습니다. AUC는 라이브러리 결과를 복사하지 않고 양성/음성 전 쌍에서 `P(pos>neg)+.5P(tie)`를 직접 계산했습니다. `AUC+P(neg>pos)+.5P(tie)=1`도 전부 확인했습니다. 최대 수치 차이는 **2.220446049×10⁻¹⁶**입니다. 고정 all5 가중치 `[.1,.1,.1,.1,.6]`와 저장 member 예측의 재조합도 전 패널에서 정확히 일치합니다.

| SF dense 음성과 비교하는 all5 양성 모드 | AUC_score | AUC_share | 합친 RMSE |
|---|---:|---:|---:|
| R9 original | 0.897949 | 0.709231 | 0.059904 |
| full-smooth12 | 0.737494 | 0.693692 | 0.075888 |
| full-smooth13 | 0.709949 | 0.679564 | 0.078373 |
| full-smooth15 | **0.467513** | 0.463295 | 0.116927 |
| stationary15 control | 0.207859 | 0.342987 | 0.134412 |
| native-long15 | **0.520109** | 0.536763 | 0.095733 |
| full-smooth15, fixed .30 | 0.715237 | 0.768917 | 0.164967 |

이는 이 고정 challenge panel에서 실제로 나타난 국소 취약성을 보여 줍니다. 숨은 Test의 생성법·유병률·실제 선박 label이나 공식 성능 하락의 원인을 확정하지 않습니다. support·궤적·참조 identity·site 구성·음성의 자연 음원이 함께 영향을 줄 수 있습니다. 특히 fixed .30은 target-level 개입이므로 나머지 모드와 같은 y의 shape 효과라고 읽지 않아야 합니다. 210행은 공유 원천을 쓰므로 독립 210회 검증이 아니며, 음성 크기가312/500으로 달라 RMSE 패널 간 비교에는 class-balanced RMSE도 함께 봐야 합니다.
