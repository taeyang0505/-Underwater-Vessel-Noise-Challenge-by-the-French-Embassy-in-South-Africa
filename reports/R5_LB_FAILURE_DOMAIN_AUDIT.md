# R5 Fold0 CNN 공개 평가 실패: TASK·단위·생성 도메인 감사

작성: 2026-10-26 KST. **공식 공개 문서와 동결된 로컬 소스의 읽기 전용 감사**입니다. 새 모델 학습·GPU/SSH·생성기/모델 수정·Test 수치/분포 분석·source matching·추가 제출은 하지 않았습니다.

## 우선 판단

사용자가 전달한 제출 `Vz5mjZEr`의 공개 점수는 `0.65199132`, 표시 RMSE는 약 `0.34800868`, 진단 AUC는 `0.539533333`입니다. 같은 모델의 자체 INNER RMSE 약 `0.090808`은 **공식 평가 성능의 신뢰할 만한 대리 지표가 되지 못했습니다.** 이것은 확인된 평가 전이 격차입니다. 공개 정답이 없어 원인별 기여량을 분해할 수는 없습니다.

현재 확인한 범위에서 **10/20 log 혼동, dB 직접 합산, 주파수 순서, 관측 셀의 분모·분자, band-power/PSD 혼동이라는 확정 오류는 발견하지 못했습니다.** 먼저 의심할 것은 자체 합성 연산자와 원천·배경 조건에 대한 의존성입니다. 다만 제출 모델 연결·전처리·후처리 자체의 재현 감사는 별도로 필요하며, 이 문서는 그 검증을 대신하지 않습니다.

이번 제출은 Fold0 CNN 단독입니다. 전체 R5 모델·다섯 fold 앙상블의 공식 결과가 아니며, 아직 작은 후보만 제출했다는 사실만으로 격차가 설명되지는 않습니다. 큰 사전학습 모델도 같은 생성 가정을 공유하면 같은 편향을 학습할 수 있습니다.

## 1. 공식 설명과 일치하는 부분

공식 평가 대상은 원래 배경의 모든 선박음을 분리하는 문제가 아니라, **통제해서 추가한 passage-reference의 에너지 비율**입니다. 참조에 녹음된 주변음도 추가 성분에 포함될 수 있고, 추가가 없는 배경에도 자연 선박음이 있을 수 있습니다. 두 합은 같은 관측 minute-band 셀을 사용하고, 각 창의 RMSE 기여는 같습니다. 순위는 share의 `1−RMSE`로 결정되며 보조 AUC는 순위 가중치가 없습니다. [공식 Info·Evaluation](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge)

| 감사 항목 | 공식 기준과 현재 구현 | 판정 |
|---|---|---|
| 격자 | 15분 × 25개 base-10 deci-decade band, 분 우선·저→고주파 순서 | 기존 schema 감사에서 일치. 이번에는 Test를 다시 열지 않음 |
| 물리량 | source-export dB이며 절대 압력 교정은 검증되지 않음 | calibrated Pa²라는 주장을 하지 않는 상대 에너지 표현이 적절 |
| dB 역변환 | `B = 10**((db−window_peak)/10)` | 기준 peak는 공통 배율이므로 비율에서 소거됨 |
| 혼합 | `E = B + alpha*R`, 그 뒤 `10*log10(E)+peak` | dB를 직접 더하지 않음 |
| 정답 | `y = sum(O*alpha*R)/sum(O*E)` | 공식 관측 셀 기반 선형 에너지 정의와 일치 |
| cell 감독·pooling | `q = alpha*R/E`, `w = O*E/sum(O*E)`, `share=sum(q*w)` | 알려진 합성 성분의 q이고 자연 선박 확률이 아님 |
| 결측 | 원 배경 NaN을 유지하고 q와 에너지 질량을 관측 밖에서 0으로 둠 | 결측을 침묵으로 라벨링하지 않음 |
| 주 손실 | 창별 share MSE를 예제 수로 평균·INNER RMSE로 선택 | 공식 RMSE의 제곱 손실과 정합. cell/presence는 보조 목적 |

공식 Data는 실제 배경에 녹음된 passage를 변형해 더한다는 수준을 설명합니다. 입력은 raw waveform이 아니며, 배경의 원래 값·시각·위치는 제공되지 않습니다. **공식 평가 배경 기간은 공개 TRAIN에서 제외됩니다.** 이는 공개 TRAIN 안의 holdout만으로 모든 평가 조건을 검증할 수 없다는 경계입니다. [공식 Data](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/data)

R4 전수 QA는 저장된 `q × energy → y`의 최대 오차 `3.1660e−7`을 기록했습니다. 이는 **우리 생성 데이터 내부의 산술적 정합성**이며, 비공개 생성기를 재현했다는 증거는 아닙니다. [전수 QA](R4_DATA_FINAL_VERIFICATION.json), [생성 코드](../pipeline/passage_sampler_r4.py)

## 2. 오디오 변환에서 확인한 실제 차이

공식 starter의 `minute_band_power`와 `pipeline/audio.py`는 공통으로 실제 sample rate, 단일 waveform 채널, 비중첩 2초 Hann periodogram, constant detrending, one-sided density PSD, `centre × 10**(±.05)` band 적분을 사용합니다. 우리 코드는 채널 0을 선택합니다. 적분에서 `sum(PSD bins) × df`를 하므로 반환값은 band mean-square power이며, PSD/Hz나 SEL을 다시 에너지로 착각한 구현이 아닙니다. 같은 기준의 완전한 1분을 비교한 과거 생성 waveform fixture의 최대 상대 오차는 `1.83e−7`이었습니다. 해당 receipt의 audio 코드·starter SHA는 현재와 같습니다. 이것은 실제 녹음기 교정의 동등성을 증명하지는 않습니다. [공식 starter 원본](../data/raw/maritime_starter_notebook.ipynb), [오디오 코드](../pipeline/audio.py), [과거 수치 parity](OFFICIAL_BASELINE_COMPARISON_FACTS.json)

**그 다음 연산은 다릅니다.**

| 단계 | 제공된 starter 변환 함수 | 우리 R4 |
|---|---|---|
| 클립 끝 처리 | 완전한 60초까지만 사용 | 모든 완전한 2초 bin 보존; 마지막 미완전 1분의 2초 bin도 사용 가능 |
| 변형에 전달하는 표현 | complete-minute 평균값만 반환 | 실제 2초 power trajectory 전체 |
| 시간·fade·배치 | 함수 자체에는 없음 | whole-source warp → 2초 fade/EQ → 2초 위치 배치 → 30-bin 평균 |

따라서 `Mean30 ∘ Warp/Fade/Place`는 일반적으로 minute-first 변형과 같지 않습니다. 세부 PSD 적분이 일치하더라도 경계 분의 스펙트럼과 시간 궤적은 달라질 수 있습니다. 이것은 **공식 helper와의 확인된 연산 차이**입니다. 그러나 숨겨진 공식 generator가 helper 반환값만으로 변형했다고 공개적으로 입증된 것은 아니므로, 곧바로 라벨 버그·규정 위반·실패의 단독 원인이라고 단정하지 않습니다.

## 3. 공식적으로 확인되지 않은 우리의 사전 가정

로컬 starter의 출처 고지는 VTUAD 계열 passage를 분 단위 주파수 요약으로 변환하고, 절대 digital gain을 버린 뒤 시간 궤적·상대 채널·에너지를 변형했다고 설명합니다. 정확한 recipe, 범위, source sampling, class/target 비율, 배치 방법은 이 고지에서 제공하지 않습니다.

| R4에서 고정한 항목 | 구현 사실 | 공식 generator와 동일하다는 증거 |
|---|---|---|
| 추가 없음/target prior | zero `round(.2N)`, 양성 share U(.005,.95)의 LHS | 없음 |
| 시간 길이 | 파생 support 120–900초의 discrete-uniform LHS | 없음 |
| source timewarp | 전체 원 궤적을 한 번 통과, ratio 1/3–3, 대역별 평균 power 보존 | 없음 |
| fade·채널 | raised-cosine amplitude² fade 최대 30초, smooth reference EQ ±4dB | 정확한 형태·범위 미공개 |
| 위치·관측 관계 | 관측된 2초 위치를 anchor로 삼아 양성 overlap을 보장 | 같은 방식이라는 증거 없음 |
| 배경·gain | released clean SA 배경, 추가 common gain 0, background blending 없음 | 숨겨진 기간·수신 조건과의 동등성 미검증 |

이 가정들은 사전 선언된 합성 실험으로서 산술적으로 유효합니다. **선언되어 있다는 사실과 공식 평가 분포에 맞는다는 사실은 다릅니다.** 목표 prior·operator 차이를 public score나 Test 코호트 통계로 역산·조정해서는 안 됩니다. [고정 R4 config](../configs/group_cv_passage_r4_data.json), [R4 요약](R4_DATA_SUMMARY.md)

또한 렌더 길이에 따라 사용할 수 있는 source 집합이 달라집니다. `m/K∈[1/3,3]` 조건으로 pool을 구성하므로, 길이·원천 성격 사이의 의존성이 생깁니다. 같은 연산자를 공유하는 TRAIN/INNER에서는 이것이 유리한 학습 단서가 될 수 있지만, 다른 생성 연산자에서도 유지된다는 검증은 없습니다.

## 4. source holdout과 simulator holdout은 다르다

R4의 전체 선택은 119 clips / 22 retained connected components / 88 hulls / 22 days입니다. Fold0 TRAIN은 69 clips / 12 components, INNER는 25 clips / 5 components입니다. 여러 hull이 겹치는 녹음·날짜가 연결되어 있으므로 행 수나 hull 수를 모두 독립 음향 도메인으로 세면 안 됩니다. 같은 ONC 계열 hydrophone의 추가 독립 현장 측정 22개라는 의미도 아닙니다. [참조 격리 감사](r4_reference_audit.json)

SA sensor/week와 24시간 purge, reference hull/day/UTC/bytes 분리는 중요한 보호입니다. 그러나 TRAIN/INNER의 source를 나누고 **같은 합성 연산자로 양쪽 정답을 만들면**, 검증은 그 연산자 안의 source 일반화를 측정합니다. 다른 연산자·새 수신 조건에 대한 일반화는 별도 문제입니다. 새 epoch마다 합성 행을 만들더라도 원천 녹음의 독립 정보가 같은 비율로 늘어나지 않습니다.

혼합만 보고 추가 에너지를 완전히 식별할 수 없는 근본적 이유도 있습니다. 관측된 양의 행렬 `E`에 대해 임의의 `c∈[0,1]`로 `A=cE`, `B=(1−c)E`를 두면 관측 `E`는 같고 정답은 `c`가 됩니다. 실제 문제는 허용된 배경·passage의 구조를 통해 이 모호함을 줄이는 것입니다. 그러므로 generator/source prior에 맞지 않으면, 더 큰 네트워크와 정확한 pooling만으로 정답을 보장할 수 없습니다.

## 5. 먼저 할 수 있는 TRAIN/INNER 전용 진단

아래는 **제안이며 이 감사에서 실행하지 않았습니다.** 현재 동결된 GPU 실행이나 데이터는 수정하지 않습니다.

1. **제출 재현을 먼저 닫기:** 같은 checkpoint·prior·raw policy의 Mac CPU INNER 예측/후처리를 source GPU와 비교하고, 제출 파일 SHA·ID 순서·물리 pooling을 확인합니다. 새 후처리는 CPU INNER로만 결정하고 Test 전에 고정합니다. 순수 변환/연결 오류를 generator 문제로 오인하지 않기 위한 절차입니다.
2. **잔차를 source·물리 조건별로 분해:** 기존 INNER 전체 RMSE와 SA sensor/week, reference component, known share, 실제/파생 duration, support, coverage, quality flag별 RMSE·bias·q 오차·진단 AUC를 함께 기록합니다. 어려운 구간을 삭제하지 않습니다. 알려진 controlled-zero는 자연 선박 부재를 뜻하지 않습니다.
3. **공통 gain의 정확한 불변성 검사:** 자체 INNER의 `db+g`는 B/A를 같은 배율로 바꾸므로 정답 비율은 불변입니다. 사전 고정한 gain에서 예측 변화를 보면 absolute-level shortcut을 진단할 수 있습니다. R5의 8채널에는 absolute dB 채널이 있어 불변성이 구조적으로 강제되지는 않습니다. Test 레벨에 맞춰 g나 보정계수를 선택하지 않습니다.
4. **operator-heldout stress 평가:** 같은 허용 원천의 TRAIN만으로 설계를 정하고, minute-first vs 2초-first, 무 warp 실제 궤적 vs bounded warp, 다른 사전 선언 fade/channel/support 조건을 분리합니다. 한 family로 학습한 모델이 다른 family에서도 작동하는지 INNER 역할의 source-isolated 실험으로 확인합니다. 각 label은 변형 후 동일 관측 셀의 실제 추가/혼합 에너지로 다시 계산해야 합니다.
5. **후보 모델 전체 비교:** 같은 INNER와 위 stress에 CNN/TFGrid/FDY/tree/foundation을 비교하고, 앙상블도 INNER에서만 선택합니다. 단독 모델보다 좋아졌는지뿐 아니라 동일 방향 bias를 공유하는지도 봅니다. 큰 모델을 먼저 넣는 것이 domain mismatch의 검증을 대신하지 않습니다.

원칙적으로 source holdout을 유지한 채 operator family까지 분리해야 합니다. 이미 반복해서 본 INNER는 개발 자료이고, 후속 판단을 독립 검증으로 부르려면 별도의 잠긴 source/설계 평가가 필요합니다. OUTER·공식 점수는 보고로 보존하며 계수·target prior·generator를 맞추는 신호로 사용하지 않습니다.

## 6. 과거 문서에서 재사용하지 않은 추론

기존 `TASK_EDA_REPORT.md`는 공식 source 고지로부터 “같은 생성 과정을 직접 만들 수 있다”고 확장 해석했습니다. 정확한 숨은 연산자까지 알 수 있다는 근거는 없습니다. 그 문서의 Test 코호트·입력 분포 비교 및 Benchmark를 통한 target moment 가정은 이번 수정 근거에서 제외했습니다. 이번 감사는 그 숫자를 재집계·재현하거나 hidden source/label을 복구하지 않았습니다.

현재 공식 Info/Data는 직접 열어 확인했습니다. 새 discussions 페이지와 두 Staff 스레드는 웹 도구에서 접근되지 않아, 관련 답변은 로컬 보관본(스레드 35370·35368, 2026-10-20)을 참고했습니다. 해당 답변은 own train/validation 사용과 Test inference-only 경계를 강조하지만, 숨은 generator recipe를 공개하지 않습니다. 최신 전체 토론을 이번에 재열람했다고 주장하지 않습니다. [보관본](../scratch/discussions/discussion_archive.txt), [수집 목록](../scratch/discussions/reading_audit.csv)

## 소스 지문

| 파일 | SHA-256 |
|---|---|
| `pipeline/passage_sampler_r4.py` | `d56e8199e5cbee5ee22a96143fbba3186bf5ff0ddff9d0ed34a5c0f6d56a0162` |
| `pipeline/audio.py` | `86853a4c144ed9077e3523ba4bee092110cf3f8ac217de6426e599168a23eec1` |
| `configs/group_cv_passage_r4_data.json` | `a0be9c9957db117a30b43379d9e5097098897a3b551bde7dff4c1b78acdff70d` |
| `data/raw/maritime_starter_notebook.ipynb` | `08242f4771c38de43ab9022d460b537b5eaa90a687bb1ddb532061b512dd6b83` |
| `reports/R4_DATA_FINAL_VERIFICATION.json` | `05669a24b06d59773e83901ff31bf1b3f7fc54c616e9f57215a83d06ee88d4da` |

본 문서는 새로운 성능 개선·공식 generator 재현·실제 압력 분리 성공을 주장하지 않습니다.
