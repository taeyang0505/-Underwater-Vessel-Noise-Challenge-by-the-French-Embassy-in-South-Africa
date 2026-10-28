# R16 support × 시간 구조 — 독립 사전 설계 검토

## 검토 범위

사용자 첨부 `d7556153-4c96-4a32-bad0-677d70f9aac0/붙여넣은 텍스트.txt`, `../r16_problem_gate_r1/FEATURE_FAILURE_REVIEW.md`, 실제 R16 소형 모델 코드와 동결된 Tree5 스트레스 결과를 대조했다. 이 검토는 새 학습·합성·Test 접근을 하지 않았고 기존 코드·데이터·checkpoint를 변경하지 않았다.

이번 목적은 두 보완을 **한 번의 사전 선언된 2×2 소형 실험**으로 분리 평가하는 것이다. 본체 전체 학습, 공식 제출, 최적 하이퍼파라미터 탐색을 승인하는 보고서가 아니다.

## 1. 확인된 사실과 아직 검증되지 않은 설명

| 주장 | 판정 | 해석 및 한계 |
|---|---|---|
| R16 소형/본체 Fold0 TRAIN의 관측된 추가 support 중앙값은 2분이며 ≥12분 양성은 0건이다 | 확인됨 | 실제 `cell_share > 0` support 및 참조 분할을 전수 센 결과다. 원천 관측 시간 길이와 추가 성분 support는 구분한다. |
| selected119 풀의 ≥12분 클립 4개는 기존 C의 2개 연결 컴포넌트에만 배정되었다 | 확인됨 | 19/20/23분 세 클립은 같은 2016-09-27 컴포넌트다. 세 클립을 세 독립 장시간 원천으로 세지 않는다. |
| 동결 Tree5도 native15 vs HF 음성에서 classifier AUC 0.2592로 역전된다 | 확인됨 | 양성 확률 평균 0.2676, 음성 0.5013, pairwise rank inversion 74.08%. 이 조건의 실패가 실측되었다. |
| 확률 포화만 해결하면 순위 실패도 해결된다 | 반례 확인 | 해당 Tree5 패널의 ≥0.99 비율은 0.8%뿐인데 AUC는 0.2592다. |
| native15는 완전 stationary가 아니다 | 확인됨 | 알려진 추가 A의 대역별 시간 CV 중앙값 0.443은 시간 변동 존재를 지지한다. rounded E와 raw q로 재구성한 기술 통계이므로 원파형 정확 재구성으로 부르지 않는다. |
| native15의 CV가 있으므로 시간 특징만으로 선박과 배경을 100% 구별할 수 있다 | 미입증 | 변동의 존재는 클래스 식별성·센서 전이·독립 참조 일반화를 보장하지 않는다. 관측 E에는 여러 B/A 비음수 분해가 가능하다. |
| 자연 HF 음성은 모두 1분짜리이고 LF/MF와 동조하지 않는다 | 미입증 | 현재 문서에는 전체 HF 음성의 지속성·cross-band correlation 분포를 검증한 수치가 없다. 새 특징으로 확인할 가설이지 정답 생성 규칙이 아니다. |
| self-subtraction이 장시간 오차의 단일 원인이다 | 유력 후보지만 미입증 | source·길이·crop·HF 음성 범위가 함께 다르다. 2×2 결과도 자료 보완 패키지 효과와 특징 효과를 구분할 뿐 모든 인과를 확정하지 않는다. |
| R16의 낮은 St Francis/HF 음성 share 평균이 실제 대회 과대추정 해결을 증명한다 | 로컬 사실만 확인 | 고정 로컬 자연 음성에서 평균 0.030/0.047은 확인되지만 실제 Test 정답으로 검증된 개선은 아니다. 이전 Public 악화의 단일 원인도 확정되지 않았다. |

`y=0`은 남아공 배경에 **우리가 추가한 참조 에너지가 없음**을 뜻한다. 자연 선박·생물·기계 소리가 전혀 없는 바다라는 뜻이 아니다. 이를 임의의 선박 존재 라벨로 바꾸지 않는다.

## 2. 원천 분할과 공정한 2×2 설계

### 새 분할 A′/B′/C′

- 새 분할은 **clip의 길이·hull/day/중첩 컴포넌트 메타데이터만으로** 고정한다. 결과 점수를 보고 어떤 장시간 컴포넌트를 TRAIN으로 옮길지 고르지 않는다.
- 현재 실행안은 2016-09-27 장시간 컴포넌트(19~23분)를 A′로 편입하고, 별도 12분 컴포넌트를 새 C′의 미관측 장시간 원천으로 남긴다. B′는 checkpoint/Tree early stopping·정책 선택 전용이며 C′를 선택에 쓰지 않는다.
- TRAIN/fit/report/도메인 배경은 source group 분리와 900초 support 기준 24시간 purge를 유지한다. 새 HF 후보도 해당 TRAIN 배경 분할 안에서만 고른다.
- selected119의 조건부 연결 그래프와 full509의 단일 연결 컴포넌트를 혼동하지 않는다. 평가용 원천 분리를 주장하는 이번 소형 실험에 full509를 무차별 편입하지 않는다. 전체 제출 학습의 원천 활용은 검증 분할과 별도로 선언해야 한다.

### 네 arm

| Arm | 자료 D | 특징 F | 비교 목적 |
|---|---|---|---|
| D0F0 | 새 A′, 기존 짧은 crop/기존 음성 방식 | 기존 relative8 / temporal700 | 새 분할에서 재학습한 공정한 대조군 |
| D1F0 | 같은 A′, 자연 support 확대 + matched HF 음성 | 기존 특징 | 자료 보완 패키지 효과 |
| D0F1 | D0와 정확히 같은 자료 | 시간 지속성·동조 보강 | 특징 효과 |
| D1F1 | D1과 정확히 같은 자료 | 시간 지속성·동조 보강 | 결합 효과·상호작용 |

네 arm은 참조 roster·양성 background ID·명시적 zero slot·연산자·seed를 공유한다. D0/D1의 참조별 추첨 횟수도 가능한 한 동일하게 두고, 긴 참조를 D0에서 짧게 crop하고 D1에서 native 길이로 보존한다. 서로 다른 원천을 동시에 바꾼 비교를 순수 crop 효과로 부르지 않는다.

HF 음성 교체에서는 background ID 자체가 바뀔 수 있다. 따라서 **모든 행이 동일 배경**이라고 주장하면 안 된다. 대신 원래 zero slot과 교체 배경의 site/sensor/recorder/정확한 관측 mask·TRAIN 분할을 일치시킨 쌍을 저장하고 미일치·미해결 slot 수를 공개한다. 표본을 부풀리려고 평가 구간에서 후보를 가져오지 않는다.

기존 동결 R16 mixed는 별도의 **역사적 기준선**으로 평가한다. 원천 A′가 바뀐 D0F0와 기존 A66으로 학습된 old mixed를 같은 인과 대조군으로 취급하지 않는다.

## 3. 자료·물리 라벨 사전 검사

- 원래 분 단위 참조를 순서대로 사용한다. 반복·시간 신축·인위적 stationary/full_smooth15·highshare 강제 배정은 금지한다.
- support는 **관측된 추가 에너지**로 계산한다. native 참조 길이, 창에 삽입한 길이, 관측 mask와 실제 겹친 support를 각각 보고한다.
- 1~4/5~10/11~15분 bin의 양성 수·참조 컴포넌트 수·정답 y 분포를 공개한다. 공급 가능한 독립 원천이 없는데 균등 비율을 맞추기 위해 stretch/repeat하지 않는다.
- 무추가 슬롯은 기존 계약과 동일하게 20%를 선언하고 실제 관측 추가 0 비율을 따로 센다. mask와 겹치지 않아 양성이 y=0이 되는 경우 재추첨 정책을 사전에 명시한다.
- 변경된 실제 `A`, `B`, `O`에서 `E=B+A`, `q=A/E`, `y=sum(OA)/sum(OE)`를 계산한다. `db`는 0.1dB 격자, 입력 weights는 반올림된 E에서 계산하며 raw 라벨과의 제한된 양자화 오차를 구분한다.
- 음성 교체는 reference 에너지를 추가한 뒤 y=0으로 덮어쓰는 조작이 아니다. 실제 추가 A가 0인 자연 남아공 배경을 사용한다.
- 자료 SHA·각 row의 seed/배경·참조·route·support·mask 및 교체 lineage를 저장한다. D0F0=D0F1, D1F0=D1F1 자료 SHA 동일성을 검사한다.

## 4. 시간 특징과 회귀/분류 분리

- 새 시간 특징은 band별 시간 중심화·정규화 이후 만든다. per-band `+c[f]` 불변성을 주장하려면, 서로 다른 offset을 갖는 band를 먼저 선형 에너지 합산한 뒤 그룹 시계열을 만드는 구현은 별도 검사가 필요하다. 각 band를 먼저 중심화한 뒤 group 집계하면 이 혼선을 피할 수 있다.
- rolling2/4/8은 실제 15분 축에서 계산한다. 결측 분을 제거해 5분 간격 두 관측을 연속 1분으로 압축하지 않는다. lag-k는 실제 k분 차이에서 둘 다 관측된 pair만 사용한다. 부족한 support와 유효 pair 수를 명시한다.
- 기울기·곡률 fit은 실제 minute index를 사용하고 순서를 보존한다. masked/unobserved padding 값을 진짜 0 에너지로 취급하지 않는다.
- `+c[f]` 불변성 검사는 0.1dB grid-aligned offset에서 정확성 기준을 적용한다. 비격자 offset은 반올림 순서에 따른 최대 0.1dB 차이를 별도 다룬다.
- 분류에는 절대 dB·PCA·LF anchor·상대 정적 profile·sqrt(w)·분모 기반 share를 넣지 않는다. 회귀는 기존 상대 profile·sqrt(w)와 물리 풀링을 유지한다.
- TRAIN 자기 예측 회귀 출력/마스크를 분류 특징에 넣으면 누수 위험이 있다. 이번 소형 비교에는 이를 넣지 않는다. 후속 사용에는 독립 원천 OOF 특징 생성과 inference 정의가 선행되어야 한다.
- NN 회귀와 분류 파라미터·gradient 경로를 분리한다. share loss + 0.25 weighted cell loss를 유지하고 contrast BCE를 mask 경로에 재투입하지 않는다.

## 5. 사전 고정할 판정 기준

아래는 **본 실험 실행 전에 root가 제안한 운영 기준**이다. 통계적으로 증명된 동등성 한계나 실제 Public 성능 보장은 아니다. 실제 실행 config/PLAN에도 같은 수치를 결박한 뒤 평가해야 한다.

1. **Primary 유지:** 별도 배경·새 별도 참조/크롭·미관측 사이트 pooled RMSE의 악화가 각 대조군 대비 `max(0.005, baseline×5%)` 이내이고 독립 classifier AUC 감소가 0.02 이내일 것. 여섯 연산자별 지표를 모두 공개하여 pooled 개선으로 특정 조건 악화를 숨기지 않는다.
2. **장시간/HF 순위:** 새 미관측 12분 참조 vs matched HF 음성 및 SF dense에서 classifier AUC≥0.75를 최소 운영 기준으로 하고 0.80을 목표로 보고한다. 알려진 R11 AUC는 노출된 스트레스 진단으로 별도 표기한다.
3. **HF 오탐:** 동일 평가 HF 음성의 독립 분류 확률 평균이 대조군보다 절대 0.10 또는 상대 25% 이상 감소할 것. 양성 확률도 함께 내려가는 전체 score 축소와 구분하도록 AUC·양성 평균·음성 평균·고정 0.5 threshold의 TPR/FPR를 함께 보고한다. 확률 평균 감소만으로 일반화를 선언하지 않는다.
4. **회귀 편향:** y>0.2, support별·site별 bias와 RMSE를 보고한다. 낮은 평균 음성 share만으로 고비율/장시간 과소추정을 숨기지 않는다.
5. **한 번의 실험:** seed·자료 수·epoch100/Adam·checkpoint 선택 역할·Tree 설정·가중치·판정 기준을 고정하고 네 arm만 실행한다. 실패하면 실패 및 원인을 보고하며 같은 report/R11로 반복 탐색하지 않는다.

### 불확실성 및 기존 평가 오염

- C에서 A′로 이동한 참조를 포함하는 기존 C 패널은 `legacy_C_exposed`로 표시한다. 그 결과가 좋아져도 새 source-unseen 성능이라고 부르지 않는다. 기존 R16 3대 수치는 역사 비교로 보존하되 source가 실제 겹치는 row 비율을 함께 표시한다.
- 새 C′ 장시간 평가는 12분 컴포넌트 하나뿐이다. **실제 장시간 원천 유효 N=1**이며, 500/1,000개 합성 조합은 500/1,000개의 독립 장시간 선박이 아니다. 12분 holdout 통과만으로 15분 미관측 통과까지 입증할 수 없다.
- 원천 bootstrap은 배경 group와 참조 component를 단위로 하고 같은 원천의 여러 합성 row를 독립 표본처럼 resample하지 않는다. 장시간 컴포넌트 하나에서는 참조 전이의 95% CI를 정직하게 추정할 수 없다. 조건부 배경 bootstrap과 컴포넌트별 원시 수치를 구분해 보고한다.
- 2×2 차이에는 D효과, F효과, D×F 상호작용을 같은 평가 row에서 paired difference로 표시한다. source 수 부족으로 신뢰구간이 넓으면 점 추정만으로 확정 표현을 하지 않는다.

## 6. 기존 R16 복원 비교 경로

- 기준 모델: `../../models/r16_clean_pilot_r1/mixed/COMPLETED.json`, `best_regression.pt`, `best_classifier.pt`, `tree_regressor.txt`, `tree_classifier.txt`.
- API: `ops.r16_clean_models.predict_arm(folder, panel)`은 best regression과 best independent classifier checkpoint를 **각각** 로드한다. 동일 checkpoint라고 가정하거나 best-regression의 classifier를 사용하면 공정 비교가 깨진다.
- 반환은 `tree_share/tree_classifier`, `nn_share/nn_classifier`, 고정 `balanced_share=0.6Tree+0.4NN`, `balanced_classifier=0.5Tree+0.5NN`이다. 새 실험에서 비교용 고정 결합을 쓰면 이 수식을 사전에 유지한다.
- 기존 API는 `safe_panel(panel)` 때문에 q/y를 읽어 라벨 무결성을 검사하지만 실제 `nn_predict`/Tree feature 입력에는 db와 weights만 전달한다. 독립 감사에서는 모델 순전파에 라벨이 들어가지 않는지 별도로 확인한다. Test 추론 API로 그대로 사용하는 함수는 아니다.
- old R16의 저장 score lambda와 새 independent classifier를 혼동하지 않는다. 공정한 2×2 분류 비교는 독립 classifier probability AUC를 주로 쓰고, share 섞인 예전 frozen score는 별도 진단으로만 보고한다.

## 최종 판정

이 두 보완은 현재 확인된 자료 support 공백과 자연 HF 음성 오탐에 직접 대응하는 합리적인 소형 실험이다. **성공 여부는 아직 미측정**이다. 기존 레시피의 문제를 단정적으로 모두 해결했다거나, 시간 특징만으로 완전 분리 가능하다거나, 새 C′에서 15분 독립 일반화를 입증했다고 미리 결론내려서는 안 된다.
