# R16 원천·생성 연산자 교차 실험 자료

작은 CPU 실험용으로 생성했으며 본체 생산 학습 자료의 준비 완료를 뜻하지 않는다. 기존 R9/R14/R15 파일과 모델은 변경하지 않았다. Test, 기존 Test EDA, Test 분포 비교 산출물을 이 생성기에서 읽지 않았고 모델 학습·추론·성능 기반 생성 파라미터 선택은 수행하지 않았다. 이미 과거 Test 기술 통계를 본 개발 단계라는 한계는 사전 선언에 남겼다.

## 행 수와 소비 API

| 자료 | 고유 배경 | 파일 수 | 파일별 행 | 참조 그룹 |
|---|---:|---:|---:|---|
| TRAIN control / mixed | 같은 4,000개 | 2 | 4,000 | A |
| local_fit | 400 | 6 | 400 | B |
| local_gate | 400 | 6 | 400 | B |
| local_report | 400 | 6 | 400 | C |
| domain_holdout | 400 | 6 | 400 | C |

순수 배경은 총 5,600개다. 채널 조건을 달리한 저장 행은 TRAIN 8,000개와 패널 9,600개로 총 17,600개이며, 반복 조건을 독립 원천 수로 세지 않는다. 각 파일은 정확히 추가0 20%, 관측 가능한 양성 80%다. control과 mixed 학습군을 각각 8,000행으로 늘리지 않았다.

```python
from ops.r16_forward_data import load_training, load_panel
manifest = 'reports/r16_cross_operator_r1/data/manifest.json'
train = load_training(manifest, arm='control')  # 또는 mixed; 정확히4,000행
fit = load_panel(manifest, 'local_fit', 'alpha_preEQ_inserted')
```

반환값은 원자료 배열 mapping이며 `db`, `unrounded_db`, `cell_share`, `share`, `presence`, `observed_mask`, `background_observed_power`, `added_full_power`, `added_observed_power`, `mixture_observed_power`, `sample_id`, `pair_id`, `site`, `background_group`, `base_window_index`, `reference_index`, `reference_component`, `family`, `explicit_no_source`, `operator`를 포함한다. 모든 NPZ는 숫자 또는 Unicode 배열이며 `allow_pickle=False`로 검사했다.

`local_report`와 `domain_holdout`은 `freeze_receipt` 없이 소비할 수 없다. 필수 receipt는 `status='completed'`, `all_models_frozen=True`, `ensemble_policy_frozen=True`, `data_manifest_sha256`, 그리고 실재하는 모델 summary·정책 파일들의 project-relative `artifact_sha256` map이다. SHA를 실제 파일과 대조한다. local_gate를 더 늦게 개방하는 실험 감독 정책은 부모 runner에서 추가로 적용할 수 있다.

## 원천 분리

배경 TRAIN은 R9 Fold0 TRAIN이고, fit/gate/report는 각각 기존 `ensemble_fit`/`ensemble_gate`/`report` 원천 역할을 유지했다. ID·background group 교집합이 없고, 모든 역할 쌍에서 같은 sensor의 900초 창 지원 구간에 24시간 이상 간격을 확인했다. 클록은 export clock이며 검증된 절대 UTC로 과장하지 않는다.

domain_holdout은 원래 공개 자료에서 따로 두었던 Saldhana Bay·Fish Hoek·Sea Point다. 새 TRAIN에는 없는 사이트이지만, 과거 개발 진단에서 이미 평가한 도메인이므로 새 blind 평가라고 부르지 않는다. 실제 관측 분의 원 row ID와 기존 boundary QC를 연결해 마스킹했고 min3분 관측을 적용했다.

참조는 과거 bridge 제외 후 선정한 VTUAD 119클립/22개 연결 성분을 기존 검증된 509 native-minute 캐시에 SHA로 연결했다. 고정 metadata SHA 순서로 A=66클립/12성분, B=19클립/5성분, C=34클립/5성분으로 나눴다. 선택된 클립 안에서는 hull·날짜·UTC 구간 겹침·bytes가 A/B/C를 넘지 않는다. 전체 525개 녹음의 그래프는 1개 연결 성분이라는 기존 한계를 유지한다. 제외된 bridge 파일은 이번 자료에 쓰지 않았다.

새 R16 모델만 A 대비 B/C를 미관측 참조로 해석할 수 있다. 기존 R9/R14 모델은 509개 공유 참조를 이미 학습했으므로 B/C에 대한 **기존 모델 평가를 reference-unseen으로 표시하면 안 된다.** fit과 gate는 서로 다른 배경을 쓰지만 같은 B 참조 그룹을 공유하므로 참조에 대한 검증 노출이 있다. report/domain은 C를 사용한다. B의 클립 수가 19개라는 제한도 유지한다.

## 고정된 6개 연산자

| 연산자 | 배율 기준 | 후속 채널 |
|---|---|---|
| control_eq6 | 전체 native 궤적의 EQ 후 broadband peak | random smooth EQ±6dB |
| mfhf_shelf6 | 같은 전체 EQ 후 peak로 다시 계산 | 위 EQ와 중·고역 shelf+6dB |
| mfhf_shelf12 | 같은 전체 EQ 후 peak로 다시 계산 | 위 EQ와 shelf+12dB |
| mfhf_shelf12_lfminus6 | 같은 전체 EQ 후 peak로 다시 계산 | 위 EQ와 shelf+12dB·LF−6dB |
| alpha_preEQ_inserted | 실제 관측된 삽입 구간의 EQ 전 peak | 25대역 로그 이득 평균0인 smooth EQ; 이후 재정규화 없음 |
| independent_subband_snr | 삽입 구간 EQ 전 peak에 대해 ≤200Hz와 >200Hz 배율을 독립 계산 | 같은 zero-mean smooth EQ; 이후 재정규화 없음 |

SNR은 모든 경우 사전 고정 U[−10,+10]dB다. 독립 서브밴드의 두 SNR도 각각 같은 범위에서 독립 추출했다. 배경 기준은 각 해당 군의 관측 분 선형 파워 중앙값이다. 새 post-EQ 범위는 TRAIN ±6dB, 모든 평가 역할 ±12dB로 사전 구분했다. zero-mean **로그 이득**은 선형 파워 총합 보존을 뜻하지 않는다.

원 R15의 EQ 후 고정 broadband SNR은 일관된 생성 법칙이다. 이를 오류로 단정하거나 새 연산자가 공식 숨은 생성기를 재현한다고 주장하지 않는다. 원 R15의 peaked crop은 EQ 전 native peak를 실제로 포함했다. 따라서 새 inserted 기준은 ‘항상 빠진 peak를 고쳤다’는 의미가 아니다. EQ 후 다른 분으로 peak가 옮겨갈 수 있는 경우와 다른 SNR 사전분포를 별도 비교하는 실험이다.

각 양성은 native 길이≤15분이면 사전 확률 25%로 자연 전체 궤적을 쓰고, 나머지는 연속1–6분을 쓴다. TRAIN/fit/gate crop은 원래 peak를 포함하고 report/domain은 원생 구간의 시작점을 균일 추출한다. 인공 bell·강제 fade·timewarp·반복·full_smooth15·강제 highshare는 없다. 실제 부족한 녹음을 늘리거나 이어 붙이지 않는다.

관측 양성을 보장하기 위해 O와 겹치는 합법적 시작점을 한 번 열거해 선택했다. 새 연산자의 inserted peak도 O 적용 후 기준이므로 alpha는 마스크에 조건화된다. 이는 원 R9의 mask-independent 배치와 다른 정책이며, 순수하게 채널만 바꾼 원인 분리 실험이라고 과장하지 않는다.

## 물리 검증과 범위

원래 B와 변형 후 A로 `E=B+A`, `q=OA/E`, `y=ΣOA/ΣOE`를 정확히 계산했다. 공통 dB offset 단위의 float64 성분을 저장하고 모델용 dB만 0.1dB로 반올림한다. 결측 외부의 A_full도 보존한다. B_observed를 결측에 0으로 두는 것은 관측 적분 규약이며 실제 미관측 B의 진실을 아는 것은 아니다. 참조는 배경음이 섞일 수 있는 전체 측정 녹음이며 순수 선박음 에너지 정답이 아니다.

자체 전수 QA는 E의 float64 합 일치, 저장 q·y 재계산, 관측 마스크, 양성 support, 20% zero, 후보 간 zero 입력 일치, 동일 pair/source ID, pickle 없는 reload, report 조기 개방 거부 및 모든 파일·원천 SHA를 검사했다. 모델 성능은 여기서 계산하지 않았다. 원천 분리는 [SOURCE_AUDIT.json](SOURCE_AUDIT.json), 수치 검증은 [SELF_QA.json](SELF_QA.json), 정확한 후보·seed는 [plan.json](plan.json), 파일 결박은 [manifest.json](manifest.json)에 있다.
