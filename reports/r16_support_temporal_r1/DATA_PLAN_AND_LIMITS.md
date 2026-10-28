# R16 자연 support × 시간 구조 소형 실험: 데이터 구성과 검증

이 문서는 **새 소형 2×2 실험**의 데이터 준비 결과다. 본체 학습 완료나 공식 Test 성능 개선 결과가 아니다. 기존 R16 소스·데이터·27 epoch 체크포인트는 수정하지 않았다. 새로운 데이터로 학습한 모델에 기존 체크포인트를 이어 붙이지 않는다.

## 1. 사전에 고정한 비교

| 구분 | D0: 새 원천 분할 위 기존 crop | D1: 자연 support + HF 음성 보강 |
|---|---|---|
| TRAIN 원천 | A′ 75클립·13컴포넌트 | 동일 A′ |
| 행 수 / 무추가 | 4,000 / 정확히 800 | 동일 |
| 양성 참조 ID | 공통 metadata-only 길이 층화 추첨 | D0와 행별 동일 |
| 양성 남아공 배경 / 관측 마스크 | 공통 | 행별 동일 |
| 생성 연산자·EQ·SNR | R16 6연산자, 공통 seed | 행별 동일 |
| 시간 배치 | 기존 25% native 전체·나머지 1~6분 crop | 짧은/중간 native는 전체 보존, 긴 native는 실제 연속 11~15분 crop |
| `y=0` 배경 | 원래 배경 | exact site/sensor/recorder/mask 일치 TRAIN HF 대조군으로 교체 |

D0는 **과거 frozen R16 mixed와 완전히 같은 데이터가 아니다.** A′ 원천과 사전 길이 층화 추첨을 D1과 맞춘 통제군이다. 과거 frozen R16 수치는 별도의 historical 행으로만 비교해야 한다. D1은 support와 HF 음성 보강을 묶은 데이터 요인이고, 두 요소 각각의 인과 기여까지 분리하는 실험은 아니다. 모델·특징 요인 F0/F1은 별도로 교차한다.

## 2. 참조 원천 분할과 한계

| 새 분할 | 클립 | 선택 그래프 컴포넌트 | 완전한 1분 구간의 클립별 합계 |
|---|---:|---:|---:|
| A′ TRAIN | 75 | 13 | 332분 |
| B fit/gate | 19 | 5 | 76분 |
| C′ report/domain | 25 | 4 | 105분 |

이 시간은 **클립별 완전한 분 수의 합**이며 중첩 녹음을 제거한 독립 녹음 시간은 아니다.

- 기존 C의 `VT_R4_REF_474fe510b4e0aa460cad`를 **컴포넌트 전체**로 A′에 이동했다. 이 컴포넌트는 19·20·23분 장시간 3클립 외 단편 6클립도 포함하므로 총 9클립이다. 장시간 클립 3개만 이동하는 잘못된 분할은 사용하지 않았다.
- `VT_R4_REF_de02fbd6bbb753c153bd`는 C′에 남겼다. 이 컴포넌트 6클립 중 독립 장시간 전용 패널에는 실제 **12분 클립 1개**만 쓴다.
- 선택된 119클립·22컴포넌트 안에서 hull/day/바이트·UTC 중복이 분할을 넘지 않는지 다시 검사했다. B는 변경하지 않았다.
- 전체 509클립을 연결한 원천 그래프는 **하나의 컴포넌트**다. 따라서 본 실험의 독립성은 중간 연결 원천을 제외한 **119클립 선택 그래프에 조건부인 독립성**이며, 전체 아카이브의 완전 독립 교차 검증으로 표현하면 안 된다.
- 현재 선택 풀에서 15분 이상 클립은 이동한 한 컴포넌트에만 있다. **미관측 15분 이상 원천을 유지한 채 15분 이상 TRAIN 원천을 공급하는 독립 분할은 만들 수 없다.** 새 독립 장시간 검증은 실제 12분 원천으로 제한된다.
- 기존 C와 R11 `native_long15`가 이동한 TRAIN 원천을 포함하므로, 새 모델의 해당 legacy 결과는 **reference-seen 진단**이다. 독립 성능으로 재사용하지 않는다. C′ report/domain과 독립 native12 패널을 새로 생성했다.

## 3. 실제 support 분포

사전의 native 길이 층화 비중은 짧은 1~4분 35%, 중간 5~10분 40%, 긴 11~15분 25%다. 실제 관측 support는 원본 결측 마스크에 따라 줄어든다. 이를 맞추려고 결측을 채우거나 참조를 재추첨하지 않았다.

| 양성 3,200건 기준 | D0 | D1 |
|---|---:|---:|
| 실제 활성 분 중앙값 | 3분 | 5분 |
| 실제 활성 12분 이상 | 0 | 500 (15.625%) |
| native 11~15분 선택 | 0 | 800 (25%) |
| `share` 평균, 전체 4,000건 | 0.145799 | 0.194907 |
| `share` RMS, 전체 4,000건 | 0.235432 | 0.298823 |

| 실제 활성 분 | D0 양성 수 | D1 양성 수 |
|---:|---:|---:|
| 1 | 723 | 113 |
| 2 | 639 | 293 |
| 3 | 645 | 621 |
| 4 | 449 | 348 |
| 5 | 327 | 430 |
| 6 | 358 | 544 |
| 7 | 21 | 71 |
| 8 | 23 | 68 |
| 9 | 0 | 3 |
| 10 | 15 | 74 |
| 11 | 0 | 135 |
| 12 | 0 | 122 |
| 13 | 0 | 138 |
| 14 | 0 | 119 |
| 15 | 0 | 121 |

D1의 높은 평균·RMS는 더 긴 실제 참조를 같은 SNR 법칙으로 누적한 결과다. 목표 `share`를 강제하거나 Test/LB 수치에 맞추지 않았다. **그러나 이 분포 변화가 일반 구간 과대추정으로 이어질 가능성은 남으므로**, 본체 승격 전 독립 참조·크롭/미관측 사이트의 RMSE와 bias 유지 여부를 반드시 확인해야 한다. 장시간 800건은 3개 원천 클립의 반복 조합이며, 독립 장시간 원천 800개를 확보한 뜻이 아니다.

## 4. Recorder/mask 일치 HF 무추가 대조군

- 후보 탐색: 원래 Fold 0 TRAIN 78,869개 배경만 사용했다.
- 고주파: 1.995~7.943 kHz(밴드 18~24) 선형 전력 합의 시간 최대 dB − 시간 중앙값 dB.
- 기준: **site × sensor_group × recorder × 실제 15bit 관측 마스크**별 TRAIN q90을 엄격히 초과한 창. 각 층의 임계치는 TRAIN 내부에서만 계산했다.
- 원래 4,000개 배경 ID를 모두 후보에서 제외하고, 안정 SHA 순서로 중복 없이 교체했다.
- 요청 800개 중 **798개 교체, 2개 유지**. 부족한 두 행은 임계치를 낮추거나 배경을 반복하지 않고 원래 `y=0` 배경을 유지했다.
- 모든 양성 배경, site/센서/녹음기/관측 마스크/양성 여부의 행별 대응이 유지됐다.
- 여기서 `y=0`은 **알려진 추가 성분 A가 0**이라는 뜻이다. 자연 선박이 없는 바다라는 원시 정답을 주장하지 않는다.

## 5. 평가 데이터와 접근 게이트

| 역할 | 자료 구성 | 행 수 |
|---|---|---:|
| local_fit | 기존 B, 6연산자별 400 | 2,400 |
| local_gate | 기존 B, 6연산자별 400 | 2,400 |
| local_report | 새 C′, 6연산자별 400 | 2,400 |
| domain_holdout | 새 C′ + 기존 미관측 사이트, 6연산자별 400 | 2,400 |
| 독립 native12 양성 | C′ 실제 12분 클립, 6연산자별 400 | 2,400 |
| 자연 HF 무추가 | 기존 report 원천 내 HF q90 초과 | 359 |
| St Francis dense 무추가 | 기존 report의 St Francis, Nobs≥13 | 168 |

독립 native12는 12분을 그대로 넣었으며 관측 활성 분은 366건이 12분, 34건이 11분이다. 참조를 15분으로 늘리지 않았다. HF/SF 비교는 양성과 음성의 수가 다르므로 각 클래스 총 가중치를 0.5씩 주는 **균형 RMSE·AUC·예측 평균**으로 평가한다. 음성 부족분을 복제하지 않는다.

TRAIN과 모든 평가 역할 사이에 원본 배경 ID/그룹 중복이 없으며, 900초 창 support 기준 **24시간 purge**를 다시 검증했다. 추가 long/HF/SF 배경까지 검사했다. domain은 센서가 분리된다. 이 자료들은 기존 개발 과정에 노출된 원천이며 완전히 새로운 blind 시험이라고 주장하지 않는다.

`local_report`, `domain_holdout`, `long_validation` 로더는 모델·앙상블 정책의 `GLOBAL_FREEZE` 파일 없이는 거부한다. freeze는 새 데이터 manifest SHA 및 실제 checkpoint/policy 파일 SHA와 결합해야 한다. 모델 결과를 본 뒤 이 데이터 레시피를 같은 버전에서 바꾸지 않는다.

## 6. 수식과 저장 규격

원래 R16 6개 정방향 생성 연산자와 EQ/SNR 범위를 그대로 썼다. 실제 1분 × 25대역 참조 전력을 사용하며 파형 반복, 시간 신축, 가짜 평탄화, 추가 bell/fade, `highshare` 강제 배정은 없다.

`E = B + A`, `q = A/E`, `y = Σ(OA)/Σ(OE)`를 반올림 전 성분으로 계산한다. 입력 `db`는 0.1 dB로 반올림하고 마스크는 원래 관측/NaN을 유지한다.

- raw 에너지 풀링 최대 오차: D0/D1 모두 **4.44×10⁻¹⁶**.
- 반올림 후 `Σ(wq)`와 raw `y`의 최대 차이: D0 0.001632 미만, D1 0.001347 미만. 각각의 엄격한 반올림 상한 0.005758보다 작다. 정답을 임의로 덮어쓰지 않았다.
- NPZ 34개 엔트리 전수 재로드: object dtype/pickle 0, 유한 타깃·관측 마스크·양성 정의 통과.
- 현재 새 데이터 폴더 약 165 MB. fit/gate 12개 엔트리는 원본 파일을 복사하지 않고 프로젝트 경로와 SHA로 재사용한다. 배열 엔트리 20,527행은 생성 연산자·paired arm을 포함한 수이므로 독립 원천 샘플 수로 해석하면 안 된다.

## 7. 파일과 재현성

- [고정 계획](data/PLAN.json)
- [새 참조 분할](data/reference_partitions.parquet)
- [원천·HF 교체 검증](data/SOURCE_AUDIT.json)
- [HF 교체 영수증](data/HF_SWAP_RECEIPTS.json)
- [추가 장시간 배경 purge 검사](data/EXTENDED_LONG_BACKGROUND_PURGE_QA.json)
- [데이터 manifest](data/manifest.json)
- [최종 loader QA](data/FINAL_DATA_LOADER_QA.json)
- [실자료 생성 로그](DATA_BUILD_SUCCESS.log)

manifest SHA: `1277393b88070b519c1001dcd81858a3ac64d4ce0f444a60107c2f60d075fe80`.

builder SHA: `1a76749e54335dc0e824a95eeca2e898c646c63fcdddf64d971293fed40b8cee`.

원천 20개 pin, 데이터 artifact 58개 pin을 모두 재검산했다. 최종 실자료 생성은 9.63초였다. 앞선 네 번의 사전 빌드는 컴포넌트 cardinality 확인 및 numpy/pandas metadata 직렬화 단계에서 중단됐고 `data_attempt00_*`~`data_attempt03_*`에 실패 로그/계획을 보관했다. **그 파일의 부분 결과를 학습에 재사용하지 않았다.**

```python
from ops.r16_support_temporal_data import load_training, load_panel, load_long_panel
data = load_training(manifest_path, arm='D0')  # 또는 D1
fit = load_panel(manifest_path, 'local_fit', 'control_eq6')
report = load_panel(manifest_path, 'local_report', 'control_eq6',
                    freeze_receipt=global_freeze_path)
long_positive = load_long_panel(manifest_path, 'native12_positive',
                    operator='independent_subband_snr', freeze_receipt=global_freeze_path)
hf_negative = load_long_panel(manifest_path, 'hf_high_negative',
                    freeze_receipt=global_freeze_path)
```

이 데이터 QA는 학습 효과의 증거가 아니다. 소형 2×2 실험에서 기존 핵심 교차 검증 유지와 독립 native12 대 HF/SF 순위 역전 개선을 함께 확인한 뒤 승격을 판단한다. 509 전체 풀 본체 학습은 선택 그래프 분할과 별도의 최종 학습 정책을 명시해야 한다.
