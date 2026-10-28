# R4 데이터 구축 완료 기록

로컬 전체 생성과 사후 전수 검증을 완료했습니다. 다섯 fold의 `ready_to_train=true`는 데이터 품질 gate를 통과했다는 상태입니다. 새 모델 구조 검토가 진행 중이며 이 데이터로 GPU·모델 학습은 실행하지 않았습니다.

확정 원천·코드의 materialize 실행은 **124.3초**, 사후 전수 검증은 **33.9초**였습니다. 생성은 고정 config/seed에서 한 번 수행했고 seed 재시도·중간 산출물 재사용은 없습니다.

## 보존한 데이터와 원천

| 항목 | 실제 값 |
|---|---:|
| TRAIN 역할 합계 | 267,837 |
| INNER / OUTER 역할 합계 | 15,000 / 15,000 |
| 전체 생성 예제 | 297,837 |
| 교차 fold 고유 primary 학습 배경 창 | 94,711 |
| 선택 VTUAD clips / retained components | 119 / 22 |
| 원 측정 길이 ≥900초 clips / components | 3 / 1 |
| 정규 NPZ 15개 | 434,330,282 bytes (414.2 MiB) |
| 정규 배열 payload | 895,893,696 bytes |

267,837은 다섯 학습 역할의 합계이며 원천이 교차 fold TRAIN에 재등장합니다. 22 retained components는 선택된 hull/day/UTC/bytes 연결 그래프의 분리 단위입니다. 같은 VT hydrophone의 새 독립 측정 원천 22개나 실측 15분 source 22개로 세지 않습니다.

기존 SA fold·sensor/week 배정·24시간 support purge와 clean TRAIN primary 순서/100% coverage를 유지했습니다. 기존 R2 파일과 원본 1,715,779행은 보존했습니다. R4 INNER/OUTER는 새 동일 생성 정책으로 만들었고 legacy 평가 배열을 active R4에 복사하지 않았습니다. 24시간 purge는 SA 배경에 대한 보장입니다. VT 참조는 선택된 hull/day/UTC/bytes 직접 격리 범위입니다.

| Fold | TRAIN | INNER | OUTER | SA 최소 support gap |
|---:|---:|---:|---:|---:|
| 0 | 53,244 | 3,000 | 3,000 | 24.0 h |
| 1 | 53,315 | 3,000 | 3,000 | 24.0 h |
| 2 | 53,522 | 3,000 | 3,000 | 24.0 h |
| 3 | 52,998 | 3,000 | 3,000 | 24.0 h |
| 4 | 54,758 | 3,000 | 3,000 | 24.0 h |

## 모든 split에 동일한 생성 법칙

양성 예제에는 VT passage 하나만 사용합니다. 먼저 독립 full-split LHS로 양성 share U(.005,.95), rendered 2초-bin 길이 discrete Uniform{60,…,450}을 정하고 zero 수는 round(.2N)로 고정·셔플합니다. u와 길이에는 독립 난수 stream/셔플을 씁니다.

원 측정 120–1380초 power trajectory 전체를 cumulative-linear-power integral로 단회 단조 통과하며 ratio는 [1/3,3]입니다. 대역별 mean power를 보존하고 source 원본 UTC/길이는 그대로 기록합니다. 파생 rendered support는 120–900초입니다. 뒤에 raised-cosine amplitude²의 2초-bin 평균 fade와 smooth reference EQ±4dB를 적용합니다. dB 보간·clip 반복·원 UTC 수정은 없습니다.

자연 SA 관측 mask를 그대로 사용합니다. 별도 gain·background blending·dropout·time reversal·hard SNR/support cap은 적용하지 않았습니다. 최종 known added-reference fraction과 cell q는 변형 후 관측 에너지에서 다시 계산합니다. 참조에 녹음된 주변음도 추가 성분에 포함되며 자연 선박 전체의 정답이 아닙니다.

## 실제 생산 데이터 품질 gate

| 검사 | 최악값 / 상태 |
|---|---:|
| saved q×energy → y 최대 오차 | 3.166e-07 |
| 검증한 예제 / 양성 recipe | 297,837 / 238,269 |
| 15 NPZ·모든 recipe SHA, 전체 mask 원 배경 일치 | PASS |
| source split/SHA/UTC/길이/단조 warp/EQ, 전체 schedule 재현 | PASS |
| 고정 code/config/원천 metadata 불변 | PASS |
| mean_share_difference | 0.00000538 |
| zero_fraction_difference | 0.00000755 |
| coarse_duration_total_variation | 0.00024881 |
| fine_duration_total_variation | 0.00060649 |
| rendered_duration_KS_statistic | 0.00042620 |
| rendered_duration_mean_difference_seconds | 0.01378989 |
| maximum_conditional_mean_difference | 0.01517957 |

관측 coverage와 원 측정 길이는 source에 따라 다르므로 별도 진단으로 남겼습니다. 파생 rendered-duration·u의 사전 gate는 실제 생산 크기에서 모두 통과했습니다. 1,024건 pilot의 conditional-u 오차 .07165 실패는 보존했고 seed를 바꾸지 않았습니다. 원 측정 duration mismatch를 없앴다고 주장하지 않습니다.

## 새 모델에서 재사용할 범위

NPZ는 모델 채널을 고정하지 않습니다. `db[N,15,25]`와 NaN mask가 관측 입력이며 `share`, `presence`, `cell_share`는 학습 정답입니다. 새 8채널/다른 특징은 이 db의 변환 또는 해당 fold TRAIN에서만 맞춘 변환으로 구성할 수 있습니다. Oracle cell_share와 reference/schedule provenance를 query 입력으로 넣지 않습니다.

공식 비공개 generator를 복원했다는 증거는 없고, 실제 대회 성능과 모델 학습 품질은 아직 측정하지 않았습니다. Test·calibration·sealed의 데이터는 사용하지 않았으며 외부 전송·공식 제출은 하지 않았습니다.

## 재현·감사 파일

- [고정 생성 config](../configs/group_cv_passage_r4_data.json)
- [CV manifest](../data/synthetic/group_cv_passage_r4_v1/cv_manifest.json)
- [생성 실행 receipt](R4_DATA_GENERATION_EXECUTION.json)
- [생성 단계 감사](R4_DATA_AUDIT.json)
- [사후 전체 NPZ·mask·recipe 검증](R4_DATA_FINAL_VERIFICATION.json)
- [Code/config/source freeze](R4_DATA_CODE_FREEZE.json)
- [독립 timewarp·생산 schedule QA](R4_TIMEWARP_INDEPENDENT_QA.json)
- [원천 선택 감사](r4_reference_audit.json)
- [처음 fade-only pilot](R4_DATA_PILOT.md)
- [bounded warp 1k pilot와 남긴 실패](R4_DATA_PILOT_WARP.md)
- [순수 숫자 fixture의 builder E2E](R4_DATA_BUILDER_WARP_FIXTURE.json)

CV source hash: `ea4a8a64dc831bcac7058df010e2ec525bd311093567877180dffb35479898ae`.

행렬/스케줄 독립 재검증: `python -m ops.verify_passage_cv_r4 --data-dir data/synthetic/group_cv_passage_r4_v1/fold_00` (fold_00부터 fold_04까지).
