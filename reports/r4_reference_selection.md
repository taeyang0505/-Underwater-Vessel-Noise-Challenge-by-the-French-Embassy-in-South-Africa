# R4 VTUAD source selection: 119 clips / 22 selected components

기존 허용된 VTUAD 218개에서 99개 bridge recording을 실제로 제외해 **119 clips, 88 hulls, 22 recording days, 22 connected components**를 구성했다. 선택된 subset 안에서는 hull/day/UTC-overlap/SHA가 서로 다른 split을 가로지르지 않는다. 전체 525-recording corpus graph는 여전히 한 component다. 22개는 선택된 pool에 조건부인 직접 격리 단위이며 새로운 hydrophone·해역·물리적 독립 원천 22개를 뜻하지 않는다.

이 작업은 source metadata 구현이다. 기존 acoustic bank의 header와 index 범위만 확인했고 수치값은 읽지 않았다. Test, calibration/domain/sealed audio·배열, 모델 예측, OOF 결과, leaderboard는 selection에 사용하지 않았다. R2 data·모델·기존 core 모델/샘플러, SA fold 배정·24h purge는 변경하지 않았다. NOAA/HMS의 품질 불량을 판정하거나 주장한 것도 아니다.

## 입력과 실제 제외

- 허용 pool: 기존 `quality_v2_native`에서 자격을 통과해 R2 CV에 편입됐던 VTUAD 218개만 사용했다. 원본 split이 train/validation이고, AIS scenario-positive vessel passage, verified/measured/full-band QC, 120–1380초, 완전한 2초 segment 60–690개를 만족한다.
- 원 calibration/domain/sealed와 `excluded_*` source는 편입하지 않았다. 이번에 제외한 99개는 **허용된 218개 내부에서 새로 뺀 실제 clip**이며 `excluded_manifest.parquet`에 기록했다.
- 같은 hull을 한 recording day에만 남기고 다른 날의 bridge clip은 제거했다. 같은 날의 모든 선택 clip은 연결 graph에서 같은 component가 된다. 실제 UTC overlap과 byte SHA도 graph edge로 재검사했다.
- 기존 ≥900초 native 3개는 모두 보존했다. 두 hull H124/H003, 같은 2016-09-27의 녹음이므로 한 component이며 분할하지 않았다.

허용 pool에는 23일·88 hull이 있다. day–hull 최대 matching은 22다. 각 분리된 selected component에는 다른 component가 쓰지 않는 day와 hull의 edge가 적어도 하나 필요하므로 22가 component 수 상한이다. 실제 선택 graph가 22에 도달했다.

선택 우선순위는 사전에 고정했다: component 상한 달성 → native 3개 보존 → 유지 clip 수 최대화 → 유지된 실제 2초 segment 수 최대화 → seed 42의 SHA tie break. 소스 메타데이터의 정수 할당이며 모델 학습·AutoML·성과 기반 hyperparameter 탐색이 아니다. **119는 ‘hull당 한 day만 유지’ 정책과 native 보존 조건 내의 clip-retention 최적값이다. 모든 임의의 22-component subgraph 중 절대 최대 clip 수가 119라는 주장은 하지 않는다.**

실제 측정 길이는 31,075초, 완전한 2초 사용 구간은 31,070초다. 남는 5초는 기존 bank의 trailing incomplete segment 처리와 일치한다. 원 `segment_file/start/count`, SHA, UTC, hull/day는 변경하지 않았다.

## 고정된 5-fold source 배정

component별 clip 수·측정 길이로 미리 정렬했다. 두 coarse duration bin을 모두 포함하는 component를 각 fold에 먼저 하나씩 배치한 후, 4–5 component의 용량 제한 안에서 clip 수·측정 시간이 적은 fold에 나머지를 배정했다. 이 배정에는 acoustic 값이나 모델 성과가 들어가지 않는다. inner는 `(outer_fold+1)%5`, TRAIN은 나머지 3개 source fold다.

| outer fold | components | clips | 120–<300초 | 300–1380초 | 실제 ≥900초 |
|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 25 | 19 | 6 | 0 |
| 1 | 5 | 25 | 20 | 5 | 0 |
| 2 | 4 | 23 | 16 | 7 | 3 |
| 3 | 4 | 23 | 14 | 9 | 0 |
| 4 | 4 | 23 | 19 | 4 | 0 |

| CV fold | TRAIN clips / components | inner clips / components | outer clips / components |
|---:|---:|---:|---:|
| 0 | 69 / 12 | 25 / 5 | 25 / 5 |
| 1 | 71 / 13 | 23 / 4 | 25 / 5 |
| 2 | 73 / 14 | 23 / 4 | 23 / 4 |
| 3 | 73 / 14 | 23 / 4 | 23 / 4 |
| 4 | 71 / 13 | 25 / 5 | 23 / 4 |

두 coarse bin은 모든 TRAIN/inner/outer에서 존재한다. fine 4-bin coverage는 다음과 같으며 희소한 길이 구간을 채우려고 hull/day 격리를 약화하지 않았다.

| outer fold | 120–<300 | 300–<600 | 600–<900 | 900–1380 |
|---:|---:|---:|---:|---:|
| 0 | 19 | 5 | 1 | 0 |
| 1 | 20 | 5 | 0 | 0 |
| 2 | 16 | 3 | 1 | 3 |
| 3 | 14 | 8 | 1 | 0 |
| 4 | 19 | 4 | 0 | 0 |

따라서 ≥900초 구간은 5-fold 전체에 균일하게 존재하지 않는다. 이를 숨기거나 short clip을 native로 재명명하지 않는다. 이 manifest는 시간 warp/repeat로 길이를 바꾼 것이 아니라 실제 120–1380초 source만 선택한 것이다.

## 파일과 schema

확정 metadata 디렉터리: `data/versions/vtuad_reference_r4_v1/` (로컬 실제 경로는 `data.nosync/versions/vtuad_reference_r4_v1/`).

- `reference_manifest.parquet`: 선택된 119개 전체와 고정 outer assignment. base `split=pool`.
- `fold_00_reference_manifest.parquet` … `fold_04_reference_manifest.parquet`: 동일 119개를 `train`, `validation`, `outer_validation`으로 표시.
- `excluded_manifest.parquet`: 제거한 99개와 유지한 hull/day 및 제외 이유.
- `component_manifest.parquet`: 실제 component별 clip/길이/bin/outer assignment.
- `manifest.json`: 입력 SHA, 출력별 SHA, selection 목적과 상한, source hash, 길이 coverage, 제한 사항.

생성기 필수 열 `source`, `clip_id`, `split`, `reference_component`, `group_key`, `original_group_key`, `hull_id`, `recording_day`, `utc_begin/end`, `sha256`, `segment_file/start/count`, `duration_seconds`, `measured_duration_seconds`, `sample_rate_hz`, `header_sample_rate_hz`, `verified`, `measured`, `quality_full_band_good`, `reference_eligible`, `role`, `provenance`를 포함한다. `parent_reference_component`, `parent_outer_fold`, `original_split`도 provenance로 보존한다. source/hull/day ID는 split isolation용이며 acoustic model 입력으로 쓰지 않는다.

`source_hash`:

```text
694cf18c9daabc69a77807f6e0ede33ca451443fbfb17d754ac191f0206463ef
```

## 독립 감사 결과

[r4_reference_audit.json](r4_reference_audit.json)의 `passed=true`를 확인했다.

- 5개 fold 모두 component/hull/day/SHA/실제 UTC overlap cross-split 위반 **0**.
- 실제 selected graph count **22**, day/hull matching 상한 **22**.
- 99개 실제 제외 clip이 모든 R4 split에 **0개**.
- 원 native 3개 보존, 원 reserved/excluded split 재편입 **0개**.
- 선택과 5-fold assignment를 metadata-only로 재실행했을 때 clip/component/fold 배정이 정확히 일치.
- 원본 2초 bank header는 `87163 × 25`, float32이며 모든 선택 offset/count가 범위 안에 있음. 수치 waveform/energy 값은 읽지 않음.
- 전체 525-corpus graph는 **1**임을 별도로 재확인. selected 직접 격리가 전체 corpus의 물리적 독립성을 증명한다는 해석은 금지.

AIS 자동 scenario class와 provider/checksum/header/QC 검증은 사람이 확인한 vessel-only ground truth와 다르다. 원 recording의 ambient sound와 미검증 pressure calibration을 유지해서 해석해야 한다.

## 재현

새 빈 metadata 버전으로만 생성한다. 기존 디렉터리가 있으면 생성기가 덮어쓰지 않는다.

```bash
python -m pipeline.vtuad_reference_r4 --output data/versions/vtuad_reference_r4_replay --seed 42 --minimum-components 20
python ops/audit_vtuad_reference_r4.py --metadata-dir data/versions/vtuad_reference_r4_replay
```

구현: `pipeline/vtuad_reference_r4.py`. 독립 감사: `ops/audit_vtuad_reference_r4.py`. metadata 설정 snapshot은 `configs/vtuad_reference_r4_v1.json`이다. 생성된 파일 SHA가 manifest와 다르거나 reserved/제외 clip이 포함되면 학습 데이터 생성 단계로 넘기지 않는다. 모델·sampler·SA 배경 정책은 이 source helper가 수정하지 않는다.
