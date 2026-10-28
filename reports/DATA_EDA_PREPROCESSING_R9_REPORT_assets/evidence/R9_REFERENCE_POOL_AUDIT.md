# R9 전체 strict VTUAD 참조 pool — 실제 구축·검증

사용자의 새 명시적 허가에 따라 **새 production 버전에서만** 기존 VT reference 예약/제외를 해제했다. 원본 R4/R5/R6 데이터·모델·패널·SA분할은 덮어쓰지 않았다. 새 pool은 **vessel-archive 측정509개 전체**를 원WAV bytes 기준으로만 중복 검사해509개를 유지하고, **dedicated ambient16개**를 별도 controls manifest로 보관한다. 추가 WAV 다운로드나 학습을 하지 않았다.

## 실제 산출물과 readiness

디렉터리: [vtuad_full_r9](../../../data/versions/vtuad_full_r9/manifest.json). `readiness.json.ready=true`와 독립 validator PASS를 확인했다. 이는 **측정 참조 pool 사용 준비 완료**이며 새 합성 generator/배경CV/모델 학습 전체 준비를 대신하는 gate는 아니다.

- 참고 bank: `data/processed/external_segments_power.npy` / shape `[87163, 25]` / dtype `float32` / SHA `026d57965c2bec7268eee2ffe37aa663eec50609452fbdbec1b503aa3128df52`.
- 새 source_hash: `d9148a5ae923d4713be5fd71294596e84394780c479d7ac763a8a50d095b50a5`.
- 509개 reference 모두 full25band finite·nonnegative·segment broadband>0,native최소1개 완전minute를 확인했다. no-minute0,원WAV byte중복0이다. 미관측 대역을 채우거나upsample/pad/repeat하지 않았다.
- reference header합 **119,388s**, 실제완전2s사용 **119,376s**, ambient header합 **31,980s**다. 합산시간은독립fieldrecording시간이아니다.
- 과거 R4선택119개/R6선택94개는전부포함되며 새개수차는각각390/415개다. 이를새로운물리녹음의수집으로세지않는다.

## 역할·약한 라벨·품질

| 기존 weak_class | vessel-archive refs/509 |
|---|---:|
| tug | 188 |
| cargo | 97 |
| passengership | 71 |
| other | 40 |
| fishing | 34 |
| pleasurecraft | 32 |
| background | 22 |
| dredger | 8 |
| sailing | 6 |
| tanker | 5 |
| rescue | 4 |
| pilotvessel | 2 |

archive_role=vessel_passage안의 **weak_class=background22개**도사용자all509지시로포함했다. 이를verified vessel-only파워로바꾸거나class명으로추가성분presence/energy label을정하지않는다. 실제추가된whole recording의알려진성분으로q/y를정해야한다. ambient16개도별도controls이지자동으로자연선박부재또는에너지0정답인것이아니다.

모든원래hull/day/UTC/providerWAVSHA/header길이·samplingfs·원래split을보존했다. 이전providerchecksum/원파형QC기록과이번convertedbank수치검증은구분한다. 525개 `large_DC_fraction_review` 경고를지우지않았다. detrend는압력calibration이나pure-vessel분리를보장하지않는다.

| 복원한기존split | vessel refs |
|---|---:|
| excluded_hull_day_conflict | 260 |
| train | 228 |
| domain_holdout | 11 |
| validation | 5 |
| calibration | 5 |

## 5개 background-CV fold에서 참조를 공유

새 user policy에서는모든509개가각fold TRAIN에들어갈수있다. `training_reference_schedule.parquet`는5×509=2545행이며각foldcycle에서509개모두정확히1회등장한다. `reference_cycle_order.npy[5,509]`와독립대조했다.

API: `reference_indices_for_epoch(reference_count,n_examples,epoch,fold,seed=20261026)`는fold별고정permutation을epoch마다중단없이이어가는**참조index cycling**이다. epoch당N을fold안에서고정하고전체epoch순서를한번생성한뒤minibatch로slice한다. N≥509면매epoch모든참조를쓴다. N<509면연속여러epoch를통해fullcycle을완료한다. 이는한합성event에서waveform을반복하는처리가아니다.

**reference 독립5fold가아니다.** 전체525corpus의연결graph는1이며모든fold가참조bank를공유한다. SAbackground분할과24hpurge는별도부모builder에서유지해야한다. 이pool로훈련한새모델에대해과거VTreferencecalibration/domain/outer기반평가는오염된개발비교이므로독립holdout/OOF라부르지않는다. 원래파일이변한것이아니라새모델의훈련접근범위가달라진것이다.

## schema와 무결성

- reference/ambient manifest는`power_cache=segment_file`로원bank를가리키고`segment_start/count`로slice한다. `duration_seconds`는원header길이,`measured_used_seconds`는완전2s사용길이다.
- `expected_segment_slice_sha256`은원float32배열의C-contiguousrawbytes SHA다. `segment_array_identity_sha256`은canonical(shape,dtype)JSON+같은rawbytes SHA다. 원WAV SHA와구분한다.
- `clip_quality_manifest.parquet`는525개각각의valid2s/minute/per-band관측count와sliceSHA를보존한다. class/hull/day는provenance이지모델target이나예측입력이아니다.
- `duplicate_WAV_bytes_manifest.parquet`는이번중복0행을명시한다.

[독립검증JSON](../../R9_REFERENCE_POOL_AUDIT.json) · [readiness](../../../data/versions/vtuad_full_r9/readiness.json) · [새pool코드](../../../pipeline/reference_pool_r9.py) · [builder/검증](../../../ops/build_reference_pool_r9.py)
