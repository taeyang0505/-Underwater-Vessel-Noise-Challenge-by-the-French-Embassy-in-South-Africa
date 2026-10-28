# R3 reference coverage 감사: 기존 5분 이상 VTUAD의 시간 변형 가능 범위

기존 파일만으로 **44개 실제 VTUAD passage를 15분 길이의 제어된 band-energy trajectory로 변형하는 후보**를 만들 수 있다. 추가 다운로드는 필요 없다. 다만 이것은 기존 관측의 형태 증강이며, **새 독립 녹음·선박·해역·장비 그룹을 추가하지 않는다.** 현재 CV에서 짧은 참조만 가진 두 하위 component를 긴 형태의 입력에 활용할 수 있다는 의미와 물리적으로 독립적인 원천이 늘어난다는 의미를 구분해야 한다.

이번 작업에서는 메타데이터와 TRAIN 원본 가용성만 감사했다. 추가 데이터 생성, 모델 학습, GPU 사용, 외부 전송, 공식 제출은 실행하지 않았다. Test 및 Test 통계, `TASK_EDA_REPORT`는 사용하지 않았다. R2의 데이터·모델·분할·평가 결과는 변경하지 않았다. 숫자와 개별 clip 목록, SHA, 검사 내역은 [기계 판독 facts](../ops/r3_reference_coverage_facts.json)에 저장했다.

## 1. 현재 원천과 길이별 후보

현재 source CV pool은 1,171 clips / 26개 연결 component다.

| 소스 | clips | 현재 CV component | 실제 측정 ≥15분 positive clips |
|---|---:|---:|---:|
| HearMyShip V1 | 945 | 19 | 0 |
| VTUAD v1.3.2 | 218 | 3 | 3 |
| NOAA extension | 8 | 4 | 8 |
| 합계 | 1,171 | 26 | 11 |

실제 측정 ≥15분 참조가 있는 component는 5개다. HMS는 10–30초 excerpt이므로 이번 15분 확장 후보에서 전부 제외한다. 반복 재생, 주기적 타일링, 10–30초의 녹음을 15분 측정으로 표기하는 방식은 사용하지 않는다.

VTUAD의 CV 편입분 218개 중 ≥300초인 실제 passage는 47개다. 그중 3개는 이미 15분 이상이므로 기존 native crop을 유지하고, **300초 이상·900초 미만인 44개**만 새 시간 변형 후보가 된다. 기존 2초 band bank 기준으로는 `150 <= segment_count < 450`이다. 359초·419초처럼 끝에 1초가 남는 녹음은 기존 변환과 동일하게 완전한 2초 segment만 사용하므로 실제 사용 길이는 각각 358초·418초다. stretch는 WAV 표기 길이가 아니라 이 사용 길이로 계산한다.

| 현재 VT CV component | 포함 clips | ≥5분 clips | 새 warp 후보 | 실제 ≥15분 | hull/day 수 | 고정 outer fold |
|---|---:|---:|---:|---:|---|---:|
| `REF_eca948a5584d1f14fbea` | 208 | 45 | 42 | 3 | ≥5분 후보 기준 33 hull / 18일 | 4 |
| `REF_61e16af34294aaca010c` | 5 | 1 | 1 | 0 | ≥5분 후보 기준 1 hull / 1일 | 4 |
| `REF_8d0211ada01d043634eb` | 5 | 1 | 1 | 0 | ≥5분 후보 기준 1 hull / 1일 | 3 |

새로 긴 형태에 활용할 수 있는 짧은 두 component의 구체적인 원본은 다음과 같다.

| clip | 원본 member | hull / 녹음일 | 측정 길이 | 900초로 변형 시 stretch |
|---|---|---|---:|---:|
| `e18fdfa0d71c0ee1` | `vessel/111.wav` | H094 / 2016-07-06 | 360초 | 2.5 |
| `595437c742858b8e` | `vessel/400.wav` | H177 / 2016-11-10 | 300초 | 3.0 |

## 2. fold별 TRAIN에서 실제로 사용할 수 있는 수

원래 component 배정과 inner/outer를 고정한 상태의 계산이다. 각 fold에서 `split == train`인 참조만 후보로 삼았다.

| fold | TRAIN clips / component | 기존 실제 native clips / component | 새 VT warp clips / component | 현재 CV 표기상 native-like component 수 | 별도 VT same-sensor 24h까지 요구할 때 warp clips |
|---:|---:|---:|---:|---:|---:|
| 0 | 779 / 16 | 7 / 3 | 44 / 3 | 3 → 5 | 44 |
| 1 | 806 / 17 | 7 / 3 | 44 / 3 | 3 → 5 | 44 |
| 2 | 803 / 16 | 7 / 3 | 43 / 2 | 3 → 4 | 39 |
| 3 | 564 / 14 | 6 / 3 | 0 / 0 | 3 → 3 | 0 |
| 4 | 561 / 15 | 6 / 3 | 1 / 1 | 3 → 4 | 0 |

위 native-like 수는 **기존 CV의 하위 component를 어떤 길이의 제어된 입력으로 사용할 수 있는가**를 나타낸다. 전체 component 수 26과 실제 독립 원천 수는 증가하지 않는다. 한 후보당 uniform warp 한 개를 만든다는 가정의 긴 template 수는 `[51, 51, 50, 6, 7]`이지만, 이를 독립 표본 수나 실제 측정 15분 녹음 수로 세면 안 된다.

특히 fold 3에는 TRAIN VT 후보가 전혀 없다. 기존 raw만 이용한 이 증강으로 다섯 fold의 source coverage를 균일하게 늘릴 수는 없다. fold 4의 유일한 후보는 더 엄격한 참조 간 24시간 조건에서는 제외된다.

## 3. 직접 분할 제약과 전체 corpus 연결 그래프는 다른 검사다

선택된 R2 참조 1,171개에 대해 5개 fold를 다시 검사했다. 모든 fold에서 다음 **직접 cross-split 위반 수는 각각 0**이다.

- 원래 source group의 split 중복
- VT hull의 split 중복과 VT 녹음일의 split 중복
- WAV SHA-256의 split 중복
- 선택된 VT 녹음의 실제 UTC interval overlap
- NOAA station의 split 중복

따라서 이번 감사는 선택된 R2 bank에서 **직접 hull/day/interval/byte 격리 위반이 확인됐다는 결론을 내리지 않는다.** 기존 직접 제약은 통과했다.

별도로 미선택·제외·reserved 메타데이터까지 포함해 더 보수적인 연결 민감도를 계산했다. 전체 VT 525개를 hull/day/실제 interval/동일 bytes로 연결하면 한 component가 된다. ambient controls를 빼고 vessel archive 509개만 사용하거나, background/unknown class를 뺀 487개만 사용해도 한 component다. 이는 선택된 세 VT component가 **미선택 녹음을 경유해 전이적으로 연결될 수 있음**을 보여준다. 동일 배경과 선박의 반복 관측을 더 보수적으로 묶었을 때의 민감도이며, 선택된 녹음끼리 직접 중복됐거나 정답이 유출됐다는 증거와 동일하지 않다.

두 개 이상의 선택 component에 직접 이어지는 미선택 bridge recording은 39개다. 예를 들어 `aaba757ab2c1b3d3`는 원래 `excluded_hull_day_conflict`, H111, 2016-07-10의 vessel passage다. 이 녹음은 큰 component와는 같은 날로, `REF_8d0211ada01d043634eb`와는 같은 hull로 연결된다. 이 녹음 자체는 R2 TRAIN·inner·outer 어느 곳에도 새로 편입하지 않는다. 구체적인 연결 clip ID와 이유는 facts의 `unselected_direct_bridge_examples`에 있다.

따라서 새 warp의 효과는 **기존 훈련 원천의 시간 형태 다양화**로 설명하고, 두 component가 새로 물리적으로 독립적인 장시간 원천이 됐다고 표현하지 않는 것이 맞다. HMS도 반복 hull 정보가 없으므로 19개 day component를 hull 독립성 보증으로 바꿔 해석하지 않는다.

## 4. 24시간 조건의 정확한 범위

현재 `pipeline/group_cv.py`의 24시간 support purge는 SA 배경의 sensor/week 분할에 적용된다. VT reference 분할은 hull/day/bytes/실제 interval 연결을 사용하며, 모든 같은 hydrophone 녹음 사이에 24시간을 보장하는 코드가 아니다. 두 조건을 혼용하거나 R2가 검증하지 않은 보장을 소급해 주장하지 않는다.

추가로 같은 VT hydrophone의 현재 heldout 참조와 최소 24시간 간격을 요구하면, 새 warp 후보는 `[44, 44, 39, 0, 0]`이다. fold 2의 4개 후보는 heldout VT와 최소 간격이 약 2.22, 6.70, 13.58, 13.72시간이다. fold 4의 유일한 후보는 약 1.72시간이다. 이 숫자는 메타데이터로 계산한 더 엄격한 선택 옵션이며, R2 결과를 변경하거나 직접 누수로 판정하는 데 사용하지 않았다.

시간 변형 시에는 **원본 UTC interval을 그대로 보존**하고 별도의 synthetic time axis를 기록해야 한다. 실제 300초 녹음의 `utc_end`를 `utc_start + 900초`로 고쳐서는 안 된다. SA background의 24시간 purge 및 원래 clean-window index/order도 그대로 유지한다.

## 5. 원본 가용성과 품질 확인

세 vessel ZIP part는 모두 로컬에 있고, `MultipartReader`로 이어 읽을 수 있다. 별도 합본 ZIP이나 새 다운로드가 필요 없다. ≥5분 후보 47개는 모두 fold 0의 TRAIN 참조다. 이 47개 WAV만 읽어 **member CRC, 전체 WAV SHA-256, provider per-WAV SHA ledger, 실제 header와 길이를 재검증**했고 전부 통과했다. 읽은 원본 합계는 1,393,794,068 bytes다. calibration/sealed/기존 OC02 평가 녹음의 raw 또는 배열은 열지 않았다.

47개 모두 PCM16 / 32,000Hz이며 25개 대역의 최대 상단 약 8,912.51Hz를 지원한다. `large_DC_fraction_review`는 47개 모두에 있다. rail fraction은 0이고 1초 이상의 digital-zero run은 없다. 기존 band conversion은 2초 Hann periodogram, `detrend='constant'`, density의 실제 주파수-bin 적분, base-10 대역 경계, 30개 segment의 선형 power 평균을 사용한다. DC flag를 없던 것으로 처리하거나 calibrated pressure·순수 선박 신호로 해석하지 않는다. 제안은 이미 de-mean 처리된 선형 band-power trajectory를 변형하는 방식이며 raw waveform의 단순 resampling/pitch shift를 제안하지 않는다.

CV에 없는 VT 307개를 이번에 다시 편입하지 않는다. 그 안에는 calibration 6개, domain holdout 15개, hull/day conflict 260개, cross-day conflict 1개와, 원 TRAIN/validation에 있었지만 ambient/모호한 vessel class로 참조 자격이 없는 25개가 포함된다. ≥5분이라는 이유만으로 excluded vessel 60개, reserved domain vessel 3개, 모호한 원 TRAIN vessel 3개 또는 긴 ambient controls를 positive 참조로 되살리지 않는다.

전체 기존 reference bank 기준 reserved는 HMS calibration 144 / domain 56, VT calibration 6 / domain 15, NOAA calibration 2 / sealed 2다. NOAA calibration은 `oc03_02_20191108T010700Z`, `oc03_02_20200306T042200Z`, sealed는 `sb01_01_20181112T190100Z`, `sb01_01_20190127T225600Z`다. 기존 OC02의 이미 평가한 세 recording과 PM NoShip quarantine도 후보가 아니다. 이번 작업에서는 이들의 메타데이터/제외 이력만 확인했다.

## 6. 다음 전처리 코드 계획 — 아직 실행하지 않은 제안

첫 후보는 **fold 0 TRAIN에 이미 있는 44개 원본만** 사용하는 controlled uniform time warp다. fold 0에는 CV의 VT inner/outer 참조가 없으므로 위 별도 같은-sensor reference 24시간 조건도 44개 모두 통과한다. 검증과 outer는 기존 파일을 byte-for-byte 유지한다. 이 후보를 pilot으로 실행할지와 혼합 비율은 root가 별도로 확정해야 한다.

1. 새 모듈 `pipeline/reference_timewarp_v3.py`에 `warp_band_power_uniform(P, target_segments=450, max_stretch=3)`을 추가하는 계획이다. 입력은 기존 TRAIN measured `P[K,25]`, `150 <= K < 450`이며 출력은 450개의 2초 power bins다. dB 보간, waveform pitch shift, clip 끝과 처음의 연결, zero padding, HMS 반복은 하지 않는다.
2. 각 대역의 누적 power 적분 `C_b(u)`를 원 2초 bin 경계에서 만들고, `u_j = j * (2K)/450`으로 원 trajectory를 한 번 연속 통과한다. 출력은 `(C_b(u[j+1]) - C_b(u[j])) / (u[j+1]-u[j])`이다. 이 방식은 uniform warp에서 대역별 평균 power를 보존한다. 시간 길이는 `s = 450/K`배이므로 총 power-time integral은 그에 따라 늘어난다. 원래와 같은 총 에너지가 유지된다고 잘못 주장하지 않는다. 5분 clip의 `s=3`은 단일 uniform map이며, 기존의 모든 coarse bin 사이에 새 독립 측정이 생긴 것이 아니다.
3. 원본 `ReferenceRecord`의 measured segments·source ID·component·hull/day·UTC를 바꾸지 않는다. R3 sampler subclass에서 별도의 `warp_eligible` 경로로만 이 함수를 호출하고 recipe kind는 `controlled_time_warped_passage_15min`처럼 구분한다. 기존 `native_contiguous_15min_crop`이나 `native_long=True`로 위장하지 않는다. `source_measured_duration`, `used_source_segments`, `derived_duration=900`, `stretch`, `parent_SHA`, `component`, `seed`, 변환 code/config SHA를 저장한다.
4. 450개 segment를 30개씩 평균해 15×25의 선형 참조 power를 만든다. 기존 reference-only smooth EQ와 common digital gain의 의미는 그대로 기록하고, 배경 `B`·관측 mask `O`와 결합한 뒤 기존 solver로 α와 실제 target을 다시 계산한다. `q=αR/(B+αR)`, `y=Σ(OαR)/Σ(O(B+αR))`를 새로 계산하고 20dB minute SNR ceiling, support cap, requested/effective share, float32 양자화 검사를 유지한다. 원래 target을 증강 입력에 그대로 복사하지 않는다.
5. 새 config와 새 data version에만 연결한다. R2 NPZ/recipe/metadata/readiness/policies/checkpoints는 덮어쓰지 않는다. TRAIN의 SA clean indices, background source assignment와 24시간 purge는 변하지 않는다. 새로운 long-like route에서는 실제 측정 native와 derived warp를 별도 count로 감사하고, corpus/component를 먼저 균형 샘플링해 큰 VT component의 42개 후보가 두 작은 component를 단순 파일 수로 압도하지 않도록 한다.

필수 numerical/quality gate는 identity·상수 power·비음수·평균 power 보존, 450×25 shape, `s <= 3`, 원 source 구간의 단회 단조 통과, deterministic replay, 원본 미수정, 관측 mask 보존, saved `q/y` energy identity, positive→zero 0건, 20dB minute ceiling과 ULP-aware 검사, source ID/분할/중복·hull/day 제약 불변, heldout 모든 SHA 불변이다. 처음부터 time trajectory·EQ·샘플링 비율을 다수 탐색하는 계획이 아니라 한 가지 데이터 전처리 가설의 제한된 pilot 후보로 남긴다.

추가로 한 clip당 uniform warp 한 개의 2초 float32 cache를 저장하면 44개에 약 1,980,000 bytes, 1분 cache는 66,000 bytes다. 이는 cache 용량일 뿐 독립 데이터 증가량이 아니다. on-the-fly 재생성도 가능하다. 현재는 이 cache나 추가 dataset을 생성하지 않았다.

## 7. 공식 starter가 뒷받침하는 범위

로컬 [public starter의 zero-based cell 9](../data/raw/maritime_starter_notebook.ipynb)는 VTUAD를 minute-level band-energy로 변환하고 time trajectory, relative acoustic channel, energy를 바꿔 SA 배경과 결합했다는 변경 고지를 포함한다. 이 사실은 task 표현에 맞춘 시간 형태 증강을 검토할 근거다. **공식 생성기가 정확히 3배 stretch나 위 누적 적분 알고리즘을 사용했다는 근거는 아니다.** Test 분포나 성능으로 warp 범위를 역추정하지 않았다.

원 출처는 [VTUAD v1.3.2](https://doi.org/10.5281/zenodo.23037895), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), upstream [Ocean Networks Canada](https://www.oceannetworks.ca/)다. adaptation 고지와 attribution을 유지해야 하며, 이 licence가 SA challenge measurement에 새로 적용된다는 뜻은 아니다. 외부 데이터 공개 고지문은 기존 사용자 지시대로 초안만 보관하며 이번 감사에서 게시하지 않았다.
