# R6: 공개 관측 TRAIN의 duty cycle·결측·duration 처리 검토

이번 검토는 코드와 기존 메타데이터만 읽었습니다. 음향 관측 배열, Test, cross/OUTER 배열, 모델 예측을 읽지 않았고 학습·수정도 하지 않았습니다. 공개 대회 자료의 원천 관측 TRAIN과 우리가 정한 내부 `split='train'`은 구분합니다.

## 확인된 결론

**5분 관측 구간을 이어 붙여 15분 연속 관측으로 만들거나, 빈 시간을 보간·반복하여 채우는 처리는 없습니다.** 원래 시각의 15분 고정 창에 실제로 존재하는 분만 넣고 나머지는 NaN으로 보존합니다. 모델 특징에서 결측을 0과 mask로 표현하는 것은 원본 관측값을 복원했다는 뜻이 아닙니다.

**현재 R6 내부 TRAIN의 St Francis 창은 0개입니다.** R6 source 메타데이터에서 St Francis 17,380창은 모두 `cross_sensor` 역할(sensor group 4)입니다. 현재 R6 TRAIN에는 Richards Bay, St Croix Island, Bird Island, Durban, Cape Recife, Riy Banks가 있습니다. 따라서 아래 St Francis 실측 메타데이터는 공개 관측 자료의 기술적 확인이며, 현 R6 모델이 St Francis TRAIN을 학습했다는 근거가 아닙니다. 원래 `sa_window_manifest`에서는 St Francis가 sensor group 4의 내부 train 21,802창·validation 2,076창·calibration 3,276창으로 나뉘었습니다. 이를 최초 준비부터 domain holdout이었다고 설명하면 틀립니다. 현재 R6는 [source builder](../pipeline/source_split_r6.py) (L126)에서 부모 R4 fold0의 TRAIN/validation만 허용한 뒤 새 역할을 부여하므로, 현재 역할 배정과 원 V1의 역할 배정은 구분해야 합니다. 이번 검토에서는 St Francis 관측값이나 `cross_sensor` 관측 배열을 열지 않았습니다.

## St Francis: 공개 관측 메타데이터에서 확인한 것

메타데이터의 site는 `7 St_Francis`, recorder는 `SM2M_Cape_StFrancis_SM2M.1.48000`, 명목 sampling rate는 48,000 Hz입니다. export-clock 범위는 2015-04-21 00:00:00–2018-02-28 14:24:00입니다. 원시 waveform sampling rate·연속 녹음 여부를 이번에 다시 확인한 것은 아닙니다.

| 항목 | 메타데이터 실측 |
|---|---:|
| 원 관측 행 | 169,488 |
| 초기 적격·고정창 포함 행 | 169,487 |
| 포함 행의 duration=60초 | 169,486 |
| 포함 행의 duration=59초 | 1 |
| 제외된 duration=1초 행 | 1 |
| 보존된 15분 시계 창 | 27,154 |
| 창당 관측 5분 | 23,782 |
| 창당 관측 12분 | 1 |
| 창당 관측 15분 | 3,371 |

포함 행의 timestamp를 recorder 안에서 정렬했을 때, 인접 시각 차이 1분은 142,333회, 16분은 23,760회, 46분은 3,362회였습니다. 5개 연속 분 이후 다음 묶음까지 16분 차이가 나는 패턴과 15개 연속 분 이후 46분 차이가 나는 패턴에 부합하지만, 이것만으로 장비 운용의 모든 구간·원인·실제 녹음 길이를 인증하지는 않습니다.

| export-clock 연도 | 보존 창의 관측 분 수별 개수 |
|---|---|
| 2015 | 15분 2,630개, 12분 1개 |
| 2016 | 5분 8,623개, 15분 741개 |
| 2017 | 5분 13,747개 |
| 2018 | 5분 1,412개 |

이번 조사에서 찾은 2022년 문헌의 2015년 15/60분, 2016–2018년 5/20분 운용 설명은 이 패턴을 해석할 단서입니다. 다만 실제 공개 메타데이터에는 2016년의 15분 관측 창도 있으므로 연도 하나로 모든 행의 스케줄을 단정할 수 없습니다. `15/60=25%`, `5/20=25%`이며, 5/20을 20%로 표현하는 것은 산술적으로 맞지 않습니다. 문헌 내용만으로 공개 자료 각 행과의 동일성을 증명한 것은 아닙니다.

## 구체적인 코드 경로

- [prepare_sa.main](../pipeline/prepare_sa.py) (L42): 공통 25대역 finite, sentinel 제외, `55<=dur<=65`, timestamp 초 부분 `<=5`인 행을 약 1분 관측으로 선별합니다. 원본 행 자체는 따로 보존합니다.
- [같은 분 중복 선택](../pipeline/prepare_sa.py) (L57): recorder별 같은 분에서 duration이 60초에 가까운 행을 우선하고 시각·원 행 순서로 결정합니다. 여러 짧은 행을 합쳐 새 1분을 만드는 로직은 없습니다.
- [시계 창 구성](../pipeline/prepare_sa.py) (L63): 원시각을 15분 clock bin으로 내림하고, `[15,25]`를 전부 NaN으로 시작해 실제 분 위치에만 값을 넣습니다. [최소 관측 조건](../pipeline/prepare_sa.py) (L68)은 3분입니다. 관측 없는 창을 별도 가상 사례로 만들어 넣지는 않습니다.
- [source_split_r6.apply_minute_qc](../pipeline/source_split_r6.py) (L31): 원 배열을 복사하고 boundary bit에 해당하는 실제 분만 추가 NaN 처리합니다. 기존 NaN이 그대로인지 검사합니다. [source builder](../pipeline/source_split_r6.py) (L149)는 `raw_observed_minutes`와 `qc_observed_minutes`를 나눠 기록하며 QC 후 3분 미만 창은 제외합니다.
- [SourceSplitR6.load_backgrounds](../pipeline/source_split_r6.py) (L256): 요청 role에 허용된 원 window index만 읽고 위 minute mask를 적용합니다. 공백의 압축·시간 재배열·보간은 없습니다.
- [robust_sampler_r6.synthesize](../pipeline/robust_sampler_r6.py) (L130): 배경의 관측 mask `o`를 유지합니다. [reference 적용](../pipeline/robust_sampler_r6.py) (L134)은 관측 셀에만 더하며, [결과](../pipeline/robust_sampler_r6.py) (L142)는 원래 관측 셀만 dB로 변환하고 나머지는 NaN입니다. [검사](../pipeline/robust_sampler_r6.py) (L148)에서 mask 동일성과 missing q=0을 확인합니다. 자연 공백에 reference가 놓이면 관측 added share=0일 수 있고, 보이는 구간으로 다시 뽑거나 CPA를 재정규화하지 않습니다.
- [R4 비교군](../pipeline/passage_sampler_r4.py) (L194)도 배경 관측 mask를 유지합니다. R4의 명시적 timewarp는 외부 reference의 파생 궤적에 적용되며, SA의 5분 관측 공백을 메우는 작업이 아닙니다.
- [relative_grid의 시간 차분](../pipeline/relative_grid_r6.py) (L96)은 실제로 인접한 두 분이 모두 관측일 때만 계산합니다. 공백을 건너뛰어 앞뒤 관측을 이웃으로 만들지 않습니다. [특징과 가중치](../pipeline/relative_grid_r6.py) (L109)는 결측 특징=0, mask=0, energy weight=0으로 만듭니다. [고정 score](../pipeline/relative_models_r6_v2.py) (L31)의 시간 최솟값도 missing을 제외합니다.

## minute 집계와 duration의 한계

SA 관측은 이미 제공된 약 1분 대역 dB 요약입니다. 현재 코드가 SA 원시 waveform에서 다시 60초 PSD를 계산한 것은 아닙니다. `dur`는 입력 적격·중복 우선순위에 사용하며, 59초를 60초로 늘리거나 각 관측을 `dur/60`으로 재가중하지 않습니다. 통과한 행을 명목상 같은 1분 셀로 다룹니다.

합성 reference의 2초 band-power는 별도입니다. [R6 minute_first](../pipeline/robust_sampler_r6.py) (L81)는 30개의 완전한 2초 power를 선형 평균해 60초를 만들고 불완전 tail은 제외합니다. [2초 placement](../pipeline/robust_sampler_r6.py) (L68)는 450개 2초 셀의 창에서 30개씩 평균합니다. reference가 존재하지 않는 부분의 0은 알려진 added component A=0이며, 관측 배경의 결측 NaN을 채우는 0이 아닙니다. 이 둘을 혼동하면 안 됩니다.

현재 target/pooling은 관측된 명목 minute×band의 선형 power 합에 근거합니다. 실제 55–65초 duration 차이까지 곱한 적분 에너지를 복원했다고 주장할 수 없습니다. 원 exporter의 duration·대역 통계 의미와 waveform 원천 정합은 별도 확인 사항입니다. export clock timezone도 기존 준비 manifest에서 미검증 상태입니다.

## recorder 경고

[boundary_audit_v2](../pipeline/boundary_audit_v2.py) (L18)는 recorder별 연속 minute run을 나누고, 시작/끝 분의 broadband excess 중앙값이 3dB보다 클 때 후보를 표시합니다. 해당 후보는 확정된 장비 transient라는 뜻이 아닙니다. 기존 [boundary audit](data_quality_boundary_audit.json)의 St Francis recorder 결과는 27,154개 run, 시작·끝 중앙값 excess 각 0dB, flagged row 0개입니다. 이것이 waveform 품질 인증은 아닙니다.

[recorder flag 승계](../pipeline/build_physics_cv_v2.py) (L22)는 실제 창에 포함된 행의 boundary 후보만 사용하며, `URN_8321_8321.1.12000`에 대한 nominal 12kHz/상위 대역 경고를 별도 컬럼으로 보존합니다. 이는 St Francis의 48kHz recorder와 다른 장비입니다. 현재 R6 내부 TRAIN 메타데이터에서 URN warning은 4,185창이며, 해당 warning만으로 전체 대역을 삭제하지 않았습니다. 경계 후보의 물리 원인은 여전히 미검증입니다.

## 확인 범위와 근거 해시

아래 파일의 메타데이터·기존 보고서와 Python 소스만 읽었습니다. R6 `background_manifest`의 TRAIN 집계와 별도로 St Francis의 역할 컬럼만 확인했고, 부모 R4 source에서도 St Francis의 역할만 확인했습니다. St Francis의 원 공개 TRAIN은 row/window 메타데이터만 확인했습니다. 다른 역할의 관측값, 모델 예측, 새로운 synthetic 사례는 읽거나 만들지 않았습니다. 이 신규 보고서 외 파일 변경·학습·추론은 없습니다.

| 메타데이터 | SHA256 |
|---|---|
| `data/processed/sa_row_manifest.parquet` | `e05004b977bf7db91f131ffe2d87d944a135de48685591d484aa53294bf29de4` |
| `data/processed/sa_window_manifest.parquet` | `7a3425d40a160fb31da0b5c5fbcba1641dccefae8fdc8035a7ad84c232d0c413` |
| `data/versions/source_split_r6_guardrails_v2/background_manifest.parquet` | `fa534e3ba31279a2982be46e80e0a9cdf89be3755f72058747e7fdafda60a714` |

이 검토는 실제 운용 스케줄을 확정하거나, missing-mask 처리만으로 대회 평가 분포와 일치함을 입증하는 검증이 아닙니다.
