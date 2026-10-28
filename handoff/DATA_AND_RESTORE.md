# 데이터 설명, 합성·분할, 출처와 복원

## 1. 대회 Task와 원본

대회: https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge
자료/규정: `/data`, `/rules`, `/discussions` 페이지를 현재 시점에 다시 확인한다. 원본은 로그인·참가 동의가 필요한 계정 자료이며 공개 재배포 허용이 확인되지 않았다.

남아공 관측 1,715,779행은 자연 바다의 소음 관측이다. 원본 소리에 실제 선박이 있더라도 ‘인위적으로 추가한 성분’의 정답은 없다. 15분×25대역(375셀) 창으로 정리하고 VTUAD의 알려진 에너지를 추가하면 추가량 정답을 계산할 수 있다. 합성 데이터 건수는 WAV 수나 독립 실제 15분 녹음 수가 아니다. 서로 겹치는 배경·공유 참조·Fold별 재합성으로 통계적으로 종속된 창이 포함된다.

Test는 1,500창, 원시 WAV 없이 375개 dB 관측 열과 ID가 제공된다. 일부 분이 결측이다. 출력은 전체 관측 에너지 중 추가 성분의 비율 share와, 추가 유무의 순위 점수 score이다. RMSE가 주요 평가이며 공개 UI의 Score는 1−Share RMSE, score AUC는 별도 진단이다.

| 원본 파일 | 로컬 bytes | 용도 |
|---|---:|---|
| data/raw/South_African_Real_Acoustics.zip | 94,916,719 | 1,715,779 관측 추적 원본 |
| data/raw/Test.csv | 2,829,614 | 마지막 추론용 관측 1,500창 |
| data/raw/sample_submission.csv | 49,541 | schema / ID 확인 |
| data/raw/maritime_starter_notebook.ipynb | 10,782 | 공식 minute_band_power helper |

이 경로는 원본 프로젝트의 data→data.nosync 심볼릭 링크 아래다. 공개 checkout에는 실제 파일이 없다.

## 2. 외부 데이터와 보유량 ≠ 현재 투입량

- VTUAD v1.3.2: https://zenodo.org/records/23037895 . 공개 레코드의 라이선스·checksum·registry를 먼저 확인한다. 기존 증거 문서에는 CC BY 4.0으로 확인돼 있다.
- strict vessel509 + strict background16 =525 WAV를 확보한 이력이 있다. vessel split archive part01~03, background ZIP, 선종별 아카이브는 중복/동일 원천 여부를 구분한다. 아카이브 수를 독립 원천 수로 세지 않는다.
- strict vessel archive 크기: part01/02 각1,237,445,900bytes, part03 1,237,445,899bytes. background926,037,631bytes. 원 레코드 `strict_interval_wav_SHA256SUMS.txt`, `registry_anonymized.csv` 대조.
- HearMyShip 1,146 WAV 확보 이력. ScienceDB DOI 10.57760/sciencedb.13099, 과거 검토 CC0. 초단기 클립 편중 문제로 현재 주력 경로에서는 제외.
- NOAA/SanctSound 일부 장시간 음원 및 평가 패널 확보 이력. ONC 파트너별 사용 조건이 달라 전체 파일 재배포를 가정하지 않는다. 현재 주력 데이터는 아래 VTUAD A′ subset이다.
- DeepShip/ShipsEar는 탐색했지만 현재 실행에 신규 확보·투입되지 않았다. 남아공 AIS가 확인되지 않아 AIS 기반 순수 선박 정답 추출도 하지 않았다.
- 외부 출처 공지/공식 게시 이력은 관련 draft·보고서에 보존했다. 게시 여부를 추측해 완료 처리하지 말고 사용자의 실제 게시 증거를 확인한다.

상세 근거: `reports/R8_EXTERNAL_DATA_EVIDENCE.md`, `reports/R6_PUBLIC_TRAINING_PROVENANCE_CATALOG.md`, 다운로드 스크립트 `fetch_vtuad_pilot.py`, `fetch_external*.py` 및 관련 inventory.

## 3. 현재 R16의 정확한 범위

**A′75클립·13원천 연결 그룹**을 사용한 nested5Fold다. B19/C′25를 전역 독립 검증에 남긴다. 보유509전체를 학습했다는 뜻이 아니다.

배경도 1,715,779행 전체를 새로 80:20한 것이 아니라 기존 R9 Fold0 TRAIN78,869창·206그룹 내부를 다시 분할했다. Fold별 61,446 / 61,368 / 61,259 / 61,546 / 61,685 train창, 조건당 합계307,304개.

| 항목 | D0 | H0 | 합계 |
|---|---:|---:|---:|
| TRAIN (5Fold 파일 합계) | 307,304 | 307,304 | 614,608 |
| CV (1,000×5) | 5,000 | 5,000 | 10,000 |
| 기본 합계 | 312,304 | 312,304 | 624,608 |
| 공유 local_fit | — | — | 추가2,400 |
| 저장 샘플 총수 | — | — | 627,008 |

본체 train/CV NPZ 약495,452,836bytes (약0.495GB; cache 제외). 단순 nominal 15분을 곱한624,608×0.25=156,152시간은 **중복을 포함한 합성 창 시간 합계**이고, 독립 녹음 시간/음원 확보량이 아니다. 실제 관측 분수도 결측 때문에 더 작다. 외부 확보 150~220시간은 과거 목표치였으며 달성한 사실이 아니다.

local_fit2400개는 400개 고유 배경×6연산자다. 원천 분리/900초 support/24시간 purge를 검사한다. 시점·센서·참조 그룹 단위 분리 후 합성한다. 일반 randomKFold로 이미 혼합된 627,008행을 나누면 누수 위험이 있다.

## 4. 처리 순서

1. 남아공 원본 행·QC·사이트·센서·run·시각을 추적. 경계 의심 분은 창 전체 삭제 대신 분 마스크로 처리하는 개편을 거쳤다. 현재 정확한 입력 lineage는 manifest SHA가 기준이다.
2. 배경 원천 그룹을 나누고 900초 창 support 기준24h purge. 고정 local_fit과 train/CV의 원천 충돌 검사.
3. VTUAD 원본을 공식 helper와 일치하는 2초 Hann PSD/25band 적분 후 **1분 선형 평균을 먼저** 만든다. uncalibrated 참조는 절대 음압 교정치로 보지 않는다.
4. 참조 원천/overlap component를 나눈 뒤 배치. D0는 주로 peak 포함1~6분 실제 연속 crop, H0는 ≤10분 전체 또는 긴 음원의11~15분 연속 crop. 가짜 반복/평탄 resampling/강제 highshare는 없다.
5. 6개 사전 정의 forward 연산자가 LF/HF 독립 세기, smooth shelf/tilt, scaling 순서 차이를 생성. 무추가20%는 D0/H0 동일 배경/마스크/정답으로 pair.
6. 선형 공간 B+A로 E, q=A/E, y=ΣOA/ΣOE 계산. 분모는 관측 셀만. 자연 배경의 선박형 과도음도 추가A없으면 y=0.
7. 입력db를0.1dB로 반올림. 실제 q/y 정밀도 보존. roundedE로 w 정규화. 이 때문에 Σwq와 rawy는 미세하게 다르며 허용gap0.0058 검증.
8. NPZ 저장→전수 재읽기→범위·NaN·presence·energy identity·hash·split검증. 학습 cache는CPU에서 만들고 MPS에는 가벼운 tensor를 전달.
9. 17채널 상대 특징과 회귀/독립 분류를 학습. 절대db/100, q0/Bhat, 학습PCA잔차/LFanchor는 배제. 상대profile/sqrtw는 회귀에 남아 있어 임의 EQ 불변을 주장하지 않는다.

수식·17채널 상세·모델 크기는 `reports/R16_INFERENCE_POSTPROCESSING_REVIEW.md`를 읽는다.

## 5. 복원: 같은 Mac

원본은 이동·삭제하지 않았다. code 공개 clone은 검토용, 실제 학습은 기존 canonical root를 계속 사용한다. `python3 handoff/doctor.py`가 현재 root/data/external/state를 확인한다. Git에 존재하지 않는 파일은 `handoff/LOCAL_ARTIFACT_INVENTORY.json`에 원래 경로/크기를 남겼다. exact current manifest 사본은 `handoff/data_manifests/manifest.json`에 있다.

## 6. 복원: 다른 Mac / GPU 서버

1. 이 repo clone. 데이터 없는 상태에서 `run_* all`을 실행하지 않는다.
2. 대회 계정으로 정식 raw파일 다운로드, 원본 SHA를 기존 manifest/증거와 대조. Zindi쿠키/토큰을 Git에 기록하지 않는다.
3. 외부 음원은 공식 레코드 checksum을 검증해 별도로 확보. rawdataset 및 **현재 고정 파생 데이터/패널**은 허가된 비공개 저장소/SSH 복사로 이관한다. 인터넷에서 다른버전 파일을 받아 같은 이름만 붙이면 안 된다.
4. 현재 생성기는 R9/R16production 및 support_temporal의 부모 manifest/Parquet/NPZ lineage를 의존한다. 공식 raw4개만 있다고 즉시 동일 R16이 재생성되지 않는다. `manifest.source_artifact_sha256`, `source_identity`, `R9Sources` 참조를 따라 요구 파일을 확보한다.
5. 공개 Release의 직접학습 모델을 별도 디렉터리에 내려받아 SHA/entry검증. dataset/panels는 이 archive에 없으므로 모델만으로 재학습은 안 된다.
6. CPU/MPS 환경 분리 설치. CUDA 이전은 별도 버전으로1step/수치동등성/속도/메모리 확인 후 본학습한다. 현재 run의 경로/코드hash를 조용히 덮어 바꾸지 않는다.

복원 불가능 항목을 데이터 ‘생성 완료’라고 보고하지 않는다. 새 환경의 portability setup은 제공하지만 current frozen runner가 모든 OS에서 무수정 실행됨을 보장하지 않는다.
