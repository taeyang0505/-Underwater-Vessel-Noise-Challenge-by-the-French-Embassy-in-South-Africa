from pathlib import Path
import datetime, json, hashlib, os, shutil

SOURCE=Path('/Users/hongtaeyang/Documents/ChatGPT/대회/underwater_vessel_noise.nosync')
OUT=SOURCE.parent/'work/underwater-vessel-noise-handoff'
NOW=datetime.datetime.now().astimezone().isoformat()
REPO='https://github.com/taeyang0505/-Underwater-Vessel-Noise-Challenge-by-the-French-Embassy-in-South-Africa'

def put(name,text):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text.strip()+'\n')

put('README.md',f'''
# Underwater Vessel Noise Challenge — 연구·실행 인계 저장소

**인계 기준: {NOW} (Asia/Seoul).** 이 저장소는 진행 중인 로컬 프로젝트의 공개 가능한 스냅샷이다. 최신 작업은 **R16 D0F1 + H0 50:50**이며, 과거 R4/R5/R9/R14 문서의 계획을 현재 완료 상태로 읽으면 안 된다.

## Claude / 다음 작업자가 가장 먼저 읽을 파일

1. [CLAUDE.md](CLAUDE.md): 현재 사용자 지시, 금지 사항, 실제 작업 위치.
2. [현재 진행 상태와 다음 행동](handoff/CURRENT_STATE.md): 진행 중 학습 / 완료 / 대기 구분.
3. [환경 설치·실행](handoff/ENVIRONMENT.md), [중단·복구 절차](handoff/RESUME_AND_RECOVERY.md).
4. [데이터·출처·복원](handoff/DATA_AND_RESTORE.md), [실험 이력과 공식 성능](handoff/EXPERIMENT_HISTORY.md).
5. [현재 추론·후처리 상세 검토](reports/R16_INFERENCE_POSTPROCESSING_REVIEW.md).
6. [남은 제한 및 검증 원칙](handoff/KNOWN_LIMITATIONS.md).

### 현재 완료와 미완료

| 항목 | 기준 시점 상태 |
|---|---|
| 현재 R16 D0/H0 데이터 구축·전수 검증 | 완료 |
| Tree 회귀·독립 분류 | D0/H0 각각 5-Fold 완료: 회귀 10개 + 분류 10개 |
| [Tree 전용 CSV](submissions/r16_multisupport_tree5_r1/submission_r16_tree5.csv) | 생성·독립 검증 완료, **공식 점수 미측정** |
| 본체 신경망 4종 × D0/H0 Fold0 | 진행 중. 현재 상태는 아래 스냅샷 및 원본 state를 확인 |
| 첫 5종 결합 CSV | 신경망 Fold0 종료 후 생성 예정 |
| 신경망 Fold1~4 | 첫 CSV 검증을 기다리는 후속 러너 등록 완료 |
| 전체 5-Fold 결합 CSV | 미생성 |

**실제 현재 상태는 원본 Mac의 state 파일이 기준이다. GitHub 파일은 자동 갱신되는 라이브 모니터가 아니다.**

```bash
python3 handoff/doctor.py
# 다른 위치에 원본 프로젝트가 있으면
python3 handoff/doctor.py --root /path/to/underwater_vessel_noise.nosync
```

## 직접 볼 수 있는 EDA / 시각화

- [대회 Task 및 심층 EDA](TASK_EDA_REPORT.md)
- [데이터 전체 설명·R9 구축 과정](reports/DATA_EDA_PREPROCESSING_R9_REPORT.md)
- [EDA 그림 폴더](eda_deep/figures), [요약 표 폴더](eda_deep/tables)
- [파이프라인 그림 폴더](figures)
- [현재 학습 화면 스냅샷](reports/r16_multisupport_full_r1/START.html)
- [전체 Fold 후속 실행 화면 스냅샷](reports/r16_multisupport_fivefold_r1/START.html)
- [그림·PDF·HTML 바로 찾기](handoff/VISUAL_ARTIFACT_INDEX.md)

GitHub에서 HTML은 소스로 표시될 수 있다. 로컬에서 `python3 -m http.server 8000 --bind 127.0.0.1` 실행 후 `http://127.0.0.1:8000/handoff/START.html`을 열면 시각 자료를 볼 수 있다. 일부 역사적 HTML의 데이터 링크는 공개에서 제외된 파일이라 열리지 않으며, 누락 목록에 이유를 남겼다.

## 공식 평가에서 확인된 이력

| 제출 | Public Score | Share RMSE | Score AUC |
|---|---:|---:|---:|
| 공개 Benchmark | 0.753867825 | 0.246132175 | 이 저장소에서 확정하지 않음 |
| R5 CNN | 0.651991320 | 0.348008679 | 0.539533333 |
| R9 Tree 5-Fold | 0.768128675 | 0.231871324 | 0.548088888 |
| **R9 Fold0 all5 (기존 최고)** | **0.782464193** | **0.217535806** | **0.555066666** |
| R14 step1 | 0.705436197 | 0.294563802 | 0.497866666 |

R16의 로컬 개선은 공식 LB 개선으로 아직 검증되지 않았다. `1 − 로컬 RMSE`를 예상 LB 점수로 표현하지 않는다.

## 공개 업로드 범위

코드, 설정, 보고서, EDA 표·그림·PDF, 제출 후보/이력, 검증 영수증을 Git에 보존한다. 직접 학습한 체크포인트·Tree 모델은 [Release 백업]({REPO}/releases/tag/handoff-2026-10-28)으로 분리했다. [모델 SHA 목록](handoff/OWNED_MODEL_RELEASE_MANIFEST.json)으로 검증한다.

**대회 원본 ZIP/Test 관측값, 외부 WAV, 합성 행렬·학습 캐시, 다운로드한 타사 사전학습 가중치, 개인키·토큰은 공개 저장소에 넣지 않는다.** 따라서 새 clone만으로 원자료가 복원되거나 현재 학습이 재개되는 것은 아니다. 동일 Mac에는 원자료가 보존돼 있고, 다른 장비에서는 [복원 문서](handoff/DATA_AND_RESTORE.md)의 비공개 데이터 이관/정식 다운로드가 필요하다.

과거 첫 시도부터의 자료를 넓게 보존했기 때문에 일부 문서에는 폐기된 모델·수치·가설이 있다. 현재 지시와 충돌하면 `CLAUDE.md` → 최신 실행 계약/코드 → 해당 실험 원문 순서로 확인한다. 공개 자료 및 코드의 제3자 권리는 각 원출처의 라이선스를 따른다. 이 저장소 전체에 임의의 재라이선스를 선언하지 않았다.
''')

put('CLAUDE.md',r'''
# Claude 인계: 여기서 바로 이어받기

## 최우선 사실

이 저장소는 2026-10-28 진행 중인 연구의 **공개 스냅샷**이다. 원본은 아래 위치에 있으며, 사용자는 현재 학습을 유지하고 Fold0 CSV 생성 후 나머지 Fold1~4까지 완료하라고 승인했다.

```
/Users/hongtaeyang/Documents/ChatGPT/대회/underwater_vessel_noise.nosync
```

이 경로의 원본이 살아 있다면 공개 clone에서 새 학습을 시작하지 말고 먼저 `python3 handoff/doctor.py`로 원본 state·PID·완료 영수증을 읽어라. GitHub snapshot의 PID와 epoch는 옛 값일 수 있다. 사용자에게 완료라고 보고하기 전에 실제 파일과 SHA를 확인한다.

## 마지막으로 확정된 사용자 지시

1. R16 **D0F1 + H0 50:50** 결합으로 본체 학습과 CSV 생성 진행. 장시간/HF 파일럿 RMSE 0.117553이 목표 0.11 미달임을 설명했고, 사용자가 미달을 명시한 진행을 승인했다. 이 관문만으로 다시 멈추지 않는다.
2. Tree 전체 Fold가 끝나면 Tree만의 CSV를 먼저 생성. **이미 완료**: `submissions/r16_multisupport_tree5_r1/submission_r16_tree5.csv`.
3. 신경망 4종의 Fold0가 끝나면 Tree 5-Fold와 결합한 첫 CSV를 만들고, 이후 신경망 Fold1~4를 이어서 학습한다. 후속 waiter가 이미 실행 중이다. 중복 실행 금지.
4. 추론 개선/후처리 검토 MD도 완료: `reports/R16_INFERENCE_POSTPROCESSING_REVIEW.md`.
5. 사용자가 직접 검토·제출할 것이다. 현재 러너는 **CSV 생성만**, 공식 업로드·토론 게시·Gemini 외부 전송을 하지 않는다.
6. 현재 신경망은 **100 epochs / Adam / dropout 0.1 / no early stopping**, 최적 회귀·분류 checkpoint 별도 저장. 과거 100epoch/ES7 계약은 이후 사용자 지시로 변경됐다. LightGBM은 별도 boosting 계약 1500trees / ES40이다.
7. 사용자 요청으로 GitHub에 상세 인계 환경을 만든다. 원본 학습 파일을 이 공개 checkout으로 이동하지 않는다.

## 수정 금지 / 정확성

- 실행 중 `pipeline/training_r16_multisupport*.py`, 데이터/모델/특징 모듈, ops 감독·최종화 모듈, frozen policy/PLAN/manifest의 내용을 바꾸지 않는다. 프로세스가 시작할 때 코드 SHA를 동결했다. 수정이 필요하면 별도 버전에서 검증 후 사용자 승인 범위에 맞게 전환한다.
- 학습 중단·PID 종료·모델/데이터 삭제·Pod 종료를 임의로 하지 않는다. 중복 감독 프로세스도 만들지 않는다.
- 비밀키·토큰·전체 환경변수·인증 파일을 읽거나 출력하지 않는다. 원자료를 공개 Git에 추가하지 않는다.
- **현 모델은 509개 전부를 쓰지 않는다. A′75클립/13그룹 안의 nested CV**다. 보유량·독립 원천 수·합성 샘플 수를 구분한다.
- 동일 창/원천이 Fold 간 혼입되지 않도록 source group + 900초 support + 24h purge를 유지한다. local_fit에서만 정책 적합, CV/locked panel은 평가용. Test 전체 통계나 LB 점수에 맞춰 합성 분포·후처리를 반복 조정하지 않는다.
- noES와 최적 checkpoint 저장이 global minimum 보장을 뜻하지 않는다. 로컬 AUC/분포 일치가 실제 LB를 보장하지도 않는다.
- 보고서의 ‘가설’이나 Gemini의 설명을 확인된 사실로 승격하지 않는다. 특히 Test 숨은 정답 share는 알 수 없다.

## 지금 점검할 파일

```
ops/r16_multisupport_full_r1/state.json
ops/r16_multisupport_full_r1/STARTED.json
models/r16_multisupport_full_r1/progress.json
models/r16_multisupport_full_r1/ALL_TREES_COMPLETED.json
ops/r16_multisupport_fivefold_r1/state.json
ops/r16_multisupport_fivefold_r1/PLAN.json
ops/r16_multisupport_fivefold_r1/CONNECTION_VERIFICATION.json
```

현재 첫 CSV 예정: `submissions/r16_multisupport_full_r1/submission_r16.csv`.
전체5Fold 예정: `submissions/r16_multisupport_fivefold_r1/submission_r16_5fold.csv`.
신경망 순서/현재 epoch는 live progress로 확인한다. 완료 전 예정 파일을 완료 산출물로 링크하거나 점수를 주장하지 않는다.

## 중단 후 재개 주의

`latest.pt`에는 optimizer/scheduler/RNG가 저장되지만 **현재 partial run을 이어가는 CLI는 구현·검증되지 않았다**. `all`을 재실행하면 기존 산출물 때문에 거부되는 것이 정상이다. checkpoint가 있다는 이유로 처음부터 덮어쓰거나 best.pt를 resume으로 쓰지 않는다. 정확한 복구가 필요하면 [RESUME_AND_RECOVERY](handoff/RESUME_AND_RECOVERY.md)의 검증 절차를 따라 별도 resume 구현을 준비한다.

## 상세 문서

- [CURRENT_STATE](handoff/CURRENT_STATE.md)
- [ENVIRONMENT](handoff/ENVIRONMENT.md)
- [DATA_AND_RESTORE](handoff/DATA_AND_RESTORE.md)
- [EXPERIMENT_HISTORY](handoff/EXPERIMENT_HISTORY.md)
- [KNOWN_LIMITATIONS](handoff/KNOWN_LIMITATIONS.md)
''')

put('AGENTS.md','''
# Agent instructions for this handoff checkout

Read CLAUDE.md and handoff/CURRENT_STATE.md before doing work. This is an exported snapshot, not the active training directory. Do not start a second training process, change frozen running source/data/policies, or treat snapshot state as live. Preserve exact code bytes because artifact ledgers pin source SHA. New utilities/docs may be added separately. Never publish raw competition data, audio, synthetic matrices, vendor checkpoints or credentials. No official competition upload, public messaging, or external consultation without a current direct user instruction. Confirm outputs and hashes before reporting completion. Follow handoff/RESUME_AND_RECOVERY.md for interruption recovery; no tested partial-run resume CLI currently exists.
''')

put('.gitignore',r'''
# Raw / licensed datasets and derived training matrices are private local artifacts.
/data/
/data.nosync/
/external/
/external.nosync/
/models/
/scratch/
/outputs/
/delivery/
/ops/server_backups/
/ops/r14_production_runtime_py/
*.wav
*.flac
*.mp3
*.zip
*.tar.gz
*.pt
*.pth
*.safetensors
*.npy
*.npz
*.parquet
*.pkl
*.joblib
*.log
__pycache__/
*.pyc
.DS_Store
.venv*/
.env
.env.*
!.env.example
id_rsa*
id_ed25519*
credentials*
*.pem
''')

state=json.loads((SOURCE/'ops/r16_multisupport_full_r1/state.json').read_text())
waiter=json.loads((SOURCE/'ops/r16_multisupport_fivefold_r1/state.json').read_text())
put('handoff/CURRENT_STATE.md',f'''
# 현재 상태 — {NOW}

**스냅샷이며 실시간 표시가 아니다.** 원본 프로젝트를 사용할 수 있다면 다시 읽어야 한다.

## 실행 상태

```json
{json.dumps(state,ensure_ascii=False,indent=2)}
```

후속 상태: `{waiter.get('phase')}`, PID `{waiter.get('PID')}`.

원본 첫 감독 PID 12773, NN child 21489, 후속 waiter 23279는 이 문서 작성 당시 값이다. PID는 재사용될 수 있으므로 `ps -p PID -o pid,ppid,etime,command`와 실제 command를 대조한다.

## 완료한 일

- paired D0/H0 최종 데이터 구축·전수 NPZ 재읽기·원천 분리·purge 및 물리 정답 검사.
- D0/H0 각각 Tree 5-Fold 회귀·독립 분류. 총 20개 booster.
- Tree 전용 1500행 CSV 생성, 10개 member 평균 일치·ID·범위·SHA 독립 검산.
- Fold0 4신경망×2조건 실행 및 이어지는 Fold1~4 32개 job 연결 구현.
- 후속 연결 CPU 실데이터 step/checkpoint reload, 40개 고유 모델 경로 및 첫 CSV barrier 검증 통과.
- 추론·후처리 리뷰 MD 작성.

## 지금 할 일 / 자동 후속

1. 학습이 살아 있으면 그대로 둔다. NN 각 100epoch 끝날 때 별도 best regression / best classifier 보존.
2. Fold0 8회 학습이 모두 끝나면 frozen policy 적합 → CV/locked panel 평가 → Test 추론 → 첫 CSV 검증.
3. 후속 waiter가 parent 완료 + 첫 CSV SHA/1500unique/schema + Fold0 모델 완료 ledger를 확인하고 Fold1~4를 시작한다.
4. 각 후속 Fold는 cache → 8회 NN → 완료 검증 → 해당 Fold의 검증된 파생 cache만 제거. 원자료·모델은 보존한다.
5. 40개 NN 학습이 모두 끝나면 아키텍처별 5Fold 평균, capped regression 결합, 독립 classification 결합으로 최종 CSV 생성.
6. 로컬 결과, 패널별 RMSE/AUC/bias와 후보 예측분포를 보고한다. 공식 제출은 자동으로 하지 않는다.

## 이미 사용 가능한 최신 CSV

`submissions/r16_multisupport_tree5_r1/submission_r16_tree5.csv`

SHA256 `589bbee56ce9c8432aabd0da174ad040dd261410a84788293992cafd9d029490`

- share 평균 0.2478607800, RMS 0.3104906998, 중앙값 0.2135238303.
- score 평균 0.8969556912, 중앙값 0.9855341819.
- score≥0.90: 1127/1500 = 75.1333%.
- 확률 보정이 검증된 ‘90% 확신’이라는 뜻은 아니다. 포화 경향의 기술 통계일 뿐 정확도/AUC를 알 수 없다.
- 이 CSV의 Public 점수는 아직 없다.

## 소요시간

역사적 Mac 측정 기반 계획 범위는 Fold0 완료까지 대략 24~40시간, 전체 5Fold 대략 120~200시간이었다. 현재 실제 남은 시간은 epoch throughput, 완료 모델 수, 발열·메모리·sleep을 재측정해야 한다. RTX4090 35~60시간은 현재 paired17채널 전 과정의 확정 실측이 아니다. GPU 대여·학습 전환을 자동으로 하지 않는다.
''')

put('handoff/DATA_AND_RESTORE.md',r'''
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
''')

put('handoff/EXPERIMENT_HISTORY.md',r'''
# 실험 이력과 교훈 — 숫자는 측정 대상과 함께 읽기

## 대회 공통

Share 회귀 RMSE가 주 평가. score는 추가유무 순위AUC. 자연 선박 유무/선종 분류와 다른 문제다. Test 숨은y는 없으며 로컬 syntheticy와 다른 target distribution·conditional mapping이 가능하다. 상수 benchmark점수 하나로 Testy모멘트/양성률을 역산하면 안 된다. adversarialAUC가0.5라도 label조건부분포 동등성을 보장하지 않는다.

## 시간순 요약

| 버전 | 주요 변경/발견 | 상태·해석 |
|---|---|---|
| 초기/R1 | 원본1715779행 EDA, 외부VTUAD/HearMyShip/NOAA 확보, 합성/Tree·smallNN·foundation 비교 | 외부자료 추가흔적·길이·도메인 편중 가능성 |
| R1 foundation | TabICL outer query-batch 재현성gate실패 | 실패partial을 정상완료로 재사용 금지 |
| R2 | CNN, FDY/iTransformer변형, TabM,Tree,TabICL/Kumo 등9후보 | CNNOOF0.238545보다9모델ensemble0.245585 악화. train의support/SNRcap와inner/out생성규칙불일치가 확인됨 |
| R3/R4 | VTUAD중심동일recipe, source/purge, 297837건 train267837+inner15000+outer15000 | 5Fold파일수합계이며 독립297837 WAV가 아님 |
| R5 | 8채널WideCNN,raw/quantization/missingpilot,본체GPU | localINNER RMSE0.090808,AUC0.99672 vs officialRMSE0.348009,AUC0.53953. 검증대표성붕괴 |
| R6 | 상대입력,절대db/q0제거,분마스크/site복원,교차센서/생성기파일럿 | 교차0.110407,일부0.059377은 각각다른패널. LB미측정;0.89LB로환산불가 |
| R7/R8 | 제출경로재계산,starter변환/source선언검토 | 변환일치와 실제생성기동일성은다름. 경쟁자의출처선언만으로모델/recipe모두알수없음 |
| R9 | strictWAV풀확장,minute-first,safeensemble,4NNFold0+Tree5 | all5제출은 Tree도Fold0:0.782464193 /RMSE0.217535806/AUC0.555066666. Tree별도5Fold는0.768128675 |
| R10 | score구성·관측밀도·셔플·참조AB·2×2site/generator·ambient진단 | starter수식차이약1.88e-7,단순저주파site전이/참조AB만으로붕괴재현안됨. 시간셔플AUC거의동일:순서의존낮음 |
| R11 | fullsupport양성vsHF자연음성,LF/HF비대칭,저장Test출력포화,StFrancis밀도 | 일부스트레스에서AUC붕괴재현. 이것이비공개Test의실제구성임을증명하진않음 |
| R12 | 장시간/마스크일치HF교체/PCAablation,미니CNN+Tree | 고SNR스트레스개선·classifiercontrast포화감소. H1mask교란발견→HM1대조군 |
| R13 | highshare교체,독립classifier,calibration/목적함수비교 | highshare가일반bias악화. minifeature673 vs본체5252차이,분류회귀분리검토 |
| R14 | 본체14채널·PCA/LFanchor·Tree5·MacMPS | local17패널개선에도step1official0.705436197/RMSE0.294563802/AUC0.497866666로악화. CNN81.05%편중. TFGrid중단후storedpredaudit |
| R15 | forward조건sensitivity/독립score/규정검토 | Test통계를강제목표로해서generator를fit하지않는경계정리. marginalbump차이만으로Test정답확정불가 |
| R16 forward | LF/HF독립변형및다중연산자,깨끗한상대입력 | smallmix3축RMSE:0.08899→0.06774,0.08256→0.05802,0.10323→0.08568. 로컬조건별개선 |
| R16 support/clean | naturalclip길이,HFreplacement누수/결합대조,F1시간지속성 | 합성flat15만으로recipe를선택하지않음. 최종D0F1+H0.5/.5,미달native12/HF0.117553 vs0.11공개후진행승인 |
| 현재R16multisupport | A′75/nested5Fold/17ch/Tree5160reg2033clf/100epAdam | 데이터·Tree완료,NNFold0진행. 이후Fold1~4자동후속대기. 새LB미측정 |

## 공식 제출 결과 및 가중치

| 버전 | Score | ShareRMSE | scoreAUC |
|---|---:|---:|---:|
| 초기CPU제출 |0.658591231|0.341408769|이번자료에서확정안함|
| R5rawCNN |0.651991320|0.348008679|0.539533333|
| R9Tree5 |0.768128675|0.231871324|0.548088888|
| R9Fold0all5 |0.782464193|0.217535806|0.555066666|
| R14step1 |0.705436197|0.294563802|0.497866666|

R9 all5 order `[bg_cnn, tfgrid_fdy_conformer, fdy, itransformer, lgbm_physics]`, weights `[.1,.1,.1,.1,.6]`。5モデル全てFold0。Tree単独5Fold提出と同一モデルではないので差をNNだけの効果と断定しない。

R14step1 WideCNN81.05%/Tree18.95%。新R16はTree≥50%/各NN≤25%に制約し、scoreはshareと独立。古い良いローカルfitにより新Test良化を断定しない。

## 調べる文書

詳細な数値・条件・否定結果は`reports/`の各R番号MD、`reports/r16_*`のMD/JSONに保全。特に `R5_CURRENT_ISSUES_AND_GAP_REPORT.md`, `R10_LOCAL_VALIDATION_GAP_DIAGNOSTICS.md`, `R11_PINPOINT_DIAGNOSTICS.md`, `R12_CPU_ABLATION_RESULTS.md`, `R13_CPU_CORRECTION_RESULTS.md`, `R14_STEP1_PUBLIC_RESULT_REVIEW.md`, `FEATURE_FAILURE_REVIEW.md`, `R16_INFERENCE_POSTPROCESSING_REVIEW.md` を参照。

過去の推測を「原因が完全に規明」と断言した引用も履歴として残すが、当該条件の対照実験で分かったことと公式Test原因の同定を分ける。現在も全domain gap解消/LB0.90を保証できない。
'''.replace('。','.' ))

put('handoff/ENVIRONMENT.md',r'''
# 実行環境・可搬性 / 실행 환경

## 1. 現在使われている環境（현재 실제 환경）

原本コードは2つのPythonを明示指定する。公開cloneで既存runnerの定数を変更しても原本学習には反映されない。原本の凍結コードは変更禁止。

| 用途 | CPU / データ・Tree | MPS / NN |
|---|---|---|
| Python | 3.12.14 arm64 | 3.14.5 arm64 |
| numpy | 2.3.5 | 2.5.4 |
| torch | 2.14.1 | 2.14.1 |
| torchvision | 不要 | 0.29.1 |
| device | CPU | Apple Metal MPS |
| 硬体 | Apple M4 Pro, GPU20cores, unified24GB | 同じMac、メモリはOS/CPU/GPUで共有 |

CPU interpreter:
```
/Users/hongtaeyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```
CPU環境:
```
PYTHONPATH=.:/private/tmp/underwater-ml-deps:/private/tmp/underwater-eda-deps
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
OPENBLAS_NUM_THREADS=1
```
MPS interpreter:
```
/Users/hongtaeyang/Documents/Codex/2026-10-27/new-chat/.venv/bin/python
```
MPS環境:
```
PYTHONPATH=.:ops/r14_production_runtime_py
PYTORCH_ENABLE_MPS_FALLBACK=0
```
`ops/r14_production_runtime_py`は一時的なpure-Python依存shim。第三者package約1977ファイルは公開にvendoringしない。新環境ではrequirementsから正式installする。MPS環境にSciPy/LightGBMがあると仮定しない。cache/TreeはCPU環境。

## 2. 新しい環境を用意する

```bash
# macOS CPU Tree依存。既に導入済みなら変更しない
brew install libomp
bash handoff/setup_env.sh cpu /path/to/new-cpu-venv /path/to/python3.12
bash handoff/setup_env.sh mps /path/to/new-mps-venv /path/to/python3.14
```

setupは依存導入だけ、学習は開始しない。pinはこのMacで観測したバージョンの再現記録。新プラットフォームのwheel提供・将来のインストール成功は保証しない。wheel取得失敗時は黙って別versionへ変更せず別lockとfixtureで検証する。pipfreeze/metadata原本は`handoff/environment/`。

CPU追加実測: pandas2.2.3/pyarrow26.0.0/scipy1.18.1/scikit-learn1.9.1/lightgbm4.7.0/soundfile0.14.0/matplotlib3.11.2/joblib1.6.0。

```bash
/path/to/new-mps-venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.backends.mps.is_built(), torch.backends.mps.is_available())
PY
```

MPS sandbox検査でunavailableでもOSレベルの本プロセスは利用可能な場合がある。学習ログ/実機1stepを根拠にする。CPUfallbackは無効の契約。

## 3. 実行コマンドは新規実行専用

現在のrunは既に動いている。**以下を重複実行しない。** 原本監督が段階を順次起動する:

```
ops/run_r16_multisupport_full.py all
```

stage=`build,trees,cache,preflight,neural,finalize`。CPUとMPSを自動で使い分けるがinterpreter定数はこのMac固有。後続 `ops/run_r16_multisupport_fivefold.py` のCLIは `--help`/source参照、既存waiterを再起動しない。

CUDA環境への無変更切替は未検証。MPSloader/torchopをCUDA対応版にする場合は既存runから分離しcodepins/数値同等性/epoch性能を測る。CUDA依存foundation/Kumo/TabICL用の旧requirements_gpu_cv.txtは現在最小環境ではない。

## 4. ファイルキャッシュと容量

24GB unifiedはRTX4090の24GB専用VRAMとは別。CPU多並列とMPS並列でメモリ競合する。現在1NNずつ。disk空きは作成時約12GiBで、後続runnerはcache作成前6GiB以上を要求し、各Fold完了後にhash検証済み派生cacheだけを削除する。原本/synthetic/modelを手動削除しない。

## 5. Macのsleep

WiFi断だけならローカルNN/Treeはネット不要で継続。sleep/蓋閉じではOS設定次第で停止・一時停止しうる。`caffeinate -i`だけでclosedlid挙動を保証しない。ユーザーはAmphetamineを自分で設定した経緯があるが、現在設定を勝手に変えない。sleepから戻ってprocessが生存しているならそのまま観測し、初期化しない。
''')
# Korean replacement to avoid mixed-language handoff writing.
put('handoff/ENVIRONMENT.md',r'''
# 실행 환경과 새 장비 준비

## 실제 사용 중인 환경

| 항목 | CPU 데이터·Tree | MPS 신경망 |
|---|---|---|
| Python | 3.12.14 arm64 | 3.14.5 arm64 |
| numpy | 2.3.5 | 2.5.4 |
| torch | 2.14.1 | 2.14.1 |
| torchvision | 불필요 | 0.29.1 |
| 장치 | CPU | Apple Metal MPS |

하드웨어는 Apple M4 Pro GPU20코어 / 통합메모리24GB. OS·CPU·GPU가 메모리를 공유하며 RTX4090의24GB전용VRAM과 다르다.

CPU interpreter:
```
/Users/hongtaeyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```
CPU 환경:
```
PYTHONPATH=.:/private/tmp/underwater-ml-deps:/private/tmp/underwater-eda-deps
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
OPENBLAS_NUM_THREADS=1
```
MPS interpreter:
```
/Users/hongtaeyang/Documents/Codex/2026-10-27/new-chat/.venv/bin/python
```
MPS 환경:
```
PYTHONPATH=.:ops/r14_production_runtime_py
PYTORCH_ENABLE_MPS_FALLBACK=0
```

현재 MPS runtime에는 SciPy/pandas/sklearn/LightGBM이 있다고 가정하면 안 된다. CPU가 featurecache를 만들고 MPS가 tensor를 사용한다. `ops/r14_production_runtime_py`는 임시 pure-Python dependency shim이며 제3자파일약1977개는 공개에 vendoring하지 않았다. 새 환경에는 정식 requirements로 설치한다.

CPU 추가 실측 패키지: pandas2.2.3, pyarrow26.0.0, scipy1.18.1, scikit-learn1.9.1, lightgbm4.7.0, soundfile0.14.0, matplotlib3.11.2, joblib1.6.0. 별도 파일에 버전/환경을 기록한다.

## 새 환경 설치 (현재 학습 환경을 건드리지 않음)

```bash
# macOS에서 이미 설치됐으면 생략
brew install libomp
bash handoff/setup_env.sh cpu /path/to/new-cpu-venv /path/to/python3.12
bash handoff/setup_env.sh mps /path/to/new-mps-venv /path/to/python3.14
```

setup은 새 venv 의존성만 설치하며 학습을 시작하지 않는다. pins는 관측 버전 기록이다. 다른OS/아키텍처의 wheel 제공이나 미래 설치 성공을 보장하지 않는다. 설치 실패시 버전을 몰래 바꾸지 말고 별도lock/fixture로 수치검증한다.

```bash
/path/to/new-mps-venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.backends.mps.is_built(), torch.backends.mps.is_available())
PY
```

샌드박스에서 MPS unavailable이어도 실제 감독 프로세스에는 사용 가능할 수 있다. 학습 로그와 실제1step을 근거로 판단한다. CPUfallback0 계약을 유지한다.

## 현재 러너 명령 / 중복 실행 주의

현재 run은 이미 실행 중이므로 아래를 다시 실행하지 않는다.

```
ops/run_r16_multisupport_full.py all
```

감독의 stage는 build→trees→cache→preflight→neural→finalize. CPU/MPS를 선택하지만 interpreter 절대경로가 이Mac에 고정돼 있다. 새 환경에서는 versioned portability 변경과 연결fixture가 필요하다. 원본실행의 frozen코드 상수를 수정하면 안 된다.

후속 `ops/run_r16_multisupport_fivefold.py`는 첫CSV barrier를 기다리는 waiter가 이미 실행 중. CLI는 source/`--help`로 확인하고 중복 시작하지 않는다. 옛 `requirements_gpu_cv.txt`의 TabM/TabICL/Kumo환경은 현재 최소환경이 아니다.

CUDA 이전은 자동 지원으로 간주하지 않는다. 별도버전으로 CUDA연산호환/정밀도/캐시동등성/1epoch 속도/메모리를 먼저 확인한다. 현재 paired17채널 전체 실행의 RTX4090시간은 확정 실측이 아니다.

## 디스크·메모리

작성시 원본디스크 여유 약12GiB. 후속runner는 Foldcache 생성전6GiB이상 요구하고 해당Fold8모델완료 후 hash검증된 파생cache만 제거한다. 원본/NPZ/model을 수동삭제하지 않는다. CPUTree/MPS동시대규모작업은 통합메모리경합 위험이 있으므로 새 병렬학습을 추가하지 않는다.

## WiFi / 덮개 / sleep

WiFi단절만으로는 로컬 학습이 멈추지 않는다. OSsleep이나 덮개닫힘은 설정에 따라 실행이 일시정지될 수 있다. `caffeinate -i`만으로 closedlid 동작을 보장하지 않는다. 사용자가 Amphetamine을 직접 설정한 이력이 있지만 현재상태를 추측하거나 몰래변경하지 않는다. 돌아온후 process가 살아있다면 재초기화하지 말고 진행log를 확인한다.
''')

put('handoff/RESUME_AND_RECOVERY.md',r'''
# 작업 중단·재접속·재개 절차

## A. 대화/IDE만 바뀌고 학습이 살아 있음

1. `python3 handoff/doctor.py`로 canonicalroot의 live state를 읽는다.
2. statePID의command를 `ps`와 대조. 프로세스가살아있으면 새학습을 시작하지 않는다.
3. 최근 epochlog·progress.json과bestcheckpoint파일mtime를 확인. 같은epoch가오래면rows_seen/optimizer_steps가움직이는지 본다. TFGrid는다른모델보다느리다.
4. 다음모델로넘어가는정상경계를기다린다. 새로운agent가 frozen코드를포맷/정리하지않는다.
5. 보고서를갱신할때live값과snapshot시각을구분한다.

읽기전용 예시 (원본 root에서):
```
cat ops/r16_multisupport_full_r1/state.json
cat models/r16_multisupport_full_r1/progress.json
cat ops/r16_multisupport_fivefold_r1/state.json
ps -p 12773,21489,23279 -o pid,ppid,etime,command
```
PID는스냅샷값. 새state의PID를확인한다.

## B. 첫Fold0가완료됨

- `models/r16_multisupport_full_r1/final/COMPLETED.json` 및 첫CSV 존재/SHA/1500unique/schema/range를확인한다.
- 후속waiter가이미있으면그대로둔다. parentcomplete만보고수동후속실행하지않는다.
- 후속`handoff()`는원본코드/weights/data핀과Fold0각100epoch완료ledger를검사한다.
- 이후32회NN(4아키텍처×D0/H0×4Fold)을독립초기화한다. Fold0best로warmstart하지않는다.

## C. 프로세스가실제로죽었음 / partial checkpoint

**현재는 tested resume CLI가 없다.** `latest.pt`에optimizer/scheduler/RNG가있어도runner의`all`은partial폴더가있으면거부한다. 이는부분학습을새완료로오인하지않기위한설계다.

1. 즉시자동재시작하지않는다. FAILED/log/lastcheckpoint/manifest/코드SHA/환경을읽기전용snapshot으로보존한다.
2. latest.pt의run/regime/model/fold/epoch/globalstep/optimizer/scheduler/RNG/checkpointdata/sourcepin을대조한다. pickle형식은신뢰하는직접학습파일만로드한다.
3. 원본파일수정없이새버전에`--resume-from`와resume출처ledger를구현한다. epoch경계인지 mid-epoch인지와sampler순서/cacheddata일치부터확인한다. 저장정보가부족한경계는정확재개라고주장하지않는다.
4. deterministic소형fixture에서중단없는Nstep과중단/재개Nstep의weights·optimizer·scheduler·RNG·best선택이같은지확인한다. MPS비결정성은허용오차/재현성한계로보고한다.
5. 완료모델은기존SHA로검증하고재학습하지않는다. 미완료모델은검증된resume또는명시적새버전fresh학습으로처리한다. 기존폴더덮어쓰기/COMPLETED위조/partialbest를완료로승격금지.
6. 새복구정책이codehash계약을바꾸므로별도runID/receipt/필요한사용자확인을남긴다. 원본완료ledger는수정하지않는다.

## D. 다른장비로옮김

Release는직접학습weights snapshot이고데이터/학습캐시가아니다. [DATA_AND_RESTORE](DATA_AND_RESTORE.md)의비공개데이터복원이먼저다. 원본sourcepins는bytes기준이므로공개문서링크수정은괜찮아도학습코드포맷변경은금지. archivedcheckpoints의running/completed구분은ledger에있다.

## 완료판정

Tree완료만으로최종전체5Fold완료가아니다. NN4종×2조건×5Fold의각100epoch완료,회귀/분류bestcheckpoint,ownfoldCV,lockedpanel,정책freeze,TestCSV/ID/SHA,최종receipt가필요하다. 각아키텍처의전체Fold평균은Test/local_fit용이고、CV에는해당Fold만사용해야누수가없다.
'''.replace('、',','))

put('handoff/KNOWN_LIMITATIONS.md',r'''
# 확인된 제한과 남은 위험

1. **공식일반화 미검증:** 최신R16의Public점수는없다. local3축개선·R11개선·분포통계가LB0.78/0.90보장은아니다. 과거R5와R14는로컬개선에도공식악화했다.
2. **음원다양성:** 현재A′75/13groups,전역heldB19/C′25. strict509전체아님. ≥15분독립그룹하나뿐,이를보류한Fold4train에는≥11분참조없음.
3. **NestedCV범위:** 기존R9Fold0TRAIN배경78869내부다. 전체남아공모든해역/센서를일괄5분할한검증이아니다.
4. **합성sample종속:** paired조건/반복배경/6연산자같은400fit배경을공유.2400fit샘플이2400독립관측아님.624608은독립실녹음624608개가아님.
5. **미달승인:** D0F1+H0.5/.5의native12/HF파일럿RMSE0.117553이0.11기준미달. 사용자가이를명시한본체진입을승인했다. 무한추가진단을이유로이미승인된본체를중단하지않는다.
6. **입력정보:** absolutelevel/q0/PCA/LFanchor는제거했지만relativeprofile/sqrtw에는정적대역분포가남는다. globaldBoffset불변성과arbitraryEQ불변성은다르다. q05/q20가항상순수배경을복원하는것도아니다.
7. **정밀도:** roundeddb의w와rawq/y에최대gap.0058허용. 정확항등식으로소개하면안된다. 데이터전수QA실측은관련JSON확인.
8. **독립분류포화:** 최신Tree의75.13%score≥.9. 출력이확률처럼보여도calibration미검증. 이를단독정상/실패판정으로쓰지않는다. score높음은share높음과별개.
9. **무제한최적화아님:**100epoch/noES/최적checkpoint는globalminima보장아님. validation반복선택으로발생하는과적합가능성은남는다.
10. **후처리범위:** 실제는clip+고정평균/cappedfit결합뿐. SAM·스무딩·backgroundPCAprojection·zero-gating·temperature/power보정/RAG는적용안됨. 미적용아이디어를실제적용으로말하지않는다.
11. **사전학습대형모델:** TimeMoE-200M/Dasheng는과거연결검증만,현재17채널본체학습에미포함. 큰모델별도GPU구매/학습시작은현재인계범위아님.
12. **재개·이식성:** 현재partialresumeCLI미검증,Mac절대interpreter경로,임시runtime/PYTHONPATH의존. 새clone에는데이터없음. setupscript만으로동일실행보장안됨.
13. **공개백업한계:** 진행중weights는한파일별원자적snapshot,여러checkpoint의전역동시시각snapshot아님. latest/best가서로다른epoch인것은정상. history와epoch검증후사용.
14. **디스크:** 원본작성시여유12GiB. 무분별한cache/데이터복사금지. 원본고정데이터를지우는우회금지.
15. **원자료권한:** 대회데이터재배포허용미확인. 공개repo는원본/합성행렬/audio/pretrainedvendorweights를제외했다. 파일목록·출처·복원절차를제공하며데이터가없어도실행가능하다고속이지않는다.
''')

put('handoff/requirements-cpu-observed.txt','''
numpy==2.3.5
pandas==2.2.3
pyarrow==26.0.0
scipy==1.18.1
scikit-learn==1.9.1
torch==2.14.1
lightgbm==4.7.0
soundfile==0.14.0
matplotlib==3.11.2
joblib==1.6.0
''')
put('handoff/requirements-mps-observed.txt','''
numpy==2.5.4
torch==2.14.1
torchvision==0.29.1
''')
put('handoff/setup_env.sh',r'''
#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "Usage: bash handoff/setup_env.sh cpu|mps NEW_VENV_PATH PYTHON_EXECUTABLE" >&2
  exit 2
fi
mode="$1"; task_venv="$2"; task_python="$3"
case "$mode" in cpu|mps) ;; *) echo "mode must be cpu or mps" >&2; exit 2;; esac
if [[ -e "$task_venv" ]]; then echo "Refusing to change an existing environment: $task_venv" >&2; exit 1; fi
"$task_python" -m venv "$task_venv"
"$task_venv/bin/python" -m pip install --upgrade pip
"$task_venv/bin/python" -m pip install -r "$(dirname "$0")/requirements-${mode}-observed.txt"
"$task_venv/bin/python" -c 'import platform, numpy, torch; print(platform.python_version(), numpy.__version__, torch.__version__); print("MPS",torch.backends.mps.is_available())'
echo "Environment only. Training was NOT started; restore private datasets and verify paths first."
''')
(OUT/'handoff/setup_env.sh').chmod(0o755)

# Own model summaries and exact receipts: no model arrays or opaque pickle.
for f in (SOURCE/'models').rglob('*.json'):
 if any(x in f.relative_to(SOURCE).parts for x in ('pretrained','__pycache__')) or f.stat().st_size>2_000_000:continue
 target=OUT/'handoff/model_metadata'/f.relative_to(SOURCE/'models')
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)

visuals=[]
for ext in ('*.pdf','*.png','*.svg','*.html'):
 for f in sorted(OUT.rglob(ext)):
  if '.git'not in f.parts:visuals.append(f.relative_to(OUT))
put('handoff/VISUAL_ARTIFACT_INDEX.md','# 시각화 / PDF / HTML 색인\n\n아래 파일은 공개 스냅샷에 포함됐다. HTML의 일부 데이터링크는 원자료 제외로 동작하지 않을 수 있다.\n\n'+'\n'.join(f'- [{p}](../{str(p).replace(" ","%20")})'for p in visuals))
put('handoff/START.html','''
<!doctype html><html lang="ko"><meta charset="utf-8"><title>Underwater Vessel Noise 인계</title>
<style>body{font:17px/1.7 system-ui;max-width:1050px;margin:40px auto;padding:20px;background:#f5f7fb;color:#182c45}section{background:white;border:1px solid #d9e3ef;border-radius:12px;padding:24px;margin:18px 0}a{color:#125a9f}strong{color:#942a24}</style>
<h1>Underwater Vessel Noise — 인계·EDA 자료</h1><p>2026-10-28 공개 스냅샷. <strong>라이브 학습 상태와 다를 수 있습니다.</strong></p>
<section><h2>현재 작업</h2><a href="../reports/r16_multisupport_full_r1/START.html">R16 Fold0 학습 스냅샷</a> · <a href="../reports/r16_multisupport_fivefold_r1/START.html">전체Fold 후속</a><p><a href="../README.md">README</a> · <a href="../CLAUDE.md">Claude 인계</a> · <a href="CURRENT_STATE.md">현재 상태 상세</a></p></section>
<section><h2>데이터·EDA·시각화</h2><a href="../TASK_EDA_REPORT.md">심층 EDA MD</a> · <a href="VISUAL_ARTIFACT_INDEX.md">전체 이미지/PDF/HTML 색인</a><p><a href="../eda_deep/figures/">EDA 그림</a> · <a href="../figures/">구조도</a> · <a href="../reports/">모든 실험 보고서</a></p></section>
<section><h2>바로 사용 가능한 후보</h2><a href="../submissions/r16_multisupport_tree5_r1/submission_r16_tree5.csv">R16 Tree5 CSV</a><p>공식 점수 미측정. 신경망 포함 최종 CSV는 아직 학습 후 생성 예정입니다.</p></section>
''')

put('THIRD_PARTY_AND_DATA_NOTICE.md','''
# Third-party and data notice

This repository preserves first-party competition research code, experiment reports, and generated visualizations. It is not a blanket license grant for third-party datasets or model weights. No repository-wide open-source license has been assigned without the owner's decision.

Competition data remains subject to Zindi/organizer terms. Raw or synthetic training matrices and audio are not distributed here. VTUAD, HearMyShip, NOAA/ONC and pretrained model projects retain their own licenses and attribution requirements. See handoff/DATA_AND_RESTORE.md and source evidence reports. Historical code imports optional vendor implementations; dependency availability does not imply their rights transfer.

The release archive contains directly trained project checkpoints and LightGBM boosters, not downloaded vendor pretrained checkpoints. PyTorch pickle checkpoints must only be loaded from trusted sources. Verify archive and entry SHA before use. Running checkpoint snapshots are not completed training claims. Private source files and secret tokens are not needed to read this repository.
''')

print(json.dumps({'docs_created':True,'snapshot_kst':NOW,'visual_artifacts':len(visuals)},ensure_ascii=False))
