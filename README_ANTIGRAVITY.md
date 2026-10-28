# Antigravity에서 이 폴더 열기

**그림과 EDA를 보려면 먼저 `START_HERE.html`을 브라우저로 여세요.** 인터넷이나 Python 실행 없이 그림·표·보고서 목록을 볼 수 있습니다. 그림을 누르면 원본 크기로 열립니다.

Antigravity에서는 이 파일이 있는 **폴더 전체**를 `File → Open Folder`로 여세요. 이 폴더에 있는 `Underwater_Vessel_Noise.code-workspace`를 열어도 됩니다.

## 무엇을 열면 되나

| 보고 싶은 내용 | 파일 / 폴더 |
|---|---|
| EDA·시각화 종합 화면 | [START_HERE.html](START_HERE.html) |
| 최신 R4 데이터 규모·생성·검증 결과 | [reports/R4_DATA_SUMMARY.md](reports/R4_DATA_SUMMARY.md) |
| 최신 R4 실제 데이터 그림 | [reports/r4_visual_review](reports/r4_visual_review) |
| 원본 스키마·결측·심화 EDA 상세 | [TASK_EDA_REPORT.md](TASK_EDA_REPORT.md) |
| 원본 EDA 그림 7개 | [eda/figures](eda/figures) |
| 심화 EDA 그림 18개 | [eda_deep/figures](eda_deep/figures) |
| EDA 원천 표 | [eda/tables](eda/tables), [eda_deep/tables](eda_deep/tables) |
| 현재 모델 후보·크기·논문·진행 상태 | [MODEL_ROSTER_AND_STATUS.md](MODEL_ROSTER_AND_STATUS.md) |
| 이전 R2 본학습 보고서·곡선 | [reports/R2_FULL_PIPELINE_REPORT.md](reports/R2_FULL_PIPELINE_REPORT.md), [reports/r2_actual_curves](reports/r2_actual_curves) |
| 기존 데이터·전처리 상세 기록 | [DATASET_CONSTRUCTION_PREPROCESSING_REPORT.md](DATASET_CONSTRUCTION_PREPROCESSING_REPORT.md) |
| 이전 EDA 노트북 | [reports/eda_pipeline.ipynb](reports/eda_pipeline.ipynb) |

`reports/report.html`, 기존 노트북, 여러 옛 파이프라인 이미지는 이전 실험의 기록입니다. 현재 상태는 **R4 데이터 완료 / R5 CPU 통합 검증 완료·GPU 본학습 대기**입니다. 새 모델을 모두 학습했다는 의미가 아닙니다.

## 폴더 구성

```text
START_HERE.html                  ← 브라우저로 여는 첫 화면
MODEL_ROSTER_AND_STATUS.md        ← 최신 모델 후보와 확인된 규모
pipeline/                        ← 생성·입력 변환·모델·학습·추론 코드
configs/                         ← 생성·학습 설정 (버전별 보존)
data/ → data.nosync/              ← 같은 폴더 안의 실제 데이터에 연결
  raw/                           ← 대회 제공 ZIP·Test·sample·공식 starter
  extracted/                     ← 압축 해제한 원본 (있는 경로 그대로 보존)
  processed/                     ← 행 manifest·band energy·외부 음원 파생 배열
  versions/                      ← 원천 선택·버전별 metadata
  synthetic/group_cv_passage_r4_v1/ ← 현재 확정 R4 TRAIN/INNER/OUTER 5-fold
external/ → external.nosync/      ← 같은 폴더 안의 수집 외부 음원·출처
models/                          ← 과거 모델·공개 pretrained 가중치
ops/                             ← 실행·검증·백업 도구와 이력
ops/server_backups/               ← 서버 학습 산출물 ZIP·검증 기록
eda/, eda_deep/                   ← EDA 표·그림·분석 코드
figures/, reports/, evidence/     ← 도식·보고서·검증 근거
delivery/, outputs/               ← 과거 전달 패키지·추론 CSV
reference_materials/              ← 제공된 참고 이미지·텍스트 보관본
DESKTOP_PACKAGE_MANIFEST.json     ← 복사 파일별 SHA256 (패키징 시 생성)
DESKTOP_PACKAGE_VERIFICATION.json ← 원본과 복사본 비교 결과
```

`data.nosync`와 `external.nosync`는 **이 폴더 안에 실제 파일로 포함**합니다. 외부 Documents 폴더에 의존하는 데이터 바로가기만 둔 구성이 아닙니다.

## 현재 확정된 데이터

- 원본 남아공 관측 **1,715,779행**을 보존했습니다. 자연 선박 성분의 실제 정답 라벨은 없습니다.
- 선택 VTUAD **119클립 / 22개 연결 그룹**, 측정 원천 시간 약 **8.63시간**입니다. 22개는 선택 그래프의 분리 그룹이며 22개 독립 hydrophone을 뜻하지 않습니다.
- R4의 5-fold TRAIN 역할 합계 **267,837건**, INNER **15,000건**, OUTER **15,000건**입니다. 교차 fold에는 같은 학습 원천이 재등장하므로 모두 고유 관측으로 세지 않습니다.
- TRAIN/INNER/OUTER 생성 법칙을 일치시킨 **297,837건 전수 검증**을 마쳤습니다. 정규 NPZ 15개는 **414.2MiB**입니다.
- 현재 범위는 완료됐습니다. 미확보 150~220시간 외부 목표를 달성한 것으로 표시하지 않습니다. 제외 VTUAD 307개 재편입·남아공 AIS 추출은 하지 않습니다.

## 실행·이동 시 주의

1. 프로젝트 루트를 현재 작업 디렉터리로 두세요. `pipeline`은 로컬 패키지입니다.
2. Python 3.12를 기준으로 개발했습니다. 기본 의존성은 `requirements.txt`, 과거 GPU 환경 기록은 `requirements_gpu_cv.txt`와 `ops/`에 있습니다. R5 대형 백본은 별도 pinned runtime/공식 revision을 사용하므로 구형 설치 파일만으로 R5 전체 실행이 보장되지는 않습니다.
3. 옛 manifest·학습 ledger의 `/workspace/...`, `/Users/...` 경로는 당시 실행의 **역사적 provenance**입니다. SHA 검증을 보존하기 위해 일괄 치환하지 않았습니다. 현재 R4 CV 경로는 프로젝트 상대 경로입니다.
4. `run_gpu.sh`와 옛 delivery README는 이전 버전 실행입니다. 이를 최신 R5 본학습 명령으로 사용하지 마세요. 현재 명령은 reports/R5_INTEGRATION_READINESS.md에 있습니다. 최종 모델 선정과 성능은 GPU 본학습 후 확인합니다.
5. EDA 페이지는 읽기 전용입니다. 열기만 해서는 학습·다운로드·외부 게시·대회 제출을 실행하지 않습니다.
6. 데스크톱 폴더는 패키징 시점의 독립 복사본입니다. Documents의 원본도 보존했습니다. 두 복사본을 동시에 편집하면 이후 동기화가 필요합니다.

## 검토 화면 재생성

의존성 준비 후 프로젝트 루트에서:

```sh
python ops/build_local_review_portal.py
```

이 명령은 R4 TRAIN/INNER의 기술 통계와 기존 감사 결과로 그림·HTML을 만듭니다. 모델을 학습하거나 Test/OUTER 관측 배열을 읽지 않습니다.

최신 도식: [R5 후보 파이프라인](figures/pipeline_r5_candidate_v1.png). CPU 사전검증: [R5_TRAINING_PREFLIGHT.json](reports/R5_TRAINING_PREFLIGHT.json).
