# Underwater Vessel Noise Challenge — 연구·실행 인계 저장소

**인계 기준: 2026-10-28T15:02:23.505861+09:00 (Asia/Seoul).** 이 저장소는 진행 중인 로컬 프로젝트의 공개 가능한 스냅샷이다. 최신 작업은 **R16 D0F1 + H0 50:50**이며, 과거 R4/R5/R9/R14 문서의 계획을 현재 완료 상태로 읽으면 안 된다.

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

코드, 설정, 보고서, EDA 표·그림·PDF, 제출 후보/이력, 검증 영수증을 Git에 보존한다. 직접 학습한 체크포인트·Tree 모델은 [Release 백업](https://github.com/taeyang0505/-Underwater-Vessel-Noise-Challenge-by-the-French-Embassy-in-South-Africa/releases/tag/handoff-2026-10-28)으로 분리했다. [모델 SHA 목록](handoff/OWNED_MODEL_RELEASE_MANIFEST.json)으로 검증한다.

**대회 원본 ZIP/Test 관측값, 외부 WAV, 합성 행렬·학습 캐시, 다운로드한 타사 사전학습 가중치, 개인키·토큰은 공개 저장소에 넣지 않는다.** 따라서 새 clone만으로 원자료가 복원되거나 현재 학습이 재개되는 것은 아니다. 동일 Mac에는 원자료가 보존돼 있고, 다른 장비에서는 [복원 문서](handoff/DATA_AND_RESTORE.md)의 비공개 데이터 이관/정식 다운로드가 필요하다.

과거 첫 시도부터의 자료를 넓게 보존했기 때문에 일부 문서에는 폐기된 모델·수치·가설이 있다. 현재 지시와 충돌하면 `CLAUDE.md` → 최신 실행 계약/코드 → 해당 실험 원문 순서로 확인한다. 공개 자료 및 코드의 제3자 권리는 각 원출처의 라이선스를 따른다. 이 저장소 전체에 임의의 재라이선스를 선언하지 않았다.
