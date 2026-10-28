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
