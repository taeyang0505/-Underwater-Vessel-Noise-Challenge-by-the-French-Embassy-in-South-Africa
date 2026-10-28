# Claude 인계: 여기서 바로 이어받기

## 2026-10-28 16:22 KST 우선 갱신 — 이전 snapshot보다 우선

- 첫 CSV 완료 뒤 사용자 지정 네 감사(score 집계/연산자 회귀/정적 입력 치환/1,000회 그룹 부트스트랩)를 별도 일회성 CPU 실행으로 연결했다. [준비 보고서](reports/R16_POSTCSV_AUDIT_READINESS.md), [현재 상태 스냅샷](handoff/POSTCSV_AUDIT_LIVE_SNAPSHOT.json)을 먼저 읽는다. CSV 및 실제 네 감사 결과는 아직 미완료다.
- 기존 학습 PID와 waiter가 종료돼 있었음을 발견했다. source/data/cache/checkpoint/Adam/scheduler/RNG 검증 후 61epoch 경계에서 복구했으며 62epoch 이상 실제 완료를 확인했다. 현재 D0 CNN 약67epoch(스냅샷). 기존 PID나 오래된 state를 살아 있는 학습으로 보고하지 않는다.
- 별도 검증된 복구 도구: `ops/resume_r16_multisupport_lifecycle.py`, `ops/continue_r16_multisupport_verified.py`. 원본 frozen trainer는 수정하지 않았다. [검증 영수증](ops/r16_multisupport_resume_20261028_r1/VERIFIED.json). 한 번 명시적으로 복구한 실행이며 재실행하면 거부한다. 또 죽었다고 기존 명령을 자동 반복하지 않는다.
- Documents 가상환경 패키지가 dataless 파일 읽기에서 정체됐다. 기존 설치된 로컬 Python3.12/PyTorch2.14.1에서 **MPS GPU를 유지**한다. 기존 최적 체크포인트의 저장 예측 재현 오차0, CPU/MPS Adam/RNG 복원 fixture 오차0. 레시피·손실·100epoch 계약은 그대로다.
- 현 recovery supervisor PID36954, child36956, Fold1~4 waiter37128, 사후 감사 waiter37129. 이는 snapshot PID이며 실제 command와 생존 여부를 확인한다. 중복 실행 금지.
- Release `handoff-2026-10-28`의 model ZIP은 **이 갱신 이전 checkpoint 스냅샷**이다. 새61epoch 복구 snapshot/이후 weights가 release에 올라갔다고 주장하지 않는다. 새 학습은 canonical 경로에만 있다.
- 공통 ln4 차감은 순위를 바꾸지 않는다. 패널별 rank는 코호트 의존 진단으로 한정한다. one-SE 권고는 기존 첫 CSV/정책을 덮어쓰지 않는다. Test 관측값/query 배열은 감사에서 읽지 않는다.


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
