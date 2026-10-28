# 최신 갱신 2026-10-28T16:23:04.812803+09:00

## 2026-10-28 16:22 KST 우선 갱신 — 이전 snapshot보다 우선

- 첫 CSV 완료 뒤 사용자 지정 네 감사(score 집계/연산자 회귀/정적 입력 치환/1,000회 그룹 부트스트랩)를 별도 일회성 CPU 실행으로 연결했다. [준비 보고서](../reports/R16_POSTCSV_AUDIT_READINESS.md), [현재 상태 스냅샷](POSTCSV_AUDIT_LIVE_SNAPSHOT.json)을 먼저 읽는다. CSV 및 실제 네 감사 결과는 아직 미완료다.
- 기존 학습 PID와 waiter가 종료돼 있었음을 발견했다. source/data/cache/checkpoint/Adam/scheduler/RNG 검증 후 61epoch 경계에서 복구했으며 62epoch 이상 실제 완료를 확인했다. 현재 D0 CNN 약67epoch(스냅샷). 기존 PID나 오래된 state를 살아 있는 학습으로 보고하지 않는다.
- 별도 검증된 복구 도구: `ops/resume_r16_multisupport_lifecycle.py`, `ops/continue_r16_multisupport_verified.py`. 원본 frozen trainer는 수정하지 않았다. [검증 영수증](../ops/r16_multisupport_resume_20261028_r1/VERIFIED.json). 한 번 명시적으로 복구한 실행이며 재실행하면 거부한다. 또 죽었다고 기존 명령을 자동 반복하지 않는다.
- Documents 가상환경 패키지가 dataless 파일 읽기에서 정체됐다. 기존 설치된 로컬 Python3.12/PyTorch2.14.1에서 **MPS GPU를 유지**한다. 기존 최적 체크포인트의 저장 예측 재현 오차0, CPU/MPS Adam/RNG 복원 fixture 오차0. 레시피·손실·100epoch 계약은 그대로다.
- 현 recovery supervisor PID36954, child36956, Fold1~4 waiter37128, 사후 감사 waiter37129. 이는 snapshot PID이며 실제 command와 생존 여부를 확인한다. 중복 실행 금지.
- Release `handoff-2026-10-28`의 model ZIP은 **이 갱신 이전 checkpoint 스냅샷**이다. 새61epoch 복구 snapshot/이후 weights가 release에 올라갔다고 주장하지 않는다. 새 학습은 canonical 경로에만 있다.
- 공통 ln4 차감은 순위를 바꾸지 않는다. 패널별 rank는 코호트 의존 진단으로 한정한다. one-SE 권고는 기존 첫 CSV/정책을 덮어쓰지 않는다. Test 관측값/query 배열은 감사에서 읽지 않는다.


---

## 이전 스냅샷(역사 보존)

# 현재 상태 — 2026-10-28T15:02:23.505861+09:00

**스냅샷이며 실시간 표시가 아니다.** 원본 프로젝트를 사용할 수 있다면 다시 읽어야 한다.

## 실행 상태

```json
{
  "phase": "neural",
  "PID": 12773,
  "child_PID": 21489,
  "elapsed_seconds": 9002.841671943665,
  "progress": {
    "stage": "neural",
    "regime": "D0",
    "kind": "bg_cnn",
    "epoch": 35,
    "rows_seen": 59264,
    "total_rows": 61446,
    "optimizer_steps": 33600,
    "elapsed_seconds": 2503.187347457744
  },
  "official_upload": false
}
```

후속 상태: `waiting_for_fold0_CSV`, PID `23279`.

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
