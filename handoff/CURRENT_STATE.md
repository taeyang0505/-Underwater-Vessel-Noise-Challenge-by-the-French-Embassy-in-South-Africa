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
