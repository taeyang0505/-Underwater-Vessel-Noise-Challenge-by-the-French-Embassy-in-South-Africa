# R7: 기존 R5 제출 경로 재현 감사

결과: **passed**. 모델·입력 SHA 계획을 추론 전에 저장했고, 원 epoch19 CNN과 TRAIN prior·정책을 그대로 사용했습니다. 원 CPU4와 이번 CPU2의 차이는 아래 수치 재현 검사로 분리했습니다.

| 검사 | 결과 |
|---|---|
| 375열 minute-major/주파수/ID/missing mask 숫자 roundtrip | bitwise 일치; 잘못된 헤더 거부 |
| raw TRAIN 준비와 query 특징·가중치 | bitwise 일치 |
| 동일 배열 vs CSV parser 경유 모델 예측 | bitwise 일치 |
| 저장 Test raw→동결 정책→원 CSV 바이트 | SHA e32b04a2…c83fe 정확 재현 |
| INNER CPU2 재현 RMSE | 0.090808147081 |
| 저장 CPU4 / GPU BF16 INNER RMSE | 0.090808147081 / 0.090849261020 |
| INNER CPU2−CPU4 share 최대차 | 0 |
| Test CPU2−CPU4 share 최대차 (추론 재현 차이만) | 0 |
| 새 CPU2 추론의 원 CSV 바이트 동일 여부 | True |

입력/가중치/정책은 전후 SHA가 같았고 학습·보정 재선택·새 제출은 하지 않았습니다. Test는 고정 추론과 기존 예측의 ID별 재현 차이 확인만 사용했으며 분포·원천 추정이나 정답 복원은 하지 않았습니다. 재추론 결과는 audit NPZ로만 저장하고 새 submission CSV는 쓰지 않았습니다.

이 결과는 구현의 특정 경로 재현 증거입니다. 숨은 생성기 정합·독립 검증의 대표성·공식 성능 격차의 원인을 확정하거나 외부 분포 오류만이 원인이라고 증명하지는 않습니다. INNER는 원 CNN epoch 선택에 이미 사용됐습니다. e32b…는 로컬 handoff CSV 해시이며 실제 Zindi 업로드 바이트의 SHA는 독립 확인되지 않았습니다.

[추론 전 동결 계획](R7_R5_SUBMISSION_PATH_AUDIT_PLAN.json) · [상세 수치·SHA 결과](R7_R5_SUBMISSION_PATH_AUDIT_RESULTS.json)
