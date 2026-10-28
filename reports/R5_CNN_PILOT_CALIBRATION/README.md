# 고정 CNN의 INNER 전용 보정 진단

선택: **identity**, a=1, b=0.

| 후보 | calibrator-crossfit RMSE | identity 대비 개선 | BG 개선 비율 | gate 통과 |
|---|---:|---:|---:|---|
| identity | 0.090808147 | +0.000000000 | 0.0% | False |
| offset | 0.091789314 | -0.000981167 | 22.0% | False |
| constrained_affine | 0.092133282 | -0.001325135 | 12.0% | False |

정해진 25개 reference×background 분할과 세 후보를 한 번만 계산했다. 모든 행은 검증에 정확히 한 번 포함됐고 각 fit의 양성 reference와 background group이 검증과 겹치지 않는다. zero는 사전 선언대로 i=0에서만 검증한다. 별도 24시간 purge는 추가하지 않았다.

기본 CNN이 전체 INNER로 early stopping됐으므로 이 수치는 독립 신경망 OOF 성능이 아니다. 합성 원천의 Test 대표성이나 LB 점수를 추정하지 않는다. 전 기준을 통과한 후보가 없으면 identity를 유지하며 재분할·gate 완화·재시도를 하지 않는다.

비identity 최종 계수는 선택 후 전체 INNER에서 한 번만 다시 fitting한다. 그 재대입 점수는 진단용이며 선택 근거가 아니다. scalar 보정 이후 원본 cell mask는 원본 share의 물리적 가중합을 설명하며, 보정된 scalar를 그대로 설명한다고 주장하지 않는다. score/presence는 수정하지 않는다.

[고정 사양](specification.json) · [진단 수치](metrics.json) · [동결 정책](frozen_policy.json)
