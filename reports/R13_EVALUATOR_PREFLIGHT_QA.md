# R13 CPU evaluator 사전 독립 QA

**PASS.** 최종 평가 코드 SHA는 `14ba4d5d09c73d85a2054c8b2a8f1262fda20ecdff109903a7c2e8f24a7457c1`입니다. 학습 중인 CNN driver와 모든 생산 source를 수정하지 않았습니다. 실제 평가 main·held-out 관측/정답·생산 checkpoint 텐서를 열거나 모델 학습/추론을 수행하지 않았습니다. 첫 검토의 기존 R10 source ID/sensor/week metadata-only 집계는 별도로 표시했습니다.

[최종 수치·source 결박 JSON](R13_EVALUATOR_PREFLIGHT_QA.json) · [초기 검토 기록](R13_EVALUATOR_PREFLIGHT_QA_INITIAL.json)

- **역할 순서:** 4개 CNN 논리조건(3fresh+1완료 control 재사용)과4개 tree 완료증명·source binding을 확인하고, FROZEN_BEFORE_GATE를 쓴 뒤 gate를 엽니다. 새 정책을 저장하고 파일 SHA를 즉시 확인한 다음에만 locked eval/high-SNR를 로드합니다. 내부 공유 loader에는 기존 R12 정책을 넘기지만, 외부 R13 check_policy가 매 locked-load 바로 전에 현재 R13 정책의 파일 SHA를 검증합니다.
- **정책 수치:** 최종 소스 AST의 실제 gate_rmse/winner/gate_auc/lambda 대입문4개를 숫자 전용 namespace에서 실행했습니다. 예상 regression winner `highshareP1`, score lambda `.25`를 얻었습니다. 동률은 canonical 첫 `baselineP0/lambda0`를 유지했습니다. score는 회귀 share/q를 바꾸지 않습니다.
- **정밀도·AUC:** gate와 보고 score가 모두 `mix_score`의 float64 가중합→float32를 사용합니다. 네 lambda 모두 bitwise 수식 일치와 pairwise AUC tie oracle을 확인했습니다.
- **동일 예산·재사용:** P0/P1은 모두11채널/26,953 parameter의 같은 모델이며 P0는 shape3채널을0으로 채웁니다. 기존 control의 완료100epoch·Adam·현재 training_r12 코드 SHA, strict-v2 manifest·원 parent manifest·basis SHA를 metadata로 대조했습니다. R13 builder가 baseline keys/dtypes/배열bytes 일치도 이미 검증했습니다.
- **모델 바인딩:** 최종 evaluator는 CNN100epoch history/strict fit minimum/ckpt epoch·arm·26953parameter·유한값과 shape toggle을 검사합니다. tree는 현재 R13 data hash/manifest/개별TRAIN/공통valfit/PCA/feature toggle binding과 strict loader를 함께 사용합니다.
- **unseen CI:** 예전 R10 unseen source manifest의1,000개 고유 base ID와 sensor×week를 원래 행 순서대로 연결합니다. 별도 sort를 하여 label/prediction 순서를 바꾸지 않습니다. 숫자 cluster fixture에서 공동 행 permutation에 대한 CI 불변성과 동일 예측 차이 CI=0을 확인했습니다. 실제 연결의 근거는 원래 frozen producer의 동일 row-order/file SHA 계약입니다. 임의로 섞인 source metadata를 ID join해 복구하는 일반화된 loader는 아닙니다.

현재 사용 경로에서 확인된 역할 누수·예산 차이·수학 blocker는 없습니다. source-pin merge는 충돌검사 없이 dict.update하므로 현재 trusted producer의 동일 SHA 계약에 의존한다는 작은 방어 한계를 기록했습니다. CIs는 source-group 조건부이며 공유 reference·반복 gate 선택·다중비교 불확실성을 전부 제거하지 않습니다. 사전 QA는 실제 성능 개선이나 공식점수의 증명이 아닙니다.
