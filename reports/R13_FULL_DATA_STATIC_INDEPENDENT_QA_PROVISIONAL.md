# R13 fullscale 자료 정적 독립 QA — provisional

**자료 관련 검산은 통과했으며, runtime policy/CLI 최종 freeze와 production readiness는 아직 확정하지 않았습니다.** 검토 대상은 새 `data/synthetic/group_cv_physics_r13_variants_v2`입니다. 예전 single-highshare v1 자료나 source audit snapshot을 현재 코드의 최종 실행 자료로 사용하면 안 됩니다.

[검산 JSON](R13_FULL_DATA_STATIC_INDEPENDENT_QA_PROVISIONAL.json)

## 원천 분리와 정답 범위

원 R9 배경 role source metadata를 읽어 각 fold의 TRAIN/selection/ensemble_fit/ensemble_gate/report 간 source ID·background group 교집합이0임을 독립 확인했습니다. sensor_group별 export-clock 순서에서 role이 바뀌는 인접 구간의 거리에900초 support를 빼는 별도 계산으로 최소24시간 purge도 확인했습니다. 저장 gap 필드를 그대로 요약한 결과가 아닙니다. Export clock을 별도 인증된 UTC라고 부르지 않습니다.

|fold|TRAIN source|TRAIN group|독립 최소 support-gap(h)|epoch1..5 zero 개수|TRAIN 3PC 설명분산|
|---|---:|---:|---:|---|---:|
|0|78,869|206|24.000|[15774, 15774, 15774, 15774, 15773]|84.87%|
|1|78,069|208|24.000|[15614, 15614, 15614, 15614, 15613]|84.84%|
|2|78,362|207|24.000|[15673, 15673, 15672, 15672, 15672]|84.31%|
|3|78,753|207|24.000|[15751, 15751, 15751, 15750, 15750]|85.53%|
|4|78,124|202|24.000|[15625, 15625, 15625, 15625, 15624]|84.21%|

완전한15분 관측만을 강요하거나 결측을 채우지 않습니다. 원 source boundary QC 뒤의15-bit effective mask와 qc_observed_minutes가 일치하고, source 순서에서 계산한 stratum key도 일치했습니다. HF 기준은 각 fold TRAIN의 site×sensor_group×recorder×effective-mask q90으로만 적합하며 엄격한 `>` 비교입니다. held-out stratum에 TRAIN 문턱이 없으면 held-out 자체 q90을 적합해 채우지 않는 코드입니다.

## 전체 primary와 zero 할당

전체 primary를 각epoch1회씩 사용합니다. N이 5의 배수가 아니면 매 epoch의 20%를 정수로 정확히 만들 수 없습니다. `floor(N/5)+I[(epoch−1)%5<N%5]` 법칙으로 5epoch의 zero 합이 N, 100epoch의 zero 합이 20N임을 확인했습니다. 각 source가 반드시 5epoch 중 한 번 zero라는 주장이 아니라 전체 비율의 정수 법칙입니다.

같은 stratum 안에서 random baseline의 zero quota를 유지하면서 HF 후보의 zero 할당을 우선합니다. 따라서 소형 12k의 outside-primary zero-background 교체와 같은 절차는 아닙니다. 각 stratum의 zero 개수와 HF priority가 실제 후보·zero에만 속하는지 확인했습니다. 실제 5epoch×5fold에서 baseline/highshare의 primary order, zero mask, HF priority, ref index, family, item seeds가 정확히 같았습니다. highshare만 일부 positive의 alpha를 바꿉니다.

## PCA와 fixed 자료

각 fold의 basis는 해당 TRAIN에 속하는 고유 배경 2,000개만 사용하며 다른 역할의 source와 겹치지 않습니다. raw TRAIN PCA input의 typed SHA, source IDs, binding을 저장 state와 대조했고 5개 fold의 basis SHA는 서로 다릅니다. A=0은 추가 성분이 없다는 뜻이며 자연 선박 부재의 정답이 아닙니다. basis는 safe band 0..22의 3PC이고 PCA 잔차도 source truth가 아닙니다. 기존 8채널과 full25 band 물리 가중치까지 upper-band에 독립이라는 뜻은 아닙니다.

selection/ensemble_fit/ensemble_gate의 두 variant metadata에서 source, mask, recorder, 순서가 같음을 확인했습니다. **report 수치·타깃·생성 metadata는 파싱하지 않고 파일 SHA만 확인했습니다.** Producer는 모든 fixed role에 대해 zero quota, positive observed support, missing q, no-pickle roundtrip을 검사하지만, 본 QA가 sealed report label을 추가로 열어 재검사한 것은 아닙니다.

두 variant의 fixed 자료는 source와 geometry가 같아도 추가 성분의 alpha와 정답 y가 다를 수 있습니다. 각 variant의 자기 report RMSE를 같은 target에서의 우열로 비교할 수 없습니다. Full runtime은 한 variant와 shape를 훈련 전에 고정하고 데이터 identity, 모델, 앙상블, 추론에 결박해야 합니다.

509개 참조 풀은 TRAIN과 held-out 역할·fold 사이에서 공유됩니다. 배경 시간 그룹 CV를 reference/hull 독립 CV라고 부르지 않습니다. 기존 vessel archive에는 weak-background 22개가 포함되며, 독립 ambient 16개는 활성 reference 풀에서 제외됩니다. 정답은 추가한 녹음 전체의 알려진 에너지이며 분리된 선박 에너지의 정답이 아닙니다.

## 현재 준비 상태와 남은 확인

v2 manifest/readiness/data code pins는 이 스냅샷과 일치하며 검산 중 코드·config 변경은 없었습니다. Runtime policy 전달, variant별 preflight 경로, tree feature toggle 연결은 담당 agent가 마무리 중입니다. Config의 production launch는 false이고 실제 CUDA 검증·훈련을 수행하지 않았습니다. 최종 source/policy SHA가 확정되면 binding을 추가 검토해야 합니다. 현재 스냅샷을 GPU 학습 완료나 production 실행 승인으로 해석하면 안 됩니다.
