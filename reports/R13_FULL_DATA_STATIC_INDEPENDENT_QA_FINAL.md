# R13 전체 자료·런타임 최종 독립 QA

현재 후보의 기술적 연결 검사는 통과했다. **production 승인과 CUDA 실측은 없으며 본학습 실행은 차단된다.** 이 문서는 품질 개선 또는 대회 성능을 보증하지 않는다.

검사 대상은 `group_cv_physics_r13_variants_v2`의 baseline/highshare 두 변형이며, 자료 해시는 `bb1dc465007a70798968ce14040b59643715538716b9e2c1f11c77e72c27d3c5`이다. 이전 [원천·메타데이터 독립 검사](R13_FULL_DATA_STATIC_INDEPENDENT_QA_PROVISIONAL.json)의 자료 해시와 DATA_CODE가 그대로여서 그 검사를 재사용하고 최종 런타임 연결을 별도로 재검증했다. 과거 runner 스냅샷을 현재 실행 증거로 간주하지 않았다.

## 원천 및 분할

| Fold | 매 epoch TRAIN primary 수 | 원천 그룹 수 | 처음 5 epoch의 A=0 개수 |
|---|---:|---:|---|
| 0 | 78,869 | 206 | 15774, 15774, 15774, 15774, 15773 |
| 1 | 78,069 | 208 | 15614, 15614, 15614, 15614, 15613 |
| 2 | 78,362 | 207 | 15673, 15673, 15672, 15672, 15672 |
| 3 | 78,753 | 207 | 15751, 15751, 15751, 15750, 15750 |
| 4 | 78,124 | 202 | 15625, 15625, 15625, 15625, 15624 |

모든 primary를 매 epoch 정확히 한 번 사용한다. 정수 개수 때문에 매 epoch의 A=0 비율이 정확히 20%인 것은 아니며, 5 epoch 합계는 정확히 전체 primary 수와 같아 전체 관측의 20%가 된다. 100 epoch 합계도 정확히 20%다. 각 원천이 5 epoch마다 정확히 한 번 A=0이 되는 규칙은 아니다.

원천 ID·background group의 역할 간 분리와 sensor/export-clock 기준 `900초 창 + 24시간 purge`를 메타데이터로 독립 재계산했다. 최소 보정 간격은 다섯 fold 모두 24시간이다. 이 시각은 export-clock이며 실제 UTC 정확성을 새로 인증한 것은 아니다. 각 fold의 PCA는 해당 TRAIN의 고유 원천 2,000개에서만 fit되며, 다섯 basis SHA가 서로 다르고 동일 fold의 평가 역할 원천과 겹치지 않는다.

HF 후보의 q90 임계값은 TRAIN에서만 계산한다. `site/sensor/recorder/effective mask`별 A=0 quota 안에서 후보를 우선 배정하며 primary를 복제하지 않는다. minute QC를 먼저 적용해 실제 관측 mask를 만든 뒤 합성·에너지 정답을 계산하고 최소 3분 관측 조건을 유지한다. baseline/highshare의 primary·참조·mask·배치·item seed는 같으며 고비율 부분집합에서 추가 성분의 스케일만 바뀐다. 양성 관측 지지 여부는 생성기 검사와 기존 연결 증거에 의존하며, 이 QA에서 잠긴 report 수치나 라벨을 새로 읽지 않았다.

## 최종 실행 바인딩

현재 런타임은 [candidate highshare/P1 v2](../configs/r13_runtime_candidate_highshare_P1_v2.json)이며 파일 SHA는 `227ae599fe830ab37de5d8395c29ca7cad65d178dba0ce0fa5917f99839151ee`다. 30개 코드 SHA, 자료 manifest, CPU 선택 근거의 프로젝트 상대경로와 SHA를 확인했다. 작은 CPU 실험의 모델 가중치나 score 혼합계수를 fullscale 실행으로 이전하지 않는다.

- runner SHA: `99c2a311b01048dd7d72a67250c5c6e71335cb6c538d3373bee6455ff1451634`
- CLI SHA: `8b4459f909edde4b1c279008f64187430fbe3663d239699ed8fd6bfc89ea35ef`
- runtime payload SHA: `32f6ea653cc802d8609671c4c88d79c5b174d1ee4ebbab8dd55cd0d169e0122c`

승인 없는 production, 다른 자료/코드, 변조된 payload, 변형과 비율이 모순인 policy는 수치 fixture에서 거부됐다. [최종 runner fixture](R13_RUNNER_FINAL_VERIFICATION.json)는 생산 실행에서 전체 0~4 fold를 요구하는 barrier도 검증한다. [최종 preflight](R13_PREPARED_PREFLIGHT_FINAL.json)와 [패키지 계획](R13_PORTABLE_PACKAGE_PLAN_FINAL.json)의 현재 핀이 일치한다.

## 검증 범위와 남은 제약

[실제 원천 4개 full-size 모델 1 Adam step 연결](R13_REAL_FULL_MODEL_CONNECTION_V3.json), [Tree/PCA 연결](R13_TREE_PCA_CONNECTION.json), [동결 추론](R13_FROZEN_INFERENCE_VERIFICATION_V2.json), [4 workers·prefetch2의 실제 TRAIN 64행 재현](R13_SPAWN_LOADER_CONNECTION.json)은 각 증거 파일의 현재 소스 범위와 대조됐다. 이것은 CPU 연결 증거이며 CUDA 메모리·epoch 시간·전체 throughput 또는 100 epoch 학습 성능의 실측이 아니다.

509개 참조 녹음은 모든 역할과 fold가 공유한다. 따라서 배경 시간 분리 검증이며 참조 녹음/hull 독립 검증은 아니다. A=0은 제어된 추가 성분이 없다는 뜻이고 자연 선박 부재 정답은 아니다. archive의 weak-background 22개는 포함되고 별도 ambient 16개는 활성 참조에서 제외된다.

baseline/highshare의 고정 패널은 참조·관측 mask가 같아도 실제 target y가 달라질 수 있다. 각 변형의 report RMSE를 그대로 동일-target paired 성능 비교로 해석하면 안 된다. fullscale의 Uniform target 추첨과 작은 12k 실험의 LHS는 같은 주변 법칙을 의도하지만 동일한 유한 표본은 아니다.

현재 `production_recipe_approved=false`이고 실제 CUDA 1 epoch benchmark도 없다. 추가 paired epoch-mixture 실험은 별도 후속 실험이며 이 후보 정책을 소급 변경하지 않는다. 원본 자료·모델·정책·학습 프로세스·Test에는 변경을 가하지 않았고, 이 QA는 Test 및 생성 report의 라벨/수치를 읽지 않았다.

정확한 코드·증거 SHA와 최종 재검사 내역은 [최종 JSON](R13_FULL_DATA_STATIC_INDEPENDENT_QA_FINAL.json)에 보존한다.
