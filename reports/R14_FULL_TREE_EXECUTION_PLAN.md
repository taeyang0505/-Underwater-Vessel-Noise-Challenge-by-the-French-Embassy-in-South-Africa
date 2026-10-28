# R14 본체 Tree CPU 실측 사전 계획

원 R12/R13 생산 코드·설정·데이터·가중치를 수정하지 않는 별도 CPU 진단이다. `production_recipe_approved=false`는 유지하며 생산 runner를 호출하지 않는다. 사용자 첨부의 원인 확정·Test 성능 해결 주장은 이 실행의 전제가 아니다.

- 대상: Fold0 baselineP1 우선, 이어 highshareP1. 각각 원 R13 epoch1 TRAIN primary **78,869개 전수**를 deterministic하게 materialize한다. global subsampling은 없다. 원 LightGBM의 tree별 bagging0.85는 유지된다.
- 특징: 원 `pipeline.trees_ensemble_r13.PhysicsLGBM`의 **5,252개**, 해당 full Fold0 TRAIN-only PCA를 사용한다. mini 실험의673특징·12k자료·공유PCA를 대체로 사용하지 않는다.
- 적합: regressor와 independent presence classifier를 각각 학습, 최대1,500/ES40, CPU2threads, 각 variant의 자기 selection1,000개만 ES에 사용한다. 두 selection은 배경·참조·기하가 paired지만 일부 highshare alpha/target은 다르므로 선택된 boosting round 차이도 이 설계 일부다.
- 실행: baseline/highshare를 별도 fresh child process에서 순차 실행해 float32 matrix·histogram working memory를 반환한다. 특징 행렬은1,656,879,952bytes(약1.54GiB)이며 전체 메모리는 별도 실측한다.
- 동결: 두 full tree learner의 summary/booster/PCA/code/source SHA가 모두 고정되기 전에17패널을 열지 않는다. 원 R13 ensemble_fit/gate/report는 사용하지 않으며, 특히 report gate를 우회하지 않는다.
- 평가: 이미 노출된 동일17개 R12/R9/R10/R11 개발 패널. share RMSE/bias/AUC, 독립 classifier AUC/Brier/포화율, 기존고정50:50 diagnostic score를 구분한다. 새 ensemble/score 정책을 선택하지 않는다. compact tree와 compact CNN/tree blend 원예측은 sample ID/정답 순서를 확인한 후 재사용한다.
- 한계: full-vs-mini 비교에서는 특징 개수뿐 아니라 행 수·원천 합성 law·fold-PCA·ES관측도 달라지므로5252특징만의 인과 효과로 단정할 수 없다. reference는공유되고 패널은개발노출 상태로 맹검 또는 공식 LB 예측이 아니다. Test/GPU/SSH/외부전송은 없다.

계획 content SHA: `280bd07ce061cc7f11c400943e0a0e730b5dde62430047c2c2a8bbfa48df31a0`

실행 경로: `models/r14_tree_full_r1`, 로그 `ops/r14_tree_full_r1.log` 및 variant별 log. 실제 종료 후 `reports/R14_FULL_TREE_RESULTS.md`와독립 검산을 발행한다.
