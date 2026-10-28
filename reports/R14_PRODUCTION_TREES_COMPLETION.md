# R14 FullTree 5-Fold 생산 학습 완료

**Fold0 완료 모델2개를 검증 재사용하고, Fold1–4의 baseline/highshare8개 learner를 실제 전수 학습했습니다. 총10개 learner·20개 booster(회귀10+독립 분류10)가 동결됐습니다.** 이 문서는 Tree 단계 완료이며 신경망·앙상블 정책·제출 파일 전체 완료를 뜻하지 않습니다.

|Fold|조건|상태|TRAIN행|특징|own selection RMSE|회귀/분류 best round|fit시간|최대RSS|
|---:|---|---|---:|---:|---:|---:|---:|---:|
|0|baseline|기존 완료 검증 재사용|78,869|5,252|0.07352335|1497/890|13.07분|6.11GiB|
|0|highshare|기존 완료 검증 재사용|78,869|5,252|0.07998436|1496/887|12.64분|5.72GiB|
|1|baseline|신규 실제 학습|78,069|5,252|0.06338644|1498/1023|12.92분|6.58GiB|
|1|highshare|신규 실제 학습|78,069|5,252|0.07168802|1500/783|12.60분|7.33GiB|
|2|baseline|신규 실제 학습|78,362|5,252|0.05999553|1500/548|11.53분|7.43GiB|
|2|highshare|신규 실제 학습|78,362|5,252|0.06621975|1499/754|12.49분|7.37GiB|
|3|baseline|신규 실제 학습|78,753|5,252|0.05574330|1497/817|12.65분|8.02GiB|
|3|highshare|신규 실제 학습|78,753|5,252|0.06078556|1480/603|11.65분|6.40GiB|
|4|baseline|신규 실제 학습|78,124|5,252|0.07004101|1500/817|12.46분|6.85GiB|
|4|highshare|신규 실제 학습|78,124|5,252|0.07711303|1496/623|11.66분|6.44GiB|

- 신규8개 fit 합계97.95분, 컨트롤러 실제 경과109.61분입니다. CPU2threads로 별도 fresh process를 순차 실행했습니다. 신규 raw epoch1 cache 합계582,169,509bytes(약0.54GiB)이며 대형5252-featurematrix는 디스크에 저장하지 않았습니다.
- 신규 worker 최대RSS는8.02GiB입니다. 이는 각 프로세스의 최대값으로 Mac 전체/MPS 동시 최고 메모리 측정값과 다릅니다. 종료 시 디스크 잔여는 약12.56GiB였습니다.
- 각 fold의 원 TRAIN primary를 epoch1에 빠짐없이 한 번씩 생성하고, 해당 fold의 TRAIN-only PCA를 사용했습니다. baseline/highshare 각자의 selection1000개만 boosting ES1500/40에 사용했습니다. 서로 다른 variant selection의 alpha/정답 일부가 다르므로 표의 RMSE는 같은 평가관측의 직접 대결이 아닙니다.
- 전체learner의 TRAIN행 합계784,354는 CV 중복과 paired variant를 포함한 학습행 수로, 독립 원천784,354개라는 뜻이 아닙니다. 509개 reference는 공유합니다. 배경 group-CV이며 reference-independent CV나 새 맹검 일반화를 주장하지 않습니다.
- 완료모델별 strict save/load·batch1/7 예측 일치, input/source/code/approval/PCA/model artifact SHA를 검증했습니다. 컨트롤러 exit0, FAILED marker없음, 실행 lock 제거, readonly verify CLI PASS입니다.
- Fold0 원본은 `models/r14_tree_full_r1/{baselineP1,highshareP1}/model` 그대로입니다. 원global freeze SHA a25811965e0edf8669555db19606be740c72f70e24a1505f867c30e731bdbdfe와각completion·artifact map을대조했고 weight복사/symlink/기존진단파일수정은없습니다.
- 신규 모델은 `models/r14_production_trees_r1/fold_01..04/{baselineP1,highshareP1}/model`에있습니다. 후속정책·추론 단계는 전역 registry의 실제model_dir을사용합니다. 각각share와raw독립classifier출력을제공합니다.
- 사용자 최종첨부 SHA54bae81c2c0aa8b5e5fa299cf6aebbbe0af98b69d68f41add3c3b1802a212ba6과새productionconfig/approvaltrue를결박했습니다. 원R13/R14진단의false승인상태와과거수치를고치지않았습니다.
- 이 단계는 새17패널/production report/Test 배열, 앙상블 가중치 선택, CSV생성, 공식업로드를수행하지않았습니다. 그 작업은별도완료모델·정책검증단계에서진행합니다.

## 고정 진입점

전역 완료 registry: `models/r14_production_trees_r1/ALL_FIVEFOLD_TREES_FROZEN.json`
파일 SHA256: `be37991e063375b80e7ff3fa501c6aef619408d1a59166c48487da1f6b5574cb`
내용 token: `084e61cc79ea18859e1118989e3abe4847bd1364d4ec11fe4ba62a885cee2949`

`models[str(fold)][variant+"P1"].model_dir`에서실제모델경로를얻고, `PhysicsLGBM.load(path, expected_binding=summary["training_binding"])`로복원합니다. `predict(db)`는share, score(기본share), score_classifier, legacy_mixed_score를반환합니다. 오래된673/3827특징Tree로대체하지않습니다.

[전체 기계 판독 결과](R14_PRODUCTION_TREES_COMPLETION.json) · [10모델 표](R14_PRODUCTION_TREES_COMPLETION.csv) · [독립 사전QA](R14_PRODUCTION_TREE_PREFLIGHT_INDEPENDENT_QA.json)

독립 완료 QA도 **PASS**했습니다. 659개 단언·141개 파일을 대상으로 원장·10개 learner/20개 booster·실제 tree count·5,252특징·승인/원천·fold별 TRAIN2000 PCA·own selection·24시간 purge를 대조했습니다. 학습기에서 share/presence를 float64로 승격하므로 raw float32/int64 label typedSHA와 다른 것은 예상한 dtype 차이이며, 원본 NPZ/db SHA는 일치했습니다. [독립 완료 QA](R14_PRODUCTION_TREE_COMPLETION_INDEPENDENT_QA.json) / [설명 MD](R14_PRODUCTION_TREE_COMPLETION_INDEPENDENT_QA.md).
