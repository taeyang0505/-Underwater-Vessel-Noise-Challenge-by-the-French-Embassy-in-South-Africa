# R9 LightGBM·앙상블 검증

R6의 기존 3,827개 상대 격자·물리 특징을 사용하는 CPU LightGBM과, INNER의 서로 다른 원천 그룹에서 채택 여부를 확인하는 앙상블 함수를 구현했습니다. 실제 대회 음향 데이터, Test, 기존 OUTER를 읽거나 본학습을 실행하지 않았습니다.

## LightGBM 계약

`PhysicsLGBM(settings=None, seed=20261026)`의 기본값은 최대 1,500 boosting rounds, early stopping 40 rounds, CPU 4스레드, 학습률 0.035, 31 leaves, 최소 leaf 50행입니다. 고정된 INNER/selection 데이터에서 회귀 모델은 **[0,1]로 자른 share의 pooled RMSE**, 존재 여부 분류 모델은 binary logloss를 기준으로 best iteration을 선택합니다. NN의 100 epochs·early stopping 없음 정책은 boosting rounds와 별개입니다.

- `model.fit(train, inner, output_dir, binding, fixture=False)` → 완료 summary 딕셔너리
- `PhysicsLGBM.load(output_dir, expected_binding=..., allow_fixture=False)` → 복원 모델
- `model.predict(db, batch_size=256)` → `share`, `score`, `score_classifier`

학습 인자는 `db/share/presence` 배열을 가진 명시적 딕셔너리입니다. 선택적으로 `role`을 주면 TRAIN에는 `train`, 선택 데이터에는 `inner` 또는 `selection`만 허용하며 `gate/report`를 허용하지 않습니다. 추론은 db 배열만 받아 라벨 딕셔너리를 거부합니다. 파일·분할에 대한 실제 접근 허용과 샘플 순서 검증은 호출자 로더가 담당합니다.

특징 변환은 기존 `cpu_ablation_r6_v2.tree_features(db, 'relative8', None)`을 그대로 호출합니다. 구성은 8×15×25 격자 3,000개, 채널·대역별 시간 mean/std/max/min 800개, R6 physics 27개입니다. 학습된 PCA나 배경 사전, 원천 ID·사이트·정답을 특징으로 사용하지 않습니다. 이름과 정확한 열 순서가 있는 feature schema 및 코드 SHA를 저장하고 복원 시 대조합니다.

TRAIN은 최대 **100,000행**이며 초과 시 임의로 줄이지 않고 거부합니다. 75,000행 특징 행렬은 약 1.069 GiB, 100,000행은 약 1.426 GiB의 float32입니다. 여기에 입력, INNER 행렬과 LightGBM histogram/working memory가 더 필요합니다. 512행씩 특징을 추출해 임시 격자 메모리를 제한하지만, 최종 학습 행렬은 메모리에 유지하므로 out-of-core 학습은 아닙니다.

회귀 출력은 `[0,1]`로 제한합니다. diagnostic score는 기존 R6과 같은 `0.5*share + 0.5*presence_classifier_probability`입니다. 자연 상태 선박 존재 확률의 새로운 보정이나 Test 기반 보정은 수행하지 않습니다.

저장 파일은 `share.txt`, `presence.txt`, `feature_schema.json`, `inner_predictions.npz`, `summary.json`입니다. summary가 완료되기 전 중단된 출력은 재사용하지 않으며, 기존 출력 폴더를 덮어쓰지 않습니다. 복원할 때 source/split binding, 코드·아티팩트 SHA, 특징 수·순서 및 점수 정책을 검증합니다. fixture 가중치는 별도 허용 없이 production으로 읽지 않습니다.

## 앙상블 정책

`fit_safe_ensemble(y, predmap, groups, fit_indices=..., gate_indices=...)`는 명시적으로 지정한 own-INNER 자료만 받습니다. fit/gate 행과 **제공된 원천 그룹 ID**는 모두 겹치면 안 됩니다. 실제 물리 원천 중복·시간 purge를 보장하는 올바른 그룹을 전달하는 책임은 상위 분할 코드에 있습니다.

사전에 정한 후보는 각 단일 모델, 전체 모델 동일 가중 평균, 트리를 제외한 신경망 동일 가중 평균입니다. 임의 실수 가중치 최적화나 하이퍼파라미터 탐색은 없습니다. 트리 키의 기본값은 `lgbm_physics`입니다.

1. fit 행에서 가장 좋은 **신경망 단일 모델**을 fallback으로 고정합니다. 신경망이 없으면 트리 단일 모델이 fallback입니다.
2. 같은 fit 행에서 사전 후보 중 가장 좋은 후보 **하나**를 고정합니다. 트리 단독 후보도 이 단계에 포함합니다.
3. 후보가 fallback의 fit RMSE를 기본 ε=0.0001 이상 개선한 경우에만, 분리된 gate 그룹에서 이 후보와 이미 고정된 fallback을 한 번 비교합니다.
4. gate에서도 같은 ε 이상 개선하면 후보를 채택합니다. 그렇지 않으면 이미 fit에서 고른 fallback을 사용합니다. gate를 보고 다른 후보를 다시 시도하거나 가중치를 재학습하지 않습니다.

따라서 트리가 fit에서만 좋다는 이유로 fallback 경로를 통해 gate를 우회하지 않습니다. 채택된 동일 가중치가 최적이라는 주장은 하지 않습니다. 이 gate 자체는 **검증 기반 선택 과정**이며, 미사용 report/outer의 독립 성능 평가를 대신하지 않습니다. 상위 러너는 모든 정책을 고정한 뒤 report/outer를 열어야 합니다.

`freeze_ensemble(policy, path)`는 payload SHA가 일치하는 정책을 새 파일로만 저장합니다. `apply_frozen_ensemble(policy, predmap)`은 라벨 없이 고정된 nonnegative simplex 가중치로 share와 diagnostic score를 각각 동일하게 평균합니다. affine 보정은 `[1,0]`으로 고정하며, 라벨·Test 자료를 인자로 받지 않습니다. 신경망끼리 동일 w를 사용할 때 scalar share 평균은 같은 가중치로 평균한 q의 물리적 풀링과 같습니다. 트리에는 q 격자가 없으므로 트리가 포함된 앙상블까지 이 등가성을 주장하지 않습니다.

## 실행한 작은 숫자 검증

합성 TRAIN 128개·selection 32개, CPU 2스레드, 최대 10 boosting rounds/ES3으로 회귀·분류 모델을 한 번씩 구성했습니다. 실제 음향 원천 학습은 아닙니다.

- 특징 수 3,827 및 행 단위/32행 단위 특징 추출이 정확히 일치했습니다.
- 모델 저장·복원 후 예측 batch 1/7/32의 share·score·classifier 출력 최대 차이가 모두 **0**이었습니다.
- 출력 범위·유한값·점수 공식을 검증했습니다.
- 라벨 딕셔너리 추론, 기존 출력 덮어쓰기, binding 변경, fixture의 production 복원, 100k/1,500 cap 초과, 모델 가중치 변조를 거부했습니다.
- 나쁜 트리 후보가 있는 합성 앙상블은 신경망 0.5/0.5, 트리 **0**을 선택했습니다. fit에서는 좋아도 disjoint gate에서 실패한 후보는 고정 fallback으로 돌아갔습니다.
- source-group/행 중복, 정책 변조, 모델 집합 변경, 정책 파일 덮어쓰기를 거부했습니다.
- 신경망 평균의 물리적 풀링 선형성 최대 오차는 `5.551115123125783e-17`였습니다.
- 원본 숫자 배열 및 기존 의존 코드 SHA는 검증 전후 동일합니다. GPU·Test·기존 OUTER 접근은 없습니다.

첫 합성 검증 기록은 `R9_TREES_ENSEMBLE_VERIFICATION_INITIAL.json`로 보존했습니다. 현재 검증 JSON은 최종 clipped-share early-stopping 기준과 엄격한 복원 검사를 적용한 소스에 대응합니다.

## 파일

- [LightGBM·앙상블 구현](../../../pipeline/trees_ensemble_r9.py)
- [합성 CPU 검증 코드](../../../ops/verify_trees_ensemble_r9.py)
- [검증 JSON](../../R9_TREES_ENSEMBLE_VERIFICATION.json)
- [R9 신경망 어댑터 검증](../../R9_MODEL_ADAPTER_VERIFICATION.md)
- [R8 특징·분할 근거](../../R8_FEATURE_PIPELINE_EVIDENCE.md)

재현: `python -m ops.verify_trees_ensemble_r9 --output reports/R9_TREES_ENSEMBLE_VERIFICATION_new.json`. 기존 결과 덮어쓰기는 거부합니다.

검증 JSON SHA256: `52e67817aeb498a8fd12e05fcd0be5bc8ca9fa4454ecca6ff53ecc322c02be66`.
