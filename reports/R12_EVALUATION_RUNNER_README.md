# R12 CPU ablation 평가 runner

현재는 **사전 계획·숫자 fixture만 완료**했습니다. 실제 gate/eval/R11 모델 평가는 실행하지 않았습니다. 기존 학습 코드·모델·데이터·정책은 변경하지 않았습니다.

- 코드: `ops/evaluate_r12_ablation.py`
- [사전 고정 계획](R12_EVALUATION_PREDECLARED_PLAN.json): 코드·config·원 데이터·기존 source/model/policy 179개 파일 SHA.
- [최종 숫자 fixture](R12_EVALUATION_RUNNER_FIXTURE_R2/VERIFICATION.json): 고정50:50 선택, 동률 baseline 유지, 정책 변조/foreign freeze 거부, locked loader의 선 정책 검증, 미완료17모델 barrier 거부, AUC tie/단일class 처리, paired source-cluster500 CI 재현을 확인했습니다. 첫 fixture는 이전 script SHA의 자료로 별도 보존했습니다.

## 실제 실행 순서

부모의 명시적 GO와9CNN+8tree 완료 이후에만 다음 두 단계가 가능합니다. Mac은 새 process를 `DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib`로 시작해야 합니다.

```sh
python ops/evaluate_r12_ablation.py freeze \
  --plan reports/R12_EVALUATION_PREDECLARED_PLAN.json \
  --output reports/R12_ALL17_GLOBAL_FREEZE.json --execute-approved
python ops/evaluate_r12_ablation.py evaluate \
  --global-freeze reports/R12_ALL17_GLOBAL_FREEZE.json \
  --output reports/r12_diagnostics/full_evaluation_r1 \
  --execute-approved --threads 2 --batch 128
```

`freeze`는 양쪽 ALL_CNN_FROZEN/ALL_TREES_FROZEN 증명서, 정확17개 summary, 전체 모델 artifact·fit code·basis·data binding, CNN100epoch/strict best checkpoint를 검증합니다. gate/eval 수치 배열을 열지 않습니다. 출력 format은 기존 tree 평가 인터페이스와 같은 `r12_all_models_frozen_v1`, `model_summary_sha256`17개를 포함합니다.

`evaluate`는 이 증명서 검증 후 다음 순서만 허용합니다.

1. 새 R12 validation_gate500에서8개의 **고정 CNN/tree50:50 share** 조합을 평가합니다. canonical 순서의 strict 최소 RMSE를 선택하므로 동률이면 L0H0P0 baseline이 유지됩니다. legacy control은 조합 선택에 포함하지 않습니다. calibration/weight/threshold를 fit하지 않습니다.
2. `selected_policy.json`과 SHA를 기록합니다. 그 이후에만 새 eval500/high-SNR-long500을 로드합니다. 새 두 패널의 R9 baseline은 원 strict helper로 **한 번의 joined inference**를 수행합니다.
3. R9 selection/기존 gate, R10의4개 고정 패널, R11의4장지원×2음성 crossing8개를 보고합니다. R9/R10/R11의 baseline은 이미 저장된 예측을 재사용합니다. 이들은 development-exposed 자료이며 blind라고 부르지 않습니다. R11 개별6개 자료는 한 번씩 예측한 뒤8개 crossing에서 재사용합니다.
4. 모든 입력·policy SHA를 다시 확인합니다. 중간 실패 산출물은 보존하며 partial reuse를 허용하지 않습니다.

## 산출물 스키마

- `results.csv`: `panel,model,stage,development_not_blind,n,positive_n,zero_n,rmse,mae,bias,prediction_mean,prediction_rms,target_mean,target_rms,auc_share,auc_score,auc_classifier,contrast_ge099_fraction,component_mixed_ge099_fraction,high_y_n,high_y_SSE,high_y_SSE_fraction,site_macro_rmse`.
- 모델 이름: `CNN_<arm>`9개, `Tree_<arm>`8개, `Blend50_<arm>`8개, `R9_frozen_all5`. 새 R12 gate는 R9 신규 추론이 필요 없어25개 출력이며 나머지 비교 패널은26개입니다.
- `selected_policy.json`: `selected_arm,baseline_arm,weights,score_policy,gate_metrics,global_freeze_sha256,policy_sha256`. R12 pure CNN와 blend의 **score는 share**입니다. 개별 tree는 share/classifier/half-share-half-classifier를 각각 보고하고, legacy CNN control만 원 derived contrast score를 씁니다.
- `paired_cluster_CI.csv`: `panel,comparison,metric,point_difference,lo95,hi95,n_clusters,cluster_key,repetitions,valid_repetitions,reason`. 선택 blend−원형 blend 및 선택 blend−R9의 RMSE/AUCshare/AUCscore 차이입니다. background_group을500회 paired bootstrap하고, 원천 ID밖에 없으면 fallback의 범위를 표시합니다. 단일class AUC는 null이고, source cluster가 부족하면 CI를 만들지 않습니다.
- `predictions/<panel>.npz`: `sample_id,target_share,model_order,share[N,M],score[N,M]`. 숫자/Unicode만 저장합니다.
- `RESULTS.json`: 정책, 단계 증명, output SHA, 새 R9 baseline inference timing, 한계.

R9의 원 derived/mixed score AUC와 **R9 share-only AUC를 반드시 함께** 보여 줍니다. new score=share로 바뀐 효과와 모델 학습 개선을 혼동하지 않기 위해서입니다. R9 all5의 `2score−share`는 NN contrast와 classifier의 가중 혼합값이며 순수 q contrast로 부르지 않습니다.

CI는 고정 패널과 모델에 조건부이며 공유 reference 의존성·다중비교·gate 선택 낙관성을 제거하지 않습니다. 반복된 R11 진단과 source group 노출도 남습니다. 전체 결과가 좋지 않은 패널을 포함해 모두 보고하며, GPU 학습이나 공식 성능 개선을 보장하는 판단으로 자동 승격하지 않습니다.
