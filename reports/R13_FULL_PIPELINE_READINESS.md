# R13 본체 5-Fold 기술 준비 상태

**기술 준비·자료 검증은 완료했지만, 생산 학습 승인은 false다.** 소형 CPU highshareP1이 모든 패널의 동시 개선 조건을 통과하지 못했고, 실제 CUDA 1epoch·VRAM 검증도 아직 없다. GPU 본학습을 실행하거나 임대가 필요하다고 확정하지 않았다. 이 문서는 성능 성공 판정이나 실제 5-Fold 학습 완료 보고서가 아니다.

`submission_r13.csv`는 아직 없다. 전체 5-Fold의 실제 학습·고정 정책·report 평가가 완료된 뒤 명시적인 infer CLI로 생성하며, 자동 업로드는 제공하지 않는다.

## 1. 실제 자료와 원천 경계

활성 bundle은 `data/synthetic/group_cv_physics_r13_variants_v2`, data hash는 `bb1dc465007a70798968ce14040b59643715538716b9e2c1f11c77e72c27d3c5`다. baseline/highshare 양쪽을 사전에 같은 배경·참조·기하·EQ·seed로 준비했다. 각 variant 15,000개, 전체 30,000개의 고정자료와 TRAIN 메타데이터·5개 PCA가 실제 저장돼 있다. highshare에서는 지정 양성의 alpha만 관측 에너지 비율 식으로 바뀐다.

|배경 fold|TRAIN primary / epoch|selection / ensemble_fit / ensemble_gate / report|해당 TRAIN PCA3 설명분산|
|---|---:|---|---:|
|0|78,869|1,000 / 500 / 500 / 1,000|84.8703%|
|1|78,069|1,000 / 500 / 500 / 1,000|84.8387%|
|2|78,362|1,000 / 500 / 500 / 1,000|84.3083%|
|3|78,753|1,000 / 500 / 500 / 1,000|85.5263%|
|4|78,124|1,000 / 500 / 500 / 1,000|84.2091%|

원래 R9의 배경 원천 역할·900초 창 support와 24시간 purge를 유지한다. 독립 검산은 각 fold의 그룹/ID 교집합 0과 최소 간격 24시간을 확인했다. **VTUAD 509개는 모든 배경 fold가 공유한다.** 전체 hull/day 연결 그래프는 하나의 component이므로 reference-independent 5-Fold나 새로운 맹검 일반화라고 부르지 않는다. 16개 dedicated ambient는 이 양성 참조풀에 넣지 않았다. 509개 안의 약한 background class도 포함된 whole-recording added-energy가 정답이며, 격리된 선박 에너지나 자연 선박 부재 정답이 아니다.

TRAIN primary는 epoch마다 전수 1회 사용한다. N이 5의 배수가 아니므로 매 epoch 정확 20% 정수 zero는 불가능하다. 사전에 고정한 floor/ceil 5epoch 주기로 정확 20%를 달성하고 100epoch도 정확하다. 각 stratum의 baseline zero 정수 quota를 유지하며 HF 우선 배정을 한다. stratum은 site/sensor_group/recorder/effective 관측 mask이고, 각 fold TRAIN만으로 q90을 구한다. 후보 부족은 명시적 shortfall 후 같은 stratum의 원 primary로 채운다. 이는 12k 자료의 외부 배경 교체와 다른 연산이다. zero를 받는 배경이 바뀌므로 양성 배경의 세부 분포까지 같다는 뜻도 아니다.

관측 양성 중 장시간은 30%, 그중 native15 계열은 25%, 나머지는 실제 5–14분 참조의 smooth bell 계열이다. native 장시간 7개와 5–14분 106개라는 제한을 남긴다. highshare 후보는 양성의 25%에 U(.25,.85)를 적용하며 `alpha = u*sum(O*B)/((1-u)*sum(O*R))`로 재계산한다. 모든 양성은 관측 support가 있도록 조건화되므로 원 R9의 무조건 geometry 법칙과 같다거나, 공식 숨은 합성기를 재현했다고 주장하지 않는다.

## 2. 본체·특징·손실

|신경망|실제 파라미터|
|---|---:|
|bg_cnn|1,491,329|
|tfgrid_fdy_conformer|2,563,813|
|fdy|130,765|
|itransformer|299,919|

P1은 relative8 + TRAIN-only PCA/곡률 3개인 11채널이다. P0도 같은 11입력 예산에서 3개를 zero-pad한다. 5개 PCA는 각 fold의 순수 TRAIN 배경 2,000개로 따로 적합했고 공유하지 않았다. 신경망은 100epoch/no early stopping/Adam/dropout.1, selection RMSE 최저 checkpoint, loss는 ShareMSE+.25×energy-weighted CellMSE다. presence/contrast BCE와 oracle q0 입력은 없다. q/y는 손실 정답만으로 쓰고 forward는 x,w만 받는다.

PhysicsLightGBM은 원 소형 CPU 실험의 466/673개와 다른 **5,252개** 특징을 사용한다. 4,125개 grid + 1,100개 시간 통계 + 27개 physics이며, 회귀와 독립 presence classifier를 별도로 적합한다. 최대 1,500 trees/ES40, Mac CPU 전용이다. TRAIN epoch1 전수를 deterministic하게 materialize하고 자동 subsampling하지 않는다. 100k행 특징만 약1.96GiB이며 histogram·모델·원자료 메모리는 별도다. 실제 본체 CPU tree 학습은 이번 준비에서 실행하지 않았다.

## 3. 실행·동결·추론 연결

`ops/run_r13_training.py`의 명시 단계는 nn fit → Mac tree → 해시/selection replay 검증 import-tree → ensemble → report다. nn fit은 사용 가능한 CUDA·생산 승인 정책·같은 장치의 실제 CUDA1epoch receipt를 모두 요구한다. CLI뿐 아니라 직접 run_training API도 이를 검사한다. 생산 NN run은 fold0..4 전체만 허용하며 subset fixture만으로 report를 여는 경로를 차단했다. 실패/부분 run 재사용과 hidden tree server fit은 허용하지 않는다.

selection 1,000개가 epoch checkpoint를 고른다. ensemble_fit 500개에서 single/equal-all/equal-neural 중 회귀 후보를 고르고 ensemble_gate 500개에서 안전 fallback을 확인한다. 별도 독립 classifier의 λ는 {0,.25,.5,1} 중 fit AUC로 고른 한 후보를 gate에서 λ0와 비교하고 미달 시 λ0다. **소형 CPU의 50:50 회귀 가중치나 λ.5를 본체에 자동 이식하지 않는다.** 같은 fit/gate를 두 유한 선택에 재사용하므로 선택 낙관 가능성이 남는다.

5개 fold의 모든 checkpoint와 정책 hash가 전역 동결된 뒤에만 report 배열을 연다. report 결과로 재적합하지 않는다. 저장된 report 집계는 배경 source-CV 개발평가이며 참조가 독립된 OOF가 아니다. 후속 infer는 완료 5fold, 원래 config bytes, runtime 정책, code closure, PCA, 모델 artifact ledger를 검증한 뒤에만 CSV를 읽는다. 각 fold 정책 출력의 동일 산술평균과 수치 clip[0,1]만 적용한다. 낡은8채널 모델/3,827 또는673특징 Tree를 몰래 대체하지 않는다.

활성 runtime 후보는 [`r13_runtime_candidate_highshare_P1_v2.json`](../configs/r13_runtime_candidate_highshare_P1_v2.json), file SHA `227ae599fe830ab37de5d8395c29ca7cad65d178dba0ce0fa5917f99839151ee`다. base config bytes는 그대로 보존했다. 이 후보의 `production_recipe_approved=false`이며 CPU 동시개선 gate 실패 이유를 기록했다. 후보 prepared 상태가 자동 launch 승인을 뜻하지 않는다.

## 4. 실제로 확인한 범위와 아직 확인하지 않은 범위

- 실제 R13 TRAIN으로 전체 크기 4NN 각각 Adam 1step, P1 입력 gradient, strict checkpoint reload bitwise, 물리 pooling 오차 0을 확인했다. 100epoch 성능·수렴 검증은 아니다.
- 숫자 fixture 2fold에서 전체 크기 4NN 1step씩, tiny Tree, Mac job import/replay, regression/score 정책, 전역 freeze 후 report, 변조·중복·부분 생산 fold 거부까지 연결됐다. fixture 모델은 생산에 사용할 수 없다.
- 실제 fold-TRAIN PCA와 5,252특징의 64행 numeric Tree 4rounds, 저장/복원·batch1/7 동일을 확인했다.
- frozen inference는 실제 Test 대신 numeric mock5fold로 32개 guard와 batch1/7 동일성을 확인했다. 실제 CUDA 모델의 batch 불변성이나 최종 Test 성능 증명은 아니다.
- 실제 TRAIN 64개를 spawn 4workers/prefetch2와 0workers로 읽어 전 tensor bitwise 동일을 확인했다. 첫 sandbox 시도는 Torch shared-memory manager 권한으로 실패했고, 허용된 로컬 실행에서 통과했다. 원자료·생산 설정을 바꾸지 않았다.

실제 Fold0 TRAIN78,869개 전체 합성+11채널 변환은 **1thread 115.44초(683.20행/초)**였다. raw 정답과 rounded-w pooling 차이 최대는 0.00209427로 고정 양자화 허용범위 안이다. 4workers의 전체 epoch 속도는 미측정이다. 64개 spawn 시작 포함 smoke는25.07초였으며 이를 전체 epoch 속도나 4배 가속으로 해석하지 않는다. 순차 속도를 100epoch×4모델×5fold에 단순 대입하면 약64 CPU시간의 전처리 작업량으로, GPU idle 병목을 점검해야 한다. 이는 GPU 학습 ETA가 아니다. 실제 CUDA1epoch에서 workers/prefetch·전처리/전송/forward/backward 전체 시간과 peak VRAM을 측정하기 전에는 기존 25시간 같은 값을 외삽하지 않는다.

## 5. 후속 명령과 보류 조건

현재 정책으로 아래 fit/tree는 승인 차단되는 것이 정상이다. 생산 승인과 실제 GPU 검증이 별도로 충족된 미래의 새 immutable 정책을 사용해야 한다. CUDA benchmark만의 1epoch 가중치는 저장하거나 본학습 재개에 사용하지 않는다.

```bash
python -m ops.run_r13_training cuda-benchmark --config configs/r13_full_training_v1.json --bundle-dir data/synthetic/group_cv_physics_r13_variants_v2 --runtime-policy APPROVED_RUNTIME.json --output reports/R13_CUDA1EPOCH.json
python -m ops.run_r13_training preflight --config configs/r13_full_training_v1.json --bundle-dir data/synthetic/group_cv_physics_r13_variants_v2 --runtime-policy APPROVED_RUNTIME.json --cuda-benchmark reports/R13_CUDA1EPOCH.json --output reports/R13_CUDA_PREFLIGHT.json
python -m ops.run_r13_training fit --runtime-policy APPROVED_RUNTIME.json --cuda-benchmark reports/R13_CUDA1EPOCH.json --preflight reports/R13_CUDA_PREFLIGHT.json --output models/NEW_R13_RUN
python -m ops.run_r13_training tree --executor mac_cpu --runtime-policy APPROVED_RUNTIME.json --output models/NEW_R13_MAC_TREES
python -m ops.run_r13_training import-tree --run-dir models/NEW_R13_RUN --jobs-dir models/NEW_R13_MAC_TREES
python -m ops.run_r13_training ensemble --run-dir models/NEW_R13_RUN
python -m ops.run_r13_training report --run-dir models/NEW_R13_RUN
python -m ops.run_r13_training verify models/NEW_R13_RUN
python -m ops.run_r13_training infer --run-dir models/NEW_R13_RUN --input EXPLICIT_QUERY.csv --output submissions/NEW_R13/submission_r13.csv
```

패키징 계획은 실행 핵심 코드30개와 보조 provenance를 포함한 Python 파일32개·prepared artifact156개·PCA·전체 bank SHA·추가 원천 provenance18개의 정확 bytes·별도 runtime/CPU evidence를 구분한다. 원시 WAV/이전 fitted weights/Test/credentials는 학습 패키지에 필요 없다. Linux CUDA에서는 해당 호스트용 Python/Torch/LightGBM 환경을 설치해야 하며 Mac runtime을 복사하지 않는다. Mac Tree 환경은 검증한 단일 Homebrew libomp를 사용하고 KMP_DUPLICATE_LIB_OK 우회는 사용하지 않는다. 패키지 전송/SSH/GPU 설치는 이번 준비에서 수행하지 않았다.

## 6. 증거와 이력

- [R13_RUNNER_FINAL_VERIFICATION.json](R13_RUNNER_FINAL_VERIFICATION.json): `1d4fe8c2ef1ab7109945a8bb3942170d208c6ab4af8acca481ff6d4f9244a583`
- [R13_REAL_FULL_MODEL_CONNECTION_V3.json](R13_REAL_FULL_MODEL_CONNECTION_V3.json): `eddfc17e0719950826eb5d1591e1de9546530b21e3539126dcd937d4766681d1`
- [R13_TREE_PCA_CONNECTION.json](R13_TREE_PCA_CONNECTION.json): `61a93cd4d046fecedad7591cf18a3fcca9934efc6804d96b74dd7c0e8e93d9e8`
- [R13_FROZEN_INFERENCE_VERIFICATION_V2.json](R13_FROZEN_INFERENCE_VERIFICATION_V2.json): `489d7b79204a3d022def92744469a773c3c4f20be758e8ad422d4af2fac65049`
- [R13_SPAWN_LOADER_CONNECTION.json](R13_SPAWN_LOADER_CONNECTION.json): `a04e0ad0a07275331e15d0f33b9cf3499391b0fb93254a7cdc183bd7ae94cff7`
- [R13_SYNTHESIS_FULL_EPOCH_BENCHMARK.json](R13_SYNTHESIS_FULL_EPOCH_BENCHMARK.json): `b5a5a1a06438c7b37648b9830bffcf66010c6ad205b4a02b10ae3ff5b6f11369`
- [R13_PREPARED_PREFLIGHT_FINAL.json](R13_PREPARED_PREFLIGHT_FINAL.json): `d92f9643bcf495a8503772162dba298d41581bc5d24fa6e115095ce52fed4408`
- [R13_PORTABLE_PACKAGE_PLAN_FINAL.json](R13_PORTABLE_PACKAGE_PLAN_FINAL.json): `94bb1e33f6e96f877aa88fc8c92b55f535e42f1084ada68879b4a3e478405f8c`
- [최종 독립 source/runtime 검산](R13_FULL_DATA_STATIC_INDEPENDENT_QA_FINAL.json): `6f5e68d80e46b9f388930f4d14d260663bc408ff9cfa36167608c1104ab6ffca`
- [CPU 성능·실패 조건](R13_CPU_CORRECTION_RESULTS.md)
- [기계 판독 readiness](R13_FULL_PIPELINE_READINESS.json)

초기 source-preflight 실패, 원 단일-variant provisional bundle, 예전 fixture/runtime v1은 이력으로 보존했다. 활성 증거는 위 final/V3/V2 파일만 사용한다. 새 bundle의 데이터 준비 완료는 CPU 전이 성능 문제를 해결했다는 뜻이 아니고, 새 전체 학습 결과도 아직 없다.
