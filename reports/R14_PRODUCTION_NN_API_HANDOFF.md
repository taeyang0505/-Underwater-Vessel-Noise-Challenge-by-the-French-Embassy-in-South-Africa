# R14 production NN API・실행 인계

신규 사용자 승인으로 별도 production14 run을 시작했다. 과거 R13/R14 진단의 승인false·지표는 바꾸지 않았다. 첫 fast3는 iTransformer CPU2 / FDY MPS / Wide MPS의 epoch-round-robin100이며 같은 raw epoch를 공유한다. 이후 TFGrid MPS50/noES/cosine50을 실행한다. 모든 모델은 unweighted c0 ShareMSE+.25cell MSE, Adam/dropout.1이다.

## 고정 API

```python
from pipeline.models_r14_production import make_production_model
from pipeline.features_r14_production import ProductionFrontend
from pipeline.training_r14_production import restore_completed_model, predict_completed_model

model, frontend, summary = restore_completed_model(model_dir, device="cpu")
x, w = frontend(torch.from_numpy(db))  # CPU FP64 계산, float32 x/w 반환
with torch.no_grad():
    share, score, cell_mask = model(x, w)

pred = predict_completed_model(model_dir, db, batch_size=128, device="cpu")
# pred = {share, score=share, cell_mask, weights}; db-only, no fitting
```

NumPy raw db의 shape은[N,15,25]이며 결측은NaN이다. 14채널은 기존11+동일anchor3 수식이다. 원fold TRAIN PCA를 사용하며 CPU frontend 전체를MPS로옮기지 않는다. score분류결합은Root의동결앙상블단계가담당한다.

## 파일 계약

run: `models/r14_production_nn_fold0_r1`. 각kind의 `best.pt`, `latest.pt`, `best_selection_predictions.npz`, `epochs.jsonl`, `summary.json`, `EPOCH1_STABILITY_GATE.json`을 기록한다. summary는 실제 전체예산을완료한뒤만생기며 `epochs_run`, `best_epoch`, `best_selection_rmse`, `architecture`, `config_sha256`, `source_code_sha256`, `data_hash`, `fold`, `basis_sha256`, `artifact_sha256`를 포함한다.

`best.pt`와`latest.pt`는 CPU Tensor로원자저장하고 optimizer/scheduler/CPU+MPS+NumPy+Python RNG, 실사용raw receipt, config/approval/첨부SHA, fold/data/PCA를묶는다. Root의CPU3.12에서weights_only로복원가능함을실제MPS사전검증에서확인했다. `restore_completed_model`은완료예산/전체artifactSHA/코드/동일fold PCA/strict14stem을검사한다. 다른채널이나소형모델로fallback하지않는다.

fast3완료: `FAST_MODELS_FROZEN.json`(models={kind:{summary_sha256,best_checkpoint_sha256}},config/data/fold/codeSHA). full4완료: `ALL_FOLD0_MODELS_FROZEN.json` 같은구조. 각registry가생기기전미완료모델을사용하지않는다. Root는fast3+5foldTree로첫CSV를만들고TF완료후별도CSV를생성할수있으며원첫CSV를덮어쓰지않는다.

## cache와 실제 첫 epoch gate

새venv3.14에는pandas/pyarrow가없으므로 검증된source Python3.12 subprocess가numeric rawcache를만들고3.14학습엔진은NPZ만읽는다. sourceproducer는고정seed의새epoch를생성하며originaldata를수정하지않는다. raw cache는mtimeFIFO 최대2epoch만남긴다. TF가epoch1로되돌아가도99/100cache가누적되지않는다. 삭제전원NPZ SHA·sourceidentity·scheduleaudit는`epoch_receipts/`에영구보존하고각history/checkpoint에receipt경로/SHA를연결한다. cache는복구용학습state가아니다.

첫실제epoch의모든78,869개 primary를방문하고finite optimizer/선택검증/원자checkpoint가완료되면모델별EPOCH1_STABILITY_GATE를남겨같은fresh학습을고정예산까지지속한다. NaN/OOM/소스·config·approval변경은즉시FAILED를남기고중단한다. 실패partial재개/설정hotpatch는없다.

## 상태와 한계

`ops/r14_production_nn_launch_r1.json`이PID/log/config/preflight/QA/code핀을보존한다. 실행중상태는run의progress/run_state, 실제모델별epoch시간·RMSE·관측driver메모리는kind/epochs.jsonl을읽는다. 3step사전속도는전체학습시간실측이아니며, 첫실제epoch가끝난뒤조건부ETA를보고한다. 이엔진은17패널/Test/공식제출을읽거나게시하지않는다.

사전검증CLI는stage1 Fold0만명시지원한다(다른fold를조용히무시하지않음). 나중fold의생산학습은해당원천/PCA에결박된별도capability profile이필요하다. 현재run source는고정하며후속확장은새entry point로구현해야기존checkpoint sourceSHA가유지된다.
