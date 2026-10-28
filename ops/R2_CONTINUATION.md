# Active R2 execution — continuation notes

User priority: finish this complete 5-fold/all-model pipeline and measure final OOF before any further improvements. No new experiments or settings changes. App automation `gpu` was DELETED at the user's request. Gemini consultation was rejected by auto-review, then explicitly declined by the user: NO external consultation; no message was sent. Do not stop/delete the Pod or make an official submission.

## Live run

- Started 2026-10-25 04:25:09 KST, unix1792869909.5714338.
- SSH root@103.196.86.97 port13475, ssh-agent (omit absent id_ed25519). Options BatchMode=yes, ConnectTimeout=15, StrictHostKeyChecking=accept-new, UserKnownHostsFile=/private/tmp/underwater_runpod_known_hosts. SSH/SCP approved with require_escalated and exact purpose. Never read keys/tokens.
- BASE /workspace/underwater_vessel_noise_r2; PROJECT BASE/project; OPS BASE/ops.
- Python /workspace/underwater_vessel_noise_r1/venv/bin/python (Torch2.8+cu128, RTX4090 24GB, effective RAM62GB).
- RUN PROJECT/models/group_cv_physics_v2_r2_adam.
- tmux underwater_train_r2, supervisor23980, driver23981.
- OPS/training_state.json, training_health.json, training_driver.log, supervisor.log, health_history.jsonl.
- Per-fold RUN/fold_XX/seed_42_training.log and seed_42/*history.json / run_manifest.json.
- Config configs/group_cv_physics_training_r2.json SHA92bcf84892839fdf9e1963d4448229923cf34a80a4dbe9ecd9eb6ae27a5452f4.
- Code ZIP delivery/r2_full_pipeline_code_v2.zip SHA40abd4c5340ca561382e6406dce9cb34ec8c34ddf5e4157ff7e5c296af43b3d2, 12 declared files. Local data/code/config audits and actual server final preflight PASSED. Do not modify pipeline/config/data during run.

## Frozen model/data configuration

Seven families/nine candidates: lgbm, lgbm_physics, lgbm_retrieval, tabm, cnn, fdy, itransformer, tabicl, kumo_large.
All NN Adam/max100epochs/early stopping7/dropout.1/seed42/batch128. No new domain augmentation in full run. CNN is the validated 8-channel background-prior residual replacement (arch4, 58,274params), all other task branches retain feature3 (TabM1258; FDY/iT6channels). Per-fold train-only B=(1-q)E aggregate32 prior, extra q0 and log1p(rawBhat/E clipped0..10)/log11; softened .5logit(clipq0 .01..99)+trained residual with .001 weighted residual penalty. CNN epoch0 may remain best; report honestly.
Prior file+embedded payload+data/package/seed/source/fold bindings checked before outer access. No reuse of fold0 prior or weights. Canonical/OPS forward/loss/gradients bitwise parity; historical12variants unchanged; tiny CLI+guarded outer+tamper tests passed.
TabICLv2(context16000,members4) and KumoLarge(context4096,members4) use fixed vendor weights, no gradient fine-tuning. Both FP32; strict seven-query-pair batch invariance gate remains atol3e-5/rtol1e-5.
Data data/synthetic/group_cv_physics_v2_r1 sourcef69509648c416cc8e8338a209ca10e728ae9430670fd3922696af40d646f9481; 267837train total; 15000inner+15000outer. Full source split +900s support/24hpurge preserved, inner/outer NPZ bytes unchanged; no Test/cal/sealed. Original1,715,779rows tracked, not certified natural-vessel groundtruth. Added-reference energy includes measured ambience.

## Completed development evidence (not final OOF)

First R1 run failed at TabICL AMP query gate; immutable source copies audit/before_native_generalization_r2, failed run/backups preserved. FP32 diagnostic fixes numerical gate.
Extra common/reference augmentation CNN pilots .320372/.319950 vs .321155; insufficient, not adopted. Augmented tree .289793 vs .283701, not adopted. Raw background map .55474, not adopted as direct estimator. Trained residual CNN pilot .288354 vs .321155, native .40750 vs .48048; adopted. BG54-feature tree .286407 vs .283701, not adopted. No more experiments.
Current R2 fold00: TabM best .289361 epoch2/stopped9. Integrated CNN best .28855447 epoch4/stopped11; FDY was at epoch2 last health message. No NaN/OOM/exception then. These are inner metrics, not OOF.

## Coordination

- scale_models: live health/convergence monitor, bounded ~60s checks; alerts/stage changes to root. No restarts/tuning.
- external_data: backup and results collection every2–3min, only completed ledger artifacts; local ops/server_backups and ops/r2_progress.
- data_quality: reports/R2_FULL_PIPELINE_REPORT.md + ops/build_r2_full_pipeline_report.py, only actual downloaded histories/curves. Portable bundle reports/R2_FULL_PIPELINE_REPORT_bundle.zip.
Root remains active until final full OOF and verified backups. Do not end after launch alone.

## Backups and finish

Existing remote helper /workspace/underwater_vessel_noise_r1/ops/backup_completed_training.py supports --run-dir RUN --backup-dir BASE/ops/backups. Collector ops/collect_r2_training_backups.py verifies ZIP SHA/CRC/all entries and ledger. It separately collects final global cv_report.json/oof_predictions.npz/oof_metadata.parquet (original backup helper excludes latter two), checking stable hashes/NPZ CRC/Parquet footer.
Final RUN/cv_report.json has pooled_oof for nine models+ensemble and fold_rmse_mean_std, plus full/boundary-excluded slices. Pooled RMSE is sqrt(pooled MSE), not arithmetic mean fold RMSE. Data/meta validation/source labels are allowed for reporting after frozen policy; never use outer to change settings.
Once all5folds complete: obtain verified local final report+OOF+all checkpoint backups; independently recompute RMSE/clean metrics from OOF arrays and metadata; have data_quality rebuild MD/charts with --final-report <absolute local cv_report.json>; open report in Codex; give actual OOF full/clean and key per-model metrics. No official submission. Mention OOF is development evidence (architecture chosen from fold0inner; sources reused), not a permanently untouched final assessment or guaranteed leaderboard score. Do not restart improvements until full measurement done. The latest human explicitly requests starting RMSE improvements AFTER the complete measurement; this is an ordered follow-on instruction, unless superseded by later steering.

## Latest live checkpoint

3/5 full folds completed; fold03 (fourth fold) has finished 3 tree models + TabM (inner .21771089, best1/stopped8). Its CNN background prior file exists, map preparation in progress; no alerts/OOM/NaN. Completed ensemble OUTER fold0 .1960051348578764 (boundary-excluded .1920638681052288), fold1 .33879642141911803 (.33848968248018874), fold2 .1692375900233939 (.16970633418522035). These are partial results only; DO NOT change settings or weights. Full9modelmetrics available local ops/r2_progress/fold_XX/outer_metrics.json, validated snapshots in ops/server_backups.
