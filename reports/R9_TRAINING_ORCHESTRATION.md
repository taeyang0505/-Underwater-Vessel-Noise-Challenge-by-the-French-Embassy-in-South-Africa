# R9 training orchestration and execution boundary

R9 data preparation and CPU connection fixtures are complete. Production CUDA training has not run. The default runner requires CUDA for neural work, and never silently fits LightGBM on the CUDA host. LightGBM is an explicit separate **Mac CPU** job. This document describes the implemented stages, rather than results from a completed R9 model run.

## Implemented stages

1. **Neural stage:** create all requested fold contexts, then fit `bg_cnn`, `tfgrid_fdy_conformer`, `fdy`, and `itransformer`. Each production model uses 100 actual epochs, no early stopping, Adam, dropout0.1 and a strict best-window checkpoint selected on the fixed `selection` role. Plateau learning-rate reduction remains enabled. Best-window, best-site-macro, last and every10th-epoch checkpoints are retained. The process exits normally as `awaiting_mac_tree` after all requested neural folds complete.
2. **Mac tree producer and verified import:** a separate fresh Mac job materializes deterministic TRAIN epoch1 in schedule order, with every admitted background primary exactly once. It fits the existing3,827-feature LightGBM regressor and diagnostic presence classifier, each capped at1,500 rounds with40-round early stopping on `selection`. There is no automatic row subsampling. The jobs need the config, prepared bundle and source banks, not NN weights; they can be performed while later GPU folds are training. Import checks completed receipts, source/config/code bindings, all artifacts and `selection` prediction replay at fixed absolute tolerance1e-6. Import never fits a model. No transport is implemented or performed by these scripts.
3. **Ensemble stage:** all requested NN and imported tree models must be complete. Every single member, equal-all and equal-neural are the finite candidates. `ensemble_fit` chooses a candidate and the best-neural fallback; only that fit-selected candidate can pass the separate `ensemble_gate` improvement test. The gate is validation selection, not a blind performance estimate. Every requested fold policy and every model artifact is then recorded in one immutable global freeze manifest.
4. **Report stage:** verify that global manifest before opening any report arrays. Evaluate every member and each frozen policy exactly once on `report`, retain unfavorable results, and write pooled-window and site-macro diagnostics. No report-dependent refit, checkpoint change or weight change occurs. A second report command fails before opening its report dataset.

The fixed role sizes per fold are1,000/500/500/1,000 (`selection`/`ensemble_fit`/`ensemble_gate`/`report`); five folds have15,000 fixed examples. Original SA background-group fold buckets and24-hour support separation remain enforced by the prepared source metadata. The TRAIN row counts are78,869/78,069/78,362/78,753/78,124. Thus100epochs×4NN×5folds means156,870,800 training-example visits. The four branches consume the same deterministic per-fold/per-epoch view schedules; this count does not mean156million independent recordings.

## Explicit commands, after deployment and GPU access

Run from the project root with the target environment's Python. The following commands are instructions; they were not executed as production training here.

```bash
python -m ops.run_r9_training preflight --device cuda --output reports/R9_GPU_PREFLIGHT.json
python -m ops.run_r9_training fit --preflight reports/R9_GPU_PREFLIGHT.json
```

On the Mac, with the exact same config/source/code/bank bytes:

```bash
python -m ops.run_r9_training tree --executor mac_cpu --output models/r9_mac_tree_jobs_v1
```

After an explicitly managed transfer of the completed Mac job directory to the CUDA project:

```bash
python -m ops.run_r9_training import-tree --run-dir models/group_cv_minute_full_r9_v1 --jobs-dir models/r9_mac_tree_jobs_v1
python -m ops.run_r9_training ensemble --run-dir models/group_cv_minute_full_r9_v1
python -m ops.run_r9_training report --run-dir models/group_cv_minute_full_r9_v1
python -m ops.run_r9_training verify models/group_cv_minute_full_r9_v1
```

`--folds 0` is available for an explicitly scoped run, but a completed single-fold run is not a five-fold production ensemble. The frozen inference contract requires the equal arithmetic mean of all five completed fold policies, using each fold's fixed member weights for both share and diagnostic score. It has no extra calibration beyond numerical clipping. This runner does not read Test or produce an official submission.

Every output is fresh. Only normal, completed stages of the declared staged workflow may feed the next stage. Failed, interrupted, partial or already-completed stages cannot be silently resumed. Per-fold and final root ledgers pin code, source identity, checkpoints, predictions and policies. One run writer and one R9 CUDA writer are allowed at a time. Errors are preserved in `FAILED.json`; there is no automatic restart, hardware control or hyperparameter adjustment.

## Validation actually performed

- [Numeric orchestration fixture](R9_RUNNER_VERIFICATION.json): two synthetic folds; four reduced-size neural architectures per fold, each with two real Adam steps; tiny CPU trees; independent tree-job import and replay; global freeze before either report; strict checkpoint restoration; complete root/fold ledgers; changed-artifact, duplicate-report, fresh-output and production-CPU rejection checks. Source-data I/O is blocked by an audit hook during this fixture. Its temporary weights are not production models.
- [Actual prepared-bundle structural preflight](R9_TRAINING_STRUCTURAL_PREFLIGHT.json): source banks and manifests verified,15,000 prepared fixed examples bound by SHA. Report files were byte-hashed only; report arrays and targets were not loaded by the orchestration preflight. Status is `structural_only_no_CUDA`, and `ready_for_production_training=false` on this Mac.
- [Real-data full-size model connection verification](R9_REAL_DATA_CONNECTION_VERIFICATION.json): a separate approved check used four actual TRAIN rows for one Adam step for each default-size architecture, then checked eight selection rows, finite gradients, pooling and strict reload parity. It is a connection fixture, not full training or a performance claim.
- [Packaging plan](R9_TRAINING_PACKAGING_PLAN.json): includes the exact R9 folders, two immutable numeric banks, source/code pins, and the historical metadata bytes required by the source manifests. Raw waveform archives, Test and old fitted weights are unnecessary. Mac runtime binaries must not be copied as a Linux environment.

The structural preflight is a source-integrity/hardware-inventory check, not a CUDA memory/throughput benchmark. Actual CUDA forward/backward, BF16 behavior, worker throughput, production100epoch behavior and server↔Mac artifact interoperability still require execution on the selected environment. No GPU rental duration is inferred from the tiny CPU fixture.

## Interpretation limits

All509 admitted VTUAD vessel-archive recordings are shared across background-fold TRAIN pools, including22 whose weak class is background. The16 dedicated ambient recordings remain separate. Targets measure the energy of the whole added measured recording; they do not verify isolated natural ship energy or natural ship presence. Restored reference recordings mean old reference holdouts cannot be claimed as independent for new R9-trained models.

Within each fold the report source groups are excluded from TRAIN, checkpoint selection, ensemble fitting and gate selection. Those background groups can occur in TRAIN in other CV folds; the common generator and shared reference bank are further limits on generalization. Report metrics therefore do not establish official Test representativeness or reference-independent performance. NN score and tree score are diagnostic combinations, not newly calibrated probabilities. The known low-share presence-loss tradeoff is retained by the predeclared loss rather than claimed harmless.

Source preparation, generation and validation parameters were not selected from Test distributions or leaderboard target recovery. No Test fitting, official submission, SSH, GPU job or external private-data transmission was performed by the orchestration work described here.
