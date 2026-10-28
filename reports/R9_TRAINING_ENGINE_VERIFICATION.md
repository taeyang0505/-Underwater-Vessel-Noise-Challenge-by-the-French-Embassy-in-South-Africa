# R9 training engine numeric verification

**PASS, CPU numeric fixtures only.** Four small fresh-weight R9 adapters each completed two real Adam steps. No source dataset, Test, OUTER, cross panel, GPU, SSH or external transmission was used. Numeric weights are temporary and must never be reused.

Checks: TRAIN set_epoch(1,2), fixed validation identities and actual input bytes, exact weighted objective with unweighted online RMSE, strict best-window/best-macro selection, saved last optimizer/scheduler steps, architecture/input/code identities, masked pooling, bitwise restored validation predictions and output reuse rejection. NaN, simulated OOM and negative loss weights stop with failed receipts. The original label is retained when rounded-weight pooling differs within its quantization bound; a larger target mismatch is rejected.

Production requires exactly100epochs with no early stopping and uses Adam1e-3/wd3e-4/dropout.1/clip1. The scheduler monitors validation window RMSE by default; site-macro is an explicit alternative. Optional TRAIN loss_weight changes the objective without duplicating backgrounds, and does not imply improved leaderboard performance.

This fixture did not run production100epochs, CUDA BF16, four spawned workers or actual source throughput. Those remain execution checks, not claims supported by this CPU fixture. Validation selection remains a development estimate, not an independent generalization guarantee.

[Detailed JSON](R9_TRAINING_ENGINE_VERIFICATION.json)
