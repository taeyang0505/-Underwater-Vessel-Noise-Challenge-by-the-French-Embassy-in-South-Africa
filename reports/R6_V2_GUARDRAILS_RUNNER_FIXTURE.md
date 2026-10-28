# R6 v2 runner numeric fixture

**PASS, numeric fixture only.** Four fresh Python execs independently fit 96 generated TRAIN and 36 generated INNER numbers: relative8/legacy8 CNN (2 epochs) and LightGBM (10 maximum trees). No competition arrays were opened; an audit hook rejects source-data I/O. Core source/config files remained unchanged.

Both tree processes remained Torch-free. Saved prediction replay, independently recomputed site-macro RMSE, TRAIN-prior tamper rejection, unexpected-prior rejection, immutable inputs and fixed-score formulas all passed. Temporary numeric weights under /private/tmp are not production models and must never be reused.

This fixture verifies the CPU utility, not the data split, hard-negative source construction, outer holdout validity or final source-driver freeze policy. Those require separate source/role audits.

[Detailed JSON](R6_V2_GUARDRAILS_RUNNER_FIXTURE.json)
