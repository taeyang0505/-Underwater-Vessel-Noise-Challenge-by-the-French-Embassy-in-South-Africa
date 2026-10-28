# R10 natural-variability background audit

**Known target is added component A=0. Natural vessels may already exist in every background.** The reported false-positive rate refers to controlled-addition evidence, not verified natural-vessel absence. The source subset is selection data, not untouched report or official Test.

## Frozen plan and panels

The plan was written before source numeric-window statistics/predictions. Fold0 selection source pool=9,417 windows; complete15-minute subset=5,452. Low bands are indices0..12, actual31.6228..501.1872Hz (nominal31.6..501.2Hz). Sum linear band energies, take10log10, then temporal population SD (ddof0); raw minute-QC observations receive no extra rounding or learned floor.

Subset q90=5.611784367dB; q50=1.854569034dB. The high panel is strictly above q90; control strictly below q50. Seed2026102605 ranks source keys by SHA256 without replacement. The fixed existing-selection syntheticpositive panel uses all452 eligible Nobs15 positives (<500), recomputed on CPUFP32. Its background IDs are excluded from both A0 candidate pools. After exclusion only499 high candidates exist, so all499 are used;499 control rows are sampled from2489 candidates. No resampling seed or target rule was retried.

Each panel has unique primary backgrounds and the three panels have zero shared background IDs. TRAIN overlap is zero by both window and background group; the original R9 sensor/export-clock24h support purge is inherited. Source groups and sites are shared across these selection panels, so this is not a group-independent or site-matched causal comparison. The panels are separate from the original TRAIN hard-negative candidate definition and were not added to training.

Prediction uses frozen Fold0 best-window checkpoints, existing db-only transforms and the same frozen all5 weights (tree0.6 and eachNN0.1). NN inference uses twoCPUthreads/FP32; the unchanged tree API retains its saved four-thread setting. Diagnostic threshold is fixed at score>=.5; share rates use strict>.03/.10/.20.

## Share error against known A=0

Each pair below is high-variability / control. Bias equals mean share because the known added share is0.

| Model | Mean share / bias | RMS share (RMSE) | Share q90 | score>=.5 rate | Conditional score AUC |
|---|---:|---:|---:|---:|---:|
|bg_cnn|0.029692 / 0.036016|0.078506 / 0.068370|0.082828 / 0.106937|0.162325 / 0.106212|0.895042 / 0.932644|
|tfgrid_fdy_conformer|0.015563 / 0.043417|0.036067 / 0.084186|0.023786 / 0.184470|0.066132 / 0.150301|0.960572 / 0.927492|
|fdy|0.021100 / 0.035466|0.051663 / 0.065996|0.046294 / 0.127739|0.140281 / 0.152305|0.890387 / 0.911753|
|itransformer|0.065970 / 0.036195|0.115889 / 0.058698|0.166402 / 0.082719|0.567134 / 0.080160|0.713830 / 0.924397|
|lgbm_physics|0.009743 / 0.018202|0.023472 / 0.037254|0.031414 / 0.048333|0.010020 / 0.018036|0.962913 / 0.929660|
|all5_frozen|0.019078 / 0.026031|0.034578 / 0.046577|0.052837 / 0.075326|0.018036 / 0.036072|0.963737 / 0.943972|

| Model | share>.03 rate high/control | share>.10 rate high/control | share>.20 rate high/control | Conditional share AUC high/control |
|---|---:|---:|---:|---:|
|bg_cnn|0.208417 / 0.288577|0.088176 / 0.104208|0.038076 / 0.022044|0.877161 / 0.833215|
|tfgrid_fdy_conformer|0.070140 / 0.254509|0.010020 / 0.128257|0.006012 / 0.084168|0.922841 / 0.807354|
|fdy|0.156313 / 0.264529|0.040080 / 0.124248|0.016032 / 0.022044|0.890015 / 0.822406|
|itransformer|0.543086 / 0.204409|0.158317 / 0.066132|0.080160 / 0.008016|0.756868 / 0.872910|
|lgbm_physics|0.112224 / 0.174349|0.008016 / 0.038076|0.000000 / 0.004008|0.900859 / 0.848221|
|all5_frozen|0.192385 / 0.234469|0.028056 / 0.048096|0.002004 / 0.006012|0.886202 / 0.852027|

## What the observations support

iTransformer is more responsive on this high-LF-variability slice: score>=.5 on56.71% versus8.02% of known-A0 windows; RMS share0.11589 versus0.05870. This is a measured association within the selected panels, not proof that low-frequency variability alone causes the response.

The pattern is not uniform across the ensemble. The frozen all5 has lower known-A0 mean/RMS and score-threshold rate on the high panel than control (1.80% versus3.61% threshold rate). TFgrid and LightGBM also do not deteriorate on the high slice by these metrics. Therefore the audit does not justify claiming that all branches universally confuse natural low-frequency variation with controlled additions. No weights or thresholds were changed after observing these results.

Conditional AUC mixes the same452 syntheticpositives with each499-row A0 panel. Those positives come from the selection population used for checkpoint choice, and R9 shares its509 reference recordings across roles. Conditional AUC is consequently a population-specific diagnostic, not an unbiased OOF/report estimate or natural-vessel-ground-truth AUC.

## Source composition and verification

| Panel | Windows | Unique backgrounds | Source groups | Site counts |
|---|---:|---:|---:|---|
|natural_high|499|499|18|1 Richards Bay:344; 4 Bird Island:69; 7 St_Francis:40; 5 St Croix Island:25; 2 Durban:21|
|natural_control|499|499|19|1 Richards Bay:299; 5 St Croix Island:120; 2 Durban:60; 4 Bird Island:17; 7 St_Francis:3|
|synthetic_positive|452|452|19|1 Richards Bay:275; 5 St Croix Island:84; 2 Durban:39; 7 St_Francis:27; 4 Bird Island:27|

The synthetic numeric oracle error was1.78e-15. Independent recomputation from saved predictions verified every mean/RMS/quantile/rate and direct pairwise AUC; max error=0. All original model/source/config pins were verified after inference and again at independent numeric verification. Original sampler, source data, policies and checkpoints remained unchanged. No Test/report, new training, network or external transmission was used.

[Plan](plan.json) · [Full results](results.json) · [All metrics CSV](metrics.csv) · [Panel selection](panel_selection.json) · [Independent numeric verification](independent_numeric_verification.json)
