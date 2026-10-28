# R5 validation failure audit

The local INNER result did not establish challenge generalization. The user-reported RMSE 0.348008679 and AUC 0.539533333 are recorded observations, not targets for fitting or Test-distribution inference.

## Verified findings

1. **No direct query q/y input was found in the audited CNN path.** Training supplies `changed.feature_db` to the common grid adapter and supplies known q/y separately as loss targets. Inference accepts db only. The seventh channel q0 comes from db plus a frozen TRAIN prior, not the query's true q. However, this prior is supervised: known TRAIN q is used to recover B=(1−q)E before fitting 32 aggregate spectral centres. It must not be described as independent of training labels.
2. **Source holdout is not simulator holdout.** Dynamic TRAIN and static INNER use the same `PassageSamplerR4`: exact 20% zero controls, independently shuffled positive share U(.005,.95), rendered durations 120–900s, whole-recording power timewarp, deterministic fade family and smooth reference EQ. About 83.64% of TRAIN positives and 85.96% of INNER positives have stretch outside 0.8–1.25. These are derived power trajectories, not newly measured independent passages or a verified propagation simulator.
3. **There is a strong same-generator spectral contrast.** A single energy-fraction diagnostic over nominal 1995.3–7943.3Hz has addition-presence AUC 0.78266 on TRAIN and 0.82793 on INNER. The ≤200Hz energy fraction correlates negatively with share (−0.60319/−0.59848). These associations are compatible with a recording-domain shortcut; they do not prove what the hidden data contain. No Test features were examined.

The positive component is an entire AIS-positive measured passage, including its ambient/instrument content. The label counts all added reference energy. Zero controls are unchanged South African backgrounds. This creates a potential foreign-recording-versus-original-background contrast. Whether the official generator shares that contrast is unknown.

## What the grouping establishes

Recorded TRAIN/INNER background-group overlap is zero and positive reference-group overlap is zero. Nevertheless, all four INNER sites and all four INNER sensor groups also occur in TRAIN. Thus the audit does not establish unseen-station, unseen-instrument or unseen-generator generalization. The selected reference pool's earlier grouping audit also must not be promoted into a claim of entirely new independent recording corpora.

## q0 is not an oracle

On all 3,000 INNER windows, the untrained frozen-prior q0 pooling has RMSE **0.51041**, presence AUC **0.23893**, and mean predicted share **0.73902** on the 600 known zero controls. q0 differs substantially from the true mask. The CNN may correct or invert this unreliable feature inside the same simulator; its presence is not evidence of valid source separation.

The fixed CNN has INNER presence AUC **0.99672**. Its contrast with the reported public AUC near chance is a strong reason to audit task/representation/generator validity rather than merely adjust a scalar offset. This does not attribute causality to one mechanism or rule out separate inference/label-definition errors being audited elsewhere.

## Interpretation limits

The best epoch used the full INNER for early stopping, so the local error is optimistic as a model-selection estimate. That bias and the subsequent calibration diagnostics are not experimentally shown to explain the entire difference between different populations. Offset and affine calibration were rejected by their predeclared source-crossfit gates; the frozen choice was identity. No new coefficients, hidden labels, Test priors or LB forecasts were derived here.

This audit read own TRAIN/INNER arrays, source metadata, frozen model/prior metadata and saved INNER predictions. It did not read Test/OUTER, fit a model, access GPU/SSH or alter the frozen source. See [numeric audit](R5_LB_FAILURE_VALIDATION_AUDIT.json). A separate predeclared CPU stress study will test physical nuisance sensitivity; it is not a reproduction of the official hidden generator.
