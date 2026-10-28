# R1 fold 0: inner-error diagnosis

Only train and inner-validation arrays and explicitly named inner artifacts were read. No frozen code, data, checkpoint, ensemble weight or running process was changed. No outer/Test/calibration/sealed predictions or labels were inspected.

## Main finding

The largest error is on held-source native 15-minute additions, not the short-event route. Native-containing recipes are 1344/3000 (44.8%), but account for 93.5% of CNN SSE and 94.7% of Kumo SSE. These synthetic rows reuse a few measured references; the row count is not an independent source count.

| Inner recipe category | Count | Mean true | CNN RMSE / mean predicted | Retrieval RMSE / mean predicted | Ensemble RMSE / mean predicted |
|---|---:|---:|---:|---:|---:|
| mixed_native_short | 347 | 0.443978 | 0.412794 / 0.105466 | 0.308743 / 0.273195 | 0.317168 / 0.243952 |
| multi_short | 92 | 0.539223 | 0.100480 / 0.512753 | 0.215110 / 0.438296 | 0.193913 / 0.449431 |
| native_only | 997 | 0.484155 | 0.480476 / 0.079616 | 0.359190 / 0.256074 | 0.371662 / 0.226901 |
| single_short | 973 | 0.489902 | 0.134741 / 0.453774 | 0.225861 / 0.416333 | 0.205751 / 0.421846 |
| zero | 591 | 0.000000 | 0.050939 / 0.027268 | 0.209581 / 0.205032 | 0.184936 / 0.176846 |

Native-only CNN mean prediction is 0.079616 against mean truth 0.484155. Kumo native-only RMSE is 0.512835, versus 0.132149 for single-short recipes. This supports a native-source transfer/identification concern. It does not prove that all native recordings or all model classes will fail.

## Baseline and frozen ensemble replay

The training-mean constant is 0.375820 and gives inner RMSE **0.317663**. The frozen inner-fitted ensemble gives **0.281359**. We reproduced its stored policy without refitting. Its nonzero weights are retrieval 0.608555, TabM 0.256617, CNN 0.134828. This is an inner fitting score, not outer CV performance or an official score.

| Model | Full inner RMSE | Mean prediction minus truth | Native-containing SSE fraction |
|---|---:|---:|---:|
| lgbm | 0.305524 | -0.083156 | 56.2% |
| lgbm_physics | 0.283701 | -0.086436 | 68.9% |
| lgbm_retrieval | 0.283760 | -0.082117 | 66.9% |
| tabm | 0.289361 | -0.094595 | 64.1% |
| cnn | 0.321155 | -0.180753 | 93.5% |
| fdy | 0.326165 | -0.179504 | 91.9% |
| itransformer | 0.344797 | -0.191482 | 88.1% |
| tabicl | 0.344146 | -0.205258 | 93.1% |
| kumo_large | 0.343103 | -0.196458 | 94.7% |
| ensemble | 0.281359 | -0.098618 | 72.7% |
| constant_train_mean | 0.317663 | -0.011861 | 38.9% |

## Feature-path interpretation

The run uses feature version 3: six spectral channels (relative dB, absolute level, square-root mixture-energy weight, observation mask, **signed** per-band 20% baseline difference, and gap-aware temporal difference). TabM uses 1,258 fixed features; physics trees add 45 and foundations use 100. The spectral output is `sum(sigmoid(cell_logits) * observed_mixture_energy_weights)`. Native/short route tags, reference identities and true cell fractions are not inference inputs. Cell fractions are supervised targets, and route/SNR categories in this report are diagnostic only. There is no missing categorical native label requirement in this architecture.

## Loss is not the logged RMSE squared

Spectral training minimizes share MSE + 0.03 presence BCE + 0.03 energy-weighted cell-fraction MSE. Cell predictions are sigmoid bounded and share is their observed-energy-weighted sum. Missing-cell weights are zero. TabM uses the mean of member losses, while its reported prediction averages members; member variance also separates that loss from ensemble MSE.

`loss` averages training-mode minibatches as parameters change. `train_rmse` is computed after the epoch on the whole training set in eval mode, with dropout off. Therefore `sqrt(loss)` is not the reported train or inner RMSE. Best checkpoints are restored by inner RMSE. Last-epoch training error must not be paired with a different epoch’s best inner error.

| Model | Best epoch | Train RMSE at best epoch | Inner RMSE at best epoch | Last epoch train / inner RMSE |
|---|---:|---:|---:|---:|
| tabm | 2 | 0.221711 | 0.289361 | 9: 0.174117 / 0.332348 |
| cnn | 6 | 0.076204 | 0.321155 | 13: 0.060159 / 0.344052 |
| fdy | 9 | 0.088893 | 0.326165 | 16: 0.075229 / 0.328448 |
| itransformer | 7 | 0.108503 | 0.344797 | 14: 0.095474 / 0.360751 |

The same-epoch gaps are large even before later overfitting. Training reduced the objective, so inadequate epoch count is not supported as the main explanation. More epochs cannot be assumed to repair unseen-source or recipe mismatch. No implementation error in loss weighting, mask pooling or best-checkpoint selection was identified here.

## Conditional recipe shifts and causal limits

| Diagnostic | Train | Inner |
|---|---:|---:|
| Target mean / SD | 0.375820 / 0.316110 | 0.387682 / 0.317442 |
| Positive support at most 2 minutes | 6.10% | 41.72% |
| Observed 13–15 minutes | 71.96% | 89.90% |
| Positive max-minute SNR >20.001 dB | 0.00% | 12.45% |
| Boundary candidate flag | 0.00% | 1.57% |

The target marginals are similar. The train-only 20 dB cap, high-target native/multi routing, reference EQ, clean-background blending and coverage policy change the conditional inputs. Existing inner recipes remain byte-frozen and can contain high-share single-short events. These shifts are real, but the route error table contradicts a claim that high-SNR short events dominate this model failure. Native transfer is the first diagnostic priority. Source identity, sensor response, persistent ambient content and augmentation are confounded in one fold. Native ship annotations do not prove isolated vessel-only energy; continuous added ambience can be difficult to distinguish from an unknown background under common-gain augmentation. That is a hypothesis requiring controlled checks, not a proven impossibility.

## Native training supervision audit

Native-positive training rows are supervised, rather than excluded or assigned zero. Every nonzero route has positive `share` and `presence=1`; its cell targets reconstruct the saved share with the measured energy weights. This checks saved numeric labels, while the source code establishes that the added component is the reference (not the background).

| Train route | Count | Mean true share | Presence=1 | Max cell→share absolute error |
|---|---:|---:|---:|---:|
| mixed_native_short | 9315 | 0.394012 | 100.0% | 2.88e-07 |
| multi_short | 14169 | 0.604085 | 100.0% | 3.62e-07 |
| native_only | 14809 | 0.470819 | 100.0% | 2.92e-07 |
| single_short | 4216 | 0.191728 | 100.0% | 2.59e-07 |
| zero | 10735 | 0.000000 | 0.0% | 0 |

The training loop permutes all rows each epoch; it does not filter native routes. Spectral models receive both share MSE and the nonzero native cell-target loss, with presence BCE. This evidence rules out missing native target supervision in the saved dataset/loop path. It does not prove that the native reference is vessel-only, that the inverse fraction is identifiable from every mixture, or that unseen native groups share the training distribution.

## Future diagnostics; no changes performed

1. Use the already frozen inner predictions to compare native source groups, target bins and coverage, reporting group counts as well as row errors. Ground-truth recipe/SNR tags are diagnostic only and must not become inference routing features.
2. Separate training-source generalization probes from held-native-source error in a later authorized experiment. Do not move inner native references into training.
3. If testing recipe alignment, create an explicitly versioned source-disjoint physics-matched validation alongside the original frozen inner set. Keep both results visible; do not replace the validation simply to lower error. The 20 dB cap is a declared synthetic assumption and needs competition-recipe justification.
4. Diagnose the numerical TabICL batch gate independently. A precision fix does not by itself resolve the native underprediction described here.

## Reproduction evidence

- [Full bounded diagnosis JSON](../ops/r1_inner_error_diagnosis.json)
- [Training loss and history implementation](../pipeline/train.py)
- [Train-only physics sampler](../pipeline/physics_sampler_v3.py)
- [Frozen heldout preservation logic](../pipeline/build_physics_cv_v2.py)

Archive source: `ops/server_backups/snapshot_20261024T180856_204764Z_56e064fd.zip`. Only the explicitly whitelisted inner members listed in the JSON were opened. No ensemble fitting or model training was performed by this analysis.
