# Fixed-CNN physical robustness stress

This is a predeclared own-INNER diagnostic, not a reconstruction of the official hidden generator. There was no fitting, policy selection or Test/OUTER/GPU use. All nine scenarios, including failures, are reported.

| Scenario | RMSE | MAE | Bias | AUC | RMSE minus identity |
|---|---:|---:|---:|---:|---:|
| identity | 0.090808 | 0.048218 | -0.009387 | 0.996721 | +0.000000 |
| common_gain_minus12dB | 0.092714 | 0.048014 | -0.018187 | 0.996585 | +0.001906 |
| common_gain_minus6dB | 0.091405 | 0.047796 | -0.013918 | 0.996678 | +0.000597 |
| common_gain_plus6dB | 0.091094 | 0.049268 | -0.004527 | 0.996684 | +0.000286 |
| common_gain_plus12dB | 0.092518 | 0.051061 | +0.000721 | 0.996553 | +0.001710 |
| common_tilt_minus4dB | 0.120976 | 0.062184 | -0.036700 | 0.994119 | +0.030168 |
| common_tilt_plus4dB | 0.072482 | 0.042686 | +0.008973 | 0.997163 | -0.018326 |
| drop_first_minute | 0.099518 | 0.055210 | -0.017507 | 0.995892 | +0.008710 |
| drop_last_minute | 0.094956 | 0.051185 | -0.009550 | 0.995789 | +0.004147 |

Common-gain cases preserve known q/y. Common-tilt cases preserve per-cell q but recompute pooled y after energy reweighting. Endpoint masks zero hidden q and recompute remaining y, including legitimate positive-to-zero transitions. Windows with fewer than three remaining observed minutes are protected by a rule independent of labels.

Per-reference AUC is undefined when that group contains only positives or only zero controls. The JSON uses null, rather than inventing a numerical AUC. Every case uses the same fixed model, prior and CPU settings.

Failure of these stress tests would establish sensitivity to the declared intervention on this synthetic population; it would not identify the actual hidden Test perturbation or explain a numerical fraction of the leaderboard error. Passing them would not establish external validity.

[Predeclared specification](R5_PHYSICAL_ROBUSTNESS_AUDIT.specification.json) · [Full metrics and source hashes](R5_PHYSICAL_ROBUSTNESS_AUDIT.json)
