# Safe local alignment audit

Challenge Test inputs were refused. Constants use synthetic TRAIN/INNER share labels; AV and KS compare only new R6 TRAIN with INNER. No main regression model, generator, ensemble or submission was changed.

## Constant predictions

| Synthetic INNER | Constant 0 RMSE | Constant .1 RMSE | TRAIN mean RMSE | Oracle INNER mean RMSE |
|---|---:|---:|---:|---:|
| R5_original_fold0_INNER | 0.491873 | 0.418974 | 0.309864 | 0.309864 |
| R6_guardrails_INNER | 0.199072 | 0.160120 | 0.159162 | 0.158872 |

MSE(c) = population variance + (c − INNER mean)² was verified. The oracle INNER mean uses its own labels and is a descriptive best-constant reference only. The unknown public Benchmark is not a target-moment constraint.

## R6 source-domain discrimination

Fixed source-group fivefold AV OOF AUC: **0.903065** (8,192 TRAIN / 1,024 INNER rows; 858 db-only features).

| Physical feature | TRAIN finite/total | INNER finite/total | TRAIN median | INNER median | KS statistic |
|---|---:|---:|---:|---:|---:|
| lf_31p6_to_501p2_bump_db | 8192/8192 | 1024/1024 | 4.923847 | 4.602042 | 0.078857 |
| high_3162_to_7943_bump_db | 8192/8192 | 1024/1024 | 4.716398 | 8.802792 | 0.201904 |
| low_high_broadband_db_pearson | 8192/8192 | 1024/1024 | 0.337112 | 0.428972 | 0.089966 |
| all25_broadband_db_population_std | 8192/8192 | 1024/1024 | 2.342279 | 2.264040 | 0.074585 |

All CV folds keep repeated primary background windows together. Source IDs are excluded from features. Physical features use complete regional minutes and report undefined values; AV adds explicit defined flags and a fixed zero fill.

This is a local source-footprint diagnostic, not a Test-gap measurement. AUC near .5 does not prove equality, and .85 is not an automatic correction threshold. KS p-values are descriptive because repeated source draws are dependent. No threshold, feature, model or generation parameter was selected from these results.

Exact definitions, fixed classifier settings, per-fold AUC, bindings and finite/undefined counts are in [JSON](R6_LOCAL_ALIGNMENT_AUDIT.json).
