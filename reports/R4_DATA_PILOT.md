# R4 fade-only data pilot

1024 fixed-seed examples × three roles × five folds; no models or timewarp.

| Fold | Role | Mean source seconds | Mean rendered seconds | Mean y | Zero fraction | Coarse TV vs TRAIN | Fine TV | Duration KS |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | train | 317.6 | 304.3 | 0.3861 | 0.1992 | 0.000 | 0.000 | 0.000 |
| 0 | validation | 295.5 | 295.5 | 0.3980 | 0.1992 | 0.046 | 0.128 | 0.140 |
| 0 | outer_validation | 276.5 | 276.5 | 0.3685 | 0.1992 | 0.040 | 0.082 | 0.141 |
| 1 | train | 277.8 | 277.8 | 0.3552 | 0.2178 | 0.000 | 0.000 | 0.000 |
| 1 | validation | 409.7 | 366.1 | 0.3887 | 0.2080 | 0.009 | 0.156 | 0.240 |
| 1 | outer_validation | 290.3 | 290.3 | 0.3874 | 0.1689 | 0.020 | 0.039 | 0.159 |
| 2 | train | 274.8 | 274.8 | 0.3839 | 0.1982 | 0.000 | 0.000 | 0.000 |
| 2 | validation | 278.9 | 278.9 | 0.3948 | 0.1758 | 0.016 | 0.033 | 0.108 |
| 2 | outer_validation | 399.5 | 360.9 | 0.3814 | 0.1963 | 0.011 | 0.168 | 0.213 |
| 3 | train | 338.1 | 322.4 | 0.3791 | 0.1973 | 0.000 | 0.000 | 0.000 |
| 3 | validation | 276.8 | 276.8 | 0.3669 | 0.2070 | 0.038 | 0.092 | 0.172 |
| 3 | outer_validation | 269.1 | 269.1 | 0.3782 | 0.1992 | 0.067 | 0.067 | 0.168 |
| 4 | train | 321.8 | 309.7 | 0.3920 | 0.2090 | 0.000 | 0.000 | 0.000 |
| 4 | validation | 275.5 | 275.5 | 0.3805 | 0.2100 | 0.022 | 0.056 | 0.180 |
| 4 | outer_validation | 281.3 | 281.3 | 0.4000 | 0.1709 | 0.019 | 0.088 | 0.111 |

Coarse bins: [120,300), [300,1380] seconds; fixed 0.5/0.5 positive prior. Fine bins: [120,300), [300,600), [600,900), [900,1380].
Rendered seconds = min(measured usable source seconds, 900); long inputs use one contiguous crop. Natural observed-mask coverage is reported separately in JSON.
Fine-duration/source differences are measured, not silently corrected. Existing R2 data and evaluations are unchanged.

[Full numeric pilot](R4_DATA_PILOT.json)
