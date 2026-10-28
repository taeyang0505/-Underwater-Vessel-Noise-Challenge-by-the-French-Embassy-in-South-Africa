# R4 bounded-timewarp data pilot

1024 fixed-seed examples × three roles × five folds; no models.

| Fold | Role | Mean source seconds | Mean rendered seconds | Mean y | Zero fraction | Coarse TV vs TRAIN | Fine TV | Duration KS |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | train | 344.1 | 510.0 | 0.3819 | 0.2002 | 0.000 | 0.000 | 0.000 |
| 0 | validation | 288.0 | 510.0 | 0.3819 | 0.2002 | 0.000 | 0.001 | 0.149 |
| 0 | outer_validation | 282.4 | 510.0 | 0.3819 | 0.2002 | 0.001 | 0.001 | 0.223 |
| 1 | train | 291.6 | 510.0 | 0.3819 | 0.2002 | 0.000 | 0.000 | 0.000 |
| 1 | validation | 425.9 | 510.0 | 0.3819 | 0.2002 | 0.000 | 0.000 | 0.283 |
| 1 | outer_validation | 288.9 | 510.0 | 0.3819 | 0.2002 | 0.001 | 0.001 | 0.112 |
| 2 | train | 286.6 | 510.0 | 0.3819 | 0.2002 | 0.000 | 0.000 | 0.000 |
| 2 | validation | 307.8 | 510.0 | 0.3819 | 0.2002 | 0.000 | 0.001 | 0.193 |
| 2 | outer_validation | 406.1 | 510.0 | 0.3819 | 0.2002 | 0.000 | 0.001 | 0.236 |
| 3 | train | 333.3 | 510.0 | 0.3819 | 0.2002 | 0.000 | 0.000 | 0.000 |
| 3 | validation | 299.4 | 510.0 | 0.3819 | 0.2002 | 0.000 | 0.001 | 0.116 |
| 3 | outer_validation | 312.6 | 510.0 | 0.3819 | 0.2002 | 0.000 | 0.000 | 0.159 |
| 4 | train | 331.4 | 510.0 | 0.3819 | 0.2002 | 0.000 | 0.000 | 0.000 |
| 4 | validation | 284.1 | 510.0 | 0.3819 | 0.2002 | 0.001 | 0.001 | 0.227 |
| 4 | outer_validation | 295.9 | 510.0 | 0.3819 | 0.2002 | 0.000 | 0.000 | 0.136 |

Coarse rendered bins: [120,300), [300,900]. Fine rendered bins: [120,300), [300,600), [600,900), {900}.
Rendered support is independently LHS-scheduled discrete Uniform{60,...,450} two-second bins. Source power passes once monotonically through cumulative-integral rebinning, ratio in [1/3,3]; original measurement UTC/duration is unchanged.
Original-source duration differences are nonblocking diagnostics. Rendered duration/target matching is checked against fixed gates; natural SA mask coverage is separate. R2 and the original fade-only pilot are unchanged.

[Full numeric pilot](R4_DATA_PILOT_WARP.json)
