# R6 source split / minute QC audit

Seed `20261026`; metadata-hash CROSS_SENSOR sensor_group `4`.
허용 raw windows 76,503; active 76,406; QC/추가purge 제외 97.
실제 선택 원 row의 boundary minute 4,356개를 가렸습니다. 전체 첫/끝 분 일괄 가림이 아닙니다.
같은 센서의 타 역할 support gap 최소: 24.0 h.
CROSS_SENSOR/CROSS_GENERATOR는 잠긴 진단 역할입니다. 후보의 물리적 고장 원인을 확정하지 않았습니다.

## Site retention

```json
[
  {
    "site": "1 Richards Bay",
    "old_clean_train_windows": 30160,
    "old_inner_windows": 11431,
    "old_boundary_excluded_train_windows": 3044,
    "new_retained_windows": 44634,
    "retained_recovered_boundary_windows": 3044,
    "train": 33204,
    "inner": 8839,
    "cross_generator": 2591,
    "cross_sensor": 0,
    "new_QC_or_purge_exclusions": 1
  },
  {
    "site": "2 Durban",
    "old_clean_train_windows": 1629,
    "old_inner_windows": 672,
    "old_boundary_excluded_train_windows": 3,
    "new_retained_windows": 2304,
    "retained_recovered_boundary_windows": 3,
    "train": 1632,
    "inner": 672,
    "cross_generator": 0,
    "cross_sensor": 0,
    "new_QC_or_purge_exclusions": 0
  },
  {
    "site": "3 Cape_Recife",
    "old_clean_train_windows": 0,
    "old_inner_windows": 0,
    "old_boundary_excluded_train_windows": 240,
    "new_retained_windows": 240,
    "retained_recovered_boundary_windows": 240,
    "train": 240,
    "inner": 0,
    "cross_generator": 0,
    "cross_sensor": 0,
    "new_QC_or_purge_exclusions": 0
  },
  {
    "site": "4 Bird Island",
    "old_clean_train_windows": 1850,
    "old_inner_windows": 0,
    "old_boundary_excluded_train_windows": 22,
    "new_retained_windows": 1872,
    "retained_recovered_boundary_windows": 22,
    "train": 1872,
    "inner": 0,
    "cross_generator": 0,
    "cross_sensor": 0,
    "new_QC_or_purge_exclusions": 0
  },
  {
    "site": "5 St Croix Island",
    "old_clean_train_windows": 6850,
    "old_inner_windows": 2986,
    "old_boundary_excluded_train_windows": 32,
    "new_retained_windows": 9772,
    "retained_recovered_boundary_windows": 32,
    "train": 6882,
    "inner": 2314,
    "cross_generator": 576,
    "cross_sensor": 0,
    "new_QC_or_purge_exclusions": 96
  },
  {
    "site": "6 Riy_Banks",
    "old_clean_train_windows": 0,
    "old_inner_windows": 0,
    "old_boundary_excluded_train_windows": 204,
    "new_retained_windows": 204,
    "retained_recovered_boundary_windows": 204,
    "train": 204,
    "inner": 0,
    "cross_generator": 0,
    "cross_sensor": 0,
    "new_QC_or_purge_exclusions": 0
  },
  {
    "site": "7 St_Francis",
    "old_clean_train_windows": 12755,
    "old_inner_windows": 4625,
    "old_boundary_excluded_train_windows": 0,
    "new_retained_windows": 17380,
    "retained_recovered_boundary_windows": 0,
    "train": 0,
    "inner": 0,
    "cross_generator": 0,
    "cross_sensor": 17380,
    "new_QC_or_purge_exclusions": 0
  }
]
```

## Reference roles

```json
{
  "train": {
    "clips": 51,
    "groups": 9
  },
  "inner": {
    "clips": 18,
    "groups": 3
  },
  "cross_generator": {
    "clips": 18,
    "groups": 3
  },
  "cross_sensor": {
    "clips": 7,
    "groups": 2
  }
}
```

기존 OUTER/calibration/domain/sealed/Test 값은 열지 않았고 모델 학습·R4/R5 수정은 하지 않았습니다.
