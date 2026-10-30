# 코드·데이터 용량 및 공개 후보 점검

측정시각: 2026-10-30T19:52:30.160814+09:00

재학습·추론·전송 없이 파일 메타데이터를 측정했다. symlink를 추적하지 않고 hardlink를 unique/allocated에서 중복 제거했다. APFS clone 공유 extents까지 독점 물리 점유량을 뜻하지는 않는다.

|구분|측정값|
|---|---:|
|regular_file_paths|106,994|
|directories|12,412|
|symlinks|5|
|logical_bytes|42.310 GiB (45,430,063,236 bytes)|
|unique_file_bytes|42.245 GiB (45,360,309,446 bytes)|
|allocated_file_bytes|42.498 GiB (45,632,274,432 bytes)|

|최상위 폴더|파일 수|논리 용량|할당 payload|
|---|---:|---:|---:|
|data.nosync|2,842|11.743 GiB|11.684 GiB|
|models|6,401|11.554 GiB|11.567 GiB|
|external.nosync|1,509|8.948 GiB|8.951 GiB|
|ops|2,948|3.817 GiB|3.823 GiB|
|reports|3,539|2.272 GiB|2.280 GiB|
|.venv_r17_eda|88,799|1.798 GiB|2.015 GiB|
|delivery|130|1.770 GiB|1.770 GiB|
|scratch|23|0.341 GiB|0.341 GiB|
|figures|41|0.024 GiB|0.024 GiB|
|reference_materials|54|0.019 GiB|0.019 GiB|
|eda_deep|39|0.008 GiB|0.008 GiB|
|submissions|51|0.006 GiB|0.006 GiB|
|pipeline|323|0.005 GiB|0.005 GiB|
|eda|24|0.002 GiB|0.002 GiB|
|audit|80|0.001 GiB|0.001 GiB|
|[root-files]|124|0.001 GiB|0.001 GiB|
|outputs|7|0.000 GiB|0.000 GiB|
|__pycache__|11|0.000 GiB|0.000 GiB|
|evidence|17|0.000 GiB|0.000 GiB|
|configs|30|0.000 GiB|0.000 GiB|
|__workflow__|2|0.000 GiB|0.000 GiB|
|external 2|0|0.000 GiB|0.000 GiB|

## 공개 후보

자동검사 통과 자체 코드/설정 529개, 6,668,779 bytes; 소형 집계근거 714개, 20,145,271 bytes; 경로/SSH 등 정제·검토필요 54개. 실제 업로드 내역이 아니다. 단순 휴리스틱으로 권리/보안을 보장하지 않아 최종 diff 검토가 필요하다.

기존 checkout은 sibling `work/underwater-vessel-noise-handoff`에 있다. 2026-10-28 게시기록은 raw/vendor weights 비공개, 자체 코드/보고서 Git과 owned-model release를 구분했다. 원본 Competition/VTUAD와 파생 음향 데이터는 재배포 조건 검토 없이 넣지 않는다. 출처·다운로드 방법·해시·스키마·소형 집계/fixture를 올리는 편이 적절하다.

## 제외 범위

- raw competition CSV/archive and source WAVs
- derived synthetic NPZ/parquet without redistribution review
- vendor weights/runtime/venv
- credentials, operational SSH state/logs/backups
- large row predictions/HTML unless explicitly curated

원본 음향 및 Test 값은 읽지 않았다.

## 현재 자체 코드·설정 크기

558파일, 7,002,497 bytes. root *.py/*.sh/requirements* and direct pipeline/*.py,ops/*.py/*.sh,configs/*.json/yaml; excludes nested snapshots and vendor runtimes

|세부 경로|파일 수|논리 용량 GiB|
|---|---:|---:|
|data.nosync/synthetic|1,019|6.040|
|external.nosync/vtuad_v1.3.2|157|5.378|
|models/pretrained_r5|4,124|5.177|
|data.nosync/cache|40|5.067|
|external.nosync/hearmyship_v1|1,152|3.095|
|ops/r16_pretrained_5090_deployment_r1|38|2.821|
|models/r16_production_feature_cache_r1|24|0.931|
|models/r16_production_feature_cache_r2|24|0.931|
|models/pretrained|4|0.910|
|ops/server_backups|255|0.896|
|models/pretrained_r16|15|0.727|
|models/r14_production_trees_r1|118|0.604|
|models/r14_production_nn_fold0_r1|142|0.594|
|reports/r17_clean_pipeline_r1|154|0.578|
|external.nosync/expansion_research|176|0.426|
|data.nosync/processed|1,680|0.399|
|models/tabicl_cpu_context16000|5|0.286|
|reports/r17_eda_audit_20261030|168|0.268|
|models/r9_mac_trees_r2|48|0.214|
|reports/r16_cross_operator_r1|44|0.173|
|reports/r16_support_temporal_r1|174|0.169|
|reports/r11_diagnostics|192|0.169|
|models/r16_multisupport_full_r1|91|0.161|
|models/r14_tree_full_r1|30|0.153|
|models/r6_fixed100_v1|566|0.150|
|models/r16_pretrained_cuda_bf16_r1|124|0.121|
|models/r16_catboost_peft_cpu8_r1|131|0.092|
|data.nosync/raw|4|0.091|
|data.nosync/extracted|15|0.089|
|reports/r13_cpu_correction_r1|22|0.086|
|reports/r15_forward_prototype_r1|13|0.086|
|models/r18_production_r1|90|0.081|
|models/tabicl_cpu_context1024|4|0.078|
|reports/r10_diagnostics|241|0.075|
|reports/r16_production_data_preflight_r1|76|0.071|
