# R9 SA background5fold·4개 검증stage 역할 — 실제 구축 감사

새버전은 **기존R4허용100,965창만** 사용한다. oldquality의TRAIN91,186+validation9,779와index집합이정확히같고SAcalibration/domain/그밖의oldpurged원천을복원하지않았다. VT509확장허가는SA예약복원허가로확대하지않았다. SA원본bank·옛dataset·모델은변경하지않았다.

## 분할 정의

기존sensor-week group258개와outerbucket0..4를그대로쓴다. 각fold는heldout한bucket을제외한4개bucket이TRAIN이므로**purge전약80/20**다. 기존nested3/5TRAIN을재사용한것이아니다. bucket원창수는{'0': 20121, '1': 20121, '2': 20170, '3': 20316, '4': 20237}다.

각heldoutbucket의whole sourcegroups를모델결과와무관한SHA순서로정렬해 **selection50%/ensemble_fit20%/ensemble_gate15%/report15%**를largest-remainder수로배정했다. 창수비율이나site비율이정확히같아진다는뜻은아니다. 작은site/group은일부역할에없을수있고이를맞추려고group을쪼개거나seed를재선택하지않았다.

모든역할에같은R6minuteQC를먼저적용했다. 원row_id와실제window/minute를join해후보분만NaN으로마스킹하고원래NaN을보존한다. 실제flag원인의물리적검증을의미하지않으며첫/끝분일괄삭제가아니다. min3observed미달은제외했다. 전체허용창중flag분6,254개/해당창6,254개,QC통과100,964개다.

## 24h·900s support purge

같은sensor의savedexportclock에서 `[start,start+900s)`사이공백이24h이상이어야한다. TRAIN은전체heldout후보와비교한다. heldout내에서는report를보존하고gate는report,fit은gate+report,selection은나머지3역할에대해경계창을제외한다. 이렇게높은우선순위는고정했으며어떤모델오차도보지않았다. clock은원exportclock이며검증된UTC로재명명하지않는다.

| Fold | purge전TRAIN | purge전heldout | QC제외 | TRAINpurge | selectionpurge | fitpurge | gatepurge | reportpurge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 80,844 | 20,121 | 1 | 1974 | 192 | 0 | 0 | 0 |
| 1 | 80,844 | 20,121 | 1 | 2775 | 96 | 0 | 0 | 0 |
| 2 | 80,795 | 20,170 | 1 | 2432 | 24 | 168 | 72 | 0 |
| 3 | 80,649 | 20,316 | 1 | 1895 | 24 | 0 | 24 | 0 |
| 4 | 80,728 | 20,237 | 1 | 2603 | 120 | 96 | 0 | 0 |

| Fold | 최종역할 | 고유창 | whole sourcegroups |
|---|---|---:|---:|
| 0 | train | 78,869 | 206 |
| 0 | selection | 9,417 | 25 |
| 0 | ensemble_fit | 3,417 | 10 |
| 0 | ensemble_gate | 3,511 | 8 |
| 0 | report | 3,584 | 8 |
| 1 | train | 78,069 | 208 |
| 1 | selection | 9,824 | 25 |
| 1 | ensemble_fit | 3,104 | 10 |
| 1 | ensemble_gate | 3,859 | 8 |
| 1 | report | 3,237 | 7 |
| 2 | train | 78,362 | 207 |
| 2 | selection | 10,271 | 25 |
| 2 | ensemble_fit | 3,432 | 10 |
| 2 | ensemble_gate | 3,274 | 8 |
| 2 | report | 2,929 | 8 |
| 3 | train | 78,753 | 207 |
| 3 | selection | 10,675 | 25 |
| 3 | ensemble_fit | 3,971 | 10 |
| 3 | ensemble_gate | 3,195 | 8 |
| 3 | report | 2,427 | 8 |
| 4 | train | 78,124 | 202 |
| 4 | selection | 10,183 | 28 |
| 4 | ensemble_fit | 3,487 | 11 |
| 4 | ensemble_gate | 3,462 | 8 |
| 4 | report | 2,889 | 8 |

독립pair별시간감사에서 **5fold모든role pair의24h미달0, sourcegroup/index교차중복0**이다. 각역할은최소2group이있고nonempty다. 생성할고정예제수보다고유창수가적다면replacement를명시해야하며생성예제수를독립원천수로세지않는다. 이builder는샘플링/합성NPZ를만들지않았다.

## stage사용과해석한계

- selection:epoch/checkpoint선택전용. ensemble_fit:미리선언한ensemble계수적합전용. ensemble_gate:동결후수용/거절판정전용. report:모든checkpoint/ensemblepolicy동결뒤최종보고전용이다.
- reportgroup은**그fold의새TRAIN/selection/fit/gate**에서제외됐다. 다른CVfoldTRAIN에서는같은source가사용될수있다. 과거프로젝트에서전혀본적없는새publiccorpus라는뜻은아니다. 여기서는report의source값을QC목적으로만읽었고모델예측을하지않았다.
- reference509는모든fold가공유한다. **reference-independentCV가아니며**,원래SA자연선박energyGT도새로생긴것이아니다. 새보고수치는배경source개발CV로표기해야한다.
- 새manifest/readiness는**backgroundsource metadata를합성할준비**라는gate다. 아직새generator정답/고정validationNPZ/실제모델학습전체ready가아니다.

## 경로·schema·재현

`fold_XX/{train,selection,ensemble_fit,ensemble_gate,report}_manifest.parquet`와heldout4역할union인`validation_manifest.parquet`가있다. 제외창은`excluded_manifest.parquet`,전체원pool은`background_pool_manifest.parquet`,모든fold사전역할은`proposed_roles_allfolds.parquet`다.

행은원래`base_window_index/window_id/background_group/site/sensor_group/start_export_clock`를보존하고`group/sensor/start/end/end_export_clock`alias,`boundary_mask_bits/masked_minute_indices/raw_observed_minutes/qc_observed_minutes`,원boundary·nominalbandwarning과새`split/role/background_fold`를제공한다. 숫자창은원bank에서globalindex로읽고동일`apply_minute_qc`를적용해야한다. 원값을0으로채우지않는다.

source_hash `5325acb42ecfc4983f9ca9fa6b54aa52bca5930129c4a1ca5d19004c88bfd4ff`. 원bankSHA `0ed303bd176424919ba7b1192950d4f95c3935f79bd79c3700aaf2daf80492a9`.

[새manifest](../data/versions/background_cv_r9/manifest.json) · [readiness](../data/versions/background_cv_r9/readiness.json) · [독립상세감사JSON](R9_BACKGROUND_CV_AUDIT.json) · [구현](../pipeline/background_cv_r9.py) · [builder](../ops/build_background_cv_r9.py)
