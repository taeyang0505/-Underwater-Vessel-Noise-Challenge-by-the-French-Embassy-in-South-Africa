# R8 데이터·특징·분할 파이프라인 근거

2026-10-26. 실제 코드와 frozen manifest를 대조했습니다. 원시 Parquet은 schema/footer, NPZ는 ZIP entry 크기와 NPY shape/dtype header만 확인했습니다. **Test 배열·cross 배열의 값/분포 통계·새 학습은 사용하지 않았고 기존 파일을 변경하지 않았습니다.** 모든 경로·함수 행·SHA와 실제 파일별 shape/bytes는 [JSON 근거](R8_FEATURE_PIPELINE_EVIDENCE.json)에 있습니다.

## 1. 원자료에서 모델 입력까지

원본은 10개 station Parquet, 총 **1,715,779행**입니다. 각 파일은 **74컬럼 = TOL 62개 + metadata/QC 12개**로 같은 schema를 갖습니다. TOL 컬럼이 존재한다는 사실만으로 해당 녹음기의 주파수 지원·calibration이 검증되는 것은 아닙니다. 현재 모델은 그중 아래의 **공통 25대역**만 사용합니다. 나머지 37개 TOL과 provenance는 보존되지만 현재 CNN 입력에는 들어가지 않습니다.

Metadata/QC는 `site`, `recorder`, `station_token`, `sample_rate_hz`, `file_kind`, `src_file`, `file_date`, `t`, `dur`, `qc_duration_ge55`, `qc_25band_finite`, `qc_25band_spread_ge6`입니다. Site·recorder·날짜·ID·reference 이름은 **분할, join, 검증, 집계용**이며 모델 predictor가 아닙니다. `sample_rate_hz`는 파일명 token에서 유도한 미검증 값이고, export-clock UTC 정합과 절대압력 calibration도 미검증입니다. [자료 설명](../data/extracted/DATA_DICTIONARY.md)

`prepare_sa.py`는 전체 행을 보존한 뒤 다음 조건으로 minute-window에 사용할 행만 고릅니다.

1. 공통25값 모두 finite, `-156.5` sentinel 없음, `55≤dur≤65초`, timestamp의 second≤5.
2. 같은 site/recorder/minute의 중복은 dur가60초에 가까운 행, 이른 timestamp, 원래 local index 순으로 하나를 선택.
3. export-clock의 고정15분 bin에 놓고 **관측된 완전한 분이3개 이상**인 window만 유지. 빈 분은 NaN으로 남기고 보간·반복·0dB 측정값 채우기를 하지 않음.

기존 준비 manifest에는 eligible rows **1,698,224**, 실제 유지 window에 포함된 rows **1,698,184**, 기본 window bank **133,641×15×25**가 기록돼 있습니다. `qc_25band_spread_ge6`는 보존된 screening flag이며 위 선택식의 선박 label이나 추가 hard filter가 아닙니다. 초기 day-hash split 표기는 이후 source-week/nested/R6 분할로 대체됐으므로 현재 TRAIN/INNER로 그대로 해석하면 안 됩니다. [실제 선택 코드](../pipeline/prepare_sa.py:31), [준비 manifest](../data/processed/sa_preparation_manifest.json)

### 공통25 대역과 index

정확한 nominal centre는 `c[i]=10^(1.5+0.1i)Hz`, `i=0..24`입니다. 코드의 audio 적분 경계는 **`[c/10^.05, c×10^.05)`**이며 centre 자체가 대역 경계는 아닙니다. 전체 적분 범위는 약28.18–8912.51Hz입니다.

| i | centre/컬럼 suffix Hz | i | centre/컬럼 suffix Hz | i | centre/컬럼 suffix Hz |
|---:|---:|---:|---:|---:|---:|
|0|31.6|9|251.2|18|1995.3|
|1|39.8|10|316.2|19|2511.9|
|2|50.1|11|398.1|20|3162.3|
|3|63.1|12|501.2|21|3981.1|
|4|79.4|13|631|22|5011.9|
|5|100|14|794.3|23|6309.6|
|6|125.9|15|1000|24|7943.3|
|7|158.5|16|1258.9| | |
|8|199.5|17|1584.9| | |

모든 index는0-based입니다. 원본 이름은 `TOL_<suffix>`, window는 시간축15×주파수축25 순서입니다.

## 2. 외부 audio 변환과 현재 실제 사용 원천

`audio_band_power`는 WAV의 실제 header를 읽어 **channel0**을 선택하고, 완전한2초 구간으로 나눕니다. 각 구간에 Hann periodogram, constant detrend, one-sided PSD density를 적용하고 대역 안의 PSD합×Δf로 선형 band power를 얻습니다. 30개2초 구간의 평균이 완전한1분입니다. 2초 미만 tail은 버리고, 1분 미만의 완전한2초 구간은 별도2초 bank에 유지합니다. 20초 음원을60초 실측으로 꾸미지 않습니다.

대역 upper edge가 Nyquist를 넘거나 유효 PSD bin이 없으면 지원 불가로 처리합니다. 단위는 **uncalibrated digital mean-square power**이며 μPa²로 검증된 값이 아닙니다. 현재 변환에는 waveform resampling·음원 반복이 없습니다. [audio 코드](../pipeline/audio.py:10)

초기 `prepare_external`에는 HMS·VTUAD 수집/변환과 초기 day/hull split이 있지만, **그 초기 split이나 전체 외부 bank가 현재 학습 pool인 것은 아닙니다.** R4의 실제 reference pool은 VTUAD v1.3.2 **119 clips/22 retained components**이고 HMS·NOAA·reserved source는 이 R4 pool에 들어가지 않습니다. R6는 그 Fold0의 기존 TRAIN+INNER만 허용해 **94 clips/17groups**를 사용합니다. 빠진25clips/5groups는 정확히 기존 Fold0 OUTER입니다. 원래 audio에 ambient가 섞일 수 있어 added reference energy를 순수 vessel-only energy로 부르지 않습니다. [R4 reference version](../data/versions/vtuad_reference_r4_v1/manifest.json), [R6 원천 감사](R6_GUARDRAILS_INDEPENDENT_AUDIT.json)

## 3. R4·R5·R6의 생성은 같은 법칙이 아님

공통으로 선형 power에서 `E=B+A`를 만들고 동일 관측 cells로 label을 계산합니다. 그러나 시간 배치·강도 결정·마스크 처리·학습 중 재생성은 다릅니다.

| 구분 | R4 정적 / R5 dynamic TRAIN | 현재 R6 law |
|---|---|---|
| 추가 강도 | explicit zero20%, 양성 목표 `u∈[.005,.95)`를 먼저 고른 뒤 `α=uΣB/((1−u)ΣR)` | full transformed reference peak-minute/background median-minute SNR을 **Uniform[−10,+10]dB**로 고르고 α 계산. 목표share 역산 없음 |
| 시간 | native linear power를120–900초로 bounded timewarp(1/3..3), 30초 fade, 관측 anchor가 있는 배치 | native 완전1분을 먼저 집계하는 family 또는 native contiguous short passage/CPA family. **timewarp 없음**, 배치 geometry는 NaN mask와 독립 |
| 주파수 응답 | smooth EQ 최대±4dB | 현재 frozen config는 **±6dB, lowpass 확률0**. 함수 default나 과거 v1의±12dB/.25를 현재 설정으로 쓰면 안 됨 |
| 결측 후 target | 양성 요청이 관측 구간에 겹치도록 구성 | source가 NaN 구간에만 놓이면 요청은 양성이어도 관측 added energy·share·presence가0일 수 있음. 이를 다시 뽑지 않음 |
| epoch | R4 저장배열. R5는 TRAIN의 clean primary를매epoch한번씩 쓰되 seed·reference/강도/순서를새로생성 | **저장한 동일 TRAIN8192행**을매epoch다시사용하며순서만바꿈 |

R6의 α는 `10^(SNR/10) × median_observed_t Σf B / max_full_reference_minute Σf R`입니다. Full-reference peak는 EQ/CPA 변환 후 window crop·관측 mask 전의 값입니다. Window 밖 peak의 tail만 보인다고 관측 tail을 다시 정규화하지 않습니다. [R4 추가식](../pipeline/passage_sampler_r4.py:198), [R5 epoch source](../pipeline/dynamic_passage_r5.py:117), [현재 R6 식](../pipeline/robust_sampler_r6.py:115), [현재 config](../configs/r6_cpu_guardrails_v2.json)

## 4. 정답·weight·score·metadata의 역할

`O`는 관측 mask, `A`는 실제 추가한 reference power입니다.

| 항목 | 정의와 역할 |
|---|---|
| `db` | 혼합물 `10log10(E)`의 모델 입력. R6는0.1dB 반올림 float32, missing은NaN |
| `unrounded_db` | R6의 반올림 전 float64 혼합물. R4 대조군 필드는 기존 sampler의float32 output을float64로cast한 것이므로 새 정밀도를 복구한 값은 아님 |
| `cell_share=q` | `A/(B+A)`; missing 밖은0. 학습 supervision이며 inference 입력 아님 |
| `share=y` | `Σ(OA)/Σ(OE)`; 반올림 전 알려진 성분 기준. 자연 배경에 이미 있던 vessel 전체의 비율 아님 |
| `presence` | `1[관측된 added energy>0]`. 자연 선박 부재 label 아님 |
| `no_source` | explicit A=0 요청 flag. 결측 이후 `presence=0`인 모든 경우와 동일하지 않을 수 있음 |
| `hard_negative` | TRAIN 원천 추출과감사용flag. Predictor나자연shipGT아님 |
| `w` | 관측 선형에너지를15×25전체합으로나눈pooling weight. R5 raw, R6 rounded Q(db)에서계산. dB 숫자합으로계산하지않음 |
| `score` | 모델의 diagnostic output. Targetlabel과다르고자연선박 calibrated probability도아님 |
| recipe/ID/group/site/time | 원천분리·재현·join·평가집계에만사용. CNN/LightGBM feature에입력하지않음 |

CNN은 `qhat=sigmoid(mask_head)`를 예측하고 `yhat=Σ(w qhat)`로 share를 계산합니다. R6 score는 band10..22에서 각 band의 관측 최소qhat을 뺀 최대상승 `contrast`를 사용해 **`.5yhat+.5contrast`**입니다. Missing은 contrast에서도 제외합니다. R6 LightGBM score는 `.5share+.5presence_classifier_probability`이며 share와 분류기는 별도로TRAIN fit합니다. [현재 CNN score](../pipeline/relative_models_r6_v2.py:21), [tree heads](../pipeline/cpu_ablation_r6_v2.py:141)

R6에서는 q/y를 raw 성분으로 유지하면서 w는 rounded값을 쓰므로 `Σ(w_Q q)=y`가 항상 exact인 것은 아닙니다. 명목±.05dB 오차의 일반 bound는 `tanh(ln(10)×.05/20)≈.0057564`이며 float32 저장오차를 추가로 검증합니다. 이 차이를 맞추려고 label을 Test나 출력 통계로 보정하지 않습니다.

## 5. R5 legacy8과 R6 current8의 정확한 채널

### R5 실제 raw 입력 계열

`x=db(float32)`, `O=isfinite(x)`, `m=관측전체dB평균`, `w_raw∝O10^(x/10)`입니다. Missing feature는0으로 채우고 별도 mask 채널로 구별합니다.

| 채널 | 식 |
|---:|---|
|0|`O(x−m)/20`|
|1|`Ox/100` — absolute dB정보 유지|
|2|`sqrt(w_raw)`|
|3|`O`|
|4|`O(x−band temporal q20)/20` — signed, ReLU 아님|
|5|인접 두분 모두관측된경우 `(x[t]−x[t−1])/20`, 첫분0|
|6|TRAIN background template의 `q0=clip(1−Bhat/E,0,1)`|
|7|`log1p(clip(Bhat/E,0,10))/log(11)`|

Prior는 알려진 TRAIN `B=(1−q)E`의 대역별 시간 median·centred log-shape를 aggregate한 최대32template입니다. Query에는 db와 frozen prior만 필요하고 q/y/ID를 받지 않습니다. High-anchor 조건 `centre≥2000Hz`는 실제 grid에서 **19..24(2511.9–7943.3Hz)**이며, `centre≥500`은12..24입니다. 이는 counterfactual feature이지 확인된 배경 정답이 아닙니다. Absolute 채널 때문에 R5 전체 표현은 common-gain invariant가 아닙니다. [legacy 채널](../pipeline/mixtures.py:157), [prior](../pipeline/background_prior_features.py:45), [query guard](../pipeline/grid_inputs_r5.py:117)

### 현재 R6 relative8

`Q=round(db.astype(float64),1)`, `med_f/q20_f/q05_f`는 관측 시간축별대역 통계, `med_win`은전체관측cell median입니다. `Δdb`는인접유효pair차이이며첫분0입니다.

| 채널 | 식 |
|---:|---|
|0|`O(Q−med_f)/10`|
|1|`O(Q−q20_f)/15`|
|2|`Δdb/10`, 유효인접pair만|
|3|band10..24에서 `relu(channel2)×tanh(mean_low(relu(Δdb))/3)`; low는0..8의유효pair평균|
|4|`O(med_f−med_win)/20`|
|5|`O`|
|6|`O(Q−q05_f)/15`|
|7|`sqrt(w_Q)`|

모든 output은float32이고, 없는 cell/대역의 feature는0이며 weight는0입니다. **TRAIN-fitted prior가 없는 db-only 개별 window 변환**입니다. q05/q20은국소통계이지자연배경정답이아닙니다. [base6](../pipeline/relative_grid_r6.py:89), [추가2](../pipeline/relative_grid_r6_v2.py:30)

Common additive dB gain은 **반올림 후** 수학적으로 상쇄되지만 임의 gain을반올림전에더하면 quantization경계를넘어달라질수있습니다. Static per-band offset은반올림이교환가능할때채널0..3/6에서상쇄되지만, spectral-profile채널4와에너지weight/sqrt(w)는상쇄되지않습니다. 모든채널이모든receiver차이에불변이라는설명은틀립니다.

## 6. 물리27개와 실제 LightGBM 특징 수

물리함수도 자체적으로float64의0.1dB반올림을적용합니다. Group minute는구성대역이모두관측될때만유효합니다. 지역별level은 `10log10(Σlinear power)`입니다.

| Group | index | 실제 centre 범위Hz |
|---|---|---|
|low|0..8|31.6–199.5|
|mid|10..19|316.2–2511.9|
|high|20..24|3162.3–7943.3|
|safe|15..22|1000–5011.9|

아래9개값에 각각 `_defined`와 `_valid_minutes`가붙어 **정확히27개**입니다. 순서는먼저값9개, 다음defined9개, 마지막count9개입니다. Undefined값은0, defined는0, count는실제유효분수입니다.

| 값 이름 | 정의 | defined 조건 |
|---|---|---|
|`bump_low_db`|low level max−median|완전분≥2|
|`bump_mid_db`|mid level max−median|완전분≥2|
|`bump_high_db`|high level max−median|완전분≥2|
|`bump_safe_db`|safe level max−median|완전분≥2|
|`corr_low_mid`|두regional dB시계열Pearson|유효pair≥3,두series중심제곱합각>1e−20|
|`corr_low_high`|동일정의,low/high|동일|
|`corr_low_safe`|동일정의,low/safe|동일|
|`std_broadband_db`|all25band level의population std(ddof0)|완전분≥2|
|`tilt_median_high_minus_low_db`|유효pair의median(highlevel−lowlevel)|pair≥1|

상수 series의 상관은“측정0상관”으로간주하지않고undefined로표시합니다. 값·defined·count가모두특징입니다. [27개정의](../pipeline/physics_features_r6_v2.py:25), [연산](../pipeline/physics_features_r6_v2.py:129)

현재 R6 LightGBM 입력은 **3827 = grid flatten3000(8×15×25) + 시간통계800(4×8×25) + 물리27**입니다. 시간통계는각채널·각대역의15분 mean/std/max/min입니다. 이미0으로채운grid에일반통계를계산하므로관측분만의nan통계가아닙니다. Mask채널도같이들어갑니다. R6 relative tree에는PCA·prototype이없습니다. Legacy8 대조tree에는legacy TRAINprior만영향을줍니다. [실제tree_features](../pipeline/cpu_ablation_r6_v2.py:116)

R5의별도LightGBM은 **1324 = fixed1258 + physics45 + TRAIN-prototype유사도8 + 추가physics13(고정12+TRAIN PCA재구성잔차1)**입니다. R7 Ridge대조군은위27개만쓰고TRAIN-onlyimputer/scaler를fit합니다. **1324·3827·27은서로다른모델/버전의입력수**입니다. [R5 tree](../pipeline/lgbm_r5.py:19), [R7대조](R7_SMALL_MODEL_CONTROLS_RESULTS.md)

최근local AV는요청에따라LF index0..12(≤501.2Hz)를사용했습니다. 이는물리27·hard-negative의LF0..8(≤199.5Hz)와다릅니다. 역사적EDA의KS/상관수치를다른정의로재현했다고표현하면안됩니다.

## 7. 원천을 먼저 분리하고 나중에 합성

R4는random-row KFold가아닙니다. SA는재사용recorder로연결된sensor_group×Monday export-clockweek단위를metadata-hash/load-balance로outerfold에배정했습니다. Outer f에대해 `(f+1)%5`가INNER, 나머지3sourcefold가TRAIN입니다. **명목3:1:1sourcefold이지정확한행60:20:20이아닙니다.** 같은sensor의15분support사이에24h이상gap을두도록purge하며OUTER를earlystop에쓰지않습니다. Reference는선택된hull/day/UTC overlap/동일bytescomponent를분리합니다. 다만VT자료의공유receiver까지독립이된것은아닙니다. [nested분할](../pipeline/group_cv.py:71), [R4계승](../pipeline/build_passage_cv_r4.py:95)

R6는기존R4 Fold0 TRAIN+INNER와복구가능한TRAIN boundary후보만허용하고기존OUTER/calibration/domain/sealed는제외합니다. Seed20261026의metadatahash로sensor_group4(StFrancis)를CROSS_SENSOR에고정했고, 다른sensor의기존INNERweek일부를CROSS_GENERATOR로분리했습니다. 같은sensor타역할support gap최소24h를재검증했습니다.

| 현재 R6 역할 | 배경원천 windows | Reference clips/groups |
|---|---:|---:|
|TRAIN|44,034|51/9|
|INNER|11,825|18/3|
|CROSS_GENERATOR|3,167|18/3|
|CROSS_SENSOR|17,380|7/2|

Reference는기존TRAIN12groups중hash로3groups를새INNER에두고9groups를TRAIN에유지했습니다. 기존INNER5groups는crosspanel3/2로분리했습니다. 이원천분리뒤에8192/1024등의합성quota를추출합니다. CROSS_SENSOR_GENERATOR는새원천role이아니라같은CROSS_SENSOR원천에다른family를적용하는패널입니다. [R6역할분리](../pipeline/source_split_r6.py:53)

Boundary처리는실제로window에선택된원row_id를분시간에join해서4,356분을가렸습니다. 첫/끝분일괄마스킹이아닙니다. 기존TRAIN제외3,545windows를복구했고97windows를추가24hpurge했습니다. source metadata의모든retainedwindow는관측분≥3입니다. TRAIN에는INNER가없는sensor_group2와일부site가있어, 모든site에서동일한분포검증을한것은아닙니다. [기존원천감사](R6_SOURCE_SPLIT_AUDIT.md), [실제v2 source](../data/versions/source_split_r6_guardrails_v2/manifest.json)

### TRAIN hard-negative와sitequota

TRAIN QC source44,034개만으로LF bump90분위수와 `LF>max(safe,high)+3dB`를정해후보4,035개를만들었습니다. 후보선택에는추가반올림을하지않으며LF0..8/safe15..22/high20..24와최소3관측분을씁니다. 이는물리27의최소2분·round.1정의와구분됩니다.

8,192개TRAIN draw중explicitno-source는1,638개이고그중819개가hard-negative입니다. 나머지7,373개는전체TRAINsite균등입니다. Hardnegative는6site에서각137/137/137/136/136/136회이며작은pool을cycling하므로 **819draws=428unique windows**입니다. 전체TRAIN8192도unique5,639입니다. Site별전체draw는1366/1366/1366/1365/1365/1364로거의균등하며, hard와baseline두균등pool을합쳐quota잔여가생깁니다. Reference는각허용role안에서group균형으로뽑습니다. 이절차는Testsite비율을사용하지않고,새자연선박정답을붙이는절차도아닙니다. [후보/추출코드](../pipeline/hard_negatives_r6_v2.py:60), [통합감사](R6_GUARDRAILS_INDEPENDENT_AUDIT.json)

## 8. 저장된 데이터 규모를 읽는 법

R4전체297,837은 **5fold TRAIN합267,837 + INNER합15,000 + OUTER합15,000**입니다. Fold별TRAIN은53,244/53,315/53,522/52,998/54,758이고각INNER/OUTER는3,000입니다. 여러fold에서원천역할이순환하므로297,837을서로독립인녹음수로부르면안됩니다. Canonical15개NPZ디스크합은434,330,282bytes(약414.21MiB),ZIP내NPY해제크기합은895,901,376bytes(약854.40MiB,header포함)입니다. `inner_validation.npz`는`validation.npz`의hardlinkalias여서다시더하지않습니다. 주의할역사적이름은 **NPZ `validation`=INNER지만 `validation_metadata.parquet`는OUTER join표**라는점입니다. INNER metadata는명시적`inner_validation_metadata.parquet`를사용해야합니다.

현재 R6경로는 `data/synthetic/r6_cpu_guardrails_v3_schemafix`입니다. 최초v1은geometry지원검사실패, 뒤버전은serialization schema교정등의이력을보존했으며새source모델fit은현재완성버전을사용했습니다. 저장된7개NPZ는:

| 파일 | 행수 | 역할 |
|---|---:|---|
|`train_r4.npz`|8192|R4-law대조TRAIN|
|`inner_r4.npz`|1024|R4-law대조INNER|
|`train_r6.npz`|8192|R6-law TRAIN|
|`inner_r6.npz`|1024|R6-law INNER|
|`cross_generator.npz`|1024|공통개발패널|
|`cross_sensor.npz`|1024|공통개발패널|
|`cross_sensor_generator.npz`|1024|공통개발패널|

합계는 **21,504 stored rows**이고NPZ디스크합은 **42,891,973bytes(약40.90MiB)**, ZIP내NPY member해제크기합은 **134,936,064bytes(약128.69MiB, header포함)**입니다. Recipe JSON·source metadata·모델checkpoint는이bytes합에포함하지않습니다.

R4/R6대조TRAIN은같은배경index·zero/hardnegative위치를공유하고,INNER도같은배경index를공유하지만각law의zero위치는다릅니다. Sensor와joint패널은같은primarydraw를공유합니다. 따라서21,504는독립원자료21,504개나25대역전체원시행수와같지않습니다. 실제100epoch5개CNN은이동일데이터를재사용한 **5개모델arm**이며새5fold전체학습을뜻하지않습니다.

현재파이프라인은unlabelled실측에자연선박정답을확정한모델teacher가아닙니다. 원래1,715,779행은보존된관측자료이고,학습정답은알려진추가성분의에너지산술에서생성합니다. 배경에원래있던선박소리와추가reference성분을구분해서설명해야합니다.
