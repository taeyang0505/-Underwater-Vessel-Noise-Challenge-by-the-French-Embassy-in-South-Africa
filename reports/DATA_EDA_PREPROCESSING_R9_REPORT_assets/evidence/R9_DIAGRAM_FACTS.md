# R9 최신 파이프라인 이미지·보고서 사실 명세

기준시각: 2026-10-26T06:18:02.949291+00:00. 이 문서는 현재 상태만 기술합니다. 이전 생성 전 상태는 현재 상태 설명에 섞지 않습니다.

## 현재 완료 상태

R9 데이터 bundle은 ready_to_train=true입니다. 고정 예제15,000개와5fold 동적 TRAIN loader를 생성·검증했습니다. runner/launcher, 분리된 Mac CPU tree 단계, ensemble global freeze, report gate도 구현됐습니다. 실제 source4개와selection8개를 잇는 기본크기4NN CPU1step 검증이 통과했습니다. **R9 full5fold100epoch CUDA 학습, 최종tree fit, 실제 ensemble/report평가는 아직 실행하지 않았습니다.**

## 원천과 크기

- SA: 1,715,779 source-export행,10sites,74열(62TOL+12metadata/QC). ZIP94,916,719B, 추출 member96,002,006B. 원 dur 합102,089,991초=28,358.3308h이며 독립적으로 중복 제거한 field시간은 아닙니다.
- SA 기본 bank:133641x15x25 float32,200,461,628B. R9는 원래 허용한100,965창/258sensor-weekgroups/7sites만 사용합니다. QC통과100,964창, 원boundary후보6,254분 중 retained에 실제 추가mask6,253분입니다.
- VTUAD strict:509reference +16ambientcontrols. header42.0467h, referenceheader33.1633h, 완전2s33.16h, 실제완전minute1978개=32.9667h입니다. 모든시간은clip합계이며 독립획득시간이 아닙니다.
- VT archive4,638,375,330B / WAV해제9,687,575,100B. 509중weakbackground22도 known whole-recording addition으로 포함하며 순수선박GT로 바꾸지 않습니다. 별도ambient16개는 자동y0가 아닙니다.
- 출처: [대회data](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/data), [VTUAD v1.3.2 / CC BY4.0](https://doi.org/10.5281/zenodo.23037895). SA 라이선스와 pressure/clock/sample-rate caveat는 원README를 따릅니다.

## 분할과 준비된 실제 데이터

5개 sensor-week bucket 중4개 TRAIN/1개 held-out으로 purge 전 약80/20입니다. held-out whole groups를 selection50%, ensemble_fit20%, ensemble_gate15%, report15% quota로 나눕니다. 같은sensor900ssupport사이24hgap을검증했으며 rolepair overlap위반은0입니다. 모든509reference는전fold공유하므로 **reference-independentCV가 아닌 배경source 개발CV**입니다.

|fold|TRAIN원창|selection원창|fit원창|gate원창|report원창|
|---|---:|---:|---:|---:|---:|
|0|78,869|9,417|3,417|3,511|3,584|
|1|78,069|9,824|3,104|3,859|3,237|
|2|78,362|10,271|3,432|3,274|2,929|
|3|78,753|10,675|3,971|3,195|2,427|
|4|78,124|10,183|3,487|3,462|2,889|

실제 고정 합성 예제는 각fold selection1000/fit500/gate500/report1000, 총15,000개입니다. NPZ20개24,707,614B, 내부NPYmember94,601,160B, 전체bundle71,244,935B입니다. 동적TRAIN은5fold합392,177개/epoch이며원창을각epoch한번씩씁니다. 100epoch방문수를독립녹음수로더하지않습니다.

## 적용된 sampler·정답·입력

실제 bundle에 적용된 `pipeline/sampler_r9.py` SHA: `cd104176b1a64c9be4b1ce9560a6af20ac41f26455adb2c92b4bb6088add4e63`. manifest 코드SHA와일치합니다. recipe는minute_first_r9_v1, native-minute cache/direct2s1527case bitwiseparity와worker4재현을확인했습니다.

Native32kHz/channel0→완전2sHannPSD/constantdetrend→25band선형power적분→30bins완전minute평균입니다. Inside/short/edge=.5/.25/.25, no-source20%, EQ<=6dB, full-native-EQpeak기준SNRUniform[-10,+10]dB, nowarp/fade/repetition입니다. 배치geometry는backgroundmask와독립입니다.

TRAIN-only LFbump>=q90 및 LF>max(safe,high)+3dB 후보를 no-source의절반에우선배치하되primary배경을복제하지않습니다. 부족분은metadata에기록하고나머지0slot을고정seed로선택합니다. naturalvesselabsenceGT는아닙니다.

E=B+A, q=A/E, y=ΣA/ΣE를관측mask안의raw성분에서계산합니다. 모델은round.1db에서relative8과같은w를만듭니다. Σw_raw*q=y와달리Σw_round*q는정확히같지않으며nominal상계약.0057564를검사합니다. 실제저장rawq/y오차max3.2465e-8, roundedgapmax.00165882는생성QA이지모델성능이아닙니다.

8채널은시간대역median/q20/q05상대level,관측인접delta,고정저주파동반상승,bandprofile,관측mask,sqrt(w)입니다. 절대db/label/ID/배경prior를입력하지않습니다.

## 모델·학습·앙상블

|모델|R9 trainableparams|
|---|---:|
|bg_cnn|1,487,873|
|tfgrid_fdy_conformer|2,562,085|
|fdy|129,901|
|itransformer|295,599|

WideCNN은parameter-freeobserved-frequencymeanresidual을qhead앞에추가했습니다. 전결측분context0,추가params0개입니다. 나머지3개도R6상대8을같이쓰는taskadaptation이며원논문fullSOTA성능주장은아닙니다.

NN:Adam.001/wd.0003/dropout.1/clip1,exact100epochs/noES,Plateau(.5/patience2/minlr1e-6). loss=siteweight*(shareMSE+.25energyweightedcellMSE+.05derived-scoreBCE). dataset의siteweights는N/(site수*site행수)이고mean1입니다. selection으로bestwindow/bestmacro를정하며last/every10도보존합니다.

CPU LightGBM:3827=3000grid+800stats+27physics,1500rounds/ES40/threads4/maxTRAIN100k. Mac CPU분리stage이며서버에서암묵적으로treefit하지않습니다. 선택기준은selectionclippedshareRMSE/classifierlogloss입니다.

Ensemblefit에서bestNNfallback과가장좋은사전후보하나를고정하고source-disjointgate에서epsilon.0001개선시에만채택합니다. gate실패시기존fallback,추가search/refit없음. 모든foldpolicy를globalfreeze한뒤에만report를엽니다. weights는nonnegative/simplex/affine[1,0]입니다.

## 고정된 추론 계약과 남은 실행

현재config는5개완료foldpolicy의산술평균을명시적으로고정합니다. perwindowround.1/relative8/w→4NN/CPUtree→각fold동결memberweight→5fold평균→[0,1]수치clipping입니다. share와diagnosticscore에동일한가중치를씁니다. publicCSVschema는ID+명시적minute-major375열입니다.

Runner의frozenpredictionhelper는구현됐지만productioncheckpoint가없으므로실제최종추론은미실행입니다. Testcohortfit/calibration/pseudo-label/source추정은없습니다. NNscore=.5share+.5maskcontrast,treescore=.5share+.5classifier이며자연선박확률보정이아닙니다.

현재bundle/source/model/engine/runnerfixture와실제source연결CPUgate는완료됐습니다. 실제GPU BF16 throughput/100epoch5fold학습및최종report평가는추후실행사항입니다. 기존R6epoch31localpilot는R9새학습결과가아닙니다.

[상세JSON·모든SHA](../../R9_DIAGRAM_FACTS.json)
