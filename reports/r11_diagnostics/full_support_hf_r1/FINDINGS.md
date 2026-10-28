# R11-1 장시간 추가 × 고주파·St Francis·경계 분 대조

기존 R9 Fold0 모델5종과 동결 all5를 CPU FP32로 실행했다. 7개양성모드×500건과 음성5패널, 총5812개입력을 한 번의결합query로추론했다. 학습·Test추론·정책변경은없다.

![장시간 support와 score 반응](full_support_diagnostics.png)

## 주요 결과: 같은 정답을 유지한 all5 비교

|양성 support|HF상위 음성: RMSE / scoreAUC / shareAUC|St Francis 음성: RMSE / scoreAUC / shareAUC|
|---|---:|---:|
|r9_original|0.073414 / 0.868836 / 0.790728|0.059904 / 0.897949 / 0.709231|
|full_smooth12|0.084569 / 0.769172 / 0.778160|0.075888 / 0.737494 / 0.693692|
|full_smooth13|0.086390 / 0.750828 / 0.769780|0.078373 / 0.709949 / 0.679564|
|full_smooth15|0.116521 / 0.589224 / 0.640988|0.116927 / 0.467513 / 0.463295|
|native_long15|0.099587 / 0.625856 / 0.685920|0.095733 / 0.520109 / 0.536763|
|stationary15_control|0.130942 / 0.362940 / 0.558100|0.134412 / 0.207859 / 0.342987|
|full_smooth15_share030|0.156760 / 0.735768 / 0.819196|0.164967 / 0.715237 / 0.768917|

장시간15분으로 바꾼 양성과 동일 StFrancis 무추가 패널의 비교에서 AUC0.897949→0.467513으로 하락했다. y평균0.141007/RMS0.213753은 창별로 정확히 같았다. 음성 score가 양성보다 더 높은 strict pair 비율은53.25%였다. 이 조건은 낮은 AUC 실패를 재현했지만 RMSE0.116927은 Public0.217536과 같지 않다. 임의로 두수치를 맞추도록 prevalence/target을조정하지않았다.

Native장시간참조에서도 StFrancis 비교 AUC0.520109였다. 현재509풀에는 완전native15분이상 참조가7개이며500개독립장시간녹음이아니다. 모든7개는weak-class tug/fishing이고4분원자료를늘린smooth실험과별도다. 동일y라도참조identity분포가달라native대조하나로장시간외다른요인을제거하진못한다.

stationary15는평균스펙트럼을15분유지하는관측가능성반례다. 실제선박통과가항상정적이라는주장은아니다. 고정y=.30의별도팔은양성분포가다르며기존y동일비교와혼동하지않는다.

## 패널·물리적 통제

- 모든 양성배경은 원 R9selection에서 완전관측500개를 선택했고음성배경과ID중복0이다. baseline가족inside/short/edge50/25/25%, EQ±6·nativepeakSNR−10..10을사전고정했다.
- full-smooth는동일원minute행렬을12/13/15분으로선형리샘플링하고0.6~1.0의완만한envelope를적용했다. waveform반복이나새실측음원을만든것이아니다. 13/15팔은순수배경분이각각2/0개이고12팔은3개로별도표시한다.
- α는실제관측추가energy로동일y를풀었다. cell_share는A/(B+A), rawy는sum(A)/sum(B+A)다. 입력만0.1dB반올림하고w를그입력에서계산했다.
- HF음성은band18..24(1995..7943Hz) 선형합의시간max−median이selection고밀도6920개q90=10.725742dB보다큰500개다. 조용한대조는q50=0.999508dB미만500개다.
- StFrancis고밀도312개는모두15분관측이며실제metadata사이트다. Test의추정StFrancis-like분류와혼동하지않는다.
- 경계대조500쌍은같은원배경에서flag가있는첫/끝분만QC이전값으로복원했다. metadata의qc_observed_minutes와복원후실제입력관측량은다르므로후자는별도CSV/QA에계산했다. flag는오류후보이지모든flag분이실제오류라는확정라벨은아니다.
- 음성정답은추가A=0이다. 자연선박부재의정답이나선박/생물음클래스를판정한것은아니다. 일부음성패널끼리배경ID는겹치므로실험독립표본수로합치지않는다.

## 해석

새로운지원범위에따른취약점은분명하지만모든score가HFbump하나에절대의존한다고일반화하지않는다. NNscore는정확히0.5share+0.5×관측q의대역별시간최대−최소대비이고, HFbump는입력dB의최대−중앙값이다. LightGBMscore의후반절반은별도분류기다. 둘은다른통계량이다.
13→15분에서q05/q20등기준선이추가성분에오염되는정도가달라진다. trueq분위수·component·target구간별AUC를보존했다. 낮은y에서만주로붕괴하는지와큰y도실패하는지를구분할수있다.
장시간양성부족과고주파자연변동음성부족을다음학습ablation의별도축으로고정할근거다. 하지만이번challenge는공식Test의동일생성법이나원인증명이아니다. 같은사이트분포·knowny에서새모델의개선을확인하기전본학습이나제출정책을바꾸지않았다.

[전210조합·역전율](crossing_metrics.csv) · [target구간별AUC](conditional_target_bin_AUC.csv) · [분위수오염·실제관측량](support_and_quantile_pollution.csv) · [score구성요소](prediction_components.csv) · [독립saved산술QA](SAVED_OUTPUT_VERIFICATION.json) · [선언계획](predeclared_plan.json)
