# 실험 이력과 교훈 — 숫자는 측정 대상과 함께 읽기

## 대회 공통

Share 회귀 RMSE가 주 평가. score는 추가유무 순위AUC. 자연 선박 유무/선종 분류와 다른 문제다. Test 숨은y는 없으며 로컬 syntheticy와 다른 target distribution·conditional mapping이 가능하다. 상수 benchmark점수 하나로 Testy모멘트/양성률을 역산하면 안 된다. adversarialAUC가0.5라도 label조건부분포 동등성을 보장하지 않는다.

## 시간순 요약

| 버전 | 주요 변경/발견 | 상태·해석 |
|---|---|---|
| 초기/R1 | 원본1715779행 EDA, 외부VTUAD/HearMyShip/NOAA 확보, 합성/Tree·smallNN·foundation 비교 | 외부자료 추가흔적·길이·도메인 편중 가능성 |
| R1 foundation | TabICL outer query-batch 재현성gate실패 | 실패partial을 정상완료로 재사용 금지 |
| R2 | CNN, FDY/iTransformer변형, TabM,Tree,TabICL/Kumo 등9후보 | CNNOOF0.238545보다9모델ensemble0.245585 악화. train의support/SNRcap와inner/out생성규칙불일치가 확인됨 |
| R3/R4 | VTUAD중심동일recipe, source/purge, 297837건 train267837+inner15000+outer15000 | 5Fold파일수합계이며 독립297837 WAV가 아님 |
| R5 | 8채널WideCNN,raw/quantization/missingpilot,본체GPU | localINNER RMSE0.090808,AUC0.99672 vs officialRMSE0.348009,AUC0.53953. 검증대표성붕괴 |
| R6 | 상대입력,절대db/q0제거,분마스크/site복원,교차센서/생성기파일럿 | 교차0.110407,일부0.059377은 각각다른패널. LB미측정;0.89LB로환산불가 |
| R7/R8 | 제출경로재계산,starter변환/source선언검토 | 변환일치와 실제생성기동일성은다름. 경쟁자의출처선언만으로모델/recipe모두알수없음 |
| R9 | strictWAV풀확장,minute-first,safeensemble,4NNFold0+Tree5 | all5제출은 Tree도Fold0:0.782464193 /RMSE0.217535806/AUC0.555066666. Tree별도5Fold는0.768128675 |
| R10 | score구성·관측밀도·셔플·참조AB·2×2site/generator·ambient진단 | starter수식차이약1.88e-7,단순저주파site전이/참조AB만으로붕괴재현안됨. 시간셔플AUC거의동일:순서의존낮음 |
| R11 | fullsupport양성vsHF자연음성,LF/HF비대칭,저장Test출력포화,StFrancis밀도 | 일부스트레스에서AUC붕괴재현. 이것이비공개Test의실제구성임을증명하진않음 |
| R12 | 장시간/마스크일치HF교체/PCAablation,미니CNN+Tree | 고SNR스트레스개선·classifiercontrast포화감소. H1mask교란발견→HM1대조군 |
| R13 | highshare교체,독립classifier,calibration/목적함수비교 | highshare가일반bias악화. minifeature673 vs본체5252차이,분류회귀분리검토 |
| R14 | 본체14채널·PCA/LFanchor·Tree5·MacMPS | local17패널개선에도step1official0.705436197/RMSE0.294563802/AUC0.497866666로악화. CNN81.05%편중. TFGrid중단후storedpredaudit |
| R15 | forward조건sensitivity/독립score/규정검토 | Test통계를강제목표로해서generator를fit하지않는경계정리. marginalbump차이만으로Test정답확정불가 |
| R16 forward | LF/HF독립변형및다중연산자,깨끗한상대입력 | smallmix3축RMSE:0.08899→0.06774,0.08256→0.05802,0.10323→0.08568. 로컬조건별개선 |
| R16 support/clean | naturalclip길이,HFreplacement누수/결합대조,F1시간지속성 | 합성flat15만으로recipe를선택하지않음. 최종D0F1+H0.5/.5,미달native12/HF0.117553 vs0.11공개후진행승인 |
| 현재R16multisupport | A′75/nested5Fold/17ch/Tree5160reg2033clf/100epAdam | 데이터·Tree완료,NNFold0진행. 이후Fold1~4자동후속대기. 새LB미측정 |

## 공식 제출 결과 및 가중치

| 버전 | Score | ShareRMSE | scoreAUC |
|---|---:|---:|---:|
| 초기CPU제출 |0.658591231|0.341408769|이번자료에서확정안함|
| R5rawCNN |0.651991320|0.348008679|0.539533333|
| R9Tree5 |0.768128675|0.231871324|0.548088888|
| R9Fold0all5 |0.782464193|0.217535806|0.555066666|
| R14step1 |0.705436197|0.294563802|0.497866666|

R9 all5 order `[bg_cnn, tfgrid_fdy_conformer, fdy, itransformer, lgbm_physics]`, weights `[.1,.1,.1,.1,.6]`.5モデル全てFold0.Tree単独5Fold提出と同一モデルではないので差をNNだけの効果と断定しない.

R14step1 WideCNN81.05%/Tree18.95%.新R16はTree≥50%/各NN≤25%に制約し、scoreはshareと独立.古い良いローカルfitにより新Test良化を断定しない.

## 調べる文書

詳細な数値・条件・否定結果は`reports/`の各R番号MD、`reports/r16_*`のMD/JSONに保全.特に `R5_CURRENT_ISSUES_AND_GAP_REPORT.md`, `R10_LOCAL_VALIDATION_GAP_DIAGNOSTICS.md`, `R11_PINPOINT_DIAGNOSTICS.md`, `R12_CPU_ABLATION_RESULTS.md`, `R13_CPU_CORRECTION_RESULTS.md`, `R14_STEP1_PUBLIC_RESULT_REVIEW.md`, `FEATURE_FAILURE_REVIEW.md`, `R16_INFERENCE_POSTPROCESSING_REVIEW.md` を参照.

過去の推測を「原因が完全に規明」と断言した引用も履歴として残すが、当該条件の対照実験で分かったことと公式Test原因の同定を分ける.現在も全domain gap解消/LB0.90を保証できない.
