# R13 paired TRAIN: 고비율 추가 에너지 보강

**완료: TRAIN 생성·물리 검증만 수행했습니다. 모델 학습·평가·Test/공통 validation 값 읽기는 0회입니다.** 기존 R12 strict site/sensor/recorder/effective-mask matched v2 baseline은 외부 원본을 그대로 참조합니다. baseline `load_control`과 R13 baseline loader의 key/dtype/array bytes가 같습니다.

## 고정 법칙

TRAIN12,000개에서 zero2,400개(20%), positive9,600개를 유지합니다. long-positive2,880개(30%)도 그대로입니다. 양성의 정확25%인2,400개만 alpha를 변경합니다: long720/standard1,680. 기존 CNN6,000개 subset에도 동일비율 highshare1,200개(long360/standard840)입니다. standard는 원래 native inside/short/edge이며 실제 오디오가 반드시 짧다는 의미가 아닙니다. source·site·sensor·recorder·정확 mask·참조·EQ·배치·sample ID·presence는 모두 보존합니다. 나머지9,600개 행의 공통 NPZ 필드 bytes는 같습니다.

목표 u는 각 long/standard×CNN membership 층에서 독립 fixed-seed LHS `Uniform(.25,.85)`로 사전 고정했습니다. 임의 max(old y,u)나 share/minute-SNR cap을 넣지 않습니다. 따라서 일부 원래 높은 y는 낮아질 수 있으며 up/down 개수를 아래에 투명하게 표시합니다. Test/LB의 비율에 맞춘 분포가 아닙니다.

## 물리식과 정보 가역성

R12의 저장 source·EQ·native power·crop/route를 다시 조립하여 기존9,600개 positive의 raw db/q/y를 전부 정확히 replay했습니다. 반올림 db에서 역추정하거나 모델로 pseudo-label을 만들지 않았습니다. 같은 관측 O에서 `alpha=u·sum(B)/[(1−u)·sum(O R)]`, `A=alpha·O R`, `E=B+A`, `q=A/E`, `y=sum(A)/sum(E)`입니다. 같은 support의 합 비율은 관측분 mean 비율과 같습니다. raw q/y는float64로 검증하고 기존규격float32로 저장하므로 serialization roundoff는 별도 검사합니다. 입력만0.1dB로 반올림하며 w도 같은 관측 격자에서 계산합니다.

정답은 추가한 **녹음 전체**의 에너지 비율입니다. 순수 선박 에너지나 자연 선박 부재 GT가 아닙니다. 현재 baseline 소스는 median-background/full-reference-peak SNR 스케일을 사용하며 명시적인 share cap은 없습니다. 현재 y분포만으로 고비율 성능 문제의 원인을 cap이라고 확정하지 않습니다. 통계적 long bell은 기존 R12 법칙을 그대로 유지하며 새로 측정한 WAV라고 부르지 않습니다.

## TRAIN 분포

|층|행 수|baseline y평균|highshare y평균|baseline y중앙값|highshare y중앙값|
|---|---:|---:|---:|---:|---:|
|all|12000|0.162751|0.232796|0.062292|0.112272|
|positive|9600|0.203439|0.290995|0.111412|0.230012|
|long_positive|2880|0.317254|0.376694|0.239981|0.349823|
|standard_positive|6720|0.154662|0.254267|0.078816|0.167339|
|support_le2_positive|3998|0.123020|0.232588|0.056123|0.127875|
|support_ge12_positive|2337|0.316162|0.375678|0.242699|0.351551|

선정2,400개 중 증가 2117개, 감소 283개, 동률 0개입니다. 전체 y≥.25/.5/.7/.85 개수는 baseline `{'0.25': 2987, '0.5': 1244, '0.7': 448, '0.85': 55}`, highshare `{'0.25': 4654, '0.5': 2347, '0.7': 945, '0.85': 43}`입니다. family·관측지원·requested peak SNR·실제 global SNR·분위수 전체는 [분포 JSON](R13_DATASET_CONSTRUCTION.json)에 있습니다.

## 격리와 준비 상태

모든 source는 기존 R9 Fold0 TRAIN에 속하며 기존 source별24h purge 증명을 계승합니다. 새 reference/source/split은 추가하지 않았습니다. 공통 validation_fit/gate/eval/high-SNR는 복사 없이 기존 R12 파일 path/SHA만 참조합니다. 이전 TRAIN-only basis의 binding과 CNN subset을 재사용합니다. Fold1..4 동적 fullscale 자료가 준비됐다는 뜻이 아니며 그 원천 검증은 별도입니다.

`pipeline.dataset_r13.load_r13_arrays(folder,'baseline'|'highshare')`가 TRAIN dict를 반환하고 `validate_r13_bundle(folder)`가 manifest·source/file SHA·readiness·독립 물리 QA를 검사합니다. 모든 arm/validation/subset 경로는 project-relative, `file_sha256`만 bundle-relative입니다. 새 data hash는 `5c30c3fec031afbae5d97de46d9e1cc4857976410e11a6be780a60b4bc752662`입니다.
