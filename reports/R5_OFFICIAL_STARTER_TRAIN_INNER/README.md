# 공식 starter 상수 방법: 현재 TRAIN / INNER 진단

공식 notebook cell 6의 share=0.1, score=0.5를 수정 없이 평가했습니다. 학습·예측 clipping은 없습니다. 전체 notebook의 Test IO는 실행하지 않았습니다.

| Fold | TRAIN 행 | TRAIN RMSE | INNER 행 | INNER RMSE | INNER MAE | INNER bias(pred−truth) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 53,244 | 0.418974020 | 3,000 | 0.418974042 | 0.329637005 | -0.281998840 |
| 1 | 53,315 | 0.418974898 | 3,000 | 0.418976452 | 0.329640402 | -0.282001585 |
| 2 | 53,522 | 0.418976813 | 3,000 | 0.418973825 | 0.329639512 | -0.281998217 |
| 3 | 52,998 | 0.418973044 | 3,000 | 0.418973453 | 0.329636934 | -0.281999759 |
| 4 | 54,758 | 0.418973118 | 3,000 | 0.418973037 | 0.329639416 | -0.281997401 |

전체 INNER 15,000행의 pooled RMSE는 **0.418974162**입니다.

INNER는 신경망 early stopping·전처리 선택에 쓰이는 내부 검증입니다. 이 값은 OUTER OOF나 공식 점수가 아닙니다. 공개 Benchmark가 이 상수 starter와 동일 알고리즘이라는 공식 매핑은 확인되지 않았습니다.

TRAIN 5fold 합계는 동일 원천이 반복된 합성 행을 포함합니다. 독립 표본 수나 신뢰구간으로 해석하지 않습니다.

[전체 지표·target bin·site/reference/background group](metrics.json) · [공식 source SHA·코드·입력·예측 manifest](source_manifest.json)

공식 starter에는 별도 학습 모델이 없습니다. waveform→band-power helper는 이번 상수 예측의 regression 모델이 아니므로 실행하거나 교체하지 않았습니다.
