# 공식 베이스라인 대비 성능 저하 진단

작성일: 2026-10-25 KST. 완료된 제출 V1, R2 및 첫 R3 결과를 구분합니다. 이번 비교에서는 Test 입력을 읽거나 모델·가중치를 학습하지 않았으며, 추가 제출도 하지 않았습니다.

**판단:** 이전 제출의 성능 저하는 단순 모델 크기보다 **낙관적인 자체 합성 검증, 실제 과제와 합성 과정의 차이, 원천 일반화 실패**를 먼저 의심해야 합니다. 현재 R2에서도 장시간 참조의 과소예측과 inner 원천에 민감한 앙상블이 확인됩니다. 라벨 수식·파일 무결성 검증 통과만으로 공식 평가와의 일치가 입증되지는 않습니다.

## 1. 공식 점수로 확정되는 비교

공식 평가는 `share_added_vessel`의 `1 − RMSE`입니다. `score_added_vessel`의 AUC는 순위에 영향을 주지 않습니다. [공식 평가 설명](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge)

| 동일한 공식 Public 평가 | 표시 점수 | RMSE = 1 − 점수 |
|---|---:|---:|
| Benchmark | 0.753867825 | 0.246132175 |
| 이전 제출 V1, cJy4iTDK | 0.658591231 | 0.341408769 |

이전 제출은 RMSE가 **0.095276594, 약 38.71% 더 높습니다.** 공식 페이지의 Benchmark RMSE 세부 표시는 0.246132174이며, 위 마지막 자릿수 차이는 표시된 점수에서 역산했기 때문입니다. [공식 리더보드](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/leaderboard), [제출 영수증](../audit/submitted_v1_cJy4iTDK/outputs/zindi_submission_receipt.json)

현재 R2·R3는 이 리더보드에 제출하지 않았습니다. R2 OOF 0.245585가 공식 Benchmark RMSE 0.246132와 비슷하다는 이유로 대등하거나 더 우수하다고 해석할 수 없습니다.

## 2. 공식 starter는 어떤 모델인가

제공된 `maritime_starter_notebook.ipynb`의 cell 6은 모든 행에 아래 두 상수를 넣습니다.

```python
submission["share_added_vessel"] = 0.1
submission["score_added_vessel"] = 0.5
```

학습·앙상블·foundation model 추론은 없습니다. 자체 파일 `pipeline/foundation_baseline.py`는 우리 구현이며 공식 baseline 코드가 아닙니다. [공식 Data 페이지](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/data), [제공된 starter 원본](../data/raw/maritime_starter_notebook.ipynb)

다만 공개 리더보드에는 Benchmark와 이 starter를 동일 제출로 연결하는 코드·파일·설명이 없습니다. 따라서 **공식 starter의 상수 방법**과 **Benchmark 점수**는 각각 확인하되, 두 방법이 동일하다고 확정하지 않습니다.

## 3. 같은 자체 OOF에서 직접 비교한 결과

모델 추가 학습 없이 기존 OOF 15,000개 ID·정답에 상수 예측을 적용했습니다. 평균 상수는 각 fold의 TRAIN 정답에서만 계산했으며, outer 정답으로 상수를 맞추지 않았습니다.

| 방법 | 동일한 자체 OOF RMSE |
|---|---:|
| 공식 starter 방법: 항상 0.1 | 0.433795 |
| 각 fold TRAIN 평균만 예측 | 0.319928 |
| 현재 R2 CNN | **0.238545** |
| 현재 R2 inner-weight 앙상블 | 0.245585 |
| 이미 완료한 R3 정규화 앙상블 | 0.240328 |

R2는 자체 생성 과제에서 상수보다 수치상 크게 나은 예측을 하지만, 이는 공식 평가로의 전이를 입증하지 않습니다. 이 OOF는 이미 모델 개발에 사용한 평가 자료이며 새로운 독립 최종 평가도 아닙니다. 상수 0.1의 성능 차이를 이용해 숨겨진 Test 정답 분포를 역산하거나 생성 파라미터를 조정하지 않았습니다.

[이번 비교 수치·입력 SHA](OFFICIAL_BASELINE_COMPARISON_FACTS.json), [재현 코드](../ops/compare_official_baseline.py), [R3 독립 검증](R3_SHRINKAGE_INDEPENDENT_VERIFICATION.json)

## 4. 문제별 판단

### A. 이전 제출의 검증 점수가 지나치게 낙관적이었다 — 확정

V1의 자체 validation RMSE는 **0.080484**, calibration 0.090265, domain holdout 0.116081이었지만 공식 RMSE는 **0.341409**였습니다. 당시 검증은 실제 점수의 신뢰할 만한 대리 지표가 되지 못했습니다. 그 차이가 전부 학습 과적합인지, 합성 생성 과정·배경·참조의 차이인지 공식 정답 없이 분해할 수는 없습니다.

V1은 train 16,000개, CPU, 신경망 각각 **실제 8 epochs**의 실행입니다. 가중치는 TabICL **53.54%**, Kumo **28.45%**, CNN **17.13%**, iTransformer **0.88%**이고 나머지는 실질적으로 0입니다. 약 **81.99%가 frozen foundation 모델**에 실렸습니다. 모든 후보가 같은 비중으로 기여한 것도, 100-epoch/early-stopping Group CV 본학습 결과도 아닙니다.

일부 소형 모델은 8번째 epoch에 best를 기록해 당시 학습 예산 부족 가능성은 있습니다. 그러나 foundation 가중치가 대부분이므로 낮은 점수 전체를 신경망 epoch 부족만으로 설명할 수 없습니다. CPU 실행이었던 V1의 성능을 이후 별도 GPU run에서 발견된 FP16 query-batch 오류 탓으로 돌릴 근거도 없습니다.

V1 배경 분할에는 같은 sensor/day 의존성이 남았다는 감사 기록도 있습니다. 이를 동일 창 중복이나 정답 누수로 확대 해석하지 않습니다. 현재 R2의 원천 분리·24시간 purge는 별도로 검증되었으므로 두 버전의 상태를 혼동하지 않습니다.

근거: [V1 run manifest](../models/fullsources_cpu_a6d7f3a5_seed42/run_manifest.json), [동결된 V1 정책·자체 검증](../audit/submitted_v1_cJy4iTDK/models/ensemble_foundation/postprocess.json), [분할 감사](data_quality_v2_audit.json).

### B. 합성 수량에 비해 독립 장시간 참조가 적고, 공식 과제와 원천 구성이 다르다 — 강한 근거

공식 starter는 추가 성분을 VTUAD에서 가져와 시간 trajectory·상대 acoustic channel·energy를 변형했다고 고지합니다. 정확한 generator, 변형 범위와 목표 share 분포는 공개하지 않습니다. 공식 passage 참조도 주변음을 포함할 수 있으므로 주변음 포함 자체가 라벨 오류인 것은 아닙니다.

현재 장시간 native 참조는 **11 clips / 5 connected components**입니다. 이 중 NOAA는 8 clips / 4 components입니다. 각 fold의 TRAIN native component는 3개뿐입니다. 합성 예제 267,837개는 독립 참조 267,837개를 뜻하지 않습니다. 따라서 현재 OOF에는 공식 VTUAD 기반 과제와 별도로 NOAA 원천 전이의 어려움이 크게 들어 있습니다.

완료된 R2에서 다음을 확인했습니다.

| 원천 구간 | OOF 행 수·비중 | 앙상블 RMSE | 전체 앙상블 SSE 비중 |
|---|---:|---:|---:|
| Native 포함 전체 | 6,695 / 44.63% | 0.324812 | **78.08%** |
| NOAA HI component | 1,322 / 8.81% | 0.495559 | 35.89% |
| NOAA FK component | 1,333 / 8.89% | 0.418137 | 25.76% |

**두 NOAA component의 17.7% 사례가 전체 SSE의 61.65%를 차지합니다.** Native 포함 구간의 정답 평균은 0.491620인데 예측 평균은 0.288295로, bias가 **−0.203325**입니다. 장시간 추가 성분을 배경으로 취급하는 방향의 일반화 실패가 주된 오차로 보입니다. 이것은 관찰된 잔차 패턴에 대한 해석이며, 특정 참조를 정답 오염으로 확정한 것은 아닙니다.

NOAA의 어려운 사례를 삭제해 기존 OOF 점수를 좋게 만드는 것은 해결이 아닙니다. 기존 평가를 유지하고, 공식 공개 가정에 맞춘 VTUAD 중심 평가와 외부 원천 stress 평가를 추가로 구분해야 합니다. Test 원본 음원·배경의 개별 정체를 찾아 맞추는 방식은 사용하지 않습니다.

근거: [원천·route별 실제 오차](R3_OOF_ERROR_AUDIT.json), [데이터 규모](group_cv_physics_data_facts.json), [추가 VTUAD 원천 감사와 한계](R3_REFERENCE_COVERAGE_AUDIT.md).

### C. Train과 고정 검증의 생성 조건이 어긋나 있다 — 확인된 설계 문제

R2가 사용하는 physics r1 TRAIN은 short-support share cap, 분당 20dB SNR cap, smooth EQ, 배경 blending을 적용했지만 이전 inner/outer 파일은 그대로 보존했습니다. 이는 기존 평가를 유리하게 바꾸지 않는 장점이 있으나, TRAIN에서 제한·축소된 조건을 평가에 남기는 조건부 차이를 만들었습니다. share의 전체 평균이 비슷해도 시간 support·참조 성격·SNR 조건은 다를 수 있습니다.

전체 recipe와 실제 저장 정답을 다시 집계한 결과입니다. TRAIN 비중은 5개 fold의 학습 역할을 합한 수치이며 독립 표본 비중으로 해석하지 않습니다.

| 조건 | TRAIN | 고정 inner |
|---|---:|---:|
| 단일 짧은 참조의 표본 비중 | 8.12% | 31.89% |
| 여러 짧은 참조의 표본 비중 | 26.98% | 3.01% |
| 단일 짧은 참조의 평균 share, fold별 범위 | 0.1892~0.1943 | 0.4803~0.5114 |
| 단일 짧은 참조 중 share > 0.4 비율 | 0% | 58.93~61.62% |

TRAIN은 높은 share를 native·다중 참조로 보내므로 단일 짧은 참조에서 이 범위를 학습하지 않습니다. 이는 생성 정책 불일치이며, inner의 고 share 사례가 물리적으로 불가능하다는 판정은 아닙니다. 또한 이 short 구간의 불일치만으로 앞의 native 원천 실패 전체를 설명할 수는 없습니다. 재현 수치와 recipe SHA는 [이번 비교 원본](OFFICIAL_BASELINE_COMPARISON_FACTS.json)에 있습니다.

이 차이는 별도 stress 평가로 명시해야 합니다. 다음 생성 설계에서는 여러 물리적 시나리오를 **train/inner만으로 사전 규정**하고 조건별 학습 coverage를 확보해야 합니다. 공식 generator가 우리의 20dB cap을 사용한다는 근거는 없습니다. Test나 리더보드 값에 맞춰 cap을 선택해서는 안 됩니다.

근거: [물리 생성 가정·실제 적용 감사](group_cv_physics_data_audit.json), [R2 고정 데이터 설명](R2_FULL_PIPELINE_REPORT.md).

### D. 앙상블 선택이 소수 inner 원천에 민감하다 — 확정

R2 앙상블 0.245585는 CNN 단독 0.238545보다 나쁩니다. 전체 fold 앙상블 RMSE도 **0.169238~0.338796**으로 크게 변합니다. 이는 15,000개의 합성 행 수보다 독립 원천 수가 제한적임을 보여줍니다.

예를 들어 fold 0 inner는 CNN 0.288554, Kumo 0.343103이어서 정책은 Kumo를 거의 배제했지만, outer에서는 CNN 0.174213, Kumo 0.118949였습니다. 이러한 순위 뒤집힘은 inner 원천 하나에서 고른 가중치가 다른 원천에 잘 전이되지 않음을 보여줍니다. Outer를 보고 Kumo 비중을 올리면 검증 누수가 되므로 그렇게 하지 않았습니다.

R3는 inner 가중치 정규화로 0.240328까지 개선했지만 CNN 단독을 넘지 못했습니다. 따라서 모델 수나 크기를 늘렸다는 이유로 해당 앙상블을 최선으로 취급하면 안 됩니다. 다음에는 outer-train 안의 **여러 원천에 대한 inner OOF**에서 조합의 안정성을 검증하는 것이 우선입니다.

근거: [모델별·fold별 결과 및 동결 가중치](R2_FULL_PIPELINE_REPORT.md), [R3 결과](R3_ENSEMBLE_REPORT.md).

### E. 지금도 단순히 학습을 덜 했는가 — 주원인이라는 근거 부족

R2는 신경망 4종 × 5 folds = **20회**를 실제 실행했습니다. Adam, 최대 100 epochs, patience 7을 적용했고 실제 합계 **270 epochs**, 각 실행 8~28 epochs에서 best checkpoint를 복원했습니다. TabICL과 Kumo는 vendor 가중치 고정·train context 구성 방식이며 gradient fine-tuning을 하지 않았습니다.

일찍 멈춘 실행이 있다는 사실만으로 underfitting을 확정할 수 없습니다. 같은 source의 train 적합도와 새로운 source의 검증 개선이 분리되는 현상이 있고, 가장 큰 잔차가 특정 native 원천에 집중됩니다. 모든 모델을 무조건 100회까지 반복하는 것보다 source 다양성·조건부 coverage·검증 원천 구성의 수정이 먼저입니다. Non-convex 신경망의 global minimum을 Adam이나 momentum만으로 보장할 수도 없습니다.

근거: [실제 학습 곡선·종료 기록](R2_FULL_PIPELINE_REPORT.md), [OOF 독립 재계산](R2_OOF_INDEPENDENT_VERIFICATION.json).

## 5. 구현 오류 여부를 구분하기 위한 추가 확인

이번에 공식 노트북의 외부 오디오 변환 함수만 추출해, 별도로 생성한 60초·32kHz waveform을 우리 `audio_band_power`와 비교했습니다. 실제 데이터·Test 입력은 사용하지 않았습니다.

양쪽 모두 한 채널, 2초 Hann PSD, 대역 경계 `centre / 10**0.05`부터 `centre * 10**0.05`, 선형 PSD 적분과 30개 segment 평균을 사용합니다. 25개 대역 power의 최대 상대 오차는 **1.83e-7**로 통과했습니다. 따라서 이 fixture에서는 band grid·PSD 적분·분 평균의 명백한 구현 차이가 발견되지 않았습니다. 모든 녹음기의 교정이나 합성의 물리적 동등성을 보장하는 검증은 아닙니다.

기존 에너지 가중 label 수식·mask·checkpoint·예측 정렬 검증도 통과했습니다. 현재 증거로는 '모든 base model이 학습되지 않았다'거나 '라벨 계산이 틀렸다'고 단정할 수 없습니다. 다만 **정확히 계산된 합성 정답도 실제 평가 과제를 잘 대표하지 못할 수 있습니다.**

## 6. 개선 우선순위

1. **검증 기준부터 고정:** 모든 새 실행에 starter 상수·TRAIN 평균·단일 CNN을 포함하고, 전체/원천/시간 support/share/SNR별 오차를 함께 기록합니다. 공식 점수와 자체 OOF 표를 분리합니다.
2. **참조·시뮬레이터 충실도 개선:** 공개 VTUAD passage 가정에 맞춘 긴 시간 형태·다양한 상대 channel을 검토합니다. 원천·hull/day·overlap·SHA 분리와 24시간 purge는 유지합니다. 기존 inner/outer는 보존하고 새 시나리오는 별도 버전으로 평가합니다. 기존 데이터의 시간 변형이 독립 원천을 늘리는 것은 아닙니다.
3. **TRAIN 조건 coverage 보강:** support cap과 고 SNR 조건의 학습 공백을 시나리오별로 계측합니다. 물리적으로 가능한 장시간·분산 추가를 확대하되 Test 통계나 public 점수로 생성 규칙을 맞추지 않습니다.
4. **앙상블 검증 재설계:** 단일 inner 원천에서 자유롭게 가중치를 고르는 대신 여러 inner source의 OOF에서 단순 조합·정규화·단일 모델 대비 개선을 확인합니다. Outer는 최종 측정에만 사용합니다.
5. **그 후 모델·손실 ablation:** CNN 배경 추정과 cell mask가 native 추가를 배경으로 흡수하는 문제를 우선 분석합니다. 모델 크기·epoch·presence 보조 손실·frozen foundation 추가는 이 기준 위에서만 비교합니다.

이 문서는 원인 진단과 후속 수정 우선순위를 정리한 것입니다. 새로운 데이터 생성·GPU 재학습·공식 제출을 완료했다고 주장하지 않습니다.

## 재현

프로젝트 루트에서 numpy·pandas·scipy·soundfile·parquet 런타임을 이용해 실행합니다.

```bash
PYTHONPATH=. python ops/compare_official_baseline.py
```

출력은 `reports/OFFICIAL_BASELINE_COMPARISON_FACTS.json`입니다. 15,000개 ID 정렬, 유효 정답 범위, 각 fold 3,000개, 입력 파일 SHA 불변, 공식 변환 함수와 생성 waveform의 수치 일치를 검사합니다. 모델 훈련·정책 변경·Test 접근은 하지 않습니다.
