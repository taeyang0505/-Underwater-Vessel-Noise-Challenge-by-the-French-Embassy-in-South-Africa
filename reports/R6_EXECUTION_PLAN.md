# R6: 원천·생성기·채널을 분리하는 Mac CPU 실험 계획

작성: 2026-10-26 KST. 새 데이터·모델 결과를 보기 전에 [실험 설정](../configs/r6_cpu_ablation_v1.json)을 고정한다. 기존 R4/R5, 체크포인트, 평가 기록은 보존한다. TFGrid/Dasheng/TimeMoE 학습, SSH/GPU, 공식 제출은 재개하지 않는다.

## 먼저 바로잡는 해석

1. Benchmark의 예측 알고리즘이 확인되지 않았으므로 RMSE²를 `E[y²]`로 쓸 수 없다. 상수 c의 MSE도 `E[y²]−2cE[y]+c²`다. `c=0`임이 확인된 평가에서만 MSE=`E[y²]`다. 따라서 Benchmark .246을 근거로 y의 RMS·범위를 역산하거나 새 validation baseline을 .24~.27에 맞추지 않는다.
2. Test의 bump/지역 비율/추정 원천을 생성 분포의 목표로 사용하지 않는다. 공식 규정은 Test inference-only다. 공개 점수와 기존 Test EDA는 이번 설계·모수 선택·분포 matching·합격 gate에 사용하지 않는다. [공식 규정](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge).
3. q0의 단독 실패와 센서 중복은 확인했지만 CNN의 암기·과대추정·특정 사이트 오판은 아직 원인 가설이다. 특히 현재 Fold0 TRAIN St Francis는 12,755/53,244≈23.96%이며 과거 EDA의 3%를 현재 R5 수치로 쓰지 않는다. Test의 site 구성은 제공되지 않으므로 EDA의 source 추정도 확정 site 정답으로 취급하지 않는다.
4. 기존 fade가 모든 양성에 순수 배경 분을 보장하는 것은 아니다. 900초 support·분 평균·자연 배경의 시간 변화 때문에 이를 수학적 보장으로 설명하지 않는다. operator-heldout 진단으로 확인한다.

## 구체적인 파일·함수 변경 범위

| 신규 파일 | 내용 |
|---|---|
| `pipeline/source_split_r6.py`, `ops/build_source_split_r6.py` | 기존 Fold0의 허용 TRAIN/INNER 원천만으로 sensor holdout 및 source-block early-stop 분리. 실제 경계 후보 minute를 NaN으로 가리고 ≥3분인 창 복구. 24h support purge·참조 group 분리 검사 |
| `pipeline/robust_sampler_r6.py` | `synthesize`: no-warp minute-first / 실제 짧은 crop / 창 경계를 지나는 passage / 2초-first 비교. 넓은 smooth EQ와 사전 고정 CPA-SNR. true labels는 알려진 unrounded 성분에서 계산 |
| `pipeline/relative_features_r6.py` | `relative_grid(db)`: prior·절대 dB 없는 6채널. `TinyMaskCNN`: 경량 동일 구조의 8채널 대조군/6채널 개편군 |
| `ops/verify_relative_features_r6.py`, `ops/verify_robust_sampler_r6.py` | shape·mask·NaN·gain 불변 범위·재현·정답·반올림 오차·모델 gradient/복원 검사 |
| `ops/run_r6_cpu_ablation.py` | 새 데이터 고정 materialize, 네 CNN arm 및 두 tree arm 학습, early stopping과 독립 평가, checkpoint·예측·hash·곡선·지표 저장 |
| `reports/R6_*` | 원천 선별, 분할 수량, 생성 분포, 실제 ablation 지표와 한계 보고 |

## 채널과 반올림

제거 대상은 absolute db/100·supervised q0·Bhat/E다. 6채널은 band median 상대값, band q20 상대값, 시간 차분, 저역≤200Hz와 중고역≥300Hz 동조 상승, band median의 창 전체 상대 profile, 관측 mask로 구성한다.

첫 네 변화 채널과 상대 profile의 불변성은 다르다. profile 및 에너지 weights는 대역별 고정 EQ까지 불변이지 않다. 이를 100% 장비 응답 불변이라고 주장하지 않는다.

모든 feature db는 0.1dB로 반올림하고 model pooling w도 같은 반올림 관측에서 계산한다. 단, 합성 y/q 정답을 바꾸지 않는다. 원래 E에서 계산한 `sum(q*w_raw)=y`는 수치 오차 내 정확하고, `sum(q*w_rounded)`는 양자화 근사다. ±.05dB 반올림에 의한 share 차이의 이론 상한은 약 .0057564다. 이를 별도 residual로 보고해 exact label integrity와 혼동하지 않는다.

## 생성 분포와 검증

새 양성은 원하는 y를 먼저 뽑지 않는다. 변형 후 참조 **전체 궤적의 창 crop/mask 전 peak-minute broadband power**가 배경의 관측 minute broadband 중앙값 대비 SNR −10..+10dB가 되도록 추가한다. CPA가 창 밖에 있으면 보이는 tail을 다시 증폭해 정규화하지 않는다. 이 정의는 숫자 fixture·실제 데이터 결과를 보기 전 물리 검토에서 정정했다. 이 범위는 사전 고정 nuisance 실험 범위이며 hidden Test의 실제 SNR이라는 주장이 아니다. 상수 baseline RMSE와 high-band 단일 통계 AUC는 결과를 그대로 보고하고 특정 숫자에 도달하도록 조정하지 않는다. 높은 y를 삭제하지 않는다.

TRAIN의 minute-first/short-crop 두 가족과 cross-generator 평가의 edge-straddling/2초-first-no-warp 두 가족을 분리한다. 원 녹음의 독립성을 보존하고 crop을 새 독립 source라고 세지 않는다. EQ±12dB/low-pass는 domain randomization 가정이지 검증된 수중 전파 시뮬레이터가 아니다.

early stopping용 INNER는 TRAIN sensor 안의 다른 source block 및 분리된 참조 source를 사용한다. 별도로 잠긴 cross-generator, cross-sensor, 둘 모두 변경한 패널은 모든 checkpoint를 고정한 뒤에만 평가한다. 이 패널을 보고 seed·설정·모델을 재선택하지 않는다. 기존 OUTER/calibration/domain/sealed는 열지 않는다.

## 공정 비교와 중단 조건

- CNN 2×2: R4법칙/R6법칙 × legacy8/relative6. 동일 tiny 구조·seed·샘플 예산·최대100·ES7·Adam·dropout.1. 각 generator의 INNER는 해당 generator 법칙으로 만들어 early stopping하며, 서로 다른 INNER 분포의 절대 RMSE를 개선율로 비교하지 않는다. 최종 비교는 모든 arm 공통의 잠긴 패널에서 한다.
- LightGBM 두 대조군: 동일 R6 데이터에서 legacy 특징과 relative 특징. 최대1500trees/ES40, Mac CPU.
- TRAIN primary는 site 동수 quota로 추출하며 실제 unique coverage·재등장 수를 보고한다. Test 지역 비율은 사용하지 않는다.
- 주 보고는 전체 window RMSE 및 site-macro RMSE, MAE/bias/AUC와 source/family별 지표다. checkpoint 선택은 같은 sensor의 source-isolated INNER site-macro RMSE다. 공식 지표가 window RMSE라는 사실은 유지한다.
- 데이터/라벨/분할 검사가 실패하면 해당 단계에서 중단하고 원인과 수정 이력을 기록한다. 실패 partial run을 완료 모델로 재사용하지 않는다.
- CPU 결과는 8192개 고정 TRAIN 조합 규모의 기전 탐색이다. 채널·생성기 대조의 입력 예산을 같게 하기 위해 이 작은 pilot에서는 고정 조합을 epoch마다 셔플한다. 전체 배경 동적 합성 production을 대체하지 않으며, 기존 1.488M WideCNN의 정밀 재현이나 최종5fold/공식 성능 검증도 아니다. 개선되지 않은 결과도 보고한다.
