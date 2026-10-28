# R5 EDA 제안의 사실·규정·검증 단계 감사

확인일: 2026-10-25. 범위는 `TASK_EDA_REPORT.md` §2.4/3.4/5.1–5.5, 해당 기존 분석 코드/요약, 현재 모델 roster와 검증 JSON, 공식 공개 규정/data page다. 이번 감사에서는 Test.csv·실제 acoustic 배열을 다시 열거나 EDA 코드를 실행하지 않았다. 새 다운로드·모델 로드·학습·GPU 실행·외부 비공개 전송도 하지 않았다. 아래 수치는 기존 산출물의 **주장·계산 경로 감사**이며 독립적인 Test 재계산은 아니다.

## 1. 공식 규정으로 확인한 경계

별도 `/rules` URL은 열리지 않았으나, [공식 대회 Info의 Full Challenge Rules → Datasets, packages and general principles](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge)에서 해당 규정을 직접 확인했다. 핵심 원문은 “Test inputs are for inference only”와 “Fixed per-window transformations are allowed”다. 같은 문단은 Test를 이용한 fit/fine-tune/calibrate/pseudo-label/학습 예제 생성을 금지하고, 공개 SA·고지된 공개 코퍼스의 증강을 허용한다. 공개 pretrained 모델도 허용한다.

이 규정은 단순히 `Test의 정답만 안 보면 된다`는 조건보다 좁다. **Test 집단의 분포에 맞추어 학습 site weights, 합성 target prior, duration/SNR/EQ 범위, 결측 주입 빈도, 후처리 CDF를 추정·최적화하는 절차는 채택하지 않는다.** Test 기반 calibration/학습 설계에 해당할 우려가 있으며 inference-only 원칙과 맞지 않는다. 대회가 이 개별 절차를 명시적으로 승인했다는 근거도 확인하지 못했다.

반면 **한 query의 관측값으로 계산하는 이미 고정된 변환**과 **Test cohort 전체로 학습 규칙을 정하는 것**은 구별해야 한다. 원 query의 finite mask·고정 정규화·TRAIN/INNER에서 동결한 feature/모델 추론은 전자의 경로다. Test에서 얻은 site 비율이나 feature quantile을 TRAIN sampling에 되먹이는 것은 후자의 경로다. 단순 기술 EDA를 별도로 허용한다는 조항은 확인하지 못했으므로, 기존 문서 서두의 `기술 통계만 냈으니 규칙 준수`도 공식 허가로 단정할 수 없다. 규정 문구를 넘어서는 회색 영역은 주최측 확인 전 허용으로 표시하지 않는다.

**안전한 통합 기준:** 공개 task/대역/target 수식과 permitted TRAIN 원천의 물리·품질 근거로 생성 정책을 정하고, TRAIN/INNER의 source-separated 검증으로 모델을 선택한다. Test EDA 숫자는 생성기·가중치·보정 파라미터의 입력으로 넣지 않는다. 규정 확인을 이유로 이번에 계정 게시나 외부 문의를 보내지는 않았다.

## 2. 네 가지 주요 주장 판정

| 주장 | 판정 | 정확한 표현 |
|---|---|---|
| 현재 5모델 모두 실제1-step 검증 완료 | **그렇게 묶어 말할 근거 없음** | 두 실제 foundation의 synthetic CPU one-window/adapter1step은 확인됨. 그 외 core/input fixtures와 현재5개를 묶은 trainer1step은 서로 다름 |
| 값이 전부0.1dB | **기존 Test finite-cell 기술 통계** | 기존 summary는 finite544,150셀의 격자 검사가 True라고 기록. NaN·전체 SA·물리 측정 분해능까지 포함한 주장이 아님 |
| KS로 true target 분포/합성 정책이 입증됨 | **아님** | Test와 TRAIN의 입력 feature 분포 차이. 알려지지 않은 `share_added_vessel` 정답 분포를 검사한 것이 아님 |
| Test St Francis32.5%/Richards Bay28.2% | **kNN 추정 비중** | 원래 site label이 아니라 TRAIN-fit domain probe가 내놓은 class 예측 구성비 |

## 3. §2.4/3.4: 공개 생성 가정과0.1dB 관측을 분리

공식 data page는 real SA background에 recorded passage-derived 성분을 controlled addition하고, 관측 minute-band cell의 선형 에너지 비율을 target으로 삼는다고 설명한다. 원래 background before addition, query station/time/AIS와 hidden source identity는 제공되지 않는다. [공식 data page](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/data), [공식 Evaluation](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge).

스타터의 출처 고지는 시간 궤적·상대 채널·에너지 변형이라는 **넓은 가정**을 지원한다. 그러나 §2.4의 이동/신축/부분 삽입의 구체 알고리즘, 특정 범위·분포·비율·rounding 순서를 실제 비공개 생성기와 동일하다고 확인해 주지는 않는다. `테스트와 같은 생성 과정을 직접 재현할 수 있다`는 문장은 `공개된 가정과 호환되는 자체 합성기를 만들 수 있다; 동일 generator라는 증거는 없다`로 한정해야 한다.

기존 `eda_deep/deep_eda.py`는 다음 경로로 quantization을 기록한다:

```text
lv = XT[np.isfinite(XT)]
all(abs(lv*10 - round(lv*10)) < 1e-3)
```

`eda_deep/summary.json`의562,500 total cells와18,350 missing cells의 차이는544,150 finite cells다. 그 유한 값에 대한 `all_on_0p1_grid=true`를 §2.4가 인용한다. 이는 float 오차 허용치가 있는 **저장값의 격자 확인**이다. 실제 acquisition 정밀도가 정확히0.1dB라는 증명, 모든 SA 값의 정밀도, 원 합성기의 rounding 시점·target 계산 순서에 대한 증명은 아니다. 이번 감사에서는 원 Test로 재확인하지 않았다.

따라서 이 관측만을 이유로 새 TRAIN을0.1dB로 반올림하거나 Test의 quantization 분포에 맞추는 작업을 바로 추가하지 않는다. 허용 TRAIN/공식 단위 명세에서 독립적인 근거를 정해 사용할 경우에도, rounding 전후의 관측 energy·cell target·overall target 정합성을 별도로 검증해야 한다.

## 4. §5.1/5.2: 결측 패턴과 site probe의 추론 한계

§5.1의 전체-minute 결측 및 특정 위치 집중은 기존 Test 기술 통계다. 여기서 `정각15분 경계로 잘린 같은 녹음기 비공개 기간`이라는 결론은 **가능한 생성 설명**이며 관측 timestamp나 recorder ID로 확인한 사실이 아니다. 데이터 명세가 query timestamps/locations를 주지 않으므로 ground truth clock alignment를 검증할 수 없다.

§5.2의 코드 경로는 SA aligned window의25차원 시간 중앙값 스펙트럼으로 `KNeighborsClassifier(n_neighbors=25)`를 fit한 뒤 Test 스펙트럼에 `predict`한 것이다. TRAIN의 site×ISO-week GroupKFold balanced accuracy0.705는 그 실험의 결과이며, Test hidden site를 맞춘 정확도가 아니다.32.5/28.2 등의 수치는 `test_pred_site_share`이고 실제 출처 비중이 아니다. 추가 성분 자체가 window median spectrum에 영향을 줄 수도 있다.

§5.4는 **실제 동일 site pair가 아니라 추정 site가 같은 집단**을 비교한다. probe 오분류·다른 recorder/calibration·미공개 기간의 조건 변화가 남아 있다. `사이트 구성 효과를 완전히 제거했다`, `차이는 추가 성분 탓임이 입증됐다`는 표현은 강하다. `추정 domain별로 나눠도 feature 차이가 남았다; 원인은 확정하지 못했다`가 맞다. nearest-neighbour distance가 작다는 사실도 실제 background source의 동일성이나 모든 query의 in-distribution 보장은 아니다.

site-balanced 평가/학습을 쓴다면 **허용 TRAIN의 실제 site label에 기반한 사전 균형**이나 INNER 검증 근거로 정한다. Test probe 예측 비중을 population weight로 변환하지 않는다. query mask를 고정 feature로 읽는 것과 Test의 전체 결측 위치 빈도를 합성 mask prior로 쓰는 것도 구별한다.

## 5. §5.3–5.5: KS·bump·스펙트럼은 target 정답이 아님

기존 `deep_eda.py`에서 KS의 두 인수는 TRAIN/Test의 `bump_high_db`, `corr_low_high`, level·centroid 등 **입력에서 계산한 feature 값**이다. `naive_share_q20` 역시 이름에 share가 들어간 휴리스틱 feature이며 공개되지 않은 true `y`가 아니다. KS가 큰 것은 해당 입력 feature들의 경험 CDF가 다르다는 뜻이다.

따라서 이것만으로 true target의 zero 비율, 고share 비중, SNR/길이/EQ 분포 또는 새 합성기의 정답 prior를 복원할 수 없다. KS값은 causal identification도 아니다. 관측들이 겹친 window/같은 source에 종속될 수 있으므로, 이를 독립 현장 표본에 대한 유의확률이나 일반화 증명처럼 제시하지 않는다.

§5.5의 `피크−window median`은 원래 배경을 제거한 실제 added component가 아니다. `밴드별 시간 범위`도 혼합 관측의 통계다. 중·고주파 성분이 풍부하다는 가설은 제시할 수 있지만, 이를 vessel-only spectrum이나 특정 선종/생성 채널의 정답으로 부르면 안 된다. `추가 성분의 확정 지문`보다 `혼합 관측의 기술 통계와 가능한 설명`이 적절하다.

별개의 문제로, **자체 synthetic TRAIN/INNER/OUTER의 생성 정책 일치**를 검사한 R4 KS/LHS QA는 Test 입력을 쓰지 않는다. 그런 mechanical QA와 EDA의 TRAIN-vs-Test feature KS를 같은 증거로 섞지 않아야 한다. 전자는 계획한 생성 정책이 split별로 지켜졌는지 확인하며, 후자는 true target validation을 제공하지 않는다.

## 6. 실제 모델 검증 상태

이번에는 모델 객체를 생성하지 않고 기존 report만 읽었다.

| 현재 후보/관련 산출물 | 확인한 증거 | 아직 그 증거가 제공하지 않는 것 |
|---|---|---|
| TimeMoEPublic | `R5_ACTUAL_TIMEMOE_PUBLIC_CPU_R1.json`: 실제 pinned pretrained 객체, synthetic one-window, adapter SGD1step, finite gradients, frozen bytes·pool·adapter 복구 통과 | 실제 Fold0 RMSE/GPU throughput/학습 수렴 |
| Dasheng1.2B | `R5_ACTUAL_DASHENG_CPU_R1.json`: 같은 범위의 실제 backbone CPU 검증 | 실제 waveform frontend 검증, Fold0 성능·GPU 실행 |
| TFGrid+FDY+Conformer2.562M | 새 `hybrid_models_r5.py` 구현과 roster 존재. 저장된 `R5_GRID_MODELS_FIXTURE.json`은 이전 TFGrid6block2.775M·BandSplit2.763M core의 수치 fixture | 해당 JSON을 현재 조합 전체의 end-to-end1step 증거로 재명명할 수 없음 |
| Wide BG CNN1.488M | 새 구현과 roster 존재. 과거 R2 hybrid parity와 현재128폭·5block은 다른 구성 | 과거 작은 CNN fixture를 현재 wide 모델의 본학습/성능 증거로 승격할 수 없음 |
| Physics/Excess Retrieval LightGBM | 후보 feature/학습 계획과 기존 tree 경로 | neural1-step와 동등한 표현 불가. 현재 R5 tree fit·저장/복구·INNER early stopping 연결은 별도 증거 필요 |
| 공통8채널/동적 generator | `R5_SHARED_INPUT_DYNAMIC_FIXTURE.json`은 replay·fold binding·TRAIN slice와 입력 변환을 확인 | report 자체가 `model_forward_or_trainer_wiring_checked_here=false`라고 명시.5모델 trainer를 실행했다는 뜻이 아님 |

따라서 현재 정확한 문구는 **`공개 foundation2개의 실제 CPU adapter1step과 별도 core/input fixtures는 완료; 현재5개 후보를 공통 trainer/검증/복원/앙상블까지 묶은 end-to-end 검증과 Fold0 본학습은 남음`**이다. core의 backward 성공을 학습 수렴이나 성능 향상으로 세지 않는다. 두 foundation의 저장 복구도 같은 검증 backbone 객체에서의 복구이며 두 번째 새 B급 객체를 재생성한 검증과는 구분한다.

읽기 중 `pipeline/hierarchical_ensemble_r5.py`에 기존 `time_moe_ultra` branch key가 남아 있어 최신 `timemoe_public` roster와 다름도 확인했다. 이는 통합이 아직 끝나지 않았다는 구체적인 예다. 이번 EDA 감사에서는 그 파일을 수정하지 않았으며 별도 담당 작업으로 정리할 사항이다.

## 7. 이번 제안을 반영할 때의 결론

1. 입력 mask·정확한 band grid·선형 energy 합산·source-separated validation은 공개 명세/허용 TRAIN 근거로 유지한다.
2. Test에서 추정한 site 비중, bump/KS/결측 빈도를 TRAIN sampling·augmentation·후처리 fitting에 넣지 않는다.
3. 생성기의 구체 prior와 모델/앙상블 선택은 사전 정책 및 TRAIN/INNER 근거로 선언하고, OUTER/Test는 동결 추론/평가 경계를 유지한다.
4. EDA에서 확인한 관측 통계·모형 추정·원인 가설·true target 사실을 표/도식에서 구분한다.
5. 현재 단계에는 새 GPU 대여가 필요하지 않다. 통합 fixture·checkpoint/앙상블 gate 검증을 마치고 실제 Fold0 처리량·VRAM을 측정할 때 GPU 필요를 알린다.

이 감사는 새로운 합성 정책이나 모델 개선을 실행한 것이 아니다. 변경 파일은 본 보고서 하나뿐이다.

후속 별도 승인 작업: 위 감사 후 `hierarchical_ensemble_r5.py`를 최신 public Time-MoE keys와 INNER-only 동결 정책으로 교체했다. 따라서 §6의 Ultra key 관찰은 **감사 시점의 발견 및 수정 근거**이며 현재 코드에 남아 있다는 뜻은 아니다. 앙상블의 synthetic 수치 검증은 `R5_ENSEMBLE_FIXTURE.json`에 따로 기록한다. 이는5모델 실제 Fold 학습/성능 측정과는 다르다.
