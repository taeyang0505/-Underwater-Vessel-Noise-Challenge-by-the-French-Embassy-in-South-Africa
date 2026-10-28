# 원본 CNN 단독 시험 제출물

이 CSV는 R5 Fold 0 원본 Wide BG Mask CNN의 완료된 최적 checkpoint(epoch 19, 1,488,002 parameters)를 Mac CPU FP32로 추론한 결과다. 최종 5fold 앙상블 결과와 구분한다.

- 파일: [submission.csv](submission.csv)
- 행/컬럼: 1,500 / `ID,share_added_vessel,score_added_vessel`
- SHA-256: `e32b04a2b876a66da6046e1ef1fdae6fd7e768eb9365bde5aa7227ad235c83fe`
- 실제 Test 1,500개 추론 시간: 5.73초
- 추가 GPU·신경망 재학습: 없음
- 공식 제출: 아직 실행하지 않음

## 전처리와 후처리

훈련된 raw 전처리와 동일한 TRAIN aggregate prior 및 8채널 입력을 유지했다. 셀 마스크를 관측 선형 에너지로 가중합하여 share를 산출했다. 진단용 score는 CNN presence head의 sigmoid 출력을 사용하며 실제 자연 선박 존재 확률로 해석하지 않는다.

| 자체 INNER 평가 | RMSE |
|---|---:|
| 저장된 서버 CUDA BF16 원본 예측 | 0.090849261 |
| Mac CPU FP32 원본 예측 | 0.090808147 |
| 편향 보정 후보의 25분할 교차 진단 | 0.091789314 |
| 제약 선형 보정 후보의 25분할 교차 진단 | 0.092133282 |

추가 보정 두 후보가 사전에 정한 기준을 통과하지 못했다. 따라서 identity `a=1,b=0` 후 [0,1] clipping만 적용했다. 이번 CSV의 share/score는 원본 CPU 예측을 그대로 보존한다. CPU와 CUDA 결과의 작은 차이는 정밀도/백엔드가 다른 실행에서 관측한 차이이며, 새 모델 개선의 근거로 주장하지 않는다.

후처리 계수 선택에는 INNER만 사용했다. 5개 양성 참조 그룹과 5개 배경 그룹 shard를 교차해 25개 보정 학습/평가를 했고 각 행을 정확히 한 번 예측했다. 기본 CNN 자체가 같은 INNER로 early stopping됐으므로 이 값은 독립 신경망 OOF가 아니다. 기존 데이터의 TRAIN/INNER 원천 분리와 24시간 purge를 유지했지만, 보정 내부 교차검증에 새로운 시간 purge를 추가한 것은 아니다.

## 앙상블과 후처리의 최종 평가

최종 실험에서는 같은 OUTER에서 개별 모델 → INNER로 고정한 앙상블 → INNER로 고정한 후처리의 RMSE를 각각 계산할 수 있다. 5fold OUTER를 합친 OOF도 같은 방식으로 비교한다. OUTER 결과를 보고 해당 가중치/보정 계수를 다시 맞추지 않는다. 실제 Test에는 정답이 없으므로 로컬 RMSE나 LB 예상 점수를 산출하지 않았다.

## 검증 근거

- [고정 모델·정책·입력·출력 SHA와 실제 추론 기록](inference_receipt.json)
- [동결된 후처리](frozen_postprocessing_policy.json)
- [CPU 재현·배치 독립성 진단](../../reports/R5_CNN_MAC_INFERENCE_PROFILE.json)
- [후처리 교차검증 보고서](../../reports/R5_CNN_PILOT_CALIBRATION/README.md)
- [이번 모델에 사용한 VTUAD 출처 고지 초안](../../reports/R5_CNN_PILOT_EXTERNAL_SOURCE_DECLARATION_DRAFT.txt)

대회 규정은 외부 출처 URL·버전·전처리·접근 방법을 공개하도록 요구한다. 사용자 계정 명의의 공개 게시는 이전 보류 지시에 따라 아직 실행하지 않았다. 이 파일의 생성 자체는 공식 게시나 제출을 의미하지 않는다.
