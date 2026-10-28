# R5 제출 실패: 추론·CSV·정답 수식 구현 감사

**현재 감사에서는 이번 격차를 설명할 CSV 오류, 시간·주파수 축 오류, 다른 전처리 적용, 가중치 복원 실패, 물리 pooling 계산 오류를 재현하지 못했습니다.** 이것이 모델의 숨은 평가 자료 일반화를 보장한다는 뜻은 아닙니다.

사용자 전달 결과는 제출 `Vz5mjZEr`, LB 0.65199132, share RMSE 0.348008679, diagnostic presence AUC 0.539533333입니다. 같은 모델의 CPU INNER RMSE는 0.090808147이며 차이는 0.257200532입니다. 이 감사는 원격 리더보드에 접근하지 않았고, 사용자 제공 결과와 명시된 로컬 제출 파일을 연결해 확인했습니다.

## 1. 실제 올린 파일과 저장 예측: 전수 일치

다음 세 CSV의 SHA256은 모두 같습니다.

`e32b04a2b876a66da6046e1ef1fdae6fd7e768eb9365bde5aa7227ad235c83fe`

- Documents의 원본 `submissions/r5_fold0_raw_cnn_cpu_pilot_r1/submission.csv`
- 별도 복사본에서 생성한 `delivery/r5_cnn_pilot_repro_r1/reproduced_submission/submission.csv`
- 사용자가 지칭한 Desktop의 `SUBMIT_NOW/submission.csv`

모두 1,500행·3열이며 열 순서는 `ID, share_added_vessel, score_added_vessel`입니다. ID 1,500개는 중복이 없고, 공개 sample submission 및 저장된 raw 예측 NPZ의 순서와 정확히 같습니다. **모든 share·score가 저장된 원본 CPU 예측과 완전히 일치**합니다. 유한값 및 [0,1] 범위 검사도 통과했습니다.

따라서 확인한 로컬 파일에서는 열 교환, ID 정렬 오류, 제출 시 다른 파일로 바뀜, scalar 후처리 오적용을 찾지 못했습니다. 원격 서비스가 보유한 업로드 바이트를 새로 내려받아 대조한 것은 아닙니다.

## 2. 후처리는 실제 identity였다

동결 정책은 `identity`, `a=1`, `b=0`입니다. 원형 적용식은 `clip(a*p+b,0,1)`이지만 원본 예측이 이미 범위 안에 있어 제출값은 raw와 전수 동일합니다. score/presence도 변경하지 않았습니다.

정책 생성은 2026-10-25 17:46:12 UTC, wrapper가 기록한 Test 개방 경계는 17:47:13 UTC입니다. 현재 정책·사전 사양·보정 코드·CPU profile의 SHA는 저장된 receipt와 일치합니다. wrapper 경계 뒤에는 query 읽기, 고정 추론·에너지 검산, 동결 정책 적용, CSV 저장만 있으며 학습·보정 fitting 호출은 없습니다.

근거: [pilot wrapper](../ops/build_r5_cnn_pilot_submission.py) 47–99행의 정책·profile·checkpoint/prior 검증, 110–142행의 경계 이후 처리; [동결 정책 적용](../ops/calibrate_r5_single_cnn.py)의 `apply_frozen_policy`.

## 3. parser와 학습·추론 경로: 실제 INNER로 최소 재현

실제 INNER의 앞 16개 창만 사용해 자체 CSV를 만들고 기존 parser에 통과시켰습니다. 실제 Test 관측값은 다시 읽지 않았습니다.

| 검증 | 결과 |
|---|---|
| 375열을 분 우선 순서로 15×25 복원 | 원래 float32 dB와 bitwise 일치 |
| NaN 위치·관측 mask | 정확히 유지 |
| 주파수 순서 | 원본 SA 준비 코드의 공통 25대역과 일치 |
| 잘못 바꾼 헤더 순서 | 조용히 해석하지 않고 거부 |
| parser 경유 vs 배열 직접 입력의 8채널·물리 가중치 | bitwise 일치 |
| parser 경유 vs 직접 입력 CNN share·score·cell mask | bitwise 일치 |
| raw TRAIN 증강 경로(p=0, 반올림 없음) vs inference 경로 | 8채널·물리 가중치·eval 예측 bitwise 일치 |
| 모델·TRAIN prior 상태 | 검사 전후 불변 |

사용한 모델은 원본 `aa222697…96d46f` 체크포인트의 epoch 19, `bg_cnn(width=128, blocks=5, dropout=.1)`입니다. 체크포인트를 원래 `restore_into`로 복원했고, 원본 core 17개 SHA를 context 및 제출 receipt와 대조했습니다. 무작위 가중치로 대체하는 fallback은 실행하지 않았습니다.

근거: [query parser](../pipeline/inference_r5.py) 261–274행은 정확한 헤더를 요구하고 float32 배열을 `(N,15,25)`로 해석합니다. [TRAIN 준비](../pipeline/neural_r5.py) 143–147행과 [추론 준비](../pipeline/neural_r5.py) 22–28행은 같은 `shared_grid_inputs`와 raw-energy 가중치를 사용합니다. 같은 파일의 65–76행은 `model.eval()` 및 CPU에서 비활성화된 CUDA autocast 경로를 사용합니다.

## 4. 에너지 가중치와 합성 target 계산

독립 계산식은 관측 셀에서 다음과 같습니다.

```
E = 10 ** ((db - window_peak_db) / 10)
w = E / sum(E)
y = sum(q * w)
```

실제 `original_energy_weights`, raw 관측 변환의 가중치, 위 독립 계산이 float32에서 bitwise 일치했습니다. INNER 16개에서 저장된 known q로 target을 복원한 최대 오차는 **8.99×10⁻⁸**, CNN의 실제 `sum(qhat*w)`와 scalar 출력 사이 최대 오차는 **5.94×10⁻⁸**입니다.

또한 원본 `PassageSamplerR4.draw`를 직접 호출한 별도의 숫자 fixture를 사용했습니다. 알려진 B/R, 관측 mask와 생성 과정에서 반환된 실제 R을 보존하고 독립 계산한 결과입니다.

| 요청 share | 독립 B/A 비율 | 저장 target | 저장 dB·q·w 재계산 오차 |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 0.27 | 0.27 | 0.2700000107 | 1.47×10⁻⁸ |
| 0.91 | 0.91 | 0.9100000262 | 2.12×10⁻⁸ |

관측 mask 보존과 미관측 q=0도 통과했습니다. 실제 산출물에는 1.05×10⁻⁷의 Test pooling residual이 기존 inference receipt에 기록되어 있습니다. 이 감사에서는 Test 입력을 다시 읽어 해당 잔차를 재계산하지 않았습니다.

근거: [R4 sampler](../pipeline/passage_sampler_r4.py) 204–207행은 `alpha=u*sum(B)/((1-u)*sum(R))`, `E=B+alpha*R`, `q=alpha*R/E` 및 관측 에너지 target을 계산합니다. [CNN 출력](../pipeline/models.py) 28–32행은 `sum(cell*energy_weights)`를 scalar로 반환합니다.

**이 검증은 우리 합성 데이터에 심은 controlled-addition 비율이 수학적으로 일관된다는 증거입니다. 대회의 공개되지 않은 생성 과정이나 숨은 정답과 일치한다는 증거는 아닙니다.**

## 5. FP32 차이가 확인된 격차 규모와 맞는가

| 동일 INNER 3,000개 비교 | 값 |
|---|---:|
| 원래 저장된 GPU BF16 RMSE | 0.0908492610 |
| CPU FP32 RMSE | 0.0908081471 |
| CPU−GPU 성능 차이 | −0.0000411139 |
| 두 예측 벡터 차이의 RMSE | 0.0005843655 |
| 두 예측 벡터 최대 절대 차이 | 0.0060639381 |
| 사용자 전달 Test RMSE−CPU INNER RMSE | +0.2572005319 |

확인한 INNER에서는 FP32 전환이 성능을 크게 악화시키지 않았고 오히려 미세하게 낮은 RMSE였습니다. 이 실측만으로 정밀도 변경을 큰 일반화 격차의 주원인이라고 판단할 근거가 없습니다. 다만 위 차이는 **같은 INNER에서의 비교**이며, 다른 입력 집합에 대한 보편적 오차 상한이라고 주장하지 않습니다. 새 GPU/Test 비교 실험은 하지 않았습니다.

## 6. 이번 감사가 확정하지 않는 것

현재 INNER는 기본 CNN의 epoch 선택에 이미 사용됐습니다. 후처리 crossfit이 그 CNN을 독립 OOF 모델로 만들어 주지는 않습니다. 최종 정책도 identity였으므로 이번 실패를 scalar 보정으로 해결했다고 볼 수 없습니다.

이 감사는 숨은 Test의 통계나 정답 prior를 추정하지 않았고, 데이터 분포를 맞추거나 라벨을 복원하지 않았습니다. 구현 경로 검증을 통과한 사실과, 합성 자료·외부 원천·독립 원천 수·검증의 대표성이 충분한지는 구분해야 합니다. **현재 증거는 후자의 문제를 해결하거나 정확한 원인을 확정하지 않습니다.**

원본 data/model/core, GPU 실행과 제출 파일은 변경하지 않았습니다. 재학습·추가 제출·원격 접속도 하지 않았습니다.

[정량 감사 JSON 및 source SHA](R5_LB_FAILURE_IMPLEMENTATION_AUDIT.json) · [원래 CPU 추론 진단](R5_CNN_MAC_INFERENCE_PROFILE.json)
