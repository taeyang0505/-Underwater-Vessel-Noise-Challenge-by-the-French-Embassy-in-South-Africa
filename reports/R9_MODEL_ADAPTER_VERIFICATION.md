# R9 R6-relative8 모델 어댑터 검증

네 기존 신경망의 에너지 마스크 경로를 R6의 동일한 8채널 입력에 연결했습니다. 모든 가중치는 새로 무작위 초기화하며, R5 체크포인트·배경 사전·사전학습 가중치는 읽지 않습니다. 이는 기존 구현을 해당 과제에 맞게 연결한 어댑터이며, 원 논문의 완전 재현이나 최신 SOTA 성능을 뜻하지 않습니다.

## 실제 매개변수 수

| 모델 키 | 기본 구성 | 기존 전체 | 제거한 별도 presence head | R9 전체 = 학습 가능 |
|---|---|---:|---:|---:|
| `bg_cnn` | width=128, blocks=5, dropout=0.1 | 1,488,002 | 129 | **1,487,873** |
| `tfgrid_fdy_conformer` | embed_dim=64, hidden_size=128, grid_blocks=5, dropout=0.1, conformer_blocks=1 | 2,562,150 | 65 | **2,562,085** |
| `fdy` | width=32, blocks=3, dropout=0.1 | 129,966 | 65 | **129,901** |
| `itransformer` | width=96, blocks=3, dropout=0.1 | 295,696 | 97 | **295,599** |

TF-grid 구성의 기존 코드 실측값은 **2,562,150개**입니다. 이전 대화에 등장한 2,562,498개 추정치와 구분합니다. WideCNN/FDY/iTransformer 기존 실측은 각각 1,488,002/129,966/295,696개와 일치합니다. 별도 존재 여부 분류 헤드만 제거했으므로 본체·에너지 마스크 헤드의 차원은 유지합니다. TF-grid+FDY+Conformer는 독립적으로 조합한 task adaptation이고, FDY와 iTransformer 역시 원래 과제를 그대로 재현한 모델이 아닙니다.

## 입력·출력 계약

`make_r9_model(kind, **settings)`로 생성하고 `model(x, w)`를 호출합니다. `x`는 `relative_grid_v2(db)`가 반환한 `[N,8,15,25]`, `w`는 같은 0.1 dB 반올림 관측값에서 계산한 `[N,15,25]` 정규화 에너지 가중치입니다.

1. 시간 대역 중앙값 대비 dB/10
2. 시간 대역 q20 대비 dB/15
3. 연속 두 관측 분 사이 dB 차이/10
4. 중고주파 양의 증가와 저주파 동반 증가의 고정 상호작용
5. 대역 중앙값과 관측 창 중앙값 차이/20
6. 관측 마스크
7. 시간 대역 q05 대비 dB/15
8. `sqrt(w)`

공장 함수는 `bg_cnn`, `tfgrid_fdy_conformer`, `fdy`, `itransformer`만 허용합니다. 주어진 텐서의 유한값·정확한 격자 크기·채널 6의 이진 관측 마스크·채널 8의 `sqrt(w)`·결측 위치의 전 채널 0·관측 위치와 양의 가중치 일치·창별 가중치 합 1을 검사합니다. 동일한 8채널 모양을 가진 과거 R5의 절대 dB/배경 사전 입력을 자동 변환하거나 허용하지 않습니다.

반환값은 `(share, score_logit, q)`입니다. `q`는 sigmoid 에너지 마스크이고 결측 위치는 정확히 0입니다. `share = sum(w*q)`를 FP32로 계산합니다. 별도 스칼라 회귀 헤드와 출력 사전은 없습니다. 점수는 기존 R6 함수와 같은 `0.5*share + 0.5*contrast`이며, contrast는 관측된 대역 인덱스 10..22에서 대역별 시간 최솟값 대비 양의 마스크 차이의 최댓값입니다. BCE용 반환값은 이 점수를 `[1e-6,1-1e-6]`로 자른 logit입니다. 이 점수는 자연 상태의 선박 존재 정답 확률로 보정된 값이 아닙니다.

`representation(x)`는 선택적으로 숨은 표현을 반환합니다. `architecture()`는 재구성 가능한 모델 설정, 입력·점수 정책과 실제 매개변수 수를 JSON 형태로 반환합니다. 체크포인트의 데이터·코드·분할 바인딩과 학습 일정 강제는 부모 러너가 담당합니다.

## 실행한 검증

실제 음향 원천과 held-out 데이터를 읽지 않은 합성 숫자 3개를 사용했습니다. 네 모델 각각 CPU 2스레드에서 Adam 업데이트 **1회**를 실행했습니다. 이 검증은 본학습이나 성능 평가가 아닙니다.

- 각 모델에서 `share MSE + 0.25*energy-weighted cell MSE + 0.05*derived-score BCE`가 유한했습니다.
- 모든 등록된 매개변수에 유한한 gradient가 연결되고 실제 가중치가 갱신됐습니다. 사용되지 않는 presence head는 남기지 않았습니다.
- 마스크 범위 `[0,1]`, 결측 마스크 정확한 0, 에너지 가중 풀링과 고정 점수 계산을 검사했습니다.
- 평가 모드에서 반복한 동일 배치는 완전히 일치했습니다. 단일 창으로 나누어 예측한 결과와의 최대 오차는 아래와 같습니다.
  - `bg_cnn`: 0
  - `tfgrid_fdy_conformer`: 2.38418579e-07
  - `fdy`: 5.96046448e-08
  - `itransformer`: 0
- 네 모델 모두 checkpoint를 메모리에서 `weights_only=True`로 저장·복원한 뒤 출력 오차가 정확히 0이었습니다.
- 잘못된 채널 계약·이진 마스크·가중치 정규화·채널 수·입력 변경 설정을 명시적으로 거부했습니다.
- 검증 전후 기존 의존 코드 SHA는 변하지 않았습니다. Test·기존 모델 가중치·GPU는 사용하지 않았습니다.

본학습의 사용자 지정 정책은 **Adam, dropout 0.1, 실제 100 epochs, early stopping 없음**입니다. 이 모듈은 학습 루프를 구현하거나 이전 학습을 재개하지 않습니다. 기존 R6 LightGBM의 3,827차원 입력 분기는 별도 CPU 모델이며, 이 네 신경망 어댑터에 포함하지 않았습니다.

## 파일과 재현

- [어댑터 구현](../pipeline/models_r9.py)
- [합성 CPU 검증 코드](../ops/verify_models_r9.py)
- [전체 검증 JSON](R9_MODEL_ADAPTER_VERIFICATION.json)
- [R6 입력·특징·분할 근거](R8_FEATURE_PIPELINE_EVIDENCE.md)

실행 예: `python -m ops.verify_models_r9 --output reports/R9_MODEL_ADAPTER_VERIFICATION_new.json`. 기존 결과 파일 덮어쓰기는 거부합니다.

검증 JSON SHA256: `70ea2c3e16c21a81ee089aad4ca7ff76baa4e1f05bef453b88b299fa4a862b5b`.

## 본학습 전 WideCNN 전역 주파수 문맥 보완

사용자 guardrail을 명시적으로 충족하도록 WideCNN만 parameter-free 전역 주파수 residual을 추가했습니다. 관측 마스크 O를 이용해 각 채널·분의 `c=sum_f(O*h)/sum_f(O)`를 구하고 `h_global=h+c`를 q head와 representation에 전달합니다. 미관측 분의 c는0이며 모든25대역에 같은 문맥을 broadcast합니다. 추가 매개변수는0개로 전체1,487,873개를 유지합니다. 이는 R9의 새 연결 방식이며 기존 R5/R6 모델을 변경하지 않습니다.

기본 CNN은 stem과5개의2-convolution residual block으로 이론상23분 temporal receptive field를 가지며 실제15분 입력 경계와 교차합니다. 매 edge output이 반드시 모든15분을 본다는 주장은 하지 않습니다. 주파수축은 masked global mean을 통해 해당 시간 hidden의25대역에 명시적으로 의존합니다.

새 oracle 검증에서 masked mean과 실제 encoding이 정확히 일치했고, 전부 결측인 분의 문맥은 정확히0입니다. 네 모델의 실제 Adam1update·유한 gradient·checkpoint reload·batch independence 검증을 다시 통과했습니다. 이전 검증은 PRE_GLOBAL_CONTEXT 이름으로 보존하며 현재 JSON만 새 소스에 대응합니다.
