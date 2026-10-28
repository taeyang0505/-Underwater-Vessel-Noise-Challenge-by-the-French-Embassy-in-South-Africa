# R14 배치 Torch 11채널 전처리 검증

**새 R14 경로의 수치·연결 검증을 통과했다.** 원형 R13 모델·자료·정책·생산 CUDA guard는 변경하지 않았다. 기존 생산 러너가 자동으로 이 경로로 바뀐 것은 아니며, 부모가 별도로 검토 중인 LF-anchor 14채널도 통합하지 않았다.

## 구현과 API

- `pipeline/features_r14_torch.py`: `transform(db, basis)->(x,w)`, 또는 반복 실행에 적합한 `TorchFeatureFrontend(basis)` 모듈. db만 받고 frozen TRAIN PCA를 buffers로 보관한다. 기본 출력은 float32 `[B,11,15,25]`와 `[B,15,25]`다.
- float64 round-to-even0.1dB, 관측값 linear quantile/median, 인접 상승·q05·sqrt(w), safe23 PCA 행렬곱·부분관측 최소제곱·곡률을 배치 Torch로 계산한다. median은 짝수 관측에서 중간 두 값을 평균해야 하므로 lower-median을 쓰지 않았다.
- `pipeline/raw_frontend_r14.py`: `RawR14Dataset(parent)`는 원 `raw_item()`으로 db/mask/q/y를 읽고 Dataset 안에서 특징을 계산하지 않는다. `RawFeatureMaskModel(backbone,basis)`는 forward에 db/mask만 받아 특징→기존 본체를 연결한다. `return_weights=True`이면 `(outputs,w)`를 반환해 원 물리 손실을 적용한다. q/y/presence/site는 forward 입력이 아니다.
- P0는 기존8채널을 그대로 두고 3채널을 zero-pad한다. PCA는 기존 fold-TRAIN 적합 결과를 읽을 뿐 새로 적합하지 않는다. 물리 정답은 재정의하지 않는다.

## 수치·학습 연결

프로토콜을 결과 전에 저장했다. 허용오차는 특징3e−6, w3e−8, 상대오차0이다. synthetic float32/64·rounding ties·NaN minute·없는 전체 band·partial safe-band와 실제 TRAIN4,096개를 비교했다. 최대 특징 차이는 **4.547e-13**, w 차이는 **0**였다. 추가3개 shape채널은23/24번 band를 바꿔도 동일했다. 기존8채널과 full25 물리 w의 상단 band 의존은 유지한다. inf/전부결측/정답dict/불일치mask는 거부했다.

전체크기 WideBG/TFGrid-FDY-Conformer/FDY/iTransformer 모두 기존 NumPy 특징 입력과 새 raw-forward 출력 차이0, 실제 TRAIN으로 Adam1step finite gradient, frontend trainable parameter0, checkpoint strict round-trip bitwise를 확인했다. 이는 모델 성능 개선이나100epoch 수렴 증거가 아니다. [원수치·모델별 검사](R14_TORCH_FRONTEND/RESULTS.json)

## CPU 비용 분리

동일4,096개·CPU2threads에서 합성만 **0.652초**, NumPy 개별 특징 **5.846초**, NumPy batch128 **3.822초**, Torch batch128 **0.1018초**였다. 배치 특징 부분의 큰 감소와 전체 합성비용을 구분해야 한다.

별도로 실제 Fold0 TRAIN **78,869개 전부**를1회 읽었다. raw 합성·collation **13.047초**, 배치 Torch 특징 **2.436초**, stream SHA 포함 전체 **16.030초**였다. known-zero15,774개와 입력·출력 stream을 기록했고 원천 SHA가 유지됐다. [전수 벤치마크](R14_TORCH_FRONTEND/FULL_EPOCH_V2.json)

이전1thread NumPy 전수 경로115.44초와 새2thread 경로는 thread/실행시점이 다르므로 엄격한 같은 조건 속도비로 부르지 않는다. 새 전수 측정에는 DataLoader multiprocessing·본체 학습·checkpoint·validation·GPU 전송이 없다. 첫 전수 측정15.817초도 이력으로 보존했고, 부하 존재를 단정하던 메타데이터를 '부하 통제하지 않음'으로 정확히 기록한 새 측정16.030초를 함께 남겼다. host 동시 부하는 통제하지 않았다. 남은 약13초의 합성비용이 사라졌거나 모든64 CPU시간이 제거됐다고 주장하지 않는다.

## GPU 경계와 원본 보존

CUDA에서 전처리까지 같은 forward 안에 두는 구현 구조는 준비했지만 **CUDA 수치·지연·VRAM은 미측정**이다. FP64 기준을 검증 없이 FP32로 바꾸지 않는다. MPS는 FP64를 지원하지 않아 별도 CPU-FP64 frontend→FP32 MPS 본체 경로를 실제 검증했다. 이 모듈 전체를 `.half()`/`.float()`로 바꿔 PCA buffer를 변형하면 거부한다. [MPS 실측](R14_MPS_BENCHMARK.md)

활성 원자료 hash는 `bb1dc465007a70798968ce14040b59643715538716b9e2c1f11c77e72c27d3c5`이고 원 R13 코드30개 SHA를 실행 전후 확인했다. 새 특징에 q0/정답/추정 site/공개 Test 분포를 넣지 않았다. 새 본학습·공식 제출은 하지 않았다. [소스 closure·기계 판독 보고서](R14_TORCH_FRONTEND.json)
