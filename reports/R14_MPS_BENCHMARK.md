# M4 Pro MPS 실제 본체 학습 가능성 진단

**M4 Pro20 GPU코어·24GB에서 실제4종 본체 모두 짧은 FP32 학습을 수행할 수 있었다.** batch8/16/32 총12조건에서 forward/backward/Adam이 finite였고 unsupported-operator CPU fallback은0으로 두었다. 이는 본학습 완료나 CUDA 대체 속도 보장이 아니다.

환경은 Python3.14.5/PyTorch2.14.1 사용자 venv다. CPU는2threads, 각 조건 warmup1step 후3개 Adam step의 중앙값이다. host 동시 CPU 부하를 통제하지 않았고 GPU 최초 compile 비용을 전체 학습시간에 외삽하지 않았다. 매 조건 fresh weights와 별도 optimizer로 시작했으며 생산 checkpoint를 변경하지 않았다. CPU/MPS eval 출력 최대 차이는 5.513e-06로 사전 tolerance 안이었다. CPU/MPS dropout RNG가 다르므로 optimizer 궤적 동일성은 주장하지 않는다.

|본체|파라미터|CPU2threads1step(초)|MPS1step(초)|CPU특징+전송+MPS1step(초)|
|---|---:|---:|---:|---:|
|bg_cnn|1,491,329|0.36044|0.04095|0.04142|
|tfgrid_fdy_conformer|2,563,813|1.55572|0.19513|0.18586|
|fdy|130,765|0.06532|0.03138|0.02891|
|itransformer|299,919|0.01637|0.02336|0.02514|

모두batch32 결과다. Wide/TFGrid/FDY는 이 조건에서 빨랐고 **iTransformer는 CPU가 더 빨랐다.** MPS 사용 여부를 모델마다 무조건 동일하게 판단할 수 없다. [batch8/16 포함12조건 원수치](R14_MPS_BENCHMARK_r2/RESULTS.json)

동기화된 step 이후 관측 driver 메모리는 최대 **3.12GiB**, process lifetime peak RSS는 **2.72GiB**였다. 이는 activation 순간 peak가 아니고 CPU/Metal 공유 메모리의 다른 앱·Tree 작업까지 포함한 안전 headroom 보장이 아니다. batch128이나 더 큰 모델의 가용성을 추측하지 않는다. 실제 full epoch·100epoch 전체 시간이 없으므로 전체 소요시간은 추정하지 않았다.

FP64 MPS 할당이 실제로 거부됐다. 따라서 R14의 NumPy-parity FP64 특징은 **CPU에서 계산한 뒤 x/w와 정답 tensor만 FP32로 MPS에 전송**했다. FP64 frontend를 MPS에서 직접 실행했다고 표현하지 않는다. [`MPS 문서`](https://docs.pytorch.org/docs/stable/notes/mps.html), [`Apple PyTorch 안내`](https://developer.apple.com/metal/pytorch/)와 별도로 실제 dtype probe 오류를 원JSON에 남겼다.

첫 진단은 venv에 tabm이 없어 모델 초기화 전에 실패했고 그대로 보존했다. 원 모델의 공통 import를 수정하지 않고 기존 순수 Python tabm/rtdl 두 파일을 SHA 그대로 workspace overlay에 제공한 fresh r2에서12조건을 실행했다. 다운로드·venv 설치·숨은 모델 대체는 없었다. [overlay receipt](../ops/r14_runtime_py/receipt.json) / [처음 실패 이력](R14_MPS_BENCHMARK/RESULTS.json)

원 R13 CUDA-only 생산 guard와 승인false는 유지한다. 이 결과는 MPS 지원을 검증하는 신규 학습엔진을 별도로 만드는 근거이며 원 러너가 지금 MPS 본학습을 지원한다는 뜻이 아니다. LF-anchor14채널도 본 진단에 넣지 않았다. [소스 closure](R14_MPS_BENCHMARK.json)
