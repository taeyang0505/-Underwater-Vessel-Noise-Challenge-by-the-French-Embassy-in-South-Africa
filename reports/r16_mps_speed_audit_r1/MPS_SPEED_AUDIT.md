# Mac MPS 가속 후보 실측

본학습과 별개인 임시 초기 가중치로 CNN/TFGrid를 짧게 측정했다. 생산 코드·가중치·설정은 변경하지 않았다. CPU Tree 본학습은 병행 중이며 전체 CPU 부하는 통제하지 않았다.

|모델|현재 FP32(ms/step)|사전 입력 검사·통계 회수 지연 FP32|step 시간 단축|같은 후보 + BF16|
|---|---:|---:|---:|---:|
|bg_cnn|65.77|57.81|12.1%|57.03|
|tfgrid_fdy_conformer|322.06|299.79|6.9%|312.78|

M4 Pro 20 GPU코어·24GB, PyTorch2.14.1/MPS, batch64, CPU2threads. 실제 TRAIN 원천 30개를 반복해 고정 크기 배치를 만들고, 조건별 임시 모델에서 warmup2step 후 6step 묶음을 3번 측정했다. 각 묶음 끝 MPS synchronize로 시간을 쟀다. CPU 입력 특징 생성·전송·전체 validation·체크포인트 저장·장시간 발열은 측정 구간 밖이다. 전체 epoch나 최종 RMSE 가속 실측이 아니다.

FP32 후보는 동일한 입력을 CPU에서 먼저 전수 검사한 뒤 배치별 GPU Python 분기 검사와 손실 스칼라 CPU 회수를 줄였다. 두 FP32 경로의 eval 출력 차이는 0, 마지막 임시 학습 손실도 같았다. 다만 이 후보는 매 step의 유한값/gradient 오류 검사 빈도를 줄인 비교이며, 그대로 생산에 적용하지 않았다. 실제 도입에는 원천/입력 전수 검사와 동일 수준 수치 오류 감지를 유지하는 새 실행 엔진 검증이 필요하다.

BF16은 CNN에서 최적화 FP32 대비 소폭 빨랐지만 TFGrid에서는 더 느렸다. 짧은 finite 검사만 통과했으며 100epoch 수렴/RMSE 동등성은 미측정이다. 생산 FP32를 변경할 근거가 없다.

사전 계산한 17채널 mmap 캐시는 이미 현 러너에 적용돼 있다. iTransformer는 기존 실측에서 CPU가 더 빨라 CPU 배치를 유지한다. 현재 본학습 source closure SHA가 벤치마크 전후 동일함을 확인했다. Test·공식 제출·생산 체크포인트 로드는 없었다.

M4 Pro MacBook Pro는 [고성능 모드](https://support.apple.com/en-us/101613)를 지원한다. 더 강한 냉각으로 일부 고부하 작업 성능에 도움이 될 수 있으나 본 학습에 대한 추가 배수는 측정하지 않았다. 시스템 전원 설정은 변경하지 않았다.

[원수치](RESULTS.json) · [독립 벤치마크 코드](../../ops/benchmark_r16_mps_speed.py)
