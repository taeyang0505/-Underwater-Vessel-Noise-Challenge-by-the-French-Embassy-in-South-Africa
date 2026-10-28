# R9 GPU 배포·시작 영수증

실제 RTX4090 24GB에서 R9 신경망 학습을 **2026-10-26 15:36:13 KST**에 시작했다. supervisor PID2256, driver PID2257이다. 고정 설정은 4모델×5fold, 각100epoch, no early stopping, Adam, dropout0.1이다. 아직 학습 완료 결과는 없다.

로컬 fresh 추출본의 structural preflight를 먼저 통과한 뒤 173개 tar entry를 서버로 전송했다. archive SHA 및172개 payload SHA를 전수 대조했고 Test·원시WAV·구형 가중치·자격증명은 포함하지 않았다. 별도 가상환경은 Torch2.8+cu128을 유지하며 프로젝트 검증 버전 패키지를 설치했고 pip check를 통과했다.

실제 TRAIN128개·4worker·기본 전체크기4모델 CUDA BF16 forward/backward/Adam1step과 FP32 physical pooling, finite gradient·parameter를 확인했다. 이 probe의 임시 가중치는 저장하지 않았다. Fold0 CNN epoch1은 실제78,869예제·617업데이트·46.62초이며 selection RMSE0.0927662다. 이는 중간 개발 지표로, 공식 점수나 완료 성능으로 해석할 수 없다.

원천·config·동결 코드는 바꾸지 않았다. 서버 LightGBM 학습은 없으며 Mac 단계는 독립이다. 실패 시 자동 재시작·resume·부분 가중치 재사용 없이 종료한다.

[숫자·환경·SHA](R9_GPU_DEPLOYMENT_LAUNCH_RECEIPT.json) · [현재 연결·PID 상태](../ops/r9_remote_state.json) · [실제4모델CUDA 검사](r9_remote_r1/R9_CUDA_FULL_MODEL_PREFLIGHT.json) · [휴대성 검사](R9_GPU_PACKAGE_PORTABILITY.json)
