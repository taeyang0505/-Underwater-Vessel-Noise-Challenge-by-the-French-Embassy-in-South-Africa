# R9 실제 자료 연결 fixture

**PASS — 기본 전체 크기 4개 모델, 각각 CPU Adam 1step만 수행한 폐기용 연결검증이다.** Fold0 TRAIN의 완전관측 2개·결측 포함 2개와 selection 고정 첫8개를 사용했다. 실제 본학습이나 성능 비교가 아니며, report·Test·GPU·SSH·WAV 접근은 없다.

채널5/7·동일 rounded energy weights·변경하지 않은 raw q/y·물리 pooling·finite loss/gradient/parameter·optimizer 1step·엄격 checkpoint reconstruction·bitwise reload·원본 입력/코드/원천 SHA 보존을 검증했다. 폐기용 weight는 /private/tmp에만 저장했고 production 재사용을 금지한다. CUDA BF16 및 실제100epoch 속도는 이 검증에 포함되지 않는다.

[숫자·모델 크기·입력/체크포인트 SHA](R9_REAL_DATA_CONNECTION_VERIFICATION.json)
