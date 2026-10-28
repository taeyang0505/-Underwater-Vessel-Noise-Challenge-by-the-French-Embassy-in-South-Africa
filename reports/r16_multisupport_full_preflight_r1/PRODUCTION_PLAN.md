# R16 본체 다중 support 실행 고정 계획

사용자는 파일럿 장시간/HF RMSE0.117553 > 기준0.11 미달을 명시하고 본체 학습·CSV 생성을 진행하도록 승인했다.

- D0F1 및 H0F1 두 전문가를 각각 새 학습한다. 기존8채널 epoch27/Tree5는 변경하지 않는다.
- Tree: 조건별5-fold, 회귀5160특징/독립분류2033특징, CPU2 threads, 최대1500trees/ES40.
- NN: 조건별4아키텍처 Fold0, relative17(기존8+F1시간9), 100epoch/Adam/noES/dropout0.1. 최적 회귀/분류 체크포인트는 local_fit에서 각각 선택한다.
- 데이터: 기존900초support+24hpurge 배경분할 계승. A′75참조/13연결그룹을 순환5-fold, B19/C′25는 전역 학습 제외. ≥15분 참조는 단1개 연결그룹이며 fold4에서TRAIN에서 빠진다.
- 양성 native 길이 계층35:40:25(1~4/5~10/11분이상), 없는 길이 계층은 가용 계층에 비례 재분배한다. D0/H0 원천·마스크·무추가20%·채널/연산자 난수는 짝지으며 H0는 원래 음성을 그대로 쓴다. 강제 highshare/target matching/flat15/PCA/LFanchor 없음.
- 조건 내부 회귀가중치만 local_fit에서 ridge simplex fit. Tree>=0.5, 각NN<=0.25. 독립분류만 사전후보비율로 선택. 조건 간 회귀 및 독립분류는 각각50:50 고정.
- 새 Test 접근은 모든 모델/정책동결 이후 추론 시에만 허용. 후처리는clip[0,1]만. 평가 패널·Test로 재가중/재보정하지 않음.
- 완료 후고정3축/native12-HF/SF 및 ownfold TreeCV를 보고한다. 최종hybrid나NN전체5-fold OOF라고 부르지 않는다.
- CSV: submissions/r16_multisupport_full_r1/submission_r16.csv. 공식 업로드 없음. 실패자동재시도/실패partial재사용 없음.
