# R14 생산 실행 사전 독립 QA

**CPU Tree·신경망 CPU 계약·실제 MPS capability·앙상블 출력 계약의 사전 검사가 통과했다.** 완료된 본학습이나 새 Test 성능을 뜻하지 않는다. 본 QA 담당자는 실제 생산 모델 학습·Test 추론을 실행하지 않았다.

## 승인과 원래 기록

새 생산 승인은 [사용자 승인 기록](../ops/r14_production_approval.json)에 따른 명시적 override다. 원래 `full15/SF AUC=0.8196987179487178`, 기준0.82라는 숫자를 반올림하여 과거 통과로 바꾸지 않았다. 기존 R13/R14 diagnostic의 false 기록은 그대로 보존하며 새 실행만 true다. 공식 업로드는 승인 범위에 포함하지 않는다.

## CPU Tree 계약

[Tree 독립 QA](R14_PRODUCTION_TREE_PREFLIGHT_INDEPENDENT_QA.json)는 validate-only CLI/API를 실제 실행했다. Fold0의 원 `a2581196…` 전체 동결 파일·completion·모델 artifact를 정확히 재사용하고, 나머지 Fold1~4는5252특징의 baseline/highshare8개 learner(회귀·분류16개 booster)를 CPU2에서 순차 학습한다. 각 fold 자신의 TRAIN에서만 fit한 PCA와 해당 변형 selection1,000의 ES1500/40을 사용한다. 원천 ID/group 분리와900초 지원+24시간 간격은 메타데이터로 확인했다. 시각은 export-clock이며 새 UTC 인증은 아니다.

## 신경망 입력·checkpoint 계약

[14채널 oracle 검사](R14_PRODUCTION_14_FEATURE_INDEPENDENT_QA.json)는 다섯 fold의 TRAIN 고유 마스크115/99/109/96/95종 전체를 대표하는 실제 합성 입력514개에서 NumPy 기준식과 Torch14 채널·w 오차0을 확인했다. TRAIN 전행의 수치 전수 비교라는 뜻은 아니다. partial-band·반올림 tie·URN 상위2대역·+20dB 공통 gain·batch 분할도 검사했다.

[CPU 계약 QA](R14_PRODUCTION_NN_INDEPENDENT_QA.json)는 네 default full-size architecture 각각 synthetic Adam1 step, 유한 gradient, CPU optimizer/scheduler 저장, strict 복원 bitwise 동일, fixture의 기본 production 거부 및 잘못된 config/fold/data/PCA/approval 거부를 검증했다. fixture 가중치는 private/tmp에 명시적으로 격리됐고 생산에 재사용되지 않는다.

생산 전처리는 CPU FP64, 본체는 CPU/MPS FP32다. fast3은100 epoch, TFGrid는 사용자가 허용한 예산50 epoch이며 early stopping을 쓰지 않는다. 손실은 c0 비가중 ShareMSE+.25 energy-weighted cellMSE이며 contrast/presence BCE가 없다. best/latest를 optimizer·scheduler·구조화된 RNG와 함께 CPU portable 형태로 원자 저장한다. source epoch receipt는 cache 삭제와 별도로 영구 보존하고 raw cache는 최대2개로 제한한다. 실패·부분 실행은 자동 resume하지 않는다.

[실제 MPS 검사](R14_PRODUCTION_NN_PREFLIGHT_FINAL_V2.json)는 별도 담당자가 수행했고 본 CPU QA와 source closure가 정확히 같다. 실제 device→CPU checkpoint 예측 차이는 최대4.40e−7, optimizer tensor는 CPU finite였다. 선택 batch는 iTransformer/FDY/BG128, TFGrid64다. TF128의 sampled driver memory가 사전8GiB 제한을 넘어 제외됐다. sampled memory는 실제 peak가 아니고 짧은 step benchmark는 전체 epoch/100epoch 시간의 실측이 아니다. 실제 생산 epoch1 안정성 gate를 통과해야 같은 실행에서 예산까지 계속한다.

## 앙상블·보고·출력 계약

[출력 독립 QA](R14_PRODUCTION_OUTPUTS_INDEPENDENT_QA.json)는 synthetic1,500행 CSV의3열·ID·float64 정확 roundtrip 및9개 잘못된 CSV/report-gate 입력 거부를 검증했다. 원 Test는 열지 않았다. nonnegative simplex 수식은2개 member의 ridge closed-form 해와1.11e−16 차이로 검산했고, gate에서 fit-best single fallback도 확인했다.

각 fold는 공통 baseline ensemble_fit500/gate500만 사용한다. 모든 Tree5 정책이 동결된 후 새 tree-only report gate가 열린다. Fold0 hybrid는 NN0+Tree0만으로 결합을 선택한다. Test에서 Tree-family 예측을5fold 평균으로 대체하는 것은 **Fold0 정책의 추론 시 bagging 확장**이며 exact hybrid5fold OOF라고 부르지 않는다. 17개 development 패널과 Test 라벨로 새 결합 가중치를 fit하지 않는다.

원17패널 비교는 과거 registry·CSV·각 NPZ SHA와 ID/target 정렬을 확인한다. Test는 정책·모델 barrier 후 기존 raw SHA `3ef819…`와 정해진375개 열을 검증해 추론에만 사용한다. 상수 벡터의 Pearson은 null로 기록한다. 이전 R9 CSV는 로컬 기준 SHA이며 실제 사용자가 업로드한 바이트 checksum은 확인됐다고 주장하지 않는다.

## 범위

모든 참조509개는 역할/fold 간 공유되므로 reference/hull 독립 검증이 아니다. synthetic report 행 수는 독립 녹음 수와 다르고, 개발에 노출된17패널은 blind 검증이 아니다. 현재 사전 QA는 실행 계약을 확인했으며 향후 학습 완료·실제 epoch 시간·최종 CSV·공식 점수는 별도로 검증해야 한다.
