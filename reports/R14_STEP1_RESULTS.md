# R14 Step 1: 빠른 본체 모델 완료·로컬 CSV 생성

Tree 5-Fold와 Fold 0 신경망 3종을 완료하고 로컬 CSV 두 개를 생성했다. TFGrid는 같은 학습 프로세스에서 계속 학습 중이다. 공식 업로드는 하지 않았다.

## 파일과 적용 범위

- [정확한 Fold 0 결합 CSV](../submissions/R14_r2/submission_r14_fold0.csv): NN Fold 0과 Tree Fold 0의 고정 정책.
- [Step 1 Tree 5-Fold bagging 확장 CSV](../submissions/r14_main_r2/submission_r14_step1.csv): Fold 0에서 고정한 같은 계열 비중을 유지하면서 Test 추론의 Tree 성분만 5-Fold 평균으로 대체.
- [Tree-only 5-Fold 대조군](../submissions/R14/submission_r14_fulltree_5fold.csv): 먼저 완료된 R1 결과를 변경 없이 재사용.

각 파일은 1,500개 ID·3개 열·수치 범위·숫자 재읽기·manifest SHA를 검증했다. Step 1 bagging 확장은 정확한 전체 NN 5-Fold OOF 평가가 아니다. 아래 17패널 표는 정확한 Fold 0 정책의 개발 진단 결과이며, bagging 확장 CSV와 평가 범위를 구분해야 한다.

## 완료 모델과 고정 가중치

| 모델 | 완료 epoch | best epoch | 최저 selection RMSE | 회귀 비중 |
|---|---:|---:|---:|---:|
| bg_cnn | 100 | 98 | 0.049400 | 81.047% |
| fdy | 100 | 94 | 0.074457 | 0.000% |
| itransformer | 100 | 100 | 0.095514 | 0.000% |
| tree_baseline | 최대 1,500 trees / ES40 | 모델별 | 모델별 | 16.716% |
| tree_highshare | 최대 1,500 trees / ES40 | 모델별 | 모델별 | 2.236% |

fit-best 단독은 nn_bg_cnn이다. 공통 baseline fit500에서 가중치를 적합하고 gate500에서 검증했다. gate RMSE는 단독 0.034606, 결합 0.034095로 사전 안전 게이트를 통과했다. iTransformer의 수치상 비중은 사실상 0이고 FDY는 0이다. 모든 후보를 학습한 사실과 최종 유효 비중을 구분한다.

진단 score는 0.75 × share + 0.25 × highshare 독립 분류기 확률이다. 가중치·score 구성은 17패널이나 Test 출력 분포를 보고 바꾸지 않았다.

## 고정 개발 패널 비교

| 패널 | 기존 미니 CNN+FullTree RMSE | 본체 결합 RMSE | 기존 score AUC | 본체 score AUC |
|---|---:|---:|---:|---:|
| R12_eval | 0.058540 | 0.039588 | 0.974600 | 0.975975 |
| R12_high_snr_long | 0.122765 | 0.068634 | 0.998525 | 0.998650 |
| R10_unseen__r9_complete | 0.049560 | 0.032882 | 0.968662 | 0.958025 |
| R11_full_smooth15_VS_st_francis_dense | 0.066746 | 0.062626 | 0.819699 | 0.820538 |
| R11_native_long15_VS_st_francis_dense | 0.057318 | 0.048319 | 0.818006 | 0.813904 |

17개 패널 중 17개에서 RMSE가 기존 결합보다 감소했다. AUC는 일부 패널에서 소폭 낮아졌다. 이 패널들은 기존 개발에 노출됐으며 공식 Test 정답이 아니다. 전체 결과는 [17패널 CSV](r14_production_r2/HYBRID_fast3_17_PANEL_METRICS.csv)에 있다.

## Test 예측 분포: 개선으로 단정하지 않는 항목

| 항목 | 기존 R9 로컬 기준 CSV | 새 Step 1 bagging CSV |
|---|---:|---:|
| share 평균 | 0.264744 | 0.303326 |
| share 표준편차 | 0.191907 | 0.240171 |
| share RMS | 0.326982 | 0.386897 |
| score 평균 | 0.607677 | 0.459316 |
| score 표준편차 | 0.121169 | 0.198580 |
| score ≥0.99 비율 | 0.000000 | 0.000000 |

새 share와 R9 share의 상관계수는 0.921199다. 새 share의 평균과 RMS가 R9보다 커졌다. 따라서 **로컬 RMSE 개선은 확인됐지만, Test 출력 크기가 정상화됐다거나 실제 대회 간극이 해소됐다고 결론 내릴 수 없다.** 이를 숨겨진 정답 없이 확인한 사실로 보고하며, 이 수치를 맞추기 위한 임의 축소·재보정은 적용하지 않았다.

기존 R9 공식 결과는 RMSE 0.217535806·AUC 0.555066666이다. 새 R14의 공식 RMSE·AUC는 미측정이다. R9 비교에 사용한 로컬 CSV는 SHA로 고정했지만 실제 업로드 바이트의 별도 checksum은 확보하지 않았다.

분위수와 모델별 통계·상관행렬은 [분포 감사 JSON](r14_production_r2/HYBRID_fast3_QUERY_DISTRIBUTION_AUDIT.json)에 있다.

## 출력 오류 수정과 보존

원 R1 출력은 정책 binding의 source_pins와 실행 pins가 같은 dict를 공유해, 실행 pins 확장 후 메모리 정책의 해시가 바뀌며 중단됐다. 가중치·학습·데이터 문제로 기록하지 않는다. 원본 실패 로그·코드·정책과 완료된 Tree CSV를 보존했다. R2는 binding 깊은 복사→정책 파일 동결·재로드→별도 실행 pins로 수정했고, 기존 fast3의 수치 정책 결정이 정확히 같음을 확인한 뒤 새 경로에서 평가·추론했다.

- [수정 설명](R14_PRODUCTION_OUTPUTS_R2_REPAIR_NOTES.md)
- [수정 독립 QA](R14_PRODUCTION_OUTPUTS_R2_INDEPENDENT_QA.json)
- [실시간 학습 상태](R14_PRODUCTION_START.html)

TFGrid 50 epoch 완료 후, 별도의 with_tfgrid 정책·17패널·CSV를 추가 생성한다. 현재 학습 중인 모델의 미완료 체크포인트를 제출물에 사용하지 않았다.
