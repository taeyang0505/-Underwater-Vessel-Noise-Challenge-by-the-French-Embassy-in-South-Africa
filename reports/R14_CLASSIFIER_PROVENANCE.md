# R14 Tree classifier 수치의 출처 검토

**0.944494→0.760763 전체를 highshare의 영향으로 해석할 수 없다.** 첫 수치는 원래 R12의 마스크·사이트 교란이 있던 H1 조건이고, 엄격히 매칭한 baseline에서는 classifier full15/SF AUC가 이미 0.770705다. 동일 strict baseline과 highshare의 차이는 −0.009942다. 이는 기술 통계이며 원인·통계적 유의성의 확정은 아니다.

## 같은 패널·같은 출력으로 비교

| 모델 조건 | 패널 | 독립 classifier AUC | 회귀 share AUC | 기존 hybrid score AUC |
|---|---|---:|---:|---:|
| R12_original / Tree_L1H1P1 | full15/SF | 0.944493590 | 0.875717949 | 0.942583333 |
| R12_original / Tree_L1H1P1 | native15/SF | 0.950814103 | 0.863974359 | 0.948711538 |
| R12_strict_matched / Tree_L1HM1P1 | full15/SF | 0.770705128 | 0.813275641 | 0.770634615 |
| R13 / Tree_highshareP1 | full15/SF | 0.760762821 | 0.776769231 | 0.754769231 |

classifier는 독립 이진 LightGBM의 확률이고, Tree hybrid score는 `.5 share + .5 classifier`다. 첨부의 baseline `0.820224`는 엄격매칭 CNN/Tree 50:50 회귀 share AUC다. 위 classifier 0.944494나 0.770705와 다른 출력이므로 직접 연결하면 안 된다.

원 수치는 [R12 CSV](r12_cpu_ablation_r1/results.csv), [strict-matched CSV](r12_mask_matched_evaluation_r2/results.csv), [R13 CSV](r13_cpu_correction_r1/results.csv)에 있다. 정확한 행·summary·TRAIN/validation typed SHA는 [추출 JSON](r14_decoupled_v1/classifier_provenance.json)과 [행 CSV](r14_decoupled_v1/classifier_provenance_rows.csv)에 보존했다.

## 데이터·특징·학습 예산

| 조건 | 특징 수 | TRAIN 행 | threads | best 회귀 / classifier iteration |
|---|---:|---:|---:|---:|
| R12_L1H1P1 | 673 | 12000 | 2 | 1484 / 212 |
| R12_strict | 673 | 12000 | 4 | 1417 / 370 |
| R13_baselineP1 | 673 | 12000 | 2 | 1417 / 370 |
| R13_highshareP1 | 673 | 12000 | 2 | 1116 / 217 |

모두 같은 TRAIN-only SA 배경 PCA3 basis를 포함한 compact673 특징, 최대1,500 trees·patience40, validation_fit1,000을 쓴다. 트리는 epoch100을 적용하는 신경망이 아니며 classifier best iteration 차이를 단순 epoch 부족으로 표현할 수 없다. strict-matched와 R13 fresh baseline의 TRAIN/fit typed 입력은 정확히 같다. 저장 classifier 확률은 17패널 모두 exact이며 model text의 차이는 num_threads4→2 항목뿐이었다. highshare와 baseline은 presence label 및 공통 validation_fit이 같지만, 추가 양성 진폭을 바꿨으므로 입력 특징과 회귀 y는 다르다.

## 마스크·사이트 교란

[기존 TRAIN-only 출처 감사](R12_TRAIN_MASK_PROVENANCE_AUDIT.json)의 실제 분모는 다음과 같다.

- 원 L1H0의 St Francis·관측15분 행: 411개 중 A=0 85개(20.681%). 원 L1H1은 987개 중 A=0 661개(66.971%)다. 양성326개는 같지만 dense 음성이 늘었다.
- **St Francis 내부** −Nobs AUC는 L1H0 0.502862(n=2,720), L1H1 0.794710(n=3,068)이다. **전체 TRAIN12,000**에서는 각각 0.503154와 0.577685다. 0.7947을 전체 TRAIN AUC로 부르면 틀린다.
- strict matched v2는 site·sensor·recorder·정확한 관측 mask·target 표를 보존하며 HF 음성1,199개를 대체했다(요청1,200 중1개 부족). 원래 양성의 bytes는 유지했다. [검산](../data/synthetic/r12_mask_matched_control_v2/audit.json)에 근거한다.

따라서 original H1의 높은 full15/SF classifier AUC에는 다른 음성 표집과 mask/site 상관이 함께 존재한다. 높은 수치를 일반화 증거로 취급하거나 strict 조건의 하락을 모두 highshare 때문이라고 단정할 수 없다. 반대로 이것만으로 original H1의 성능 전부가 mask 하나 때문이었다고 확정한 것도 아니다.

## 물리·인과 해석에서 주의할 점

PCA 입력에서 한 분의 공통 평균을 빼도 `db_f−db_g`는 변하지 않는다. 원래 상대8채널의 band-profile도 대역 간 상대 크기를 포함한다. 따라서 “평균 차감으로 모든 magnitude 정보가 사라졌다”는 단정은 맞지 않다. 추가 앵커 특징은 같은 관측의 재표현이므로 학습을 도울 수 있지만 새로운 측정 정보나 자연 선박 정답을 만드는 것은 아니다. 저역 앵커가 순수 배경이라는 보장도 없다.

시간상 완전히 일정한 추가 성분은 시간 차감 특징을 약하게 만들 수 있으나, 이것이 실제 모델 오차 전부의 원인이라는 실험적 증명은 별도다. highshare는 관측 에너지에서 q/y를 정확히 재계산한 제어된 증강이며 이번 감사에서 라벨 오류를 발견하지 않았다. “오염·비현실적 폭증·대회 실패 원인 확정”으로 부를 근거는 확보되지 않았다. 17패널은 이미 development에 노출됐고 참조 녹음도 공유하므로 blind 성능이나 공식 Test의 인과 설명으로 확대하지 않는다.

새 분리 결합의 [실측 결과](r14_decoupled_v1/RESULTS.md)는 동일 목표 동시 통과에 실패했으며 원 정책을 소급 변경하지 않았다.
