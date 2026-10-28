# R15 원인 주장 검토

이 문서는 기존 소스와 저장된 예측만 읽은 검토이다. 실제 Test 입력·숨은 정답을 새로 읽거나 모델 학습·추론·가중치 변경을 하지 않았다. 1,500개 출력의 분포는 **예측값의 분포**이며 정답의 분포가 아니다.

| 주장 | 확인 가능한 범위와 한계 |
|---|---|
| 네 번의 Public 결과로 특정 전처리·모델·가중치가 원인이라고 확정할 수 있다 | R5 CNN, R9 Tree5, R9 all5, R14 step1은 입력 표현·학습 자료/생성법·모델·epoch·배합·Fold 범위·score 정책이 함께 달라진 비교이다. R14가 이번 Public에서 악화된 사실은 확인되지만, 하나의 요인의 인과효과로 분해되지 않는다. Public 점수를 본 뒤 scale/배합을 맞추는 작업은 수행하지 않았다. |
| KS 차이가 크면 높은/낮은 모델 AUC를 의미한다 | KS는 두 집단의 특정 변수 경험분포 차이이다. 대회 AUC는 숨은 양성/음성에 대한 **예측 순위**이다. 입력 한 변수의 KS나 별도 domain-classifier AUC는 대회 AUC가 아니다. 입력 주변분포가 같아도 조건부 정답 관계가 다를 수 있고, nuisance 차이만 커도 domain 구분이 쉬울 수 있다. |
| median이 한쪽에 치우쳤으면 모든 예측이 포화됐다 | median은 중앙 순위만 알려준다. 전체 포화 여부에는 분위수와 0.01 이하/0.99 이상 비율이 필요하다. 본 감사는 mean/std/RMS/9개 분위수와 실제 임계값 비율을 함께 저장했다. mean·RMS가 커졌다는 사실로 숨은 정답 대비 과대예측이나 최적 보정계수를 추론할 수 없다. |
| temporal median/q05/q20 채널이면 수신기·EQ·모든 도메인에 불변이다 | 관측 창 내부의 band별 시간 통계를 빼는 일부 채널은 고정 band offset을 제거한다. 그러나 band profile 채널, normalized energy weights, sqrt(weights), 새 PCA/anchor 채널을 포함한 전체 입력은 모든 frequency EQ에 불변이라고 증명되지 않았다. 공통 gain의 정확 불변도 반올림 이후 또는 반올림과 gain 이동이 교환되는 조건에 한한다. q05/q20은 관측 통계이며 검증된 순수 배경이나 source separation 정답이 아니다. |
| mixture 전체 EQ와 추가 reference만 EQ는 같은 증강이며 기존 y를 유지할 수 있다 | 두 조작은 다른 물리 가정이다. 전체 mixture에 공통 **주파수별** gain을 적용해도 cell q는 유지되지만 전역 energy-share y는 일반적으로 바뀐다. reference에만 EQ를 적용하면 cell q와 y가 모두 바뀐다. 아래 물리식을 다시 계산해야 한다. 공통 scalar gain을 B와 A에 함께 적용한 경우에만 전역 share가 보존된다. |
| R14 NN score는 별도 선박 존재 classifier이다 | 현재 R14 production NN에는 별도 presence/contrast classifier head가 없다. 반환값은 `return p, p, q`이고 저장 점수는 share와 동일하다. R14 최종 ensemble score에는 별도 Tree classifier 혼합이 있으므로 NN score와 최종 score를 구분해야 한다. |

B는 알려진 배경 에너지, A는 알려진 추가 녹음의 에너지, O는 관측 mask이다. 주파수별 gain g>0을 전체 mixture에 적용하면

\[
q'_{tf}=\frac{g_f A_{tf}}{g_f(B_{tf}+A_{tf})}=q_{tf},\qquad
y'=\frac{\sum O g A}{\sum O g(B+A)}.
\]

g가 모든 cell에 같은 scalar인 특수한 경우를 제외하면 y'=y를 일반적으로 보장하지 않는다. 추가 reference만 g로 변형하고 scale α를 적용하면

\[
E'=B+\alpha g A,\quad q'=\frac{\alpha g A}{E'},\quad
y'=\frac{\sum O\alpha g A}{\sum O E'}.
\]

target share를 다시 맞추는 α solve를 사용하더라도 support·관측 mask·cap을 검사하고 raw 에너지 기준 q/y를 다시 계산해야 한다. 실제 reference 녹음 전체를 추가한 에너지 비율은 ‘자연 선박음만의 정답 비율’과 동일하다고 주장하지 않는다.

소스 근거:

- [R14 model: score=share, physical pooling](../../pipeline/models_r14_production.py) 85–97행.
- [R6 relative-grid 불변 조건·quantization 한계](../../pipeline/relative_grid_r6.py) 3–33행.
- [q05와 static-band-offset 불변의 한계](../../pipeline/relative_grid_r6_v2.py) 3–13행.
- [R14 Tree query 저장 항목](../../ops/run_r14_production_outputs.py) 180–191행: component share와 최종 score만 저장한다.
- [R14 hybrid query 저장 항목](../../ops/run_r14_production_outputs_v2.py) 294–314행: classifier 확률은 계산 중 사용하지만 개별 raw 확률은 NPZ에 저장하지 않는다.
- [이번 감사 결과](RESULTS.json), [모델별 통계](distribution_summary.csv), [예측 Pearson](pearson.csv), [예측 Spearman](spearman.csv).

예측 Pearson/Spearman은 **모델 출력 사이의 상관**이다. 숨은 정답에 대한 성능, 잔차 상관, 원천별 실제 AUC로 해석하지 않는다. 새 classifier를 학습하거나 미저장 R14 classifier를 역산해 원 확률이라고 표시하지 않았다.
