# R3 첫 RMSE 개선 결과

R2 전체 5-fold 실행을 완료하고 독립 재계산한 뒤에만 이 실험을 시작했습니다. 동일한 15,000개 OOF와 9개 후보를 사용했습니다. R2 데이터·모델·정책은 수정하지 않았습니다.

| 평가 | R2 동결 앙상블 | R3 정규화 앙상블 | R2 CNN 단독 |
|---|---:|---:|---:|
| 전체 OOF RMSE | 0.245585 | 0.240328 | **0.238545** |
| 경계 이상치 제외 | 0.247995 | 0.242822 | **0.238023** |

R3는 R2 앙상블 대비 RMSE를 약 2.14% 낮췄지만 CNN 단독보다 나빠 최선 모델로 승격하지 않았습니다. 공식 대회 제출 점수가 아니며 새 독립 최종 평가도 아닙니다.

## 수정 및 검증

각 fold의 INNER 예측만 사용해 `MSE + 0.05 × ||w − uniform9||²`를 최소화했습니다. 가중치는 음수가 아니고 합이 1이며 affine는 [1,0]입니다. λ=0.05 후보 한 개를 사전 고정했고, λ 탐색이나 outer 정답에 대한 보정은 하지 않았습니다.

5개 정책과 입력 해시를 모두 저장한 뒤 OOF 예측을 열어 한 번 평가했습니다. 부모 검증기가 15,000개 ID·정답 정렬, 전체/경계 제외 RMSE, 각 fold의 저장 가중치와 예측의 완전 일치를 다시 확인했습니다.

## 확인된 주요 오차

R2 OOF에서 native 참조가 포함된 사례는 6,695/15,000개(44.6%)이며, 앙상블 제곱오차의 **78.1%**를 차지했습니다. 해당 구간 RMSE는 0.3248이고 실제 평균 0.4916을 평균 0.2883으로 과소예측했습니다. 데이터 수량뿐 아니라 장시간 참조의 원천 다양성과 분포를 다음 개선의 우선 검토 대상으로 둡니다. 이 진단으로 기존 검증셋을 제거하거나 점수를 유리하게 바꾸지는 않았습니다.

## 재현 자료

- [완료된 R2 보고서](R2_FULL_PIPELINE_REPORT.md)
- [R2 독립 검증](R2_OOF_INDEPENDENT_VERIFICATION.json)
- [R3 실행 코드](../ops/ensemble_shrinkage_r3.py)
- [R3 fit 입력·정책 해시](../models/ensemble_shrinkage_r3/fit_inputs_manifest.json)
- [R3 원본 평가](../models/ensemble_shrinkage_r3/evaluation_report.json)
- [R3 독립 재계산](R3_SHRINKAGE_INDEPENDENT_VERIFICATION.json)
- [OOF 원천·route별 오차](R3_OOF_ERROR_AUDIT.json)

## 추가 참조의 품질·원천 감사

[기존 VTUAD 원천 보강 감사](R3_REFERENCE_COVERAGE_AUDIT.md)에서 시간 변형 후보 44개와 원본 47개에 대한 CRC/SHA를 확인했습니다. 선택된 R2 참조의 직접 hull/day/overlap/byte 분할 위반은 모두 0건입니다. 시간 변형은 새 물리적 독립 원천을 만드는 것이 아니며, 별도 신규 데이터 생성이나 GPU 학습은 아직 실행하지 않았습니다.
