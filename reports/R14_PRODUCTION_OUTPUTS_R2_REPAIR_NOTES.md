# R14 출력 단계 R2 수정

원본 출력 감독기는 `Policy content changed`로 종료했다. 완료된 Fast3 모델이나 Tree 모델의 학습 실패는 아니다. 원본 드라이버, 실패 단계의 정책 파일, 기존 Tree 전용 CSV·평가 결과, 완료 Fast3 체크포인트는 보존했다.

## 원인과 수정

기존 드라이버는 `binding['source_pins']`에 실행용 `pins` 딕셔너리를 그대로 전달했다. 정책의 canonical SHA를 계산하고 파일로 저장한 다음 실행용 딕셔너리에 정책 파일 SHA를 추가하면서, 메모리 안의 정책도 함께 바뀌었다. 디스크 정책은 유효했지만 메모리 정책은 자기 SHA와 달라져 추론 전 검증에서 거부됐다.

새 `ops/run_r14_production_outputs_v2.py`는 다음 순서로 처리한다.

1. fit/gate binding을 깊게 복사해 소유권을 분리한다.
2. 정책 SHA를 검사하고 새 경로에 저장한다.
3. 저장한 JSON을 다시 읽어 내용과 SHA를 확인한다.
4. 정책 파일 SHA를 추가한 새 실행용 딕셔너리를 만든다. 정책 내부 딕셔너리는 바꾸지 않는다.

Fast3에서는 원래 저장된 정책과 binding·SHA를 제외한 모든 필드를 정확히 대조한다. 가중치, 후보 선택, fit/gate 수치 또는 score 결정을 바꾸면 실행을 중단한다. 앙상블 core, 후보 목록, 안전선, 후처리 수식은 변경하지 않았다.

## 출력 격리와 실행

CLI action은 `hybrid-fast3`, `hybrid-full4` 두 개다. Tree 전용 재생성 action은 제공하지 않는다. 원래 Tree 전용 결과는 기존 경로에 그대로 남는다.

| 항목 | 새 경로 |
|---|---|
| 정책·동결 장벽·완료 receipt | `models/r14_production_outputs_r2/hybrid_fast3` 또는 `hybrid_full4` |
| 17패널 결과·예측분포 감사 | `reports/r14_production_r2` |
| 정확한 Fold0 혼합 CSV | `submissions/R14_r2/submission_r14_fold0.csv` |
| Tree 5fold 평균 확장 CSV | `submissions/r14_main_r2/submission_r14_step1.csv` |
| TF 포함 CSV | 위 파일명에 `_with_tfgrid` 접미사 |

기존 R2 scope나 해당 CSV가 하나라도 존재하면 부분 결과를 재사용하거나 덮어쓰지 않는다. Fast3는 완료된 원래 registry를 strict load한다. 회귀·분류 선택은 기존 동일 Fold0 fit/gate만 사용한다. 17패널과 Test는 fitting에 사용하지 않는다. Tree 5fold 평균을 결합한 CSV는 Fold0 정책의 추론 확장이며, 정확한 5fold 신경망 앙상블 OOF로 표시하지 않는다.

## 이번 검증 범위

`reports/R14_PRODUCTION_OUTPUTS_R2_REPAIR_FIXTURE.json`은 합성 숫자만 사용한 회귀 검증이다. 기존 alias 실패를 재현했고, 수정 후 원래 예측 대비 share·score 최대 절대 차이는 모두 0이었다. 정책·실행 소스 변조, 기존 파일, 부분 scope 재사용, 원래 의사결정 변경을 포함한 10개 거부 검사를 통과했다. 보호 대상으로 기록한 기존 파일 28개의 SHA도 변하지 않았다.

이 검증에서 실제 데이터 배열, Test 또는 17패널을 열지 않았고, 모델을 로드하거나 학습·추론하지 않았다. 실제 R2 출력 실행과 독립 검토 receipt는 별도 단계다. 기존 실패한 감독기를 재시작하지 않았다.
