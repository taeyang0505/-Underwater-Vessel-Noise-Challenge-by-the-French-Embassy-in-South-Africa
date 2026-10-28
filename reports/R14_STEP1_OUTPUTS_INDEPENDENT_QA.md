# R14 Step1 fast3 출력 독립 검증

결과 **PASS**. 저장 산출물만으로 474개 검사를 통과했습니다. 모델 로드·학습·추론·보정 및 원 Test 파일 읽기는 하지 않았습니다.

- 두 CSV 모두 정확한3열·1,500개 고유 ID·원본 순서를 유지합니다. 숫자는 유한한[0,1] 범위이며 저장 query NPZ의 share/score와 float64 CSV round-trip까지 완전히 같습니다.
- 두 manifest는 완료 영수증과 완전히 같고 source·정책·모델의 고정 SHA가 현재 파일과 일치합니다. R2정책 canonical token이 유효하며 기존 실패 단계의 모든 non-binding 선택 결과도 그대로입니다.
- 저장된17개 개발 패널은 원 비교 NPZ와 sample ID·정답 배열이 완전히 같습니다. RMSE/MAE/bias/AUC 등204개 값을 독립 재계산했으며 최대 오차는 0.0입니다.
- 저장 member share의 고정 가중합도 검산했습니다. 분류기 벡터는 NPZ에 없어 score 결합식 자체를 저장 배열만으로 끝까지 재생하지는 못합니다. CSV와 저장 score의 일치 및 score AUC는 검산했습니다.

| 파일 | SHA256 |
|---|---|
| `submissions/R14_r2/submission_r14_fold0.csv` | `81ec7e4f4324f6585c97d3f8e1140b62e401fc37251d833674e565c17fdb9978` |
| `submissions/r14_main_r2/submission_r14_step1.csv` | `947c71dcc52d50fa2e3f63a6bbc3859cca9413169847697a509199c4d5da70a4` |

정확한 Fold0 혼합과 Tree5 bagging 확장 파일을 구분해야 합니다. 뒤의 파일은 Fold0에서 고정한 계수에 query에서만 Tree5 평균을 넣은 결과이며 전체 신경망5fold OOF 검증값이 아닙니다.17패널은 이미 노출된 개발 자료이고 공식 Test 성능이 아닙니다.

이 검증은 실제 공식 업로드를 수행하거나 업로드 바이트를 확인하지 않았습니다. 원 Test는 열지 않고 기존 R9 로컬 query CSV와 ID 순서만 대조했습니다.

[전체 기계 판독 결과](R14_STEP1_OUTPUTS_INDEPENDENT_QA.json)
