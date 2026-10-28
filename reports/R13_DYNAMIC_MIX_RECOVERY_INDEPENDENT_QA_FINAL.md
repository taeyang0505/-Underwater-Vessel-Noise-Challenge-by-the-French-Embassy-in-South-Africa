# R13 paired epoch mixture 복구 독립 QA

**완료·무결성 검사는 통과했다.** 기존 R12/R13 자료·모델·선택 정책은 변경하지 않았다. 이 검사는 새 조건의 성능이나 production 승인을 판단하지 않는다.

## 실패와 복구 경계

원래 트리 실행은 `Mapping` import 누락으로 binding 검사에서 실패했다. 해당 위치는 출력 폴더 생성과 `reg.fit`/`clf.fit` 이전이다. [원래 실패 로그](../models/r13_dynamic_mix_r1/tree.log)와 [상위 실행 FAILED 기록](../models/r13_dynamic_mix_r1/FAILED.json)을 보존했다. 이후 runtime namespace 주입 helper는 데이터 사전 검사 중 종료130으로 중단됐고, [중단 영수증](../models/r13_dynamic_mix_r2/ABORTED_DYNAMIC_HELPER_BEFORE_FIT.json)에 당시 tree 폴더 없음·학습0을 기록했다.

읽기 전용 프로세스 조회 시 원래 CNN과 새 정적 트리만 실행 중이었으며 주입 helper는 없었다. 현재 실행한 [정적 helper](../ops/train_r13_dynamic_mix_tree_static.py)는 `collections.abc.Mapping`을 명시적으로 import하며 namespace를 바꾸지 않는다. 원래 R12 learner의 fit AST와 비교해 TRAIN 행 guard `12000 → 24000` 외 연산·하이퍼파라미터 차이는 없었다. 소스 SHA는 `fb87bdd170f6c5d296735896072c2e8f09f57f9add68f0418c455b79b2e2df87`이다.

## TRAIN 및 일정 검산

baseline/highshare TRAIN 12,000행을 직접 검사했다. 원천·관측 mask·site·family·presence는 같고, 2,400개 양성 행만 추가 성분 스케일이 다르다. 나머지 9,600행은 db·원정밀 db·cell q·y가 동일하다. 각 변형의 실제 원정밀 에너지 가중 q 합과 저장 y의 최대 차이는 각각 `3.509e-8`, `3.738e-8`이었다. 알려진 A=0은 양쪽 모두 2,400개이고 q도 0이다.

CNN은 기존 고유 primary 6,000개를 매 epoch 한 번씩 사용한다. A=0 1,200개, long 1,440개를 유지하며 변경된 양성 1,200개 중 600개만 각 epoch highshare를 사용한다. 저장된 100×6,000 선택표를 독립 재현해, 매 2 epoch마다 각 변경 행이 baseline/highshare를 정확히 한 번씩 받는 것을 확인했다. 100 epoch 전체에서는 각각 50번이다.

트리는 두 12,000행 TRAIN을 연결한 24,000개 증강 행으로 fit한다. 고유 primary는 12,000개이며 새 독립 관측이 24,000개 생긴 것은 아니다. 원천 9,600개는 같은 값으로 두 번, 변경 원천 2,400개는 두 스케일로 들어간다. variant suffix를 붙인 sample ID 24,000개는 고유하고, 연결한 모든 배열의 typed SHA가 실제 fit plan과 일치했다. validation_fit은 원래 1,000행 파일과 같은 SHA다. 이 QA는 그 원시 값을 열지 않고 파일 SHA와 실행 코드·바인딩으로 확인했다.

## 완료 모델 재사용 및 동결

원래 CNN은 중단·재시작 없이 1~100의 연속 history를 남겼다. 모든 epoch에 primary 6,000/high 600이 기록됐고, best checkpoint는 최초 validation-fit 최솟값의 epoch59와 일치한다. checkpoint 메타데이터·SHA와 모든 parameter tensor의 finite 여부를 검사했다. 이 writer는 optimizer-step state를 저장하지 않으므로 optimizer 상태까지 독립 증명했다고 주장하지 않는다.

완료 후에만 [원 CNN 폴더](../models/r13_dynamic_mix_r1/cnn/summary.json)를 r2 staging으로 복사했다. 6개 파일 전부 바이트 SHA가 일치한다. 이는 완료된 독립 CNN component의 재사용이며 실패한 상위 실행의 부분 checkpoint에서 학습을 이어간 것이 아니다.

정적 트리의 실제 완료 summary는 `fixture_only=false`, TRAIN24,000/validation_fit1,000/특징673이다. regressor와 classifier의 모델·schema·fit plan·basis·예측 artifact SHA를 모두 검사했다. provenance의 `actual_tree_booster_fits=1`이라는 필드명은 정확하지 않다. 실제 뜻은 **같은 recipe의 learner.fit 1회이며, 회귀 booster 1개와 분류 booster 1개를 각각 fit한 것**이다.

새 [전체 동결 영수증](../models/r13_dynamic_mix_r2/ALL_MODELS_FROZEN.json)은 14개 모델 산출물 SHA, 같은 사전 계획, 원래 소스 핀과 일치한다. 파일 SHA는 `d6fd0a39817c210c3a0cd6833d2a2e1be9fef3996ae0a5db3cf72b089d065e05`다. [복구 완료 provenance](../models/r13_dynamic_mix_r2/CORRECTED_EXECUTOR_COMPLETION.json)는 원 실패 기록 SHA·정적 helper SHA·완료 CNN 복사 SHA를 결박한다.

상세 수치·입력 SHA는 [사전 검사](R13_DYNAMIC_MIX_RECOVERY_INDEPENDENT_QA_PREFLIGHT.json)와 [최종 검사 JSON](R13_DYNAMIC_MIX_RECOVERY_INDEPENDENT_QA_FINAL.json)에 있다. 본 QA는 추가 fit/추론, Test, raw heldout 값 또는 새 17패널 평가 결과를 읽지 않았다. 알려진 development 패널의 후속 성능 평가와 원래 R13 production 후보 판단은 별도다.
