# R12 소형 CNN 학습 엔진 독립 검증

**PASS — 숫자로 만든 fixture만 사용했습니다.** 실제 음향 데이터 학습·Test·gate·eval 읽기는 없었습니다. 모델 크기는 모든 설정에서 동일한 **26,953 parameters / 11채널**입니다. shape를 끈 설정은 기존8채널 뒤에 0채널3개를 붙입니다.

[상세 검증 JSON](R12_TRAINING_ENGINE_VERIFICATION.json) · [사전 검증 계획](R12_TRAINING_ENGINE_VERIFICATION_PLAN.json)

- 8개 long/HF/shape 조합과 legacy contrast control, 총9개 설정에서 실제 forward/backward/Adam1step을 수행했습니다. gradient·parameter는 모두 finite였고, q는 미관측 cell에서0이며 share는 `sum(w*q)`와 일치했습니다. pure 설정의 score는 share와 같습니다.
- 순수 목적함수는 `(p−y)²+.25Σw(q−q*)²`입니다. contrast 함수를 강제로 실패하도록 막아도 pure loss가 정상 계산됐습니다. q에 대한 실제 gradient가 `2w[(p−y)+.25(q−q*)]`와 일치했습니다. 같은 두 cell의 작은 왜곡에 대해 pure loss gradient는 정답을 복원하는 방향, legacy BCE 포함 gradient는 왜곡을 키우는 방향임을 확인했습니다.
- uniform/LF-dominant 에너지 × y=.05/.2/.5의6개 수학 반례에서 pure loss는 정확 q를, legacy loss는 일부 왜곡된 q를 더 낮은 loss로 평가했습니다. 이는 목적함수 검증이며 실제 R9 학습 원인이나 R12 성능 개선의 증명은 아닙니다.
- **실제100epoch/200Adamstep**의 소형 숫자 loop를 실행했습니다. 숫자 fixture의 validation RMSE만 사전 수열 `.4,.3,.2,.2,…`로 대체하여 checkpoint 정책을 검사했습니다. 첫 strict minimum인 **epoch3**이 보존됐고 동률·장기 plateau에도100epoch까지 실행했습니다. 저장 state가 epoch3에서 별도로 캡처한 state와 bitwise 같았습니다.
- best checkpoint의 재로드·저장 validation 예측이 일치했고 batch1 대3 최대 차이는0이었습니다. 학습률은 첫 epoch .001에서 epoch100의 `1.0244252619e−5`까지 고정 cosine으로 감소했습니다. epoch100에서 scheduler.step을 한 이후 목표 eta_min=.00001에 도달하는 설정입니다.
- TRAIN/validation 숫자 배열·검토 소스 SHA는 전후 동일했습니다. 임시 숫자 checkpoint는 `/private/tmp`의 임시 폴더와 함께 삭제했습니다. 생산 모델로 재사용하지 않습니다. core의 STARTED 기록에 있는 `fixture:false`는 이 임시 숫자 실행을 실제 데이터 실행으로 바꾸지 않으며, 이 검증 보고서가 fixture 범위를 명시합니다.

검증은 구현의 수학·제어 흐름을 확인합니다. scripted validation 수치는 모델 RMSE가 아니며, long/HF 음향 데이터의 factorial 구성·독립성은 데이터 구축 검증의 대상입니다. 실제9개 ablation의 개선 여부는 별도로 동결된 성능 측정 결과로 판단해야 합니다.

읽기 검토 중 부모에게 전달한 보강 사항은 target q의 finite/range/missing-zero 명시 검사와 run 종료시 trainer/driver/basis SHA 재대조입니다. 현재 확인한 코드의 올바른 입력에 대한 결과와 이 입력 방어 보강은 구분됩니다. 이 에이전트는 기존 학습 소스를 수정하지 않았습니다.
