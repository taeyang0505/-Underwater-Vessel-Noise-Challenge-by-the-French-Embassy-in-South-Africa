# R12 HF 보강의 site·sensor·recorder·관측 마스크 일치 대조자료

최종 자료는 `data/synthetic/r12_mask_matched_control_v2`다. 기존 v3 `L1_H0`를 기본으로 무추가 배경만 교체했다. 새 자료 준비와 독립 검증을 완료했으며, 이 단계에서 새 모델을 학습하거나 성능 향상을 측정하지 않았다.

|항목|전체 TRAIN|기존 공통 CNN subset|
|---|---:|---:|
|행 수|12,000|6,000|
|무추가 A=0|2,400|1,200|
|관측 추가 양성|9,600|4,800|
|HF 교체 요청|1,200|600|
|실제 교체|1,199|599|
|후보 부족으로 원래 행 유지|1|1|

교체 위치는 기존 H1의 무추가 지정 위치를 그대로 썼다. 새 후보는 R9 Fold0 TRAIN만 사용하고, 원래 12,000개 source ID를 모두 제외했다. `site + sensor_group + recorder 문자열 + QC후 실제 관측분 15-bit pattern`이 정확히 같은 층에서 HF18..24(약1995~7943Hz) 시간 max−median dB가 해당 **TRAIN 층의 q90을 엄격히 초과**하는 후보만 골랐다. 고정 seed `2026102713`과 source-ID SHA 순서를 사용했고, 후보는 중복 없이 한 번만 썼다. 모델 오차나 Test 통계로 후보·문턱·방향을 고르지 않았다.

부족했던 한 행은 row index9579, Richards Bay/5708 recorder, 관측 mask28671이다. 원래 무추가 행을 그대로 유지했고 q90을 낮추거나 다른 recorder/관측량으로 대체하지 않았다. 새 배경도 자연선박 부재 정답을 가진 것은 아니다. 알려진 사실은 통제된 추가 성분이 A=0이라는 점뿐이다.

## 보존한 내용

모든 행의 site·sensor·recorder·정확한 관측 mask·q·y·presence는 H0와 같다. 모든 양성의 NPZ 공통 필드 bytes가 전부 같고, long/native 참조·샘플 순서·sample ID도 보존했다. 전체 source ID는 12,000개 고유하며 교체된 1,199개 ID는 원래 12,000개 ID와 겹치지 않는다. 기존 CNN subset 파일도 bytes 그대로 복사했다.

|마스크 단독 사전고정 방향 `−Nobs`|변경 전|변경 후|차이|
|---|---:|---:|---:|
|TRAIN12,000|0.5031543403|0.5031543403|0|
|CNN6,000|0.4963710938|0.4963710938|0|

site×sensor×recorder×effective-mask×presence의 전체 교차표도 정확히 같다. 이는 기존 H1의 마스크·사이트별 class 비율 변화를 통제한 것이다. 배경의 실제 음향 질감이나 녹음일·background group 구성까지 동일하게 만든 실험은 아니며, 모델이 어떤 특징을 학습했는지는 후속 동결 평가에서 확인해야 한다.

## v1과 엄격한 v2의 구분

현재 TRAIN에서는 site별 sensor_group이 한 값이지만 실제 recorder가 반드시 한 대인 것은 아니었다. Richards Bay와 St Croix에는 recorder label이 각각3개다. 초기 site+mask v1은 sensor_group은 같았지만 recorder가449행 바뀌었다. 이 메타데이터 발견으로 원래 v1 자료와 builder snapshot을 보존하고, v1 부분 모델 run은 부모가 중단·보존해 재사용하지 않기로 했다.

v2는 별도 `ops/build_r12_mask_matched_control_v2.py`와 새 출력에서 strict recorder 층까지 적용했다. 기존 v1 builder는 원래 SHA로 정확히 복원했고 기존 R9/v3/주요17개 모델 학습 정책을 수정하지 않았다. v2 후보 선정은 메타데이터에 근거한 후속 대조이며 원래8조건과 함께 처음부터 사전등록된 조건이라고 주장하지 않는다.

## 검증과 로드

독립 검증36항목이 통과했다. 전체 행의 실제 센서·recorder·mask·정답 일치, 양성 bytes 일치, TRAIN 원천 격리/고유성, 실제 원천의 HF 점수와 모든 층 q90 재계산, zero 물리 정답, 반올림 입력과 원천 QC 배열 일치, 마스크 AUC 차이0, 원본 SHA 불변을 포함한다. 새 validation_fit/gate/eval/locked 패널 값, Test, 네트워크, 모델은 열지 않았다.

```python
from ops.build_r12_mask_matched_control_v2 import load_control
train, manifest = load_control(
    'data/synthetic/r12_mask_matched_control_v2'
)
# CNN에는 동일 폴더의 cnn_subset_indices.npy를 적용한다.
# validation_fit와 TRAIN PCA basis는 원래 v3의 것을 그대로 사용한다.
```

신규 TRAIN의 source binding에는 이 대조자료의 새 `manifest.json` SHA와 data hash를 명시해야 한다. `basis_training_binding`은 원래 TRAIN-only PCA의 binding을 그대로 유지한다.

* data hash: `d7ae91b0b8b653927a83627eceb61eeee52210d246f3b3671a5bba79e2052cbd`
* manifest SHA: `ef8a08e159f3abab0fb2b7ff66f01cddaf4b5e2a88984b33a5722f9d2157143e`
* TRAIN NPZ SHA: `cf4bee7a67003dda0806e3dfc6cc8b214480a72835793cb5cdd4a22a82b6612e`
* builder v2 SHA: `8d108c47f24dffe93524467fed1368a453015676c8397df0299b3106db98f91d`

[독립 검증 결과](../data/synthetic/r12_mask_matched_control_v2/independent_verification.json) · [독립 검증 코드](R12_MASK_MATCHED_CONTROL_INDEPENDENT_QA_source.py) · [원래 마스크 교란 감사](R12_TRAIN_MASK_PROVENANCE_AUDIT.md)
