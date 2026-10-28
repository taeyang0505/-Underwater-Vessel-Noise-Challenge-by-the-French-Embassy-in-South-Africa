# R12 CPU ablation 자료 구축·사전 검증

최종 자료는 `data/synthetic/r12_cpu_ablation_v3`이다. 이는 기존 R9/R11을 덮어쓰지 않은 Fold0 소형 CPU 실험용 자료다. 자료 구축과 검증만 완료했으며, 이 보고서는 모델 개선 성능을 주장하지 않는다.

## 네 학습 조건과 공통 검증

|조건|장시간 양성|HF/SF 고밀도 무추가 보강|학습 행|명시적 A=0|관측 추가 양성|
|---|---|---|---:|---:|---:|
|L0_H0|없음|없음|12,000|2,400|9,600|
|L0_H1|없음|있음|12,000|2,400|9,600|
|L1_H0|양성의 30%|없음|12,000|2,400|9,600|
|L1_H1|양성의 30%|있음|12,000|2,400|9,600|

같은 행 순서와 명시적 무추가 스케줄을 사용한다. HF 변경이 양성의 배경·입력·참조·정답을 바꾼 행은 0개다. 장시간 변경이 미지정 행을 바꾼 행도 0개다. 각 조건 내부의 배경은 12,000개 모두 고유하고 R9 TRAIN에 속한다.

CNN용 `cnn_subset_indices.npy`는 네 조건에 같은 6,000행을 적용한다. `SeedSequence([2026102712,6000])`의 사전 고정 순열로 무추가 1,200행, 양성 4,800행을 뽑았다. L1에서는 장시간 양성 1,440행, H1에서는 특수 무추가 600행이다. Tree는 12,000행 전체를 사용하도록 준비했다. 모델 출력이나 검증 오차로 행을 고르지 않았다.

공통 검증 자료는 fit 1,000행, gate 500행, eval 500행이다. 양성 중 일반/장시간을 각각 절반으로 구성하고 명시적 무추가는 정확히 20%다. 별도 `high_snr_long.npz`에는 기존 eval 500행과 다른 고유 배경 500개를 사용했다. 양성은 모두 장시간이며 요청 peak SNR은 U[10,20] dB다. 이 추가 패널은 eval과 source group을 공유하는 사후 동결 평가용 challenge이며 새로운 독립 원천으로 세지 않는다.

## 원천 분리와 노출 한계

R9 Fold0 TRAIN은 78,869행이며 selection은 9,417행/25개 background group이다. R11 실제 원천 parquet와 R11-3이 분석한 기존 R9 selection 1,000행의 원천 ID를 모두 제외하면 미사용 배경 ID 4,903개가 남는다. 이를 전체 background group 단위로 나누고 같은 sensor의 900초 창 support에 추가 24시간 purge를 적용했다.

|역할|샘플링 전 허용 배경|background group|고정 생성 행|
|---|---:|---:|---:|
|validation_fit|2,454|12|1,000|
|validation_gate|1,227|5|500|
|eval|1,144|6|500 + 별도 high-SNR 500|

내부 경계 78행을 purge했고 모든 활성 역할의 최소 support 간격은 24.0시간이다. TRAIN 대 검증, 검증 세 역할 상호 간의 ID/group 교집합은 0이다. 시각은 원본 export clock 기준이며 독립 검증된 UTC라고 주장하지 않는다.

그러나 R11이 이미 selection의 25개 group 모두를 사용했다. 따라서 새 ID와 TRAIN 분리는 성립하지만, R11에 전혀 노출되지 않은 새 selection group은 0개다. 이 자료는 개발 검증이며 blind/OOF 또는 대회 Test와 같은 분포라는 주장을 하지 않는다. 509개 VTUAD 참조 역시 TRAIN/검증에 공유되므로 참조 독립 검증이 아니다.

## 장시간 참조와 관측 support

실제 ≥15분 참조는 7개이고, 실제 5~14분 참조는 106개다. L1의 장시간 양성 2,880행 중 720행은 7개 native 참조에서 12~15분 연속 crop, 2,160행은 106개 참조의 native 1분 행렬을 12~15분으로 보간하고 `.6+.4*sin²` bell power envelope를 적용했다. 후자는 새 실측 WAV도, 새 독립 녹음도 아니다. 실제 106개와 7개 참조가 모두 사용됐음을 확인했다.

같은 hull의 UTC 끝/시작이 정확히 일치하는 쌍은 8개였으나 합산 길이는 6~14분이었고, per-clip 공통 recorder/hydrophone 획득 식별을 확인할 수 없어 이어붙이지 않았다. 시간 인접성만으로 실제 연속 녹음이라고 만들지 않았다.

모든 양성에 대해 관측된 추가 에너지 합이 양수인 유한 배치 후보를 먼저 구한다. 짧은 native_edge 참조가 중간 관측 블록에 닿을 수 없으면 동일 참조를 native_inside로 배치하고 recipe에 남긴다. 무한 seed 재시도, 양성의 0 재라벨링, 예제 삭제는 없다. 일반 조건의 fallback은 131행, 장시간 조건은 97행이다.

이는 mask-independent placement였던 원형 R9와 다른, 사전에 선언한 support-conditioned 실험법이다. 명시적 20% A=0과 모든 양성의 관측 support를 동시에 충족하기 위한 변경이다. 또한 12~15분짜리 잠재 궤적이라도 결측 배경에서 실제 관측된 양성 support가 12분 이상인 것은 아니다. 관측 support별 recipe를 보존한다.

## HF 무추가와 PCA 원천

HF 후보는 **TRAIN만**의 관측량 ≥13분 배경에서 band18..24(약 1995~7943 Hz)의 선형 합→dB 시간 max−median이 TRAIN q90을 초과하거나, 실제 StFrancis 고밀도 배경인 경우다. 자연 선박 부재 정답이 아니며 알려진 정답은 추가 성분 A=0뿐이다. H1 TRAIN에서는 요청한 특수 무추가 1,200행을 모두 충족했다. SF/HF 조건은 일부 교차하므로 두 조건의 개별 개수를 더해 고유 배경 수로 세지 않는다.

새 검증에는 R11이 사용한 StFrancis 고밀도 312개를 다시 넣지 않았다. TRAIN q90을 그대로 적용한 새 검증 후보가 부족해 fit/gate/eval에서 특수 HF 무추가는 각각 28/4/10행이다. 요청 대비 부족한 72/46/40행은 기존 무추가 배경을 유지했다. 후보를 복제하거나 문턱을 낮추지 않았고 전체 무추가 비율은 20%를 지켰다.

`background_basis.npz`는 합성 입력이 아니라 실제 R9 TRAIN SA 배경 2,000개다. 중복 없이 가능한 범위에서 site를 균형화했다: Richards Bay312, Durban311, Cape Recife240, Bird Island311, St Croix311, Riy Banks204, StFrancis311. 원본 minute QC 이후 배열과 byte 단위로 같음을 검증했다. PCA fitting은 별도 담당이며 `manifest.json`의 `basis_training_binding`을 그대로 사용하도록 제공한다.

## 물리식·검증·재현성

EQ는 ±6 dB 이내이며 기본 peak SNR은 U[-10,10] dB다. `alpha=10^(SNR/10)*median_observed_minute(B)/peak(full_EQ_reference)`로 알려진 성분을 합성하고, 관측 mask 위에서 `E=B+alpha*R`, `q=alpha*R/E`, `y=sum(alpha*R)/sum(E)`를 계산한다. 장시간 파생 궤적은 해당 파생 궤적의 배치·mask 전 peak를 명시한다. 정답은 반올림 전 성분에서 계산하고 입력은 0.1 dB로 반올림한다. 모델 가중치는 반올림된 관측에서 계산하므로 두 가중치의 차이를 이론 bound로 검사한다. share 상한을 맞추기 위한 clipping이나 정답 prior 맞춤은 없다.

숫자 fixture 120개 양성/32개 무추가/짧은 edge fallback/후보 부족 중복 거부가 통과했다. 완성 후 별도 saved-array 검증 423항목이 통과했다. 네 조건 페어 일치, 정확한 무추가/양성 support, 원천 역할 분리, raw 에너지/q 정답 항등식, 반올림 bound, 64개 고정 샘플의 정확 재현, PCA 2,000개 원천 배열 일치, 원본 R9 코드·bank SHA 불변을 포함한다. 이 검증은 모델 fitting/추론, Test 또는 미사용 report 값 접근을 하지 않았다.

최초 v1은 validation HF 후보 부족 시 중복 배경을 만드는 상황을 사전 검사에서 차단해 `FAILED.json`과 부분 자료를 보존했다. 부분 자료를 재사용하지 않았다. 후보를 기존 전체 primary ID 밖에서 고르도록 수정하고 R11-3의 기존 selection 원천도 제외한 뒤 v2를 새로 생성했다. v3는 subset seed/선정법의 manifest 서술을 추가해 다시 생성한 최종판이며 v2/v3의 모든 NPZ/NPY/parquet는 SHA가 동일하다. 어떤 모델 결과를 보고 수정하거나 seed를 바꾸지 않았다.

최종 논리 크기는 207,367,801 bytes다. 핵심 고정값:

* data hash: `42e51eedf28723335cb75af7c9229d2ffb1796ff549a0ffd8533f30a4c1ddfab`
* manifest SHA: `06a8a1c8196a3cb172277f120219975b3c2d251696d7042419ec7155b2548e98`
* generator SHA: `4ad08edb149745f1b5c04c2672cf588c47ba0e80b5e1dcf7bd35ef01f634cd14`
* builder SHA: `43e6d8e6e2227dacecf2c846dd003012cabe38aa730c3e1f5ea5577969eec86a`

[423항목 실제자료 검증](R12_DATASET_VERIFICATION.json) · [검증 코드](R12_DATASET_VERIFICATION_source.py) · [최종 자료 manifest](../data/synthetic/r12_cpu_ablation_v3/manifest.json)
