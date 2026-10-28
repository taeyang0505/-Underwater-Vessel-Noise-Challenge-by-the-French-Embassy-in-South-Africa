# R9 실제 데이터 준비·검증 결과

**PASS — 새 데이터 준비 완료, 본학습은 실행하지 않았다.** 기존 R4/R5/R6 자료·모델은 변경하지 않았다. Test·WAV·GPU·SSH 접근 없이 공개 SA의 허용 원천과 변환된 VTUAD power bank만 사용했다.

- Bundle: `data/synthetic/group_cv_minute_full_r9_v1` · `9788ec783be139d51a45a49ec3b22539f10a8b9eff85f3acb501127ae82a4092`.
- 고정 자료: fold당 selection 1,000 / ensemble_fit 500 / ensemble_gate 500 / report 1,000, 전체 15,000예제. 숫자·Unicode NPZ와 별도 metadata/recipe로 저장했다.
- Bundle 71,244,935 bytes, 이 중 압축 NPZ 24,707,614 bytes. 기존 원천 bank는 이 크기에 포함하지 않는다.
- 509개 reference의 원본 2초 power→분 단위 평균과 캐시 경로를 3개 배치 family, 총 1,527회 같은 seed로 비교했다. db·q·share·presence·recipe가 전부 bitwise 일치했다.

| Fold | TRAIN 배경/epoch | 명시 no-source/epoch | LF 후보 A=0/epoch | 목표10% 대비 부족 |
|---|---:|---:|---:|---:|
| 0 | 78,869 | 15,774 | 6,947 | 940 |
| 1 | 78,069 | 15,614 | 6,835 | 972 |
| 2 | 78,362 | 15,672 | 6,904 | 932 |
| 3 | 78,753 | 15,751 | 6,877 | 998 |
| 4 | 78,124 | 15,625 | 6,865 | 947 |

5개 fold TRAIN 역할 합은 epoch당 392,177예제다. 독립 원천 수는 이 합이 아니다. 역할 전체의 고유 허용 배경은 100,964개이며, 원래 공개 자료 1,715,779행을 새 독립 정답 자료로 바꾼 것이 아니다.

각 epoch는 TRAIN의 모든 primary를 permutation으로 정확히 한 번 사용한다. 명시적 no-source는 round(0.2N)개다. LF 후보는 R6의 고정 정의를 각 fold TRAIN에서만 다시 계산했다: 저역 indices0..8의 시간 peak−median bump가 TRAIN q90 이상이고, low bump > max(safe15..22, high20..24) bump +3dB. 목표 0.1N보다 후보가 부족하므로 모든 후보를 한 번씩 A=0으로 사용하고 부족분은 일반 no-source에서 채운다. 배경을 중복해 quota를 맞추지 않는다. 이 의도적 TRAIN 조건부 표본 정책은 고정 평가 역할의 무조건 zero 배정과 차이가 있다. 자연 선박 부재 라벨이라고 해석할 수 없다.

양성 요청 슬롯은 509개 recording index를 균형 cycle로 배정한다. 모든 100epoch×5fold schedule에서 509개가 포함되고 사용 횟수 차이는 최대 1이다. 같은 epoch·index는 접근 순서와 worker 수에 무관하게 같은 혼합을 생성하며 다른 epoch는 바뀐다. 다만 schedule 검증은 계획된 coverage이며, 실제 학습이 중단됐는데 epoch를 완료했다고 인증하지 않는다.

원천의 관측 결측은 보존한다. 모든 역할에 동일한 실제 row 기반 minute QC를 적용하고 관측 최소3분을 요구한다. 허용 원천 100,965개 중 부족한 창 1개를 제외한 100,964개를 수치 검증했고, 이 활성 풀에서 6,253분을 추가 마스킹했다. 원래 source manifest의 6,254개 flag와 1 차이는 해당 제외 창 때문이다. source group 역할 중복 0, 같은 sensor의 명목15분 support 기준 역할 간 간격 최소24시간을 원천 manifest로 재검증했다.

고정 자료 전수의 raw 에너지-q-y 최대 차이는 3.246515567e-08다. 0.1dB 반올림 입력의 가중치로 고정 raw q를 pooling할 때 최대 차이는 0.001658822213이며 사전 quantization bound 이내다. true q/y를 rounded 값에 맞춰 고치지 않았다. x8은 db만 받아 만들고 channel5=관측 mask, channel7=sqrt(동일 rounded w)다.

동일 seed 재실행, pickle 재실행, 4개 spawned CPU worker의 8개 실제 TRAIN 입력, 역순 접근, epoch 변경, 원본 bytes 보존, 고정 자료 재로드를 통과했다. report Dataset은 model/policy freeze gate가 없으면 NPZ를 열기 전에 거부한다. 전역 5fold freeze 시점 관리는 production runner가 맡는다.

509개는 모든 fold에서 공유하므로 reference-independent 검증이 아니다. vessel archive의 weak-background class 22개를 표시한 채 포함했고 별도 ambient 16개는 학습 reference에서 제외했다. 정답은 측정된 전체 recording을 추가한 에너지 비율이다. 측정 recording을 선박 단독 신호라고 주장하지 않으며, 자연 배경에 이미 존재하던 선박을 라벨링한 것도 아니다. geometry와 결측이 만나 양성 요청이 실제 관측 A=0이 되는 경우는 그대로 보존한다. 모델 성능·공식 점수 개선을 이 데이터 QA로 증명한 것은 아니다.

[구체 수치·해시](R9_DATA_READINESS.json) · [4worker·재현성 검증](R9_DATASET_READY_VERIFICATION.json) · [원천·500epoch schedule 검증](../data/synthetic/group_cv_minute_full_r9_v1/source_verification.json) · [최신 모델의 학습 엔진 숫자 fixture](R9_TRAINING_ENGINE_VERIFICATION_GLOBAL_FREQUENCY.json)

재생성은 `python ops/prepare_r9_training.py --config configs/r9_full_training_v1.json --output <새 경로>`로만 수행한다. 기존 출력·실패 폴더 재사용이나 덮어쓰기는 거부한다. 현재 배포물은 이미 생성·검증되었으므로 다시 만들 필요가 없다.
