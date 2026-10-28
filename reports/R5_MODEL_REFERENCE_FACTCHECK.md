# R5 대형 모델·외부 원천 추천문 검증

확인일: 2026-10-25. §1–7은 공개 논문·저자 저장소·모델 카드·대회 공식 페이지와 로컬 메타데이터를 읽은 원문 검증이다. §8의 작은 CPU protocol fixture를 거쳐, **별도 후속 승인에 따른 실제 공개 체크포인트 다운로드·CPU 검증은 최신 §9에 기록**했다. 계정 행동, Gemini/비공개 자료 전송, Test 실행, 실제 Fold 학습, 공식 제출은 하지 않았다. R2/R4 데이터·모델·분할은 변경하지 않았다.

**핵심 판정:** Dasheng-1.2B와 Time-MoE Ultra 2.4B는 실재한다. 그러나 Time-MoE Ultra의 활성 크기는 공식 논문 기준 **1.1B**로, 300M이라는 수치는 틀리다. 대형 오디오 모델의 분류/표현학습 성적은 이 대회의 추가 에너지 비율 회귀 성능을 입증하지 않는다. SA 원본은 오디오가 아닌 1분 대역 요약이며 AIS 자료도 제공하지 않는다. 아래에서 확인된 사실과 현재 데이터에 대한 판단을 구분한다.

## 1. 오디오 모델의 크기·입력·권리 표시

| 모델 | 공개 원문으로 확인한 규모 | 원래 입력과 목적 | 공개 권리 표시 | 현재 15×25 입력에 대한 판정 |
|---|---|---|---|---|
| BEATs | 논문 모델 90M | 16kHz waveform을 128-bin mel filterbank로 변환; 코드의 25ms frame/10ms shift. 일반 오디오 표현/분류 | Microsoft 공식 코드 MIT | 25개 deci-decade band×1분 grid는 원래 입력과 다름. 단순 resize를 사전학습 입력과 동등하다고 볼 수 없음 |
| EAT | 원 논문 88M. 저자 공개 base 카드 약90M, large 카드 약0.3B. large config: depth24, width1024 | 16kHz, 128mel, 25ms Hann/10ms shift. 공식 예시는 10초에 약1024 frames; latent/분류용 | 공식 repo·저자 base/large 카드 MIT | 시간·주파수 해상도와 대역 정의 차이가 큼. 입력 어댑터와 해당 회귀 문제의 재학습/검증 필요 |
| Dasheng | Base86M / 0.6B600M / 1.2B1200M | 16kHz audio. 공식1.2B 카드 예시는 64mel/약101frames per1s, embedding 출력. 약272k시간 일반 오디오 SSL | 저자 repo·Xiaomi 공식 카드 Apache-2.0 | 실재하는 B급 encoder이나 선박 추가 에너지 정답을 생성하도록 검증된 모델은 아님 |
| Whisper-large-v3 | 전체 encoder-decoder 약1.55B. **encoder만 약635M**(아래 계산) | 16kHz, 128mel, 10ms hop, 보통30초; ASR/번역 | OpenAI 원 repo MIT; OpenAI HF 카드 Apache-2.0 표시. 사용하는 배포본의 파일별 고지를 유지 | 선박 에너지 회귀용 모델이 아님. 전체1.55B 수치를 encoder-only branch 크기로 표시하면 과장 |

BEATs의 90M과 입력 사양은 [원 논문 §4.2](https://arxiv.org/html/2212.09058), [공식 BEATs.py](https://github.com/microsoft/unilm/blob/master/beats/BEATs.py), [공식 MIT LICENSE](https://github.com/microsoft/unilm/blob/master/LICENSE)에서 확인했다. EAT의 원 논문 수치와 최근 공개 checkpoint 크기는 서로 다른 버전이므로 구분했다. [EAT 논문 Table1·§4.1.1](https://arxiv.org/html/2401.03497), [저자 base 카드](https://huggingface.co/worstchan/EAT-base_epoch30_pretrain), [large 카드](https://huggingface.co/worstchan/EAT-large_epoch20_pretrain), [large config](https://huggingface.co/worstchan/EAT-large_epoch20_pretrain/blob/main/config.json), [공식 repo](https://github.com/cwx-worst-one/EAT).

Dasheng의 1.2B는 논문/저자 repo의 명목 크기다. HF 화면의 반올림된 `1B`와 혼동하지 않아야 한다. [저자 저장소 모델표·사용법](https://github.com/RicherMans/Dasheng), [Xiaomi 1.2B 모델 카드](https://huggingface.co/mispeech/dasheng-1.2B), [원 논문](https://arxiv.org/abs/2406.06992).

Whisper의 전체 규모·목적은 [OpenAI 공식 model card](https://github.com/openai/whisper/blob/main/model-card.md), 실제 입력은 [공식 audio.py](https://github.com/openai/whisper/blob/main/whisper/audio.py), large-v3 구조는 [OpenAI config](https://huggingface.co/openai/whisper-large-v3/blob/main/config.json)로 확인했다. encoder-only 크기 **635,048,960**은 다운로드/로드 실측이 아닌, [공식 AudioEncoder와 attention/MLP 정의](https://github.com/openai/whisper/blob/main/whisper/model.py)를 config의 width1280/layers32/n_mels128에 대입한 계산이다:

```text
32 × (12×1280² + 12×1280)
+ (128×1280×3 + 1280)       # conv1
+ (1280×1280×3 + 1280)     # conv2
+ 2×1280                   # final LayerNorm
= 635,048,960
```

공식 PyTorch 구현의 sinusoidal position은 parameter가 아닌 buffer다. 다른 구현이 고정 position table까지 parameter로 계수하면 수치가 조금 달라질 수 있다. 따라서 보고서에는 `encoder 약0.64B; 전체 약1.55B`로 표기하는 것이 적절하다.

## 2. 시계열 대형 모델의 실제 규모와 목적

| 모델/배포본 | 확인된 크기 | 공개 상태와 원래 task | 이 대회에서의 판단 |
|---|---|---|---|
| Time-MoE Base | 총113M / 활성50M | 공식 공개 link `TimeMoE-50M` | 이름의50M은 total이 아님 |
| Time-MoE Large | 총453M / 활성200M | 공식 공개 link `TimeMoE-200M` | 작은 forecasting feature encoder 후보로 별도 검증 가능 |
| Time-MoE Ultra | **총2.4B / 활성1.1B** | 논문 Table1 실험 모델. 확인 시점 공식 README 공개 링크는 base/large 두 개이며 Ultra download는 확인하지 못함 | `총2.4B/활성300M이고 바로 사용 가능`이라는 주장은 정정 필요 |
| Timer / Timer-XL 공개 quickstart가 가리키는 `timer-base-84m` | 84M(카드 집계84.1M), 8layers, patch96 | 최대context2880의 univariate zero-shot forecasting, Apache-2.0 | `Timer-XL 자체가 B급`이라는 주장은 이 공개본과 맞지 않음. 15분×25대역을 곧바로 energy-share로 내놓지 않음 |
| Time-LLM | 단일 고정 크기 모델이 아닌 reprogramming framework. 논문 default Llama-7B; 코드에 GPT-2/BERT 옵션 | frozen LLM + 학습 input/output layers; future forecasting. repo Apache-2.0와 backbone 이용조건은 별도 | 7B는 backbone 계열명. 실행 `llm_layers`에 따라 실제 로드 크기가 달라지므로 항상7B라고 할 수 없음 |

Time-MoE 수치는 [공식 논문 v4 Table1](https://arxiv.org/html/2409.16040v4)에서 확인했다. 공식 README는 context 최대4096의 autoregressive forecasting과 base/large 공개본을 안내하며 covariate·classification 지원은 TODO에 있다. [공식 repo](https://github.com/Time-MoE/Time-MoE). 따라서 아직 확인하지 못한 Ultra 공개 가중치를 이미 확보/통합한 것처럼 표시하면 안 된다.

Timer의 구체적인 공개 사양은 [THUML 모델 카드](https://huggingface.co/thuml/timer-base-84m), Timer-XL의 multivariate forecasting 구조와 Apache-2.0 표시는 [저자 repo](https://github.com/thuml/Timer-XL)에서 확인했다. 같은 계열의 구조 연구와 특정 공개 checkpoint 크기는 구분해야 한다.

Time-LLM은 [원 논문](https://arxiv.org/html/2310.01728)에서 forecast를 목표로 하고 default Llama-7B를 사용한다. [공식 README](https://github.com/KimMeen/Time-LLM)는 GPT-2/BERT 지원도 명시한다. [현재 모델 코드](https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py)는 `num_hidden_layers=configs.llm_layers`로 구조를 설정하고 backbone을 freeze한다. [현재 runner](https://github.com/KimMeen/Time-LLM/blob/main/run_main.py)의 default는6layers다. 실제 가동 전체/학습 파라미터 수는 선택 checkpoint·layer count·입출력 adapter를 포함해 별도 세야 한다. 원 코드의 forecast-only output을 energy-share 회귀 head와 관측 mask에 맞추는 작업도 필요하다.

## 3. 교사 모델·LoRA·MoE·프롬프트에 대한 사실과 가설

**확인:** BEATs/EAT/Dasheng은 오디오 표현/분류 연구이며, Whisper는 ASR 계열이고, 위 시계열 모델들은 forecasting 중심이다. 이 원문들에서 본 대회의 추가 에너지 비율 RMSE 개선은 입증하지 않았다. 각 연구의 benchmark SOTA라는 표현을 `이 대회 SOTA 확정`으로 옮길 수 없다.

**현재 데이터로부터의 판단:** 1분×25개 band power는 원래 waveform 또는 10ms×64/128mel과 정보가 다르다. 보간/resize로 shape를 맞추어도 사라진 위상·협대역 구조·분 단위 이하 시간 정보를 복원하지 못한다. 오디오 encoder를 실제 외부 WAV의 품질/유사도/표현 teacher로 쓰고 coarse-input student로 distillation하는 방안은 실험 가능한 **가설**이다. 다만 추론 때 없는 raw-waveform teacher를 필수 입력으로 두어서는 안 되고, 학습 fold 내부 원천만 teacher/검색 index에 사용해야 한다.

**에너지 정답의 의미:** 알고 있는 성분 `E=B+αR`에 대해서는 `y=Σ_O αR/Σ_O E`를 정확히 계산할 수 있다. 반면 일반 encoder의 ship class 확률, embedding 유사도, 언어 답변은 `αR`의 물리적 에너지 정답이 아니다. 실제 관측 `E`만 있고 자연 성분 분해가 없으면 teacher 출력은 약한 추정값이다. 이 구분은 공식 대회가 목표를 controlled addition으로 정의하고 자연 배경의 선박 소리와 구별하는 설명과 일치한다. [공식 task 설명](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge).

**LoRA:** frozen weight에 저랭크 학습 갱신을 넣어 학습 파라미터·메모리를 줄인다는 것은 [LoRA 원 논문](https://arxiv.org/abs/2106.09685)의 결과다. 하지만 이 데이터에서 과적합을 방지한다는 보장은 아니다. **MoE:** token별 일부 expert 활성화는 계산량·capacity 설계다. 이것 역시 자동 일반화 보장이 아니다. 해당 자료들에는 15step×25band/22selected reference components라는 현재 조건에서의 과적합 상한 증명이 없다. 모델이 커도 frozen representation+작은 head가 도움이 될 수 있고, 작아도 증강 패턴/원천 shortcut을 외울 수 있으므로 source-held-out 검증이 판단 기준이다.

**프롬프트/Audio RAG:** Time-LLM의 Prompt-as-Prefix는 존재한다. 그러나 현재 공개 query에는 station/time/AIS가 없으므로 실제로 주어지지 않은 정보를 prompt로 채울 수 없다. 공통 물리 task 설명·대역 grid·mask 같은 관측 가능한 정보와 학습 원천의 집계 retrieval은 후보지만, 성능 이득은 별도 ablation이 필요하다. [공식 입력 설명](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/data).

## 4. SA AIS 제공 여부

**제공하지 않음 확인.** 공식 data page는 SA acoustic summaries/metadata를 나열하고 AIS는 별도로 얻는 경우 clock alignment를 확인하라고 명시한다. 원래 오디오, verified vessel labels도 제공하지 않는다. 외부 AIS의 누락은 선박 부재 증명이 아니고 근접성은 가청성 증명이 아니다. Test에는 station/time/AIS tracks가 없다는 공식 명세이므로 SA AIS를 향후 확보하더라도 제출 query와 직접 join하는 경로는 없다. [공식 data page](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/data).

로컬 `data/raw/South_African_Real_Acoustics.zip`의 directory는10개 acoustic Parquet + stations.csv/recorders.csv/README/DATA_DICTIONARY/SHA ledger였다. AIS extract 파일은0개이며 ZIP 문서도 AIS 미포함을 명시한다. 파일 목록/문서만 확인했으며 acoustic table 값은 이번 감사에서 읽지 않았다. 공개 page의 언급과 실제 제공 파일이 일치한다.

## 5. VTUAD 307개를 전부 복구한다는 주장

**307=예약 holdout307이 아니다.** frozen 로컬 manifest를 read-only로 다시 비교했다.

| R2 reference218에서 빠진 VT525의 나머지 | 개수 | 현재 의미 |
|---|---:|---|
| excluded_hull_day_conflict | 260 | hull/day 격리 충돌로 제외 |
| excluded_cross_day_conflict | 1 | cross-day 충돌로 제외 |
| calibration | 6 | 기존 예약 calibration |
| domain_holdout | 15 | 기존 예약 domain assessment |
| original train24 + validation1, weak_class=background | 25 | positive passage 참조로 미편입 |
| 합계 | **307** | 서로 다른 제외 이유 |

따라서 `307 모두 고품질 positive로 복구하면 된다`는 결론은 성립하지 않는다. 공개 VTUAD 이용권과 현재 실험의 독립 평가 경계는 다른 문제다. 공개적으로 이용할 수 있더라도 기존 reserved를 학습에 넣으면 그것을 새 untouched 검증처럼 보고할 수 없다. conflict clip을 무조건 넣으면 selected-pool hull/day 분리도 다시 설계해야 한다. 전체525 graph가 연결된다는 로컬 결과를 무시하고 fake split을 만들 수 없다.

R4는 허용218개 안에서 metadata-only selection으로 **119clips/22components**를 실제 선택했다. 나머지99개를 모든 R4 역할에서 제외했고, 원 reserved/excluded는 재편입하지 않았다. 원천별 직접 hull/day/UTC/bytes 격리와 전체525 graph 연결이라는 한계를 구분해 이미 감사했다. [R4 selection](r4_reference_selection.md), [R3 coverage audit](R3_REFERENCE_COVERAGE_AUDIT.md). 22는 선택된 원천 pool의 조건부 구성 수이며 물리적 recording corpus22개가 아니다.

공개 provider도 AIS 자동 라벨은 인간 청취 라벨이 아니며 default recording split보다 hull split이 강한 격리라고 설명한다. 또한 v1.3 strict WAV가 AIS-qualified 구간이고, 일부 과거 container에는 시작부분을 반복하는300초 extension이 있어 strict view 사용을 권고한다. [VTUAD v1.3.2 공식 deposit](https://zenodo.org/records/23037895). 따라서 원천을 늘릴 때 strict provenance·직접 격리를 먼저 만족해야 한다.

로컬 검산 근거:

```text
data/processed/external_clip_manifest.parquet
SHA256 aa4affc7bd61afdea226bfec6f34d5b6c6e519cd23af55a463531bd8a67c624f
data/synthetic/group_cv_physics_v2_r1/fold_00/sources/reference_manifest.parquet
SHA256 65cad52303800b7ffca2813a8b57375de039b72e371b32a593529bfa5264d0f5
VTUAD525 anti-join R2 reference218 on clip_id =>307
```

## 6. NOAA 품질·DeepShip 권리에 대한 과한 단정

**NOAA 전체가 품질 불량이라는 근거는 없다.** NOAA SanctSound 공식 raw catalog는 standardized/calibrated equipment와 시각·시간/주파수 level QA를 명시한다. 동시에 사용자 task 적합성·완전성은 보장하지 않는다고 한다. raw/products의 공개 권리 표시는 CC0-1.0이다. [공식 raw80342, 특히 Data Quality/Access Information](https://www.fisheries.noaa.gov/inport/item/80342), [products80343](https://www.fisheries.noaa.gov/inport/item/80343).

현 프로젝트에서 NOAA를 R4 positive pool에 사용하지 않는 것은 설계/분포/원천 격리 결정이며 archive 전체의 불량 판정과 다르다. deployment별 calibration/sample-rate·Good interval·ship annotation·부분 crop의 checksum 한계는 개별 감사해야 한다. ship annotation도 자연 선박-only 에너지 정답을 뜻하지 않는다. [로컬 외부 확장 사실](dataset_document_external_facts.json).

**DeepShip의 자유 이용 또는 상업적 이용 금지 중 어느 쪽도 이번 근거만으로 확정하지 않는다.** 저자 repo는 일부 데이터와 전체본을 이메일로 요청하는 안내를 제공하지만 열람 시 별도 LICENSE가 보이지 않았다. 2024년 license 요청 issue도 공개되어 있고 명시적인 허가 답변을 확인하지 못했다. [저자 repo](https://github.com/irfankamboh/DeepShip), [license issue6](https://github.com/irfankamboh/DeepShip/issues/6). 공개 다운로드 가능·다른 논문에서 사용했다는 사실만으로 재배포/대회 이용 권한이 자동 확정되지는 않는다. 반대로 별도 확인 없이 `상업적 이용은 금지`라고 단정하는 것도 부정확하다. 현재처럼 명확한 provider 권리 증거를 우선하고, DeepShip은 permission/terms가 확보되기 전 채택 확정으로 표시하지 않는다.

## 7. R5 도식/계획에 반영할 정정

1. `1B teacher`는 **후보 오디오 표현/품질 teacher**라고 표시한다. 선박 에너지 GT 생성기로 표기하지 않는다.
2. 현재 배포 가능한 후보와 논문에만 확인된 규모를 구분한다. Time-MoE Ultra는 총2.4B/활성1.1B이며 공개 checkpoint 미확인 상태다.
3. 입력 경로를 `외부 WAV→native mel encoder→학습 fold teacher feature`와 `대회15×25 summaries→학생 회귀 모델`로 나눈다. resize가 raw 정보를 복원한다고 쓰지 않는다.
4. LoRA/MoE는 선택적 학습/계산 기법이다. overfitting 방지 확정 표시를 하지 않고 source-split 검증 관문을 둔다.
5. SA AIS는 `제공 없음; 별도 확보·clock 검증 시 연구용 weak evidence 후보`로 표시한다.
6. R4 119/22와 307 제외 사유를 보존한다. NOAA 불량·DeepShip 상업금지 같은 검증되지 않은 문구는 제거한다.

이 문서는 사실 검증과 설계 판단만 제공한다. 위 대형 모델의 실제 추가·학습 또는 R5 dataset 생성은 수행하지 않았다.

## 8. 후속 R5 실제 backbone adapter 구현에서 확인한 API와 한계

2026-10-25 후속 요청으로 [foundation_masks_r5.py](../pipeline/foundation_masks_r5.py)의 **객체 주입 adapter**를 구현하고 [CPU tiny fixture](R5_FOUNDATION_ADAPTER_FIXTURE.json)를 실행했다. 실제 B급 객체·가중치는 생성/다운로드하지 않았으며, 이 fixture는 pretrained 실행 성공이나 예측 성능의 증거가 아니다. 위의 원문 검증 결과와 구분한다.

- Time-MoE 공식 `TimeMoeModel.forward`는 `inputs_embeds`, `use_cache`, `return_dict`를 받는다. bare decoder 또는 `TimeMoeForPrediction.model`을 주입하고 새200→D stem에서 15개 time token을 만든 뒤 decoder의 `last_hidden_state`를25개 band mask로 변환한다. 공식 `layers[i].self_attn.q_proj/v_proj`에 LoRA16을 넣고, 선택적으로 `ffn_layer.gate` router에도 LoRA를 넣는다. 전문가 FFN과 기존 norm/embedding은 freeze한다. 원래 causal attention을 보존한다. [공식 modeling_time_moe.py](https://github.com/Time-MoE/Time-MoE/blob/main/time_moe/models/modeling_time_moe.py).
- Dasheng HF 공식 class는 `DashengModel.encoder = AudioTransformerMAE_Encoder`이며1.2B config는 **D1536/depth40/heads24**다. `encoder.blocks`와 `encoder.norm`, 각 block의 fused `attn.qkv`가 확인된다. 원래 mel patch·pos를 우회하고8→D Conv3×3, 학습2D time15/band25 position, 375tokens, 공식blocks/norm, scalar cell head로 구성했다. fusedqkv에 LoRA8을 적용한다. 기존 mel position을 새 grid에서 같은 물리 의미라고 재사용하지 않는다. [공식 HF 모델 소스](https://huggingface.co/mispeech/dasheng-1.2B/blob/main/modeling_dasheng.py), [config](https://huggingface.co/mispeech/dasheng-1.2B/blob/main/config.json).
- 공식 저자 loader는 `dasheng_12B()`에서1.2B를 로드한다. URL의 `12b`는 이 함수의 명명이며12B 모델을 뜻하지 않는다. 확인한 weight URL은 [저자 Zenodo dasheng_12b.pt](https://zenodo.org/records/11511780/files/dasheng_12b.pt?download=1)와 [공식 HF model.safetensors](https://huggingface.co/mispeech/dasheng-1.2B/blob/main/model.safetensors)다. HF 파일 표시 크기는4.54GB다. URL 존재만 확인했고 실제 내용·전체SHA는 이번에 검증하지 않았다. [저자 loader 코드](https://github.com/RicherMans/Dasheng/blob/main/dasheng/pretrained/pretrained.py).
- Ultra의 공개 pretrained artifact가 확인되지 않아 **production 생성 gate는 닫혀 있다**.453M 대체나 random Ultra를 pretrained로 표시하는 fallback은 없다. Dasheng 실제 객체는 caller가 검증한 source revision/checkpoint SHA 증거와 공식 구조가 필요하다. 객체에 주입된 weight가 그 artifact와 동일한지 검증하는 실제 loading/smoke 단계는 남아 있다.
- 두 branch 모두 alpha=2r/dropout.1의 self-contained LoRA를 사용한다. 원 energy weights는 detach하여 변경/학습하지 않고 `share=Σ(wq)`로 집계한다. 새 stem/head/position과 LoRA만 학습하며 presence는 별도 logit head다. Adapter 저장에는 frozen vendor weight를 넣지 않는다. CPU에서는 작은 같은-API protocol 객체로 gradient·원 weight 불변·물리 pooling·adapter 저장/재로드를 검증했다. GPU memory, 실제 vendor runtime 호환성, source-held-out 성능은 검증하지 않았다.

**CLAP 7B 표현도 정정한다.** LAION-CLAP은 audio/text dual encoder 계열이며 공식 공개 구성은 HTSAT audio encoder와 text encoder/projection이다. `7B CLAP`을 이 프로젝트의 확인된 모델 크기로 쓸 근거가 없다. 구체 checkpoint를 정한 뒤 실제 total/audio-only parameter 수를 따로 세어야 한다. Qwen 같은7B audio-language model과 CLAP을 같은 모델로 묶지 않는다. 이번 확정 구성에서 Qwen/CLAP은 별도 **offline 품질/메타정보 후보**이며 제출 추론에 요구하지 않는다. [LAION 공식 CLAP repo](https://github.com/LAION-AI/CLAP), [HTSAT-base config](https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/clap_module/model_configs/HTSAT-base.json), [공식 audio/text model 코드](https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/clap_module/model.py).

Whisper encoder 약635M과 Time-MoE Ultra 활성1.1B 정정은 각각 §1, §2의 원문·계산 근거를 유지한다. 이들을 전체1.55B 또는 활성300M으로 되돌려 표기하지 않는다.

## 9. 최신 상태: 공개 Time-MoE와 Dasheng 실제 CPU 연결 검증 완료

후속 사용자 승인으로 공개 두 checkpoint를 `models/pretrained_r5/`에 받아 **한 모델씩 별도 CPU process**에서 실제 공식 객체를 로드했다. Ultra2.4B 실행은 제외했고 기존 Ultra gate는 계속 닫혀 있다. `TimeMoEPublicMask`는 이름·manifest·shape를 분리하여 공개200M-active/453M-total 모델을 Ultra라고 부르지 않는다. Whisper는 다운로드/로드하지 않았다.

| 검증 항목 | 공개 TimeMoE-200M | Dasheng-1.2B |
|---|---:|---:|
| 공식 HF revision | `794591bfeb1225fdf742cec0f4c71f20c3f3b87e` | `4ae6b990f87bf40579b97bab065f4e47920b28dc` |
| 원 checkpoint bytes | 906,450,104 | 4,536,241,360 |
| safetensors 전체 tensor elements | 453,196,800 | 1,134,047,489 |
| native dtype | BF16 | FP32 |
| 실제 grid wrapper 전체 parameters | 453,891,866 | 1,136,190,082 |
| 학습 parameters(stem/head/newpos/LoRA) | 775,706 | 2,142,722 |
| 그 중 LoRA parameters | 589,824 | 1,966,080 |
| nonzero·finite LoRA-B gradient | 24/24 projection | 40/40 projection |
| frozen base parameter bytes 전후 일치 | 통과 | 통과 |
| 원 energy-weighted mask 집계 오차 | 0 | 0 |
| adapter 저장·같은 검증 backbone 복구 | bitwise 일치 | bitwise 일치 |
| 실제 CPU 검증 total(해당 process 기록) | 2.76초 | 12.13초 |
| peak CPU RSS | 약3.06GB | 약6.42GB |

Tensor elements에는 persistent buffer가 포함될 수 있어 parameter 수와 구분했다. Dasheng1.2B는 공개 모델의 명목 이름이며, 실제 수치가 정확히1,200,000,000이라고 주장하지 않는다. Time-MoE wrapper는 원 forecasting head 대신 새 grid head를 사용하므로 checkpoint tensor 수와 wrapper parameter 수가 다르다.

두 weight의 **전체 SHA-256이 provider LFS SHA와 일치**했다:

```text
TimeMoE-200M: 28d1cbf480c1f63d4ad56d8b7649d9860b9afa9f31abeedff9decea48d9aee57
Dasheng-1.2B: b3b9316fe0ea051eefafff51ee2341629914b8db5f78cbe991803dc86f5669df
```

소스·config를 revision과 SHA로 pin하고 공식 Apache-2.0 LICENSE도 저자 repo commit에 고정했다. [로더](../pipeline/load_foundation_r5.py)는 기본 local-only이며 명시적 `--download`에서만 공개 익명 urllib GET을 사용한다. Hugging Face 인증 API와 `from_pretrained`를 호출하지 않으므로 implicit token lookup이 없고, 실제 CPU load/forward 단계에는 offline 환경과 socket 차단 audit hook도 적용했다. 공식 원본 소스를 수정하지 않고 meta 객체에 검증 safetensors를 strict assign했다. Time-MoE의 비영속 RoPE cache만 공식 constructor로 CPU에서 재생성했다.

실제8×15×25 synthetic window 한 개에 대한 forward와 adapter SGD1step을 검증했다. 모든 trainable gradient는 finite였고 fixed base/expert에는 gradient가 없었다. frozen parameter 전체 바이트 SHA가 step 전후 동일했고, 원 energy weights도 변하지 않았다. 이 검증은 **실제 pretrained backbone blocks와 새 Stem/LoRA 연결이 작동한다는 증거**다. raw waveform encoder forward, 두 번째 새 B급 객체에 대한 adapter 재로드, GPU 메모리/속도, Fold0 RMSE 또는 일반화 성능은 아직 측정하지 않았다. 채택은 실제 Fold0 학습·source validation 결과를 보고 결정해야 한다.

- [Time-MoE 실제 CPU receipt](R5_ACTUAL_TIMEMOE_PUBLIC_CPU_R1.json)
- [Dasheng 실제 CPU receipt](R5_ACTUAL_DASHENG_CPU_R1.json)
- [로컬 weights/runtime 안내](../models/pretrained_r5/README_CPU_VERIFICATION.md)

로컬 runtime은 macOS ARM64용 Transformers4.40.1과 호환 dependency를 프로젝트 안에만 설치한 것이다. Linux GPU 서버에는 `runtime_requirements.txt`로 해당 OS의 wheel을 새로 설치해야 하며 이 macOS binary directory를 그대로 import해서는 안 된다. 위 peak RSS는 CPU 실측이며 GPU VRAM 추정치가 아니다.
