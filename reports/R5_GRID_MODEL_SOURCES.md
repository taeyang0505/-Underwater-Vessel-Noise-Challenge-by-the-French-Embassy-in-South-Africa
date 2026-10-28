# R5 grid 모델: 원문 검증과 실제 구현 범위

검증일: 2026-10-25. 새 구현은 [grid_models_r5.py](../pipeline/grid_models_r5.py), CPU 검증은 [R5_GRID_MODELS_FIXTURE.json](R5_GRID_MODELS_FIXTURE.json)이다. 본학습·GPU 실행·vendor weights 다운로드는 하지 않았다. 이 문서는 음악/음성 분리 논문의 결과를 현재 대회 SOTA로 옮겨 주장하지 않는다.

## 원논문 크기와 첨부 주장 검증

| 모델 | 원논문에서 확인한 입력과 규모 | 이번 코드와의 관계 |
|---|---|---|
| TF-GridNet | waveform을 복소 STFT로 변환한다. TASLP 논문 Tables II–III의 소형 설정은 2.6–3.6M이며, Table IV의 대표 최우수 설정은 14.5M이다. | 1.5–4M을 전체 원모델의 고정 크기로 쓰지 않는다. 실제 R5 adaptation은 **2,774,914** parameters다. [원논문](https://arxiv.org/html/2211.12433) |
| BSRNN | 원 음악 논문은 복소 STFT 입력, N=128, 12개 시간/대역 모듈과 총 24개 residual BLSTM, H=256을 명시한다. 원문에서 총 parameter 숫자는 확인하지 못했다. | 첨부의 1–3M을 원논문 공식 수치로 확인할 근거가 없다. 실제 R5 adaptation은 **2,762,691** parameters다. [원논문 §II·IV-A4](https://arxiv.org/html/2209.15174) |
| BS-RoFormer | 8초 stereo waveform을 STFT/62개 대역으로 투영한다. L=6의 작은 버전도 72.2M, L=12는 93.4M이다. | 이번 파일에는 구현하지 않았다. 첨부의 1–3M을 원 BS-RoFormer 크기로 표기하면 부정확하다. [원논문 §4.4·4.6](https://arxiv.org/html/2309.02612v2) |

위 수치는 서로 다른 구성·입력·과제의 숫자다. 실제 R5 수는 `sum(p.numel() for p in model.parameters())`로 기본 factory 모델을 생성해 직접 계산했다. 작거나 큰 모델의 우수성을 이 숫자만으로 판정하지 않는다.

## 공식 구현과 라이선스의 정확한 범위

TF-GridNet 논문은 저자들의 ESPnet 공개 [PR #5395](https://github.com/espnet/espnet/pull/5395)를 직접 연결한다. [현재 separator](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tfgridnet_separator.py)에는 waveform/STFT encoder와 complex-domain 출력이 포함되어 있다. ESPnet 저장소의 코드는 [Apache-2.0](https://github.com/espnet/espnet/blob/master/LICENSE)이다.

원 음악 BSRNN 논문에서 저자 공식 GitHub/weights 및 그 weights의 라이선스는 확인하지 못했다. [ESPnet BSRNN separator](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/bsrnn_separator.py)는 후속 음성 향상 BSRNN 논문을 인용하는 구현이며, 원 음악 BSRNN과 동일한 배포물로 소개하지 않는다. ByteDance의 `music_source_separation` 저장소는 [README](https://github.com/bytedance/music_source_separation)의 실제 모델이 ResUNet143/MobileNet이므로 원 BSRNN 공식 저장소로 연결하지 않는다.

널리 사용되는 [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)는 Phil Wang의 재구현이다. 이 코드의 [MIT 라이선스](https://github.com/lucidrains/BS-RoFormer/blob/main/LICENSE)는 확인했지만, 이를 논문 저자 공식 배포 또는 별도 checkpoint 사용 허가로 확대하지 않는다.

R5 파일은 논문의 공개 연산 구성을 참고해 직접 작성했으며 위 저장소 코드를 복사하거나 import하지 않는다. 표준 `torch`와 Python 표준 라이브러리만 사용한다. 따라서 알려지지 않은 vendor checkpoint나 그 사용 조건에 의존하지 않는다.

## 입력·출력에서 달라진 부분

현재 입력은 복소 STFT가 아닌 `[B,8,15,25]`의 실수 특징이다. 15개 시간 위치는 분 단위이고, 25개 위치는 이미 집계된 deci-decade 대역이다. 위상·세밀한 FFT bin·waveform은 입력에서 복원할 수 없으므로 공식 pretrained separator를 그대로 적용하거나 iSTFT/음성 SI-SDR를 계산하는 구현으로 표기하지 않는다.

공통 출력은 `(share, presence_logits, cell_mask)`다. `cell_mask`는 `[B,15,25]`의 직접 sigmoid 에너지 분율이며, share는 `sum(energy_weights * cell_mask)`다. caller의 weights는 관측된 에너지에 대해 정규화되어 있어야 한다. 모듈은 이를 새로 fit하거나 재정규화하지 않는다. prior 값을 출력 logit에 강제로 더하지 않고 residual penalty도 포함하지 않는다. 8개 채널의 생성과 fold TRAIN-only prior binding은 외부 파이프라인의 책임이다.

### TFGridEnergyMask

기본 설정은 E=64, BLSTM H=128, 6 blocks, dropout=0.1, attention 4 heads다. 8→E stem과 1-channel mask/presence heads는 대회용 변경이다. 주파수 unfolding은 논문 소형 실험의 I=J=1 경우를 사용한다.

각 block에는 다음 세 경로가 모두 실제로 존재한다.

1. 각 시간 위치에서 25개 대역을 처리하는 full-band frequency BLSTM.
2. 각 대역에서 15개 시간 위치를 처리하는 shared sub-band temporal BLSTM.
3. 각 frame의 25개 대역을 모두 Q/K/V에 반영하는 full-band cross-frame attention. attention matrix는 `[B,H,15,15]`이며 주파수 평균으로 대체하지 않는다. Q/K의 head당 차원은 `25 × ceil(512/25) = 525`다.

정규화와 residual dropout은 현재 작은 실수 energy grid에 맞춘 명시적 adaptation이다. 원 논문 전체 waveform 모델의 재현이나 동일한 분리 성능을 주장하지 않는다.

### BandSplitEnergyMask

기본 설정은 E=128, BLSTM H=128, 4 blocks, dropout=0.1이다. 그룹별 LayerNorm/linear, shared temporal BLSTM와 ordered band-axis BLSTM의 교대 residual 경로, 그룹별 LN→MLP→tanh→GLU decoder를 사용한다. GRU로 대체하지 않았다. 각 decoder가 5개 서로 다른 logits를 생성하므로 최종 주파수 해상도는 원래의 25개 대역이다.

| 그룹 | 0-based 반개구간 | 입력 헤더 중심 주파수(Hz) | 정확한 base-10 대역 support(Hz, 근사 표기) |
|---|---|---|---|
| 0 | `[0,5)` | 31.6, 39.8, 50.1, 63.1, 79.4 | 28.1838–89.1251 |
| 1 | `[5,10)` | 100, 125.9, 158.5, 199.5, 251.2 | 89.1251–281.8383 |
| 2 | `[10,15)` | 316.2, 398.1, 501.2, 631, 794.3 | 281.8383–891.2509 |
| 3 | `[15,20)` | 1000, 1258.9, 1584.9, 1995.3, 2511.9 | 891.2509–2818.3829 |
| 4 | `[20,25)` | 3162.3, 3981.1, 5011.9, 6309.6, 7943.3 | 2818.3829–8912.5094 |

정확한 중심은 `10**(1.5+0.1*j)`, 각 대역 경계는 중심 지수의 ±0.05로 정의한다. 그룹은 선형 Hz에서 같은 폭이 아니며 half-decade support다. 전체 float 값과 입력 헤더의 반올림 값은 fixture JSON에 함께 기록했다.

`encode`는 그룹 embedding을 해당 5개 band 위치에 broadcast하여 `[B,E,15,25]` 인터페이스를 제공한다. 이 broadcast를 새로운 대역별 잠재 관측이나 주파수 보간으로 해석하지 않는다. forward에서는 그룹별 5-output decoder가 원래 25-band mask를 직접 만든다.

## 통합 API와 CPU 검증

두 class는 `encode(x)`와 global `representation(x)`를 제공한다. factory는 `make_grid_model('tfgrid_mask'|'bandsplit_mask', **kwargs)`다. 저장 가능한 실제 constructor 설정은 `model.grid_config`, kind는 `model.grid_kind`에 있다. 학습 objective·SSL projector·prior fit·checkpoint source binding은 부모 파이프라인에서 관리한다.

실제 기본 크기 두 모델을 Torch 2.14.1 CPU에서 생성한 숫자 fixture로 검증했다. shape, 모든 parameter의 유한 gradient, 세 TF-grid 경로 및 두 band-split BLSTM 경로의 비영 gradient, sigmoid 경계, 정확한 에너지 pooling, query batch 독립성, state/config 저장·복원의 bitwise 일치를 통과했다. 단독/배치 최대 차이는 TF-grid `5.96e-8`, band-split `1.49e-8`이며 다른 batch sample을 바꿨을 때 query 변화는 둘 다 0이었다.

Torch 2.8에서 사용할 수 있는 표준 연산으로 작성했지만 현재 실제 실행 검증은 2.14.1 CPU에서만 수행했다. 사용자 지시에 따라 원격 Torch 2.8/GPU 추가 probe는 실행하지 않았다. fixture의 backward는 임의 숫자의 수치 검사이며 실제 데이터 학습 또는 모델 성능 측정이 아니다.

재현 명령:

```bash
python -m pipeline.grid_models_r5 --self-test --report reports/R5_GRID_MODELS_FIXTURE.json
```
