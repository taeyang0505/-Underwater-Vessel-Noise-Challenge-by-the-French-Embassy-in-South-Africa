# 현재 모델 후보·크기·출처와 남은 작업

기준: 2026-10-25. **아래는 R5 후보 구성입니다. 아직 새 데이터로 학습한 최종 앙상블이 아닙니다.** 최종 포함 여부와 가중치는 source-separated INNER 검증 후 동결합니다.

## 현재 설계한 5개 후보

| 후보 | 실제 구현 크기 | 학습하는 크기 | 역할·확인 상태 | 원 논문 / 공개 구현 |
|---|---:|---:|---|---|
| 공개 **TimeMoE-200M + Grid Stem/LoRA** | 공식 checkpoint 453,196,800개; 사용 branch 약 **453.892M** | **775,706개 ≈0.776M**, LoRA 589,824개 포함 | 15개 시간 token, 각 token에 25대역의 8채널 입력. 공개 가중치 실제 CPU 로드·forward/backward·frozen base 불변·adapter 재로드 통과 | [Time-MoE 논문](https://arxiv.org/abs/2409.16040), [공식 코드](https://github.com/Time-MoE/Time-MoE), [공개 checkpoint](https://huggingface.co/Maple728/TimeMoE-200M) |
| **Dasheng-1.2B + Grid Stem/LoRA** | 명목 **1.2B**; 공식 checkpoint tensor elements **1,134,047,489개**, 사용 branch **1,136,190,082개 ≈1.136B** | **2,142,722개 ≈2.143M** | D1536·40 blocks·24 heads. 15×25 coarse grid를 새 stem으로 연결. 실제 공개 가중치 CPU 로드·LoRA gradient·물리 pooling·frozen base 불변·재로드 통과 | [Dasheng 논문](https://arxiv.org/abs/2406.06992), [저자 코드](https://github.com/RicherMans/Dasheng), [공식 checkpoint](https://huggingface.co/mispeech/dasheng-1.2B) |
| **TFGrid + FDY + Conformer Mask** | **2,562,150개 ≈2.562M** | 전체 **2.562M** | Grid 5 blocks, D64·H128, FDY 4 bases, Conformer 1 block, dropout 0.1. 대회 입력에 맞춘 새 조합 | [TF-GridNet](https://arxiv.org/abs/2211.12433), [Frequency Dynamic Convolution](https://www.isca-archive.org/interspeech_2022/nam22_interspeech.html), [Conformer](https://arxiv.org/abs/2005.08100) |
| **Wide BG Mask CNN** | **1,488,002개 ≈1.488M** | 전체 **1.488M** | 폭128·5 residual blocks·공통8채널·dropout0.1. q0는 입력 특징만 사용하고 출력 강제 logit 결합·잔차 억제 페널티 없음 | [자체 구현](pipeline/hybrid_models_r5.py). 특정 논문의 공식 모델이나 대회 SOTA로 표시하지 않음 |
| **Physics / Excess Retrieval LightGBM** | 신경망 파라미터 M/B와 직접 비교 불가 | 부스팅 tree·leaf를 학습 | 물리 특징·TRAIN에서만 만든 초과 에너지/고순도 prototype. 실제 tree 수는 INNER early stopping 이후 기록 | [LightGBM 원 논문 · NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), [공식 코드](https://github.com/lightgbm-org/LightGBM) |

**TimeMoE 이름의 200M은 총 크기가 아니라 명목 활성 크기입니다.** 현재 사용한 것은 공개 Large checkpoint이며 Ultra 2.4B가 아닙니다. Ultra 공개 가중치 미확인 gate는 닫혀 있습니다.

checkpoint tensor element 수는 파일 안의 전체 tensor 합계입니다. wrapper 파라미터 수와는 원래 출력 head/unused embedding 제외, 새 adapter 추가 등의 이유로 다릅니다. 특히 Dasheng의 명목 1.2B와 실제 합계의 차이를 구분해 기록했습니다.

Dasheng 검증은 실제 공식 pretrained blocks를 새 coarse-grid stem과 연결한 결과입니다. 원래 오디오 mel 입력과 coarse 1분×25대역 입력의 의미가 같다고 증명한 것이 아니며, CPU 1개 숫자 fixture 성공은 대회 RMSE 개선의 증거도 아닙니다.

## 비교군과 최종 선정

기존 경량 **CNN, FDY-CRNN, iTransformer, LightGBM**과 동일 source split·동일 생성 정책에서 비교합니다. 기존 6채널 FDY/iTransformer 구현은 각각 129,390개 / 292,816개였습니다. 공통8채널로 재구성한 새 비교군의 최종 수는 학습 manifest에 다시 기록해야 합니다.

1. 모든 neural branch는 동일한 TRAIN-fitted 8채널 변환과 관측 에너지 가중치로 입력을 받습니다. 정답·recipe provenance는 query 입력으로 쓰지 않습니다.
2. 모든 neural branch는 `q_hat[15,25]`를 sigmoid로 예측하고 `share_hat = sum(w * q_hat)`로 집계합니다.
3. 기본 앙상블은 개별 share의 가중 평균입니다. 확률 마스크의 선형 평균 후 pooling은 이것과 동치입니다.
4. logit-mask 평균 후 pooling은 별도 후보입니다. 오직 INNER에서 실제 이득이 있을 때 선택합니다.
5. Dasheng·TimeMoE 등 큰 백본은 우선 Fold0 INNER에서 경량 비교군보다 개선되는지 확인합니다. 공개 가중치 로드에 성공했다는 이유로 최종 가중치를 강제 배정하지 않습니다.
6. OUTER OOF는 동결 정책 평가에만 사용합니다. OUTER/Test를 보고 ensemble·학습 설정을 다시 맞추지 않습니다.

Kumo Large, TabICLv2, TabM은 현재 최종 후보에서 제외했습니다. Whisper-large-v3 encoder는 Dasheng 연결 실패 시의 대안이었으며, 지금 별도 활성 branch로 추가하지 않았습니다. Qwen/CLAP은 최종 추론 앙상블에 포함하지 않고, 그 점수로 `share`/`cell_share` 정답을 바꾸지 않습니다.

## 이미 완료한 것

- 확정 VTUAD 원천: **119클립 / 22개 선택 그룹**, 측정 약 **8.63시간**.
- R4 동일 레시피 5-fold 데이터: TRAIN 역할 합계 **267,837**, INNER **15,000**, OUTER **15,000**.
- 전체 **297,837건** 생성·전수 검증, 정규 NPZ15개 **414.2MiB**.
- 공개 TimeMoE·Dasheng 파일 합계 약 **5.44GB** 확보, 공식 revision/LFS SHA·license 고지 보존.
- 실제 두 백본의 coarse-grid adapter CPU 1-window·1-step 검증. **실제 학습 데이터나 Test를 사용한 본학습이 아닙니다.**

실제 검증 근거는 [TimeMoE CPU R1](reports/R5_ACTUAL_TIMEMOE_PUBLIC_CPU_R1.json), [Dasheng CPU R1](reports/R5_ACTUAL_DASHENG_CPU_R1.json), [R4 데이터 전수 검증](reports/R4_DATA_FINAL_VERIFICATION.json)에 있습니다.

## 남은 시간 예상

| 범위 | 상태 / 예상 |
|---|---|
| 현재 확정 범위 데이터 확보·생성·전수 검증 | **완료**. 150~220시간 외부 확보를 기다리지 않음 |
| 공개 가중치 다운로드·CPU 연결 검증 | **완료**. 별도 GPU 학습은 아직 아님 |
| R5 전체 학습·추론 runner, 공통 입력/동적 합성, checkpoint 복원·앙상블 gate 통합 검증 | **CPU 통합·재현 검증 완료**. 실제 GPU 검증과 Fold 본학습은 별도 대기 |
| 새 파이프라인 최종 도식 | [새 R5 도식](figures/pipeline_r5_candidate_v1.png) 생성 완료. R2 그림과 구분 |
| GPU Fold0 및 5-fold 본학습 | **새 구조 미실측**. 접속 가능한 GPU에서 실제 step·epoch 처리량과 VRAM을 측정한 뒤 시간 산정 |

Neural 본학습 기본은 사용자 지시대로 **Adam, 최대100 epochs, early stopping patience7**입니다. CPU adapter fixture의 단일 업데이트를 충분한 본학습이나 수렴으로 세지 않습니다. 글로벌 최소값 도달·과적합 없음·리더보드 개선을 사전에 보장하지 않습니다.


최신 CPU 통합 상태와 실행 계약은 [R5 통합 준비 보고서](reports/R5_INTEGRATION_READINESS.md)를 확인하세요. 모든 실제 GPU epoch/OOF/Test 추론은 아직 미실행입니다.
