# R9 Fold0 경계 정지·독립 handoff

**PASS: Fold0의 네 신경망 학습 완료와 정지를 확인하고 네 모델 모두 Mac에 보존했다. 전체5fold 및 tree·ensemble·report 단계의 완료를 뜻하지 않는다.**

실제 guard 정지 시각은 2026-10-26 20:41:49 KST다. driver2257/start86168117과 supervisor2256/start86168115가 T상태이고 뒤 Fold1–4의 model 디렉토리가 없는 것을 독립 확인했다. 이 백업 작업은 신호를 보내거나 원본 코드·데이터·설정·원장을 수정하지 않았다.

| 모델 | Best window epoch | Selection RMSE | Last100 RMSE |
|---|---:|---:|---:|
| bg_cnn | 5 | 0.0903609320 | 0.0999112709 |
| tfgrid_fdy_conformer | 17 | 0.0794150793 | 0.0835195457 |
| fdy | 16 | 0.0810336636 | 0.0899701027 |
| itransformer | 16 | 0.0965233053 | 0.1016631214 |

각100epoch·61,700회 Adam 업데이트를 완료했다. 모델마다 best-window·best-macro·last 및10epoch 간격10개(총13개) 체크포인트를 보존했고, Mac CPU에서 기본전체모델 strict restore/finite weights와 저장된1,000개 selection 예측의RMSE를 검산했다. 이는 개발용 checkpoint 선택 지표이며 공식 성능이나 독립 test 성능이 아니다.

이번 마지막 백업은 iTransformer와 최종 metadata만 포함한다. 기존 CNN·TFGrid·FDY 가중치는 각 기존 백업을 그대로 참조한다. 마지막 archive는14,523,929bytes,62개 entry이며 SHA `2ab4217a2ee952df54352b306a61f0bbb2799f46ac922de8df1c714f6dba5485`다. 원래 execution/context, 원장 snapshot, guard/실제STOP증거, all4 summary, Fold1–4의 기존 context, 환경·배포·최종stdout·고정config·bundle manifest/readiness를 함께 보존했다.

**원래 원장에는 `train/itransformer`가 없고, Fold0 `neural_completed.json`과 root `neural_stage_completed.json`도 없다.** summary가 저장된 직후 외부 guard가 멈췄기 때문이다. 이 파일들을 새로 만들어 원래 runner의 완료 산출물로 주장하지 않았다. 대신100epoch/history/체크포인트/summary artifact/source closure를 독립 검증한 producer receipt로 완료 모델 보존을 인증한다. 원래 RUNNING.lock·run_state는 정지 당시 snapshot이며 새 실행의 live lock으로 복원하면 안 된다.

새 GPU의 Fold1–4는 fresh run으로 학습해야 한다. 현재 runner에는 완료 NN import 기능이 없으므로, 보존한 Fold0와 합치려면 원본 SHA·동일 config/source/seed/selection binding을 검증하는 새 버전의 fold import/결합 manifest가 필요하다. 원래 실행을 resume하거나 원장을 임의 보완하는 작업은 이 결과에 포함되지 않는다. 모든 최종5fold 모델·정책의 전역 freeze 이전에는 report를 열지 않는다.

[네 모델의 정확한 경로·해시·결합 binding](R9_FOLD0_COMPLETED_HANDOFF_MANIFEST.json) · [마지막 모델·최종metadata의 전송/복원 영수증](R9_FOLD0_BOUNDARY_HANDOFF_BACKUP.json) · [guard 상태 원본 snapshot](../ops/server_backups/r9_gpu_r1/fold_00_itransformer_boundary_handoff_r1/restored/handoff_metadata/guard_status.json)
