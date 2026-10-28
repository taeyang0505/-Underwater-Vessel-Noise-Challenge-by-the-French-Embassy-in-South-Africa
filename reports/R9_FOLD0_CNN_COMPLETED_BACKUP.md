# R9 Fold0 CNN 완료·백업 독립검증

**PASS.** 100epoch·61,700회 Adam 업데이트를 완료한 CNN만 백업했다. 활성 TFGrid·데이터·설정·프로세스는 변경하지 않았다.

- Best window: epoch5, selection RMSE **0.0903609320**. 저장된1,000개 selection 예측으로 재계산해 일치했다.
- Best site-macro: epoch36, **0.0595335072**.
- Last: epoch100, window RMSE **0.0999112709**. 마지막 epoch를 best로 대체하지 않았다.

Best-window·best-macro·last 및10epoch 단위10개, 총13개 체크포인트와 history/summary/started/progress/selection 예측을 보존했다. 원장·원본 source context·동결 코드도 함께 저장했다. 서버 archive86,051,667bytes SHA `85225927b66ec7fa5b6217272bc4446ab1658a2a6e72b38ae9679317715c5784`를 gateway로 Mac에 복사하고42개 tar entry 전체 SHA를 대조했다. 로컬 CPU에서도13개 체크포인트를 weights_only로 읽어 기본전체1,487,873개 파라미터 모델에 strict restore하고 모두 finite임을 확인했다. 최종 optimizer61700step과100epoch 동안 같은 validation input hash도 확인했다. 새 추론·학습·Test/report 평가는 하지 않았다.

이것은 전체5fold 완료가 아닌 완료된 Fold0 CNN 한 모델의 보존 영수증이다. selection 지표는 개발·checkpoint 선택 결과이며 공식 성능을 뜻하지 않는다. 원장도 당시 완료 stage의 snapshot이다.

[전송·entry SHA 검증](R9_FOLD0_CNN_COMPLETED_BACKUP.json) · [로컬13체크포인트 복원 검증](R9_FOLD0_CNN_LOCAL_RESTORE_VERIFICATION.json) · [백업 원본 archive](../ops/server_backups/r9_gpu_r1/fold_00_bg_cnn_completed_r1/r9_fold00_bg_cnn_completed_r1.tar.gz)
