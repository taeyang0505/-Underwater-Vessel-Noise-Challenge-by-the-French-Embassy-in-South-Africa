# R9 Fold0 TFGrid·FDY 완료 백업

**PASS.** 새로 완료된 TFGrid와FDY만 독립 gateway로 Mac에 회수했다. 기존 CNN 백업과 활성 iTransformer는 변경하지 않았다. 각100epoch·61,700업데이트, best-window/best-macro/last 및10epoch 간격10개(총13개) 체크포인트, 학습로그/summary/selection예측/원장snapshot/동결코드를 보존했다. archive 및42개 entry SHA, 로컬 기본전체모델 strict restore/finite weights, 저장된selection예측의RMSE 재계산을 확인했다. 새 학습·추론·Test/report 평가는 없다.

- tfgrid_fdy_conformer: best-window epoch17, RMSE 0.0794150793; last100 0.0835195457.
- fdy: best-window epoch16, RMSE 0.0810336636; last100 0.0899701027.

이것은 두 완료 모델의 보존 결과이며 전체5fold 완료 또는 공식 성능을 뜻하지 않는다.

[수치·파일 경로·원장·SHA](R9_FOLD0_ADDITIONAL_MODELS_BACKUP.json)
