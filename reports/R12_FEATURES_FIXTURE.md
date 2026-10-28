# R12 특징 수학·물리 검증

**PASS — 숫자 fixture만 실행했습니다. 실제 음향 데이터 fit·모델 학습·추론·Test/validation 읽기는 0회입니다.**

기존 relative8과 rounded physical w는 bitwise 동일합니다. 옵션 shape는 signed PCA 잔차/10, positive 잔차/10, `(2db[f]−db[f−1]−db[f+1])/10` 곡률의 3채널을 추가하여11채널입니다. PCA는 TRAIN 순수SA A=0의 완전 관측 safe0..22 minute spectra만 사용하고, query는 저장 basis로 관측 band만 projection합니다. 누락 분을 채우지 않습니다. 추가 residual/curvature는 band23/24의 값에 영향을 받지 않았습니다. **기존8채널과 full25-band w는 그대로이므로 그 값까지 upper-band 독립이라는 뜻은 아닙니다.**

TRAIN source IDs는 중복 불가이며 input db·IDs·binding SHA가 기록됩니다. 숫자만으로 실제 TRAIN/A0 provenance를 증명할 수 없어 builder의 원천검증이 필요합니다. A=0은 추가성분 없음이며 자연 선박 부재 GT가 아닙니다. residual도 ship energy나 참 배경의 추정 정답이 아닙니다.

0.1dB-grid common gain 불변성, 임의 pre-quant gain의 수치차이, per-band static offset에 대한 비불변성, PCA repeat fit·basis reload·chunk 일관성·NaN mask·양성잔차·곡률 이웃조건·알려진 raw target의 rounded pooling bound를 확인했습니다. TRAIN 외 fit, 합성 source-kind, 중복 ID, state/file 변조, binding 불일치, 덮어쓰기를 거부합니다. 상세 수치는 [검증 JSON](R12_FEATURES_FIXTURE.json)에 있습니다. fixture NPZ는 명시적으로 DO_NOT_USE_FOR_TRAINING입니다.

API: `fit_spectral_basis(db_background, source_ids, split='train', training_binding=...)`; `transform_channels(db,basis=None,include_shape=False)` → x8/11,w; `compact_relative_features(...)` →466/673열; `compact_feature_names(include_shape)`; `save_spectral_basis` / `load_spectral_basis(expected_sha256=...,expected_training_binding=...)`.
