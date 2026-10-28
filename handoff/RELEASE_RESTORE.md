# 직접 학습한 모델 Release 복원

Release: https://github.com/taeyang0505/-Underwater-Vessel-Noise-Challenge-by-the-French-Embassy-in-South-Africa/releases/tag/handoff-2026-10-28

`owned-trained-models-2026-10-28.tar.gz`와 동일 이름의 JSON ledger를 받는다. 저장소의 `OWNED_MODEL_RELEASE_MANIFEST.json`도 동일한 SHA/entry 목록이다.

```bash
# gh가 있으면. 새로운 별도 다운로드 폴더 사용.
gh release download handoff-2026-10-28 \
  --repo taeyang0505/-Underwater-Vessel-Noise-Challenge-by-the-French-Embassy-in-South-Africa \
  --pattern 'owned-trained-models-*' --dir /path/to/release-download

# archive 전체 SHA + 706개 entry SHA 검사만. torch/pickle 로드하지 않음.
python3 handoff/restore_owned_artifacts.py /path/to/release-download/owned-trained-models-2026-10-28.tar.gz

# 새 폴더에만 추출. 기존 canonical 학습 폴더에 덮어쓰지 않음.
python3 handoff/restore_owned_artifacts.py /path/to/release-download/owned-trained-models-2026-10-28.tar.gz \
  --destination /path/to/new-owned-model-snapshot
```

원래 모델 경로와 완료 영수증은 `model_metadata/`에 있다. 새폴더에 archive를 풀었다고 데이터까지 복원되거나 현재partial 학습이 자동재개되지 않는다. 모델파일은 완료모델과 진행 중 latest/best snapshot이 섞여 있으며 ledger에 구분했다. 신뢰하는 파일만 `torch.load`한다. archive안에는 학습matrix,외부WAV,타사 pretrainedweight가 없다.
