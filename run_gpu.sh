#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# A passed V2 package is required. This script never regenerates synthetic data.
data_dir="${1:-data/synthetic/quality_v2_native_pilot}"
run_tag="${2:-gpu_quality_v2}"
training_options=(
  --epochs 100 --patience 7 --optimizer adam --scheduler plateau --dropout 0.1
  --contrastive-weight 0 --batch 256 --width 384 --k 32 --trees 1600 --threads 8
  --models lgbm lgbm_physics lgbm_retrieval tabm cnn fdy itransformer
)
# Check every destination and dataset before requesting CUDA or doing training.
for seed in 42 43 44; do
  python -m pipeline.train --data-dir "$data_dir" --tag "$run_tag" --seed "$seed" --validate-only "${training_options[@]}"
done
python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU required"; print(torch.cuda.get_device_name())'
for seed in 42 43 44; do
  python -u -m pipeline.train --data-dir "$data_dir" --tag "$run_tag" --seed "$seed" --device cuda "${training_options[@]}"
done
# These are three seeds on one held-out package, not Group K-fold CV.
# Group CV orchestration, OOF combination and source-wise calibration are separate.
# No challenge inference, sealed evaluation, posting or submission is performed.
