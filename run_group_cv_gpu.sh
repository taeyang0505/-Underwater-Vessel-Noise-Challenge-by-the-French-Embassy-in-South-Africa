#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Default action checks all hashes and source gates without starting training.
# Add --execute to train 5 folds; the model policy is 100 epochs / patience 7 / Adam.
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python -u -m pipeline.train_group_cv "$@"
