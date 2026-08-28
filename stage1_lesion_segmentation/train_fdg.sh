#!/usr/bin/env bash
# Stage 1 training — FDG branch.
# Usage: ./train_fdg.sh DATASET_ID FOLD PRETRAINED_MAE_CHECKPOINT
set -euo pipefail

DATASET_ID="${1:?usage: train_fdg.sh DATASET_ID FOLD PRETRAINED_MAE_CHECKPOINT}"
FOLD="${2:?usage: train_fdg.sh DATASET_ID FOLD PRETRAINED_MAE_CHECKPOINT}"
PRETRAINED_WEIGHTS="${3:?usage: train_fdg.sh DATASET_ID FOLD PRETRAINED_MAE_CHECKPOINT}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$REPO_ROOT/nnUNet-2.2/nnunetv2/run/run_finetuning_stunet.py" \
    "$DATASET_ID" 3d_fullres "$FOLD" \
    -tr MyCustomCurriculumTrainer \
    -pretrained_weights "$PRETRAINED_WEIGHTS"
