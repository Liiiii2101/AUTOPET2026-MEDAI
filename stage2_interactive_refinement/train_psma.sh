#!/usr/bin/env bash
# Stage 2 training — PSMA branch. Continues from a Stage-1 checkpoint for the same fold.
# Usage: ./train_psma.sh DATASET_ID FOLD STAGE1_CHECKPOINT
set -euo pipefail

DATASET_ID="${1:?usage: train_psma.sh DATASET_ID FOLD STAGE1_CHECKPOINT}"
FOLD="${2:?usage: train_psma.sh DATASET_ID FOLD STAGE1_CHECKPOINT}"
PRETRAINED_WEIGHTS="${3:?usage: train_psma.sh DATASET_ID FOLD STAGE1_CHECKPOINT}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$REPO_ROOT/nnUNet-2.2/nnunetv2/run/run_finetuning_stunet.py" \
    "$DATASET_ID" 3d_fullres "$FOLD" \
    -tr MyCustomCurriculumTrainerSegPreSkel \
    -pretrained_weights "$PRETRAINED_WEIGHTS"
