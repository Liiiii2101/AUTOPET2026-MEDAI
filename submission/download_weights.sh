#!/bin/bash
# Downloads the trained model weights used by the submitted Docker image into
# submission/weights/, in the layout Dockerfile expects. Run this once before
# building the image.
#
# Requires: gdown (pip install gdown)
set -euo pipefail
cd "$(dirname "$0")"

# TODO: replace with the real Google Drive file ID once uploaded.
# (The ID is the long token in the share link: drive.google.com/file/d/<ID>/view)
DRIVE_FILE_ID="14L-ccBAYIwjIphBO9CLFsyNi3QtDr9pn"

if ! command -v gdown >/dev/null 2>&1; then
    echo "Installing gdown..."
    python -m pip install --user gdown
fi

ARCHIVE=autopet_model_weights.tar.gz
echo "Downloading model weights (~5.5GB)..."
gdown "https://drive.google.com/uc?id=${DRIVE_FILE_ID}" -O "$ARCHIVE"

echo "Extracting into submission/weights/ ..."
mkdir -p weights
tar -xzf "$ARCHIVE" -C weights
rm -f "$ARCHIVE"

echo "Done. Expect: submission/weights/nnUNet_results/ and submission/weights/classifier/tracer_classifier.pt"
ls -la weights/
