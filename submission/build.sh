#!/bin/bash
# Build the Grand Challenge submission image.
#
# Run from anywhere; this script cd's to the repo root itself since the build
# context must see both nnUNet-2.2/ (top level) and submission/.
#
#   bash submission/build.sh
#
set -e
SCRIPTPATH="$( cd "$(dirname "$0")/.." ; pwd -P )"   # repo root
IMAGE="autopet_interactive_submit"

if [ ! -d "$SCRIPTPATH/submission/weights/nnUNet_results" ] || \
   [ ! -f "$SCRIPTPATH/submission/weights/classifier/tracer_classifier.pt" ]; then
    echo "Trained weights not found under submission/weights/ - fetching..."
    bash "$SCRIPTPATH/submission/download_weights.sh"
fi

echo "Building Docker image: $IMAGE (from submission/Dockerfile, repo-root build context)"
# Use buildx: on a native amd64 host this is equivalent to `docker build`, but
# it also works when building from an arm64 host (cross-compiles via QEMU),
# which a plain `docker build --platform=linux/amd64` cannot do.
docker buildx build --platform=linux/amd64 \
    -f "$SCRIPTPATH/submission/Dockerfile" \
    -t "$IMAGE" \
    --load \
    "$SCRIPTPATH"
echo "Done. Image tagged '$IMAGE'."
