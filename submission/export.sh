#!/bin/bash
# Build, then save the image as a gzipped tarball for upload to Grand Challenge.
#
#   bash submission/export.sh
#
set -e
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"   # submission/
IMAGE="autopet_interactive_submit"

bash "$SCRIPTPATH/build.sh"

echo "Saving image to ${IMAGE}.tar.gz (this can take a while and be several GB)..."
docker save "$IMAGE" | gzip -c > "$SCRIPTPATH/${IMAGE}.tar.gz"
echo "Created $SCRIPTPATH/${IMAGE}.tar.gz"
