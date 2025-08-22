#!/bin/bash
set -e

# Build script for Pharmacogenomics ML Platform Docker images
# Usage: ./scripts/build.sh [target] [tag]

TARGET=${1:-production}
TAG=${2:-latest}
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo "Building Pharmacogenomics ML Platform"
echo "Target: $TARGET"
echo "Tag: $TAG"
echo "Build Date: $BUILD_DATE"
echo "VCS Ref: $VCS_REF"

# Build the Docker image
docker build \
    --target "$TARGET" \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg VERSION="$TAG" \
    --build-arg VCS_REF="$VCS_REF" \
    -t "pharmacogenomics-platform:$TAG" \
    -t "pharmacogenomics-platform:$TARGET-$TAG" \
    .

echo "Build completed successfully!"
echo "Image: pharmacogenomics-platform:$TAG"

# Show image size
docker images pharmacogenomics-platform:$TAG --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"