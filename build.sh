#!/bin/bash

# FLUX.1-Schnell Docker Build Script
# Usage: ./build.sh YOUR_DOCKERHUB_USERNAME

set -e

DOCKERHUB_USER=${1:-"tolgatasci"}
IMAGE_NAME="flux-schnell"
VERSION="1.0"

echo "========================================"
echo "Building FLUX.1-Schnell Docker Image"
echo "========================================"
echo ""
echo "Image: ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}"
echo ""
echo "WARNING: This will download ~25GB model during build!"
echo "Make sure you have enough disk space."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Build
echo ""
echo "Building Docker image..."
docker build -t ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION} .

echo ""
echo "Build complete!"
echo ""

# Push?
read -p "Push to Docker Hub? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pushing to Docker Hub..."
    docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}
    echo ""
    echo "========================================"
    echo "SUCCESS!"
    echo "========================================"
    echo ""
    echo "Container Image: ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}"
    echo ""
    echo "Next steps:"
    echo "1. Go to https://console.runpod.io/serverless"
    echo "2. Create New Endpoint"
    echo "3. Container Image: ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}"
    echo "4. GPU: 24 GB (RTX 4090 / A10 / L40S)"
    echo "5. Deploy!"
else
    echo ""
    echo "Image built locally: ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}"
    echo "Run 'docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}' to push later."
fi
