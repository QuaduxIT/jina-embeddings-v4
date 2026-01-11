#!/bin/bash
# Copyright © 2025-2026 Quadux IT GmbH
#    ____                  __              __________   ______          __    __  __
#   / __ \__  ______ _____/ /_  ___  __   /  _/_  __/  / ____/___ ___  / /_  / / / /
#  / / / / / / / __ `/ __  / / / / |/_/   / /  / /    / / __/ __ `__ \/ __ \/ /_/ /
# / /_/ / /_/ / /_/ / /_/ / /_/ />  <   _/ /  / /    / /_/ / / / / / / /_/ / __  /
# \___\_\__,_/\__,_/\__,_/\__,_/_/|_|  /___/ /_/     \____/_/ /_/ /_/_.___/_/ /_/

# License: Quadux files Apache 2.0 (see LICENSE), Jina model: Qwen Research License
# Author: Walter Hoffmann
#
# Jina Embeddings v4 API - Docker Start Script (Linux)
# Usage: ./start.sh [--cpu]
# The container embeds the model, so no external volumes are necessary.

set -euo pipefail

IMAGE_NAME=jina-embeddings-v4
CONTAINER_NAME=jina-embed-v4
HOST_PORT=8090
CPU_ENV=""
GPU_FLAG="--gpus all"

if [[ "${1:-}" == "--cpu" ]]; then
    CPU_ENV="-e FORCE_CPU=1"
    GPU_FLAG=""
    echo "CPU mode forced via --cpu flag"
else
    echo "GPU mode (default)"
fi

echo "Stopping existing container..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Building Docker image..."
docker build -t quaduxit/$IMAGE_NAME build/
docker push quaduxit/$IMAGE_NAME

echo "Starting container..."
docker run -d \
    --name "$CONTAINER_NAME" \
    --restart always \
    $GPU_FLAG \
    -p "$HOST_PORT":8000 \
    $CPU_ENV \
    quaduxit/$IMAGE_NAME

echo
echo "Container started! Waiting for API to become ready (Uvicorn needs ~10s)."
echo

MAX_WAIT=150
WAIT_COUNT=0
while true; do
    sleep 2
    if docker logs "$CONTAINER_NAME" 2>&1 | grep -q "Uvicorn running on http://0.0.0.0:8000"; then
        echo
        echo "========================================="
        echo "API is ready! Available at http://localhost:$HOST_PORT"
        echo "=========================================\nEndpoints:"
        echo "  GET  /health       - Health check"
        echo "  POST /embed/text   - Text embeddings"
        echo "  POST /embed/image  - Image embeddings"
        echo
        break
    fi
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if (( WAIT_COUNT >= MAX_WAIT )); then
        echo
        echo "ERROR: Timeout waiting for API to start!"
        echo "Check logs with: docker logs $CONTAINER_NAME"
        break
    fi
    printf "Waiting for API... (%d/%d)\r" "$WAIT_COUNT" "$MAX_WAIT"
done
