# Copyright © 2025-2026 Quadux IT GmbH
#    ____                  __              __________   ______          __    __  __
#   / __ \__  ______ _____/ /_  ___  __   /  _/_  __/  / ____/___ ___  / /_  / / / /
#  / / / / / / / __ `/ __  / / / / |/_/   / /  / /    / / __/ __ `__ \/ __ \/ /_/ /
# / /_/ / /_/ / /_/ / /_/ / /_/ />  <   _/ /  / /    / /_/ / / / / / / /_/ / __  /
# \___\_\__,_/\__,_/\__,_/\__,_/_/|_|  /___/ /_/     \____/_/ /_/ /_/_.___/_/ /_/
# License: Quadux files Apache 2.0 (see LICENSE), Jina model: Qwen Research License
# Author: Walter Hoffmann

# PyTorch 2.9.1 with CUDA 13.0 for RTX 5090 (Blackwell, sm_120)
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Model cache paths (offline mode set AFTER download)
ENV HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models \
    HF_MODULES_CACHE=/app/models/modules \
    HUGGINGFACE_HUB_ALLOW_CODE_EVERYWHERE=1 \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    MODEL_NAME=jinaai/jina-embeddings-v4

WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Download model during build (makes container offline-capable)
COPY download_model.py /tmp/download_model.py
RUN python /tmp/download_model.py && rm /tmp/download_model.py

# Keep modules cache for true offline mode!
# Dynamic Python modules (trust_remote_code) are cached in /app/models/modules

# Enable offline mode AFTER download - container is now fully offline-capable
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Install fonts for document image generation (after model download to preserve cache)
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

COPY app.py /app/app.py

EXPOSE 8000

# Healthcheck: prüft /health endpoint alle 30 Sekunden
# Bei API_HOST=0.0.0.0 wird localhost verwendet, sonst der konfigurierte Host
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://$(if [ "$API_HOST" = "0.0.0.0" ]; then echo "localhost"; else echo "$API_HOST"; fi):${API_PORT}/health || exit 1

# Configuration only via environment variables:
# Usage: docker run ... jina-embeddings-v4 --cpu
# Or set FORCE_CPU=1 environment variable
ENTRYPOINT ["python", "app.py"]
CMD []
