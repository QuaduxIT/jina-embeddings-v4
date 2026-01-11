<!--
Copyright © 2025-2026 Quadux IT GmbH
   ____                  __              __________   ______          __    __  __
  / __ \__  ______ _____/ /_  ___  __   /  _/_  __/  / ____/___ ___  / /_  / / / /
 / / / / / / / __ `/ __  / / / / |/_/   / /  / /    / / __/ __ `__ \/ __ \/ /_/ /
/ /_/ / /_/ / /_/ / /_/ / /_/ />  <   _/ /  / /    / /_/ / / / / / / /_/ / __  /
\___\_\__,_/\__,_/\__,_/\__,_/_/|_|  /___/ /_/     \____/_/ /_/ /_/_.___/_/ /_/

License: Quadux files Apache 2.0 (see LICENSE), Jina model: Qwen Research License
Author: Walter Hoffmann
-->

# Jina Embeddings v4 API

[![Version](https://img.shields.io/badge/version-1.2.0-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Docker Hub](https://img.shields.io/badge/docker-quaduxit%2Fjina--embeddings--v4-blue.svg)](https://hub.docker.com/r/quaduxit/jina-embeddings-v4)
[![GitHub](https://img.shields.io/badge/github-quaduxit%2Fjina--embeddings--v4-black.svg)](https://github.com/quaduxit/jina-embeddings-v4)

Docker container for Jina Embeddings v4 with GPU support (CUDA 13.0, RTX 5090/Blackwell sm_120).

## Why This Project?

We created this software to easily leverage the **duality of the Jina v4 model**: The same model provides both **text and image embeddings** in the same vector space. Without this container, you would need to host separate services for each modality – with corresponding overhead for deployment, maintenance, and resources.

With this project, a single container starts that covers both use cases:

- 📝 **Text Embeddings** for semantic search, RAG, duplicate detection
- 🖼️ **Image Embeddings** for visual search, cross-modal retrieval
- 🔗 **Cross-Modal** – Text and images directly comparable (same embedding space)

## Features

- ✅ **Text Embeddings** - Semantic vectors for texts
- ✅ **Image Embeddings** - Visual vectors for images
- ✅ **Cross-Modal** - Text and images in the same embedding space
- ✅ **Long Context** - Up to 32,768 tokens input length (tokenizer supports 128K)
- ✅ **Matryoshka Dimensions** - Flexible dimensions (128, 256, 512, 1024, 2048)
- ✅ **GPU & CPU Support** - CUDA or CPU fallback
- ✅ **Docker Healthcheck** - Automatic health monitoring
- ✅ **Offline-capable** - Model is downloaded during build

## Quick Start

### Build Docker Image

The model (~5 GB) is automatically downloaded during build and stored in the image.
Afterwards, the container is fully offline-capable.

```bash
# Local build
docker build -t quaduxit/jina-embeddings-v4 .

# Or from Docker Hub:
docker pull quaduxit/jina-embeddings-v4:latest
```

### Start Container

```bash
# GPU mode (default) - no volume needed!
docker run -d --name jina-embed-v4 \
  --gpus all \
  -p 8090:8000 \
  quaduxit/jina-embeddings-v4

# CPU mode
docker run -d --name jina-embed-v4 \
  -p 8090:8000 \
  -e FORCE_CPU=1 \
  quaduxit/jina-embeddings-v4
```

### With Batch Script (Windows)

```cmd
start.bat          # GPU mode
start.bat --cpu    # CPU mode
```

### With Shell Script (Linux/macOS)

```bash
./start.sh          # GPU mode (default)
HOST_PORT=9090 ./start.sh --cpu  # CPU mode on different port
```

The script checks whether `docker` and `grep` are available before starting, builds the image if necessary, starts the container (GPU by default, `--cpu` forces `FORCE_CPU=1`), and waits until Uvicorn is ready. It prints the completed endpoint, explains the built-in API routes, and can be configured to use a different host port via the `HOST_PORT` environment variable.

**Why this script?** The software aims to provide dual capacity for text and image embeddings from the same model instance, so users don't need to host separate services for each modality. The script ensures the container starts exactly this use case quickly and reproducibly.

## API Endpoints

### Single-Vector Endpoints (for Vector Databases)

Single-vector embeddings generate one vector per input (2048 dimensions, Matryoshka truncation possible).

| Endpoint       | Method | Description                  |
| -------------- | ------ | ---------------------------- |
| `/health`      | GET    | Health status                |
| `/embed/text`  | POST   | Text embeddings (2048 dims)  |
| `/embed/image` | POST   | Image embeddings (2048 dims) |

**Parameters (Single-Vector):**

- `task`: "retrieval" | "text-matching" | "code"
- `dimensions`: Optional, 128-2048 for Matryoshka truncation

### Multi-Vector Endpoints (for MaxSim/Late Interaction)

Multi-vector embeddings generate multiple 128-dim vectors per input (one vector per token/patch).
Used for MaxSim similarity and Visual Document Retrieval.

| Endpoint         | Method | Description                               |
| ---------------- | ------ | ----------------------------------------- |
| `/embed/textMV`  | POST   | Text multi-vector (N tokens × 128 dims)   |
| `/embed/imageMV` | POST   | Image multi-vector (N patches × 128 dims) |

**Parameters (Multi-Vector):**

- `task`: "retrieval" | "text-matching" | "code"
- ⚠️ `dimensions` is NOT supported for multi-vector (always 128 dims per token/patch)

## Curl Examples

### Health Check

```bash
curl http://localhost:8090/health
```

**Response:**

```json
{
  "status": "ok",
  "device": "cuda",
  "model": "jinaai/jina-embeddings-v4"
}
```

### Text Embedding

```bash
# Simple text embedding (2048 dimensions)
curl -X POST http://localhost:8090/embed/text \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "Hallo Welt"],
    "task": "text-matching"
  }'

# With Matryoshka (512 dimensions)
curl -X POST http://localhost:8090/embed/text \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world"],
    "task": "text-matching",
    "dimensions": 512
  }'
```

**Response:**

```
{
  "embeddings": [
    [0.0115, 0.0002, 0.0153, ...],
    [0.0089, 0.0045, 0.0201, ...]
  ]
}
```

> Dimension: 2048 (default) or chosen Matryoshka dimension

### Image Embedding

```bash
# Simple image embedding (2048 dimensions)
curl -X POST http://localhost:8090/embed/image \
  -F "file=@image.png" \
  -F "task=text-matching"

# With Matryoshka (256 dimensions)
curl -X POST http://localhost:8090/embed/image \
  -F "file=@image.png" \
  -F "task=text-matching" \
  -F "dimensions=256"
```

**Response:**

```
{
  "embeddings": [
    [0.4180, -0.1855, -0.8398, ...]
  ]
}
```

> Dimension: 2048 (default) or chosen Matryoshka dimension

### Text Multi-Vector (for MaxSim)

```bash
curl -X POST http://localhost:8090/embed/textMV \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["What is the total revenue?"],
    "task": "retrieval"
  }'
```

**Response:**

```
{
  "embeddings": [
    [[0.012, -0.034, ...], [...], ...]
  ],
  "shapes": [[8, 128]]
}
```

> Format: N tokens × 128 dims (here: 8 tokens)

### Image Multi-Vector (for MaxSim/Visual Document Retrieval)

```bash
curl -X POST http://localhost:8090/embed/imageMV \
  -F "file=@document.png" \
  -F "task=retrieval"
```

**Response:**

```
{
  "embeddings": [
    [[0.042, -0.018, ...], [...], ...]
  ],
  "shapes": [[27, 128]]
}
```

> Format: N patches × 128 dims (here: 27 image patches)
> }

````

> **MaxSim Similarity:** Text and image multi-vectors are compared using MaxSim:
> For each text token, the maximum cosine value over all image patches is calculated,
> then the average is computed. This enables fine-grained token-to-patch matching.

## Matryoshka Dimensions

Jina v4 is trained with Matryoshka training. The full 2048-dimensional embeddings can be truncated to smaller dimensions without significant quality loss.

**The `dimensions` parameter accepts any value from 1 to 2048.** The following values are typical examples:

| Dimension | Storage       | Use Case           |
| --------- | ------------- | ------------------ |
| 2048      | 8 KB/vector   | Maximum quality    |
| 1024      | 4 KB/vector   | Good balance       |
| 512       | 2 KB/vector   | Resource efficient |
| 256       | 1 KB/vector   | Fast search        |
| 128       | 0.5 KB/vector | Minimal storage    |

> **Note:** You can use any value ≤ 2048 (e.g., `dimensions=768` for compatibility with other models). Embeddings are automatically L2-normalized after truncation.

## Tasks

The `task` parameter selects the LoRA adapter for different use cases:

| Task            | Description                                             |
| --------------- | ------------------------------------------------------- |
| `retrieval`     | Asymmetric search (query → document) - for DB retrieval |
| `text-matching` | Semantic similarity (symmetric) - duplicate detection   |
| `code`          | Code retrieval and code similarity                      |

## Environment Variables

| Variable     | Default                     | Description        |
| ------------ | --------------------------- | ------------------ |
| `API_HOST`   | `0.0.0.0`                   | Bind address       |
| `API_PORT`   | `8000`                      | API port           |
| `FORCE_CPU`  | -                           | `1` for CPU mode   |
| `HF_HOME`    | `/models`                   | Hugging Face cache |
| `MODEL_NAME` | `jinaai/jina-embeddings-v4` | Model name         |

## Tests

```bash
cd test
node test.js
````

Tests:

- ✅ Health endpoint
- ✅ Text embeddings (single-vector)
- ✅ Image embeddings (single-vector)
- ✅ Text multi-vector
- ✅ Image multi-vector
- ✅ Text similarity
- ✅ Image similarity
- ✅ Text-image cross-modal
- ✅ Matryoshka dimensions

## Directory Structure

```
jina-embeddings-v4-docker/
├── Dockerfile          # Container definition
├── app.py              # FastAPI server
├── requirements.txt    # Python dependencies
├── LICENSE             # Apache 2.0 license (Quadux files)
├── NOTICE              # Third-party notices (Jina AI, Qwen/Alibaba)
├── start.bat           # Windows start script
├── start.sh            # Linux/macOS start script
├── README.md           # German documentation
├── README.en.md        # This documentation (English)
├── CHANGELOG.md        # Change log
└── test/
    ├── README.md       # Test image documentation with sources
    ├── test.js         # Main test suite
    └── *.jpg/*.png     # Test images (Unsplash, public domain)
```

## Technical Details

- **Base Image:** `pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime`
- **GPU:** RTX 5090 (Blackwell, sm_120) support
- **Model:** `jinaai/jina-embeddings-v4` with PEFT wrapper
- **Base Model:** Qwen2.5-VL-3B-Instruct (3.8B parameters)
- **Embedding Dimension:** 2048 (Matryoshka: 128-2048)
- **Max. Context Length:** 32,768 tokens (officially supported)
- **Tokenizer Max:** 131,072 tokens (128K, from base model)
- **Framework:** FastAPI + Uvicorn

## Context Length & Token Limits

The model `jina-embeddings-v4` is based on Qwen2.5-VL-3B-Instruct and supports very long inputs:

| Limit               | Value          | Description                       |
| ------------------- | -------------- | --------------------------------- |
| **Official Limit**  | 32,768 tokens  | Recommended and tested by Jina AI |
| **Tokenizer Limit** | 131,072 tokens | Technical maximum from base model |
| **API Limit**       | 32,768 tokens  | Configured in this implementation |

> **Note:** We limit to 32K tokens as this is the officially supported and tested limit by Jina AI. Longer inputs are automatically truncated (`truncation=True`).

### Token Estimation

As a rule of thumb: 1 token ≈ 4 characters (English) or ≈ 2-3 characters (German).

| Tokens | Approximate Text Length         |
| ------ | ------------------------------- |
| 1,000  | ~4,000 characters (~1 page)     |
| 8,192  | ~32,000 characters (~8 pages)   |
| 32,768 | ~130,000 characters (~32 pages) |

## License

This Docker container and the Quadux-specific container controls are released under the **Apache License 2.0** (see [LICENSE](LICENSE)). Apache 2.0 applies to all files that are not explicitly from upstream sources.

### Jina Embeddings v4 Model

The model `jinaai/jina-embeddings-v4` is based on Qwen2.5-VL-3B-Instruct and is licensed under the **Qwen Research License**:

- ✅ **Commercial use permitted** for entities with <100 million monthly active users
- ⚠️ Entities with ≥100M MAU require a separate license from Alibaba Group

Further details:

- **[NOTICE](NOTICE)** – Third-party notices for Jina AI and Qwen/Alibaba
- **[Qwen License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE)** – Full text of the Qwen Research License

### Test Images

Test images are from Unsplash and are subject to the [Unsplash License](https://unsplash.com/license) (free commercial use, no attribution required, but credits provided in [test/README.md](test/README.md)).

## Support & Disclaimer

> &nbsp;
>
> ⚠️ **No Support by Quadux IT GmbH**
>
> This project is provided "as is" without warranty or support. We cannot answer **questions about the Jina model itself** and do not provide **technical support** for this package.
>
> For questions about the model `jina-embeddings-v4`, please contact:
>
> - **Hugging Face:** [huggingface.co/jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4)
> - **Jina AI:** [jina.ai](https://jina.ai) – the company behind Jina Embeddings
>
> &nbsp;
