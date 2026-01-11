[![Quadux IT Logo](https://quadux.it/Logo.png)](https://quadux.it/)

Copyright © 2025-2026 Quadux IT GmbH

License: Quadux files Apache 2.0 (see LICENSE), Jina model: Qwen Research License
Author: Walter Hoffmann

# Jina Embeddings v4 API

[![Version](https://img.shields.io/badge/version-1.2.0-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Docker Hub](https://img.shields.io/badge/docker-quaduxit%2Fjina--embeddings--v4-blue.svg)](https://hub.docker.com/r/quaduxit/jina-embeddings-v4)
[![GitHub](https://img.shields.io/badge/github-quaduxit%2Fjina--embeddings--v4-black.svg)](https://github.com/quaduxit/jina-embeddings-v4)

Docker-Container für Jina Embeddings v4 mit GPU-Support (CUDA 13.0, RTX 5090/Blackwell sm_120).

## Warum dieses Projekt?

Wir haben diese Software geschrieben, um die **Dualität des Jina v4-Modells** einfach nutzen zu können: Dasselbe Modell liefert sowohl **Text- als auch Bild-Embeddings** im gleichen Vektorraum. Ohne diesen Container müsste man separate Services für jede Modalität hosten – mit entsprechendem Aufwand für Deployment, Wartung und Ressourcen.

Mit diesem Projekt startet ein einzelner Container, der beide Anwendungsfälle abdeckt:

- 📝 **Text-Embeddings** für semantische Suche, RAG, Duplikaterkennung
- 🖼️ **Bild-Embeddings** für visuelle Suche, Cross-Modal-Retrieval
- 🔗 **Cross-Modal** – Text und Bilder direkt vergleichbar (gleicher Embedding-Space)

## Features

- ✅ **Text Embeddings** - Semantische Vektoren für Texte
- ✅ **Image Embeddings** - Visuelle Vektoren für Bilder
- ✅ **Cross-Modal** - Text und Bilder im gleichen Embedding-Space
- ✅ **Long Context** - Bis zu 32.768 Tokens Eingabelänge (Tokenizer unterstützt 128K)
- ✅ **Matryoshka Dimensionen** - Flexible Dimensionen (128, 256, 512, 1024, 2048)
- ✅ **GPU & CPU Support** - CUDA oder CPU-Fallback
- ✅ **Docker Healthcheck** - Automatische Gesundheitsprüfung
- ✅ **Offline-fähig** - Modell wird beim Build heruntergeladen

## Schnellstart

### Docker Image bauen

Das Modell (~5 GB) wird automatisch beim Build heruntergeladen und im Image gespeichert.
Danach ist der Container vollständig offline-fähig.

```bash
# Lokales Build
docker build -t quaduxit/jina-embeddings-v4 .

# Oder von Docker Hub:
docker pull quaduxit/jina-embeddings-v4:latest
```

### Container starten

```bash
# GPU-Modus (Standard) - kein Volume nötig!
docker run -d --name jina-embed-v4 \
  --gpus all \
  -p 8090:8000 \
  quaduxit/jina-embeddings-v4

# CPU-Modus
docker run -d --name jina-embed-v4 \
  -p 8090:8000 \
  -e FORCE_CPU=1 \
  quaduxit/jina-embeddings-v4
```

### Mit Batch-Script (Windows)

```cmd
start.bat          # GPU-Modus
start.bat --cpu    # CPU-Modus
```

### Mit Shell-Script (Linux/macOS)

```bash
./start.sh          # GPU-Modus (Standard)
HOST_PORT=9090 ./start.sh --cpu  # CPU-Modus auf anderem Port
```

Das Script prüft vor dem Start, ob `docker` und `grep` verfügbar sind, baut das Image falls nötig, startet den Container (GPU per Default, `--cpu` erzwingt `FORCE_CPU=1`) und wartet, bis Uvicorn bereit ist. Es druckt den fertigen Endpunkt, erklärt die eingebauten API-Routen und lässt sich per `HOST_PORT`-Umgebungsvariable auf einen anderen Host-Port umstellen.

**Warum das Script?** Die Software soll die duale Kapazität für Text- und Bild-Embeddings aus derselben Modellinstanz liefern, damit Anwender nicht separate Services für jede Modalität hosten müssen. Das Script sorgt dafür, dass der Container genau diesen Anwendungsfall schnell und reproduzierbar startet.

## API Endpoints

### Single-Vector Endpoints (für Vector-Datenbanken)

Single-Vector Embeddings erzeugen einen Vektor pro Eingabe (2048 Dimensionen, Matryoshka-Truncation möglich).

| Endpoint       | Methode | Beschreibung                |
| -------------- | ------- | --------------------------- |
| `/health`      | GET     | Gesundheitsstatus           |
| `/embed/text`  | POST    | Text-Embeddings (2048 dims) |
| `/embed/image` | POST    | Bild-Embeddings (2048 dims) |

**Parameter (Single-Vector):**

- `task`: "retrieval" | "text-matching" | "code"
- `dimensions`: Optional, 128-2048 für Matryoshka-Truncation

### Multi-Vector Endpoints (für MaxSim/Late Interaction)

Multi-Vector Embeddings erzeugen mehrere 128-dim Vektoren pro Eingabe (ein Vektor pro Token/Patch).
Verwendung für MaxSim-Similarity und Visual Document Retrieval.

| Endpoint         | Methode | Beschreibung                             |
| ---------------- | ------- | ---------------------------------------- |
| `/embed/textMV`  | POST    | Text Multi-Vector (N Tokens × 128 dims)  |
| `/embed/imageMV` | POST    | Bild Multi-Vector (N Patches × 128 dims) |

**Parameter (Multi-Vector):**

- `task`: "retrieval" | "text-matching" | "code"
- ⚠️ `dimensions` wird bei Multi-Vector NICHT unterstützt (immer 128 dims pro Token/Patch)

## Curl Beispiele

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
# Einfaches Text-Embedding (2048 Dimensionen)
curl -X POST http://localhost:8090/embed/text \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "Hallo Welt"],
    "task": "text-matching"
  }'

# Mit Matryoshka (512 Dimensionen)
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

> Dimension: 2048 (Standard) oder gewählte Matryoshka-Dimension

### Image Embedding

```bash
# Einfaches Bild-Embedding (2048 Dimensionen)
curl -X POST http://localhost:8090/embed/image \
  -F "file=@image.png" \
  -F "task=text-matching"

# Mit Matryoshka (256 Dimensionen)
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

> Dimension: 2048 (Standard) oder gewählte Matryoshka-Dimension

### Text Multi-Vector (für MaxSim)

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

> Format: N Tokens × 128 dims (hier: 8 Tokens)

### Image Multi-Vector (für MaxSim/Visual Document Retrieval)

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

> Format: N Patches × 128 dims (hier: 27 Image-Patches)
> }

````

> **MaxSim Similarity:** Text- und Bild-Multi-Vektoren werden mit MaxSim verglichen:
> Für jeden Text-Token wird der maximale Cosinus-Wert über alle Bild-Patches berechnet,
> dann wird der Durchschnitt gebildet. Dies ermöglicht feinkörnige Token-zu-Patch Matching.

## Matryoshka Dimensionen

Jina v4 ist mit Matryoshka-Training trainiert. Die vollen 2048-dimensionalen Embeddings können auf kleinere Dimensionen gekürzt werden, ohne signifikant Qualität zu verlieren.

**Der `dimensions` Parameter akzeptiert beliebige Werte von 1 bis 2048.** Die folgenden Werte sind typische Beispiele:

| Dimension | Speicher      | Anwendung           |
| --------- | ------------- | ------------------- |
| 2048      | 8 KB/Vektor   | Maximale Qualität   |
| 1024      | 4 KB/Vektor   | Gute Balance        |
| 512       | 2 KB/Vektor   | Ressourceneffizient |
| 256       | 1 KB/Vektor   | Schnelle Suche      |
| 128       | 0.5 KB/Vektor | Minimaler Speicher  |

> **Hinweis:** Sie können jeden beliebigen Wert ≤ 2048 verwenden (z.B. `dimensions=768` für Kompatibilität mit anderen Modellen). Die Embeddings werden automatisch L2-normalisiert nach dem Kürzen.

## Tasks

Der `task` Parameter wählt den LoRA-Adapter für verschiedene Anwendungsfälle:

| Task            | Beschreibung                                              |
| --------------- | --------------------------------------------------------- |
| `retrieval`     | Asymmetrische Suche (Query → Dokument) - für DB-Retrieval |
| `text-matching` | Semantische Ähnlichkeit (symmetrisch) - Duplikaterkennung |
| `code`          | Code-Retrieval und Code-Ähnlichkeit                       |

## Umgebungsvariablen

| Variable     | Default                     | Beschreibung       |
| ------------ | --------------------------- | ------------------ |
| `API_HOST`   | `0.0.0.0`                   | Bind-Adresse       |
| `API_PORT`   | `8000`                      | API-Port           |
| `FORCE_CPU`  | -                           | `1` für CPU-Modus  |
| `HF_HOME`    | `/models`                   | Hugging Face Cache |
| `MODEL_NAME` | `jinaai/jina-embeddings-v4` | Modellname         |

## Tests

```bash
cd test
node test.js
````

Testet:

- ✅ Health Endpoint
- ✅ Text Embeddings (Single-Vector)
- ✅ Image Embeddings (Single-Vector)
- ✅ Text Multi-Vector
- ✅ Image Multi-Vector
- ✅ Text Similarity
- ✅ Image Similarity
- ✅ Text-Image Cross-Modal
- ✅ Matryoshka Dimensionen

## Verzeichnisstruktur

```
jina-embeddings-v4-docker/
├── Dockerfile          # Container-Definition
├── app.py              # FastAPI Server
├── requirements.txt    # Python-Abhängigkeiten
├── LICENSE             # Apache 2.0-Lizenz (Quadux-Dateien)
├── NOTICE              # Third-Party-Hinweise (Jina AI, Qwen/Alibaba)
├── start.bat           # Windows Start-Script
├── start.sh            # Linux/macOS Start-Script
├── README.md           # Diese Dokumentation (Deutsch)
├── README.en.md        # Englische Dokumentation
├── CHANGELOG.md        # Änderungsprotokoll
└── test/
    ├── README.md       # Testbilder-Dokumentation mit Quellen
    ├── test.js         # Haupt-Testsuite
    └── *.jpg/*.png     # Testbilder (Unsplash, Public Domain)
```

## Technische Details

- **Base Image:** `pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime`
- **GPU:** RTX 5090 (Blackwell, sm_120) Support
- **Modell:** `jinaai/jina-embeddings-v4` mit PEFT Wrapper
- **Basis-Modell:** Qwen2.5-VL-3B-Instruct (3.8B Parameter) - Built with Qwen
- **Embedding Dimension:** 2048 (Matryoshka: 128-2048)
- **Max. Kontextlänge:** 32.768 Tokens (offiziell unterstützt)
- **Tokenizer Max:** 131.072 Tokens (128K, vom Basis-Modell)
- **Framework:** FastAPI + Uvicorn

## Kontextlänge & Token-Limits

Das Modell `jina-embeddings-v4` basiert auf Qwen2.5-VL-3B-Instruct und unterstützt sehr lange Eingaben:

| Limit                 | Wert           | Beschreibung                           |
| --------------------- | -------------- | -------------------------------------- |
| **Offizielles Limit** | 32.768 Tokens  | Von Jina AI empfohlen und getestet     |
| **Tokenizer Limit**   | 131.072 Tokens | Technisches Maximum vom Basis-Modell   |
| **API Limit**         | 32.768 Tokens  | In dieser Implementierung konfiguriert |

> **Hinweis:** Wir begrenzen auf 32K Tokens, da dies das von Jina AI offiziell unterstützte und getestete Limit ist. Längere Eingaben werden automatisch abgeschnitten (`truncation=True`).

### Token-Schätzung

Als Faustregel: 1 Token ≈ 4 Zeichen (Englisch) bzw. ≈ 2-3 Zeichen (Deutsch).

| Tokens | Ungefähre Textlänge           |
| ------ | ----------------------------- |
| 1.000  | ~4.000 Zeichen (~1 Seite)     |
| 8.192  | ~32.000 Zeichen (~8 Seiten)   |
| 32.768 | ~130.000 Zeichen (~32 Seiten) |

## Lizenz

Dieser Docker-Container und die Quadux-spezifischen Container-Steuerungen stehen unter der **Apache License 2.0** (siehe [LICENSE](LICENSE)). Apache 2.0 gilt für alle Dateien, die nicht explizit von upstream-Quellen stammen.

### Jina Embeddings v4 Modell

Das Modell `jinaai/jina-embeddings-v4` basiert auf Qwen2.5-VL-3B-Instruct und steht unter der **Qwen Research License**:

- ✅ **Kommerzielle Nutzung erlaubt** für Unternehmen mit <100 Millionen monatlichen Nutzern
- ⚠️ Unternehmen mit ≥100M MAU benötigen eine separate Lizenz von Alibaba Group

Weitere Details:

- **[NOTICE](NOTICE)** – Third-Party-Hinweise zu Jina AI und Qwen/Alibaba
- **[Qwen License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE)** – Volltext der Qwen Research License

### Testbilder

Die Testbilder stammen von Unsplash und unterliegen der [Unsplash License](https://unsplash.com/license) (kostenlose kommerzielle Nutzung, keine Attribution erforderlich, aber Credits in [test/README.md](test/README.md) angegeben).

## Support & Haftungsausschluss

> &nbsp;
>
> ⚠️ **Kein Support durch Quadux IT GmbH**
>
> Dieses Projekt wird „as is" ohne Gewährleistung oder Support bereitgestellt. Wir können **keine Fragen zum Jina-Modell selbst** beantworten und leisten **keinen technischen Support** für dieses Paket.
>
> Bei Fragen zum Modell `jina-embeddings-v4` wenden Sie sich bitte an:
>
> - **Hugging Face:** [huggingface.co/jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4)
> - **Jina AI:** [jina.ai](https://jina.ai) – die Firma hinter Jina Embeddings
>
> &nbsp;
