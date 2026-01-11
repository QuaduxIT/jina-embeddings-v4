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

# Changelog

Alle wichtigen Änderungen an diesem Projekt werden in dieser Datei dokumentiert.

Das Format basiert auf [Keep a Changelog](https://keepachangelog.com/de/1.0.0/),
und dieses Projekt folgt [Semantic Versioning](https://semver.org/lang/de/).

## [1.2.0] - 2026-01-06

### Added

- **LICENSE-Datei** - Explizite Apache 2.0-Lizenzdatei im Verzeichnis hinzugefügt (Quadux-Dateien). Jina-Modell: Qwen Research License.
- **NOTICE** - Third-Party-Hinweise für Jina AI und Qwen/Alibaba Komponenten
- **README.en.md** - Englische Übersetzung der Dokumentation
- **Linux Start-Script** - `start.sh` für Linux/macOS mit identischer Funktionalität wie `start.bat`:
  - Prüft Abhängigkeiten (`docker`, `grep`)
  - Baut das Image lokal
  - Unterstützt `--cpu` Flag und `HOST_PORT` Umgebungsvariable
  - Wartet auf API-Bereitschaft und zeigt Endpoints an
- **Testbilder-Dokumentation** - `test/README.md` mit Quellenangaben und Fotografen-Credits (Unsplash)
- **Warum-Sektion** - README erklärt jetzt die Motivation: Dualität des Modells für Text und Bild nutzen ohne separate Services

### Changed

- **Testbilder ersetzt** - Alle Testbilder durch neue Public-Domain-Bilder von Unsplash ersetzt (min. Full HD)
- **Lizenz-Header** - Alle Dateien mit einheitlichem Copyright-Header und Lizenz-Hinweis (Apache 2.0 + Qwen Research)
- **Erweiterte Docstrings** - `/embed/text` Endpoint mit ausführlicher Dokumentation wie andere Endpoints
- **Verzeichnisstruktur** - README zeigt aktualisierte Dateiliste inkl. LICENSE, start.sh, CHANGELOG

## [1.1.0] - 2026-01-05

### Added

- **Multi-Vector Endpoints** - Neue Endpoints für Late Interaction / MaxSim:
  - `POST /embed/textMV` - Text Multi-Vector (N Tokens × 128 dims)
  - `POST /embed/imageMV` - Image Multi-Vector (N Patches × 128 dims)
- **API-Übersicht im Startup** - Container zeigt beim Start alle verfügbaren Endpoints

### Changed

- **Endpoint Naming** - `/embed/imageSV` zurück zu `/embed/image` umbenannt (Konsistenz mit `/embed/text`)
- **Erweiterte Dokumentation** - README mit Multi-Vector Beispielen und MaxSim Erklärung
- **Test Suite überarbeitet** - Document Cross-Modal Test (Single-Vector) entfernt, da Single-Vector Cross-Modal per Design niedrige Similarity hat (~0.02-0.08). Neuer Multi-Vector Endpoint Test hinzugefügt.

## [1.0.3] - 2026-01-04

### Fixed

- **NaN bei langen Texten (bfloat16)** - float32 ist jetzt der Standard statt bfloat16. Das bfloat16-Format führte bei längeren Texten (>150 Zeichen) zu NaN-Werten, die als Null-Vektoren zurückgegeben wurden.

### Changed

- **float32 als Standard** - GPU verwendet jetzt immer float32 für maximale Stabilität. Die FORCE_FLOAT32 Umgebungsvariable ist nicht mehr nötig.

## [1.0.2] - 2025-12-31

### Fixed

- **NaN Checks für CPU-Version** - Erweiterte NaN-Prüfungen für CPU-Modus behoben, die auf AMD-Servern zu Problemen führten

## [1.0.1] - 2025-12-31

### Changed

- **Explizite Kontextlänge** - `max_length=32768` wird jetzt explizit beim Tokenizer gesetzt
- **Version im Health-Endpoint** - `/health` gibt jetzt auch die API-Version zurück
- **Version im Startup-Banner** - Zeigt die aktuelle Version beim Container-Start

### Improved

- **Dokumentation** - README erweitert mit Kontextlängen-Infos und Token-Schätzungstabelle
- **Matryoshka Beschreibung** - Klarstellung dass beliebige Dimensionen 1-2048 möglich sind
- **Testbilder** - `test_city.jpg` durch echtes Stadtbild ersetzt für korrekte Cross-Modal-Tests

### Added

- **CHANGELOG.md** - Änderungsprotokoll hinzugefügt
- **Versionierung** - Semantische Versionierung eingeführt (1.0.x)

## [1.0.0] - 2025-12-31

### 🎉 Initial Release

Erste stabile Version des Jina Embeddings v4 API Containers.

### Added

- **Text Embeddings** - Semantische Vektoren für Texte mit bis zu 32.768 Tokens Kontextlänge
- **Image Embeddings** - Visuelle Vektoren für Bilder (PNG, JPEG, WebP, GIF)
- **Cross-Modal Support** - Text und Bilder im gleichen Embedding-Space
- **Matryoshka Dimensionen** - Flexible Dimensionen von 1 bis 2048 (empfohlen: 128, 256, 512, 1024, 2048)
- **Task-spezifische Adapter** - Optimierte Embeddings für verschiedene Anwendungsfälle:
  - `text-matching` - Semantische Ähnlichkeit
  - `retrieval.query` - Suchanfragen
  - `retrieval.passage` - Dokumente/Passagen
  - `separation` - Cluster-Trennung
  - `classification` - Klassifikation
- **GPU & CPU Support** - CUDA 13.0 (RTX 5090/Blackwell sm_120) oder CPU-Fallback
- **Docker Healthcheck** - Automatische Gesundheitsprüfung
- **Offline-fähig** - Modell wird beim Build heruntergeladen (~5 GB)
- **L2-Normalisierung** - Automatische Normalisierung bei Matryoshka-Dimensionskürzung

### Technical Details

- Base Image: `pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime`
- Modell: `jinaai/jina-embeddings-v4` (Qwen2.5-VL-3B-Instruct, 3.8B Parameter)
- Framework: FastAPI + Uvicorn
- Tokenizer Max Length: 131.072 Tokens (128K), API limitiert auf 32K

### API Endpoints

| Endpoint       | Methode | Beschreibung      |
| -------------- | ------- | ----------------- |
| `/health`      | GET     | Gesundheitsstatus |
| `/embed/text`  | POST    | Text-Embeddings   |
| `/embed/image` | POST    | Bild-Embeddings   |

---

[1.2.0]: https://github.com/quadux-it/q-pex/releases/tag/jina-embeddings-v1.2.0
[1.1.0]: https://github.com/quadux-it/q-pex/releases/tag/jina-embeddings-v1.1.0
[1.0.3]: https://github.com/quadux-it/q-pex/releases/tag/jina-embeddings-v1.0.3
[1.0.2]: https://github.com/quadux-it/q-pex/releases/tag/jina-embeddings-v1.0.2
[1.0.1]: https://github.com/quadux-it/q-pex/releases/tag/jina-embeddings-v1.0.1
[1.0.0]: https://github.com/quadux-it/q-pex/releases/tag/jina-embeddings-v1.0.0
