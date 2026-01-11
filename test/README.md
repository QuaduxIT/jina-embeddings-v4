<!--
Copyright © 2025-2026 Quadux IT GmbH
   ____                  __              __________   ______          __    __  __
  / __ \__  ______ _____/ /_  ___  __   /  _/_  __/  / ____/___ ___  / /_  / / / /
 / / / / / / / __ `/ __  / / / / |/_/   / /  / /    / / __/ __ `__ \/ __ \/ /_/ /
/ /_/ / /_/ / /_/ / /_/ / /_/ />  <   _/ /  / /    / /_/ / / / / / / /_/ / __  /
\___\_\__,_/\__,_/\__,_/\__,_/_/|_|  /___/ /_/     \____/_/ /_/ /_/_.___/_/ /_/

License: Apache 2.0 (see ../LICENSE)
Author: Walter Hoffmann
-->

# Jina Embeddings v4 – Test Assets

Dieses Verzeichnis enthält Testbilder und Skripte für die Jina Embeddings v4 API.

## Testskripte

| Datei     | Beschreibung                                            |
| --------- | ------------------------------------------------------- |
| `test.js` | Haupt-Testsuite (Text, Image, Multi-Vector, Matryoshka) |

## Testbilder – Quellen

Alle Bilder stammen von [Unsplash](https://unsplash.com) und sind unter der [Unsplash-Lizenz](https://unsplash.com/license) frei verwendbar (auch kommerziell, keine Namensnennung erforderlich).

| Datei                   | Beschreibung              | Fotograf              | Quelle                                                                     |
| ----------------------- | ------------------------- | --------------------- | -------------------------------------------------------------------------- |
| `test.png`              | Technologie / Schaltkreis | Alexandre Debiève     | [unsplash.com/photos/FO7JIlwjOtU](https://unsplash.com/photos/FO7JIlwjOtU) |
| `test_cat1.jpg`         | Katze (Porträt)           | Manja Vitolic         | [unsplash.com/photos/gKXKBY-C-Dk](https://unsplash.com/photos/gKXKBY-C-Dk) |
| `test_cat2.jpg`         | Katze (Close-up)          | Amber Kipp            | [unsplash.com/photos/75715CVEJhI](https://unsplash.com/photos/75715CVEJhI) |
| `test_cat_cross.jpg`    | Katze (Cross-Modal Test)  | Jae Park              | [unsplash.com/photos/7GX5aICb5i4](https://unsplash.com/photos/7GX5aICb5i4) |
| `test_nature.jpg`       | Berglandschaft            | Kalen Emsley          | [unsplash.com/photos/kGSapVfg8Kw](https://unsplash.com/photos/kGSapVfg8Kw) |
| `test_city.jpg`         | Stadt-Skyline             | Pedro Lastra          | [unsplash.com/photos/Nyvq2juw4_o](https://unsplash.com/photos/Nyvq2juw4_o) |
| `test_bar_chart.png`    | Dashboard / Bar-Chart     | Campaign Creators     | [unsplash.com/photos/pypeCEaJeZY](https://unsplash.com/photos/pypeCEaJeZY) |
| `test_line_chart.png`   | Linien-Diagramm           | Ilya Pavlov           | [unsplash.com/photos/OqtafYT5kTw](https://unsplash.com/photos/OqtafYT5kTw) |
| `test_pie_chart.png`    | Datenvisualisierung       | Hitesh Choudhary      | [unsplash.com/photos/JNxTZzpHmsI](https://unsplash.com/photos/JNxTZzpHmsI) |
| `test_sales_chart.png`  | Analytics / Sales-Chart   | Carlos Muza           | [unsplash.com/photos/hpjSkU2UYSU](https://unsplash.com/photos/hpjSkU2UYSU) |
| `test_invoice.png`      | Finanzdokument / Rechnung | Ash Edmonds           | [unsplash.com/photos/Koxa-GX_5zs](https://unsplash.com/photos/Koxa-GX_5zs) |
| `test_table.png`        | Tabelle / Notizen         | Glenn Carstens-Peters | [unsplash.com/photos/RLw-UC03Gwc](https://unsplash.com/photos/RLw-UC03Gwc) |
| `test_architecture.png` | Technischer Arbeitsplatz  | Glenn Carstens-Peters | [unsplash.com/photos/RLw-UC03Gwc](https://unsplash.com/photos/RLw-UC03Gwc) |

## Auflösung

Alle Bilder wurden mit einer Breite von 1920 px (Full HD) heruntergeladen.

## Tests ausführen

```bash
cd devops/jina-embeddings-v4-docker/test
node test.js
```

Voraussetzung: Der Jina Embeddings Container muss unter `http://localhost:8090` erreichbar sein (oder `JINA_API_URL` setzen).
