# OpenCLIP for a Photography Gallery

Three small projects that together turn a folder of landscape photographs into:

1. A neatly categorised gallery (zero-shot classification with OpenCLIP).
2. A searchable vector index (image embeddings stored in pgvector).
3. A web app that finds similar photos by text query or by dropping in a sample image.

All three share the same source images in `images/`.

## Subprojects

### [`classification/`](./classification/)

A single Python script that uses OpenCLIP's zero-shot classification to sort every photo in `images/` into one of six themed buckets (🌌 Starlit Wonders, 🌿 Wild Horizons, ⏳ Time in Motion, 🌆 Urban Glow, 🌊 Ocean Whispers, 🏞️ Liquid Cascades). Writes a CSV with per-image scores and optionally copies files into per-category subfolders.

Read [`classification/README.md`](./classification/README.md) for usage.

### [`embeddings/`](./embeddings/)

A Postgres database (running in Docker, with the `pgvector` extension) plus a Python script that turns every image into a 1024-dim embedding using OpenCLIP and stores it in the database. This is the data layer that powers the search app.

Read [`embeddings/README.md`](./embeddings/README.md) for setup.

### [`search-app/`](./search-app/)

A simple web app with a FastAPI backend and a Vite + TypeScript + Tailwind frontend (no React, no framework). Lets you:

- Type a natural-language query (`"foggy mountain at dawn"`) and get the most similar photos back.
- Drop an image into the browser and get visually similar photos back.

Read [`search-app/README.md`](./search-app/README.md) for how to run it.

## Repository layout

```
OpenClip/
├── README.md                   # this file
├── images/                     # source photos shared by every subproject
├── classification/             # Part 1 — zero-shot categorisation
├── embeddings/                 # Part 2 — pgvector + bulk embedding indexer
└── search-app/                 # Part 3 — FastAPI + Vite search UI
```

## Hardware

Built and tested on an M1 Max with 64 GB unified memory. The Python scripts use Apple's MPS GPU backend automatically; they will fall back to CPU on other platforms.
