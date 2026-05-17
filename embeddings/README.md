# Embeddings — pgvector + OpenCLIP

Stands up a Postgres database with the `pgvector` extension in Docker, then encodes every photo in `../images/` with **OpenCLIP `ViT-H-14 / dfn5b`** and stores a 1024-dim L2-normalised embedding per image. This is the data layer that powers `../search-app/`.

## Why this model

`ViT-H-14 / dfn5b` is currently the strongest publicly-loadable CLIP. About 632 M parameters, 1024-dim output, ~83.4% ImageNet zero-shot. The weights are downloaded automatically on first run (~3.8 GB into `~/.cache/huggingface/hub/`).

## Why pgvector with HNSW + inner product

- We L2-normalise every embedding before insert.
- After normalisation, **cosine similarity equals inner product** up to a constant.
- Inner product (`vector_ip_ops`) is ~30% faster than cosine in pgvector, so the HNSW index uses it.
- HNSW gives sub-millisecond approximate-nearest-neighbour queries with high recall on libraries from hundreds to millions of vectors.

## Prerequisites

- Docker + Docker Compose
- [`uv`](https://docs.astral.sh/uv/) (`brew install uv`)

## 1. Start the database

From this directory:

```bash
cp .env.example .env   # already done on first checkout, but harmless to redo
docker compose up -d
```

The container is named `openclip-pgvector`. Data persists in `./pgdata/`. The first start runs `init.sql`, which enables the extension and creates the `image_embeddings` table + the HNSW index.

Tail the logs while it boots:

```bash
docker compose logs -f postgres
```

You can connect with `psql`:

```bash
docker compose exec postgres psql -U openclip -d openclip
```

## 2. Run the embedding indexer

From this directory:

```bash
uv run generate_embeddings.py
```

This:

1. Loads `ViT-H-14 / dfn5b` on MPS (or CPU fallback).
2. Walks `../images/` recursively for every supported image extension.
3. Encodes each image, L2-normalises the embedding.
4. Looks up `category` and `confidence` from `../classification/classification_results.csv` if available and stores them as metadata.
5. Upserts into `image_embeddings` (idempotent — re-running just updates).

## Schema

| Column      | Type           | Notes |
|-------------|----------------|-------|
| `id`        | `BIGSERIAL`    | Primary key. |
| `filename`  | `TEXT UNIQUE`  | Basename only. Used as the conflict target for upserts. |
| `path`      | `TEXT`         | Absolute path on disk at index time. |
| `embedding` | `VECTOR(1024)` | OpenCLIP ViT-H-14 output, L2-normalised. |
| `category`  | `TEXT`         | Optional, from the classification CSV. |
| `confidence`| `REAL`         | Optional, from the classification CSV. |
| `created_at`| `TIMESTAMPTZ`  | First insert. |
| `updated_at`| `TIMESTAMPTZ`  | Touched on every upsert. |

## Tearing it down

```bash
docker compose down              # stop the container, keep the data
docker compose down -v           # stop and delete the volume (data still in ./pgdata/)
rm -rf pgdata                    # nuke the data entirely; next `up -d` re-runs init.sql
```
