# Search App — OpenCLIP + pgvector + Vite

A tiny web app that searches a photo library by text query or by example image. Same OpenCLIP model on both ends of the indexing/querying loop, so the vector space matches and the search actually works.

```
┌────────────────────────┐    /api/*    ┌─────────────────────────┐
│  Vite + TypeScript     │ ───────────► │   FastAPI               │
│  + Tailwind v4         │              │   (loads ViT-H-14/dfn5b │
│  (no framework)        │ ◄─────────── │    once at startup)     │
└────────────────────────┘    JSON      └──────────┬──────────────┘
                                                   │ SQL
                                                   ▼
                                        ┌──────────────────┐
                                        │  pgvector        │
                                        │  (Docker, run    │
                                        │   from           │
                                        │   ../embeddings/)│
                                        └──────────────────┘
```

## Prerequisites

- The Postgres + pgvector stack from `../embeddings/` must be running.
- The `image_embeddings` table must be populated (run `uv run ../embeddings/generate_embeddings.py` first).
- [`uv`](https://docs.astral.sh/uv/) for the Python backend.
- Node 20+ and npm for the Vite frontend.

## Running it (two terminals)

### Terminal 1 — backend

```bash
cd backend
uv run main.py
```

Loads the OpenCLIP model (~5-15 s) and starts FastAPI on http://127.0.0.1:8000. Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

### Terminal 2 — frontend

```bash
cd frontend
npm install
npm run dev
```

Opens Vite on http://localhost:5173. The dev server proxies `/api/*` and `/images/*` to the FastAPI backend, so both look like they live on the same origin in the browser.

## What you can do

- Type a phrase like `"foggy mountain at dawn"` and hit Search → 24 best matches by cosine similarity.
- Drag an image (jpg/png/webp) onto the drop zone, or click "choose a file" → 24 visually similar shots.
- Hover any result tile for filename and predicted category. Click to open the full-size image.

## API surface

| Endpoint                  | Method | Body                                | Returns                |
|---------------------------|--------|-------------------------------------|------------------------|
| `/api/health`             | GET    | -                                   | model + device info    |
| `/api/search/text`        | POST   | `{ query: string, limit?: number }` | `SearchHit[]`          |
| `/api/search/image`       | POST   | multipart: `file`, `limit`          | `SearchHit[]`          |
| `/images/{filename}`      | GET    | -                                   | the photo file         |

`SearchHit`:

```ts
{
  filename: string;
  category: string | null;
  confidence: number | null;
  distance: number;    // pgvector <#> output (negative inner product, smaller = closer)
  url: string;
}
```
