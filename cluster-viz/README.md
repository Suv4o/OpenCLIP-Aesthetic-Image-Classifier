# Cluster Visualisation — UMAP + Three.js

A 3D interactive view of every photo's OpenCLIP embedding, projected from 1024 dimensions down to 3 with **UMAP** and rendered as a coloured point cloud with **Three.js**. Click any point to open the photo and its metadata.

```
1024-dim vectors (pgvector)
         │
         ▼  UMAP n_components=3, metric=cosine
   coords.json (110 × {filename, category, x, y, z})
         │
         ▼  Vite + vanilla TS + Tailwind + Three.js
   3D scatter cloud, coloured by category, click-to-open
```

## Honest caveat to know upfront

Projecting 1024 dimensions into 3 is always lossy - same fundamental constraint as flattening a globe onto a paper map. UMAP does a great job of keeping the *clusters* visible, but the precise distance between two points in 3D is **not** the same as their true cosine similarity in 1024D. For exact similarity, use the search app (`../search-app/`); for visual gestalt, use this.

## Why UMAP

- **PCA** is linear; flattens the curved embedding manifold into a featureless blob.
- **t-SNE** preserves clusters but distorts between-cluster geometry and is stochastic.
- **UMAP** preserves both local clusters *and* meaningful global structure, runs in under a second on hundreds of photos, and is reproducible with a fixed `random_state`.

## Prerequisites

- The pgvector stack from `../embeddings/` running, with the `image_embeddings` table populated.
- The search-app backend from `../search-app/backend/` running on `:8000` (used for image serving via Vite proxy).
- [`uv`](https://docs.astral.sh/uv/) for the Python projection script.
- Node 20+ and npm for the frontend.

## 1. Generate the 3D coordinates (one-off)

```bash
uv run generate_coords.py
```

This reads all embeddings from pgvector, runs UMAP, and writes `frontend/public/coords.json`. Re-run any time you add or remove photos in the database. Takes a second or two for a few hundred images.

## 2. Run the visualisation

```bash
cd frontend
npm install      # first time only
npm run dev
```

Opens Vite on `http://localhost:5174`. The dev server proxies `/images/*` to the search-app backend on `:8000`, so the same image-serving endpoint is reused.

## Controls

- **Drag** to orbit the camera around the cloud.
- **Scroll** to zoom in/out.
- **Right-drag** (or two-finger drag) to pan.
- **Hover** any point to see the filename in a tooltip.
- **Click** any point to open the side panel with the full image + metadata.

## Customising the look

Constants near the top of `generate_coords.py`:

| Setting | Effect |
|---|---|
| `N_NEIGHBORS` | Higher (~30-50) emphasises global structure. Lower (~5-10) emphasises tight local clusters. Default 15. |
| `MIN_DIST` | Lower packs clusters tighter. Higher spreads them out. Default 0.1. |
| `METRIC` | `"cosine"` is the right choice for normalised CLIP embeddings; do not change unless you know why. |
| `RANDOM_STATE` | Any integer. Same seed = same layout. |

In the frontend, `src/scene.ts` has `CATEGORY_COLORS` if you want to recolour the categories, and `POINT_SIZE` to scale the dot size.
