#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "psycopg[binary]>=3.2",
#     "pgvector>=0.3",
#     "umap-learn>=0.5",
#     "numpy",
#     "python-dotenv>=1.0",
# ]
# ///
"""
Reads every embedding from the pgvector `image_embeddings` table,
projects them from 1024-D down to 3-D with UMAP, and writes the
result to ./frontend/public/coords.json for the Three.js viewer.
"""

import json
import os
from pathlib import Path

import numpy as np
import psycopg
import umap
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
OUTPUT = PROJECT_ROOT / "frontend" / "public" / "coords.json"

# UMAP knobs. See README for what each one does.
N_NEIGHBORS = 15
MIN_DIST = 0.1
METRIC = "cosine"
RANDOM_STATE = 42


def connect_db() -> psycopg.Connection:
    load_dotenv(REPO_ROOT / "embeddings" / ".env")
    conn = psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        user=os.environ.get("POSTGRES_USER", "openclip"),
        password=os.environ.get("POSTGRES_PASSWORD", "openclip"),
        dbname=os.environ.get("POSTGRES_DB", "openclip"),
    )
    register_vector(conn)
    return conn


def fit_umap_3d(vectors: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric=METRIC,
        random_state=RANDOM_STATE,
    )
    return reducer.fit_transform(vectors)


def normalise_to_unit_cube(coords: np.ndarray) -> np.ndarray:
    """Scale coordinates so the cloud sits roughly inside [-1, 1] on every axis."""
    center = coords.mean(axis=0)
    centered = coords - center
    radius = np.max(np.linalg.norm(centered, axis=1))
    if radius == 0:
        return centered
    return centered / radius


def main():
    print("Connecting to Postgres…")
    conn = connect_db()
    with conn, conn.cursor() as cur:
        cur.execute("""
            SELECT filename, category, confidence, embedding
            FROM image_embeddings
            ORDER BY filename
        """)
        rows = cur.fetchall()

    if not rows:
        raise SystemExit(
            "No rows in image_embeddings - run ../embeddings/generate_embeddings.py first."
        )

    print(f"Read {len(rows)} embeddings from the database.")
    embeddings = np.array([row[3] for row in rows], dtype=np.float32)
    print(f"Embedding matrix shape: {embeddings.shape}")

    print(f"Running UMAP (n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST}, metric={METRIC})…")
    coords = fit_umap_3d(embeddings)
    coords = normalise_to_unit_cube(coords)
    print(f"3D coords shape: {coords.shape}")

    payload = [
        {
            "filename": filename,
            "category": category,
            "confidence": float(confidence) if confidence is not None else None,
            "x": float(coord[0]),
            "y": float(coord[1]),
            "z": float(coord[2]),
        }
        for (filename, category, confidence, _), coord in zip(rows, coords)
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Wrote {len(payload)} points → {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
