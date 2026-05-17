#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi",
#     "uvicorn[standard]",
#     "python-multipart",
#     "open_clip_torch",
#     "torch",
#     "pillow",
#     "psycopg[binary]>=3.2",
#     "pgvector>=0.3",
#     "python-dotenv>=1.0",
# ]
# ///
"""
FastAPI backend for the OpenCLIP search app.

- Loads OpenCLIP ViT-H-14 / dfn5b once at startup.
- /api/search/text   — POST { query, limit } → top-k similar images.
- /api/search/image  — POST multipart image → top-k similar images.
- /images/{filename} — serves the actual photo file from ../../images/.
- /api/health        — sanity check.

Run with:  uv run main.py
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import open_clip
import psycopg
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pgvector.psycopg import register_vector
from PIL import Image
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent.parent
IMAGE_DIR = REPO_ROOT / "images"

MODEL_NAME = "ViT-H-14"
PRETRAINED = "dfn5b"
DEFAULT_LIMIT = 12
MAX_LIMIT = 60

# Loaded at startup, cleared at shutdown.
state: dict[str, Any] = {}


def _connect_db() -> psycopg.Connection:
    load_dotenv(REPO_ROOT / "embeddings" / ".env")
    conn = psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        user=os.environ.get("POSTGRES_USER", "openclip"),
        password=os.environ.get("POSTGRES_PASSWORD", "openclip"),
        dbname=os.environ.get("POSTGRES_DB", "openclip"),
        autocommit=True,
    )
    register_vector(conn)
    return conn


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[startup] device={device}")
    print(f"[startup] loading {MODEL_NAME} / {PRETRAINED} …")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    state["device"] = device
    state["model"] = model
    state["preprocess"] = preprocess
    state["tokenizer"] = tokenizer
    state["db"] = _connect_db()
    print("[startup] ready")

    yield

    state["db"].close()
    state.clear()


app = FastAPI(lifespan=lifespan, title="OpenCLIP Search")

# Vite dev server runs on a different port. Allow it through.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextSearchRequest(BaseModel):
    query: str
    limit: int = DEFAULT_LIMIT


class SearchHit(BaseModel):
    filename: str
    category: str | None
    confidence: float | None
    distance: float
    url: str


def _query_neighbours(embedding, limit: int) -> list[SearchHit]:
    limit = max(1, min(MAX_LIMIT, limit))
    sql = """
        SELECT filename, category, confidence, embedding <#> %s AS distance
        FROM image_embeddings
        ORDER BY embedding <#> %s
        LIMIT %s
    """
    with state["db"].cursor() as cur:
        cur.execute(sql, (embedding, embedding, limit))
        rows = cur.fetchall()
    # pgvector's <#> returns negative inner product; smaller (more negative) is closer.
    # We pass it through as-is and let the client treat it as an ordering signal.
    return [
        SearchHit(
            filename=filename,
            category=category,
            confidence=confidence,
            distance=float(distance),
            url=f"/images/{filename}",
        )
        for filename, category, confidence, distance in rows
    ]


@torch.no_grad()
def _embed_text(text: str):
    tokens = state["tokenizer"]([text]).to(state["device"])
    feats = state["model"].encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy()


@torch.no_grad()
def _embed_image(img: Image.Image):
    tensor = state["preprocess"](img).unsqueeze(0).to(state["device"])
    feats = state["model"].encode_image(tensor)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy()


@app.get("/api/health")
def health():
    return {"status": "ok", "model": f"{MODEL_NAME}/{PRETRAINED}", "device": state.get("device")}


@app.post("/api/search/text", response_model=list[SearchHit])
def search_by_text(body: TextSearchRequest):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")
    embedding = _embed_text(query)
    return _query_neighbours(embedding, body.limit)


@app.post("/api/search/image", response_model=list[SearchHit])
async def search_by_image(
    file: UploadFile = File(...),
    limit: int = Form(DEFAULT_LIMIT),
):
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc
    embedding = _embed_image(img)
    return _query_neighbours(embedding, limit)


@app.get("/images/{filename}")
def get_image(filename: str):
    # Prevent path traversal.
    if "/" in filename or "\\" in filename or filename.startswith(".."):
        raise HTTPException(status_code=400, detail="invalid filename")
    path = IMAGE_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"not found: {filename}")
    return FileResponse(path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
