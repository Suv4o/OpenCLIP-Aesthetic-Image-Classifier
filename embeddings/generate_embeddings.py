#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "open_clip_torch",
#     "torch",
#     "pillow",
#     "tqdm",
#     "psycopg[binary]>=3.2",
#     "pgvector>=0.3",
#     "python-dotenv>=1.0",
# ]
# ///
"""
Bulk indexer: encode every image in ../images/ with OpenCLIP ViT-H-14 / dfn5b
and upsert the 1024-dim L2-normalised embedding into pgvector.

Run after `docker compose up -d` in this directory.
"""

import csv
import os
from pathlib import Path

import open_clip
import psycopg
import torch
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
IMAGE_DIR = REPO_ROOT / "images"
CLASSIFICATION_CSV = REPO_ROOT / "classification" / "classification_results.csv"

MODEL_NAME = "ViT-H-14"
PRETRAINED = "dfn5b"
EMBED_DIM = 1024

BATCH_SIZE = 16   # H-14 is heavier than L-14 — keep batches modest on MPS
NUM_WORKERS = 4

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


class ImageFolderFlat(Dataset):
    def __init__(self, root: Path, transform):
        self.transform = transform
        self.paths = [
            p for p in sorted(Path(root).rglob("*"))
            if p.suffix.lower() in IMAGE_EXTS
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), str(path)
        except Exception:
            return None, str(path)


def collate(batch):
    good = [(img, p) for img, p in batch if img is not None]
    failed = [p for img, p in batch if img is None]
    if not good:
        return None, [], failed
    imgs, paths = zip(*good)
    return torch.stack(imgs), list(paths), failed


def load_classification_metadata() -> dict[str, tuple[str, float]]:
    """Map basename -> (category, confidence) from the classification CSV if present."""
    metadata: dict[str, tuple[str, float]] = {}
    if not CLASSIFICATION_CSV.exists():
        print(f"[note] No classification CSV at {CLASSIFICATION_CSV} — category/confidence will be NULL")
        return metadata
    with open(CLASSIFICATION_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = Path(row["path"]).name
            try:
                metadata[filename] = (row["category"], float(row["confidence"]))
            except (KeyError, ValueError):
                continue
    print(f"[note] Loaded classification metadata for {len(metadata)} images")
    return metadata


def connect_db() -> psycopg.Connection:
    load_dotenv(PROJECT_ROOT / ".env")
    conn = psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        user=os.environ.get("POSTGRES_USER", "openclip"),
        password=os.environ.get("POSTGRES_PASSWORD", "openclip"),
        dbname=os.environ.get("POSTGRES_DB", "openclip"),
    )
    register_vector(conn)
    return conn


UPSERT_SQL = """
INSERT INTO image_embeddings (filename, path, embedding, category, confidence, updated_at)
VALUES (%s, %s, %s, %s, %s, NOW())
ON CONFLICT (filename) DO UPDATE
    SET path       = EXCLUDED.path,
        embedding  = EXCLUDED.embedding,
        category   = EXCLUDED.category,
        confidence = EXCLUDED.confidence,
        updated_at = NOW();
"""


def main():
    if not IMAGE_DIR.exists():
        raise SystemExit(f"Image directory not found: {IMAGE_DIR}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {MODEL_NAME} / {PRETRAINED} (this can be slow on first run — ~3.8 GB)…")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()

    metadata = load_classification_metadata()

    dataset = ImageFolderFlat(IMAGE_DIR, preprocess)
    print(f"Found {len(dataset)} images under {IMAGE_DIR}")
    if len(dataset) == 0:
        raise SystemExit("No images found — drop files into ../images/ first.")

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate
    )

    print("Connecting to Postgres…")
    conn = connect_db()
    failed_paths: list[str] = []
    inserted = 0

    with conn, conn.cursor() as cur, torch.no_grad():
        for imgs, paths, failed in tqdm(loader, desc="Embedding"):
            failed_paths.extend(failed)
            if imgs is None:
                continue
            imgs = imgs.to(device)
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)   # L2-normalise
            feats_np = feats.cpu().numpy()                     # (B, 1024)

            rows = []
            for path, vec in zip(paths, feats_np):
                filename = Path(path).name
                cat, conf = metadata.get(filename, (None, None))
                rows.append((filename, path, vec, cat, conf))

            cur.executemany(UPSERT_SQL, rows)
            inserted += len(rows)

    print(f"\nUpserted {inserted} embeddings into image_embeddings.")
    if failed_paths:
        print(f"{len(failed_paths)} images failed to load:")
        for p in failed_paths[:10]:
            print(f"  - {p}")
        if len(failed_paths) > 10:
            print(f"  …and {len(failed_paths) - 10} more")


if __name__ == "__main__":
    main()
