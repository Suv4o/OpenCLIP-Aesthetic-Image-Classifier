#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "open_clip_torch",
#     "torch",
#     "pillow",
#     "tqdm",
# ]
# ///
"""
Aesthetic image classifier using OpenCLIP on Apple Silicon (M1 Max).
Sorts a folder of images into 6 themed categories.
"""

import csv
import shutil
from collections import Counter
from pathlib import Path

import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_ROOT.parent / "images"
OUTPUT_CSV = PROJECT_ROOT / "classification_results.csv"
COPY_INTO_SUBFOLDERS = True
COPY_DEST = PROJECT_ROOT / "sorted_output"

MODEL_NAME = "ViT-L-14"
PRETRAINED = "datacomp_xl_s13b_b90k"

BATCH_SIZE = 32
NUM_WORKERS = 6
UNCERTAIN_MARGIN = 0.05

CATEGORIES = {
    "🌌 Starlit Wonders": [
        "a photo of the night sky full of stars",
        "astrophotography of the milky way",
        "a starry night sky over a landscape",
        "stars and galaxies in the night sky",
        "a long exposure photo of star trails",
    ],
    "🌿 Wild Horizons": [
        "a landscape photo of wild nature",
        "a scenic view of mountains and forests",
        "an untouched wilderness landscape",
        "a wide nature vista with green hills",
        "a national park scenery photo",
    ],
    "⏳ Time in Motion": [
        "a long exposure photograph showing motion blur",
        "light trails from moving traffic at night",
        "a photo with motion blur showing the passage of time",
        "a time-lapse style long exposure photograph",
        "blurred movement captured in a single photograph",
    ],
    "🌆 Urban Glow": [
        "a city skyline glowing at night",
        "an urban street with neon lights at night",
        "city lights and skyscrapers after dark",
        "a nighttime cityscape photo",
        "a glowing urban scene at dusk",
    ],
    "🌊 Ocean Whispers": [
        "a calm ocean seascape",
        "waves on the sea coastline",
        "a photo of the open ocean",
        "a tranquil beach and sea view",
        "the surface of the ocean at golden hour",
    ],
    "🏞️ Liquid Cascades": [
        "a photo of a waterfall",
        "water cascading down rocks",
        "a flowing river with rapids",
        "a tall waterfall in a forest",
        "cascading water in nature",
    ],
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


class ImageFolderFlat(Dataset):
    def __init__(self, root, transform):
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


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {MODEL_NAME} / {PRETRAINED} ...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    @torch.no_grad()
    def build_text_embeddings():
        names, embeddings = [], []
        for name, prompts in CATEGORIES.items():
            tokens = tokenizer(prompts).to(device)
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            mean = feats.mean(dim=0)
            mean = mean / mean.norm()
            names.append(name)
            embeddings.append(mean)
        return names, torch.stack(embeddings)

    category_names, text_embeds = build_text_embeddings()
    print(f"Built {len(category_names)} category embeddings.")

    dataset = ImageFolderFlat(IMAGE_DIR, preprocess)
    print(f"Found {len(dataset)} images under {IMAGE_DIR}")
    if len(dataset) == 0:
        raise SystemExit("No images found -- check IMAGE_DIR.")

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate
    )

    results = []
    failed_paths = []

    with torch.no_grad():
        for imgs, paths, failed in tqdm(loader, desc="Classifying"):
            failed_paths.extend(failed)
            if imgs is None:
                continue
            imgs = imgs.to(device)
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

            sims = feats @ text_embeds.T
            probs = (100.0 * sims).softmax(dim=-1)
            top = probs.topk(2, dim=-1)

            for i, path in enumerate(paths):
                best_idx = top.indices[i, 0].item()
                best_p = top.values[i, 0].item()
                second_p = top.values[i, 1].item()
                results.append({
                    "path": path,
                    "category": category_names[best_idx],
                    "confidence": round(best_p, 4),
                    "margin": round(best_p - second_p, 4),
                    "uncertain": (best_p - second_p) < UNCERTAIN_MARGIN,
                    "all_scores": {
                        category_names[j]: round(probs[i, j].item(), 4)
                        for j in range(len(category_names))
                    },
                })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["path", "category", "confidence", "margin", "uncertain"] + category_names
        )
        for r in results:
            writer.writerow(
                [r["path"], r["category"], r["confidence"], r["margin"], r["uncertain"]]
                + [r["all_scores"][name] for name in category_names]
            )

    print(f"\nWrote {len(results)} rows to {OUTPUT_CSV}")
    if failed_paths:
        print(f"{len(failed_paths)} images failed to load.")

    if COPY_INTO_SUBFOLDERS:
        for r in results:
            cat_folder = COPY_DEST / r["category"]
            cat_folder.mkdir(parents=True, exist_ok=True)
            src = Path(r["path"])
            shutil.copy2(src, cat_folder / src.name)
        print(f"Copied images into category subfolders under {COPY_DEST}")

    counts = Counter(r["category"] for r in results)
    uncertain = sum(1 for r in results if r["uncertain"])
    print("\nSummary:")
    for name in category_names:
        print(f"  {name}: {counts.get(name, 0)}")
    print(f"\n  {uncertain} images flagged 'uncertain' (low margin) -- worth a manual look.")


if __name__ == "__main__":
    main()
