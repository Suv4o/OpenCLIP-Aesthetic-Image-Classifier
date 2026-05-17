# OpenCLIP Aesthetic Image Classifier

Sorts a folder of images into 6 themed categories using zero-shot classification with OpenCLIP (`ViT-L-14` / `datacomp_xl_s13b_b90k`). Runs on Apple Silicon via the MPS backend.

Categories:

- 🌌 Starlit Wonders
- 🌿 Wild Horizons
- ⏳ Time in Motion
- 🌆 Urban Glow
- 🌊 Ocean Whispers
- 🏞️ Liquid Cascades

## Prerequisites

- macOS (Apple Silicon recommended — falls back to CPU on other platforms)
- [`uv`](https://docs.astral.sh/uv/) — install with `brew install uv`

No `pip install`, no venv. Dependencies are declared inline in the script (PEP 723) and `uv` handles the rest.

## Usage

1. Put your images in the `images/` folder (subfolders are walked recursively).
2. Run:

   ```bash
   uv run aesthetic_classifier.py
   ```

First run downloads `torch`, `open_clip_torch`, etc. (~250 MB) and the CLIP weights (~1.7 GB). Subsequent runs reuse the cache and start instantly.

## Output

- **`classification_results.csv`** — one row per image with the winning category, confidence, top-2 margin, an `uncertain` flag (margin < 0.05), and the score for every category.
- **`sorted_output/<category>/`** — physical copies of each image grouped by predicted category. Set `COPY_INTO_SUBFOLDERS = False` in the script to skip this.

## Configuration

Edit the constants at the top of `aesthetic_classifier.py`:

| Constant | Purpose |
| --- | --- |
| `IMAGE_DIR` | Folder to scan (default: `./images`) |
| `OUTPUT_CSV` | Where to write the CSV |
| `COPY_INTO_SUBFOLDERS` | If `True`, also copy files into per-category folders |
| `COPY_DEST` | Destination for the copies |
| `MODEL_NAME` / `PRETRAINED` | OpenCLIP model + weights (see below) |
| `BATCH_SIZE` | Images per forward pass (default 32) |
| `NUM_WORKERS` | DataLoader workers (set 0 if you hit multiprocessing issues) |
| `UNCERTAIN_MARGIN` | Top1 − top2 probability gap below which an image is flagged uncertain |
| `CATEGORIES` | Map of display name → list of prompts. Multiple prompts are averaged (prompt ensembling). |

### Tuning the categories

CLIP doesn't understand poetic labels like "Starlit Wonders" directly — it matches against the **prompts**, not the display name. To improve accuracy, edit the prompt lists inside `CATEGORIES` to be more concrete and descriptive of what the photos actually look like. Re-run; iterate.

### Swapping the model

For maximum accuracy (slower, but fine on 64 GB):

```python
MODEL_NAME = "ViT-H-14"
PRETRAINED = "dfn5b"
```

For faster prototyping:

```python
MODEL_NAME = "ViT-B-16"
PRETRAINED = "datacomp_xl_s13b_b90k"
```
