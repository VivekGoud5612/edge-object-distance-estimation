"""Dataset statistics for the 'review the data' step.

Things worth knowing BEFORE training:
  * instances per class (imbalance -> per-class AP will differ a lot),
  * box size distribution in pixels (tiny boxes are hard to detect AND give
    noisy distances, because 1 px of error on a 10 px box is 10 %),
  * images with no labels (negatives),
  * aspect ratios (a "barrier" that is 10x wider than tall is a different
    object than a cone, so expect the detector to need more data for it).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.classes import CLASS_NAMES
from src.data.yolo_labels import image_to_label_path, list_images, read_yolo_labels


def box_table(images_dir: str | Path, max_images: int | None = None) -> pd.DataFrame:
    """One row per box: image, class, width/height in pixels."""
    rows = []
    images = list_images(images_dir)[: max_images or None]
    for img in images:
        h, w = _image_size(img)
        for b in read_yolo_labels(image_to_label_path(img)):
            rows.append({"image": img.name, "source": img.name.split("__")[0],
                         "class": CLASS_NAMES[b.class_id] if b.class_id < len(CLASS_NAMES) else b.class_id,
                         "w_px": b.w * w, "h_px": b.h * h, "img_w": w, "img_h": h})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, tiny_px: float = 16) -> pd.DataFrame:
    """Per class: count, median box height, % of boxes shorter than `tiny_px`."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("class")
    return pd.DataFrame({
        "instances": g.size(),
        "median_h_px": g["h_px"].median().round(1),
        "p10_h_px": g["h_px"].quantile(0.10).round(1),
        f"pct_h<{tiny_px:g}px": (g["h_px"].apply(lambda s: (s < tiny_px).mean() * 100)).round(1),
        "median_aspect_w/h": (g.apply(lambda d: (d["w_px"] / d["h_px"].clip(lower=1)).median())).round(2),
    })


def _image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path))
    return (img.shape[0], img.shape[1]) if img is not None else (1, 1)


def sample_grid(images_dir: str | Path, out_path: str | Path, n: int = 16, seed: int = 0,
                tile: int = 320) -> Path:
    """Save a grid of random labelled images so you can eyeball label quality."""
    from src.types import Detection
    from src.visualize import draw_detections

    rng = np.random.default_rng(seed)
    images = list_images(images_dir)
    if not images:
        raise FileNotFoundError(f"no images in {images_dir}")
    picks = rng.choice(len(images), size=min(n, len(images)), replace=False)
    tiles = []
    for i in picks:
        img = cv2.imread(str(images[i]))
        h, w = img.shape[:2]
        dets = [Detection(b.class_id, 1.0, *b.to_xyxy(w, h))
                for b in read_yolo_labels(image_to_label_path(images[i]))]
        img = draw_detections(img, dets)
        tiles.append(cv2.resize(img, (tile, tile)))
    cols = int(np.ceil(np.sqrt(len(tiles))))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(len(tiles) // cols)])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    return out_path
