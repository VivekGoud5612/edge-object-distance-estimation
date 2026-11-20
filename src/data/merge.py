"""Merge several remapped sources into one YOLO dataset.

Output layout (what Ultralytics expects):

    data/merged/
        images/{train,val,test}/<source>__<original_name>.jpg
        labels/{train,val,test}/<source>__<original_name>.txt
        data.yaml                 -> used by scripts/train.py
        test_<source>.txt         -> per-source test lists (e.g. "cone test")
        stats.csv                 -> instance counts per split/source/class

Design choices
  * File names get a "<source>__" prefix, so two datasets that both have
    "img_001.jpg" can't overwrite each other, and every file can be traced
    back to where it came from.
  * Source splits are respected when the source has them (Roboflow does).
    Re-splitting would leak near-duplicate frames and Roboflow's augmented
    copies from train into test, which inflates mAP.
  * Images are hard-linked when possible (no extra disk), copied otherwise.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.classes import CLASS_NAMES
from src.config import save_yaml
from src.data.yolo_labels import YoloBox, write_yolo_labels

SPLITS = ("train", "val", "test")


@dataclass
class Sample:
    source: str
    split: str               # train | val | test
    image_path: Path
    boxes: list[YoloBox]     # already remapped to our class ids


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:               # different filesystem, etc.
        shutil.copy2(src, dst)


def write_merged_dataset(samples: list[Sample], out_dir: str | Path,
                         keep_empty: bool = False) -> dict:
    """Write images + labels + data.yaml. Returns a small summary dict.

    keep_empty=False drops images whose boxes were all removed by the remap
    (e.g. a Roboflow image that only had "Drum"). Intentional background
    images (COCO negatives) are marked with source suffix "_bg" and kept.
    """
    out_dir = Path(out_dir).resolve()
    counts: Counter = Counter()          # (split, source, class_name) -> instances
    image_counts: Counter = Counter()    # (split, source) -> images
    per_source_test: dict[str, list[str]] = {}
    dropped = 0

    for s in samples:
        if s.split not in SPLITS:
            raise ValueError(f"unknown split {s.split!r}")
        is_background = s.source.endswith("_bg")
        if not s.boxes and not (keep_empty or is_background):
            dropped += 1
            continue
        name = f"{s.source}__{s.image_path.name}"
        img_dst = out_dir / "images" / s.split / name
        _link_or_copy(s.image_path, img_dst)
        write_yolo_labels(out_dir / "labels" / s.split / Path(name).with_suffix(".txt"), s.boxes)

        image_counts[(s.split, s.source)] += 1
        for b in s.boxes:
            counts[(s.split, s.source, CLASS_NAMES[b.class_id])] += 1
        if s.split == "test":
            per_source_test.setdefault(s.source, []).append(str(img_dst))

    for source, paths in per_source_test.items():
        (out_dir / f"test_{source}.txt").write_text("\n".join(sorted(paths)) + "\n")

    # data.yaml uses an absolute path: Ultralytics resolves relative paths
    # against its own datasets dir, which is a common source of confusion.
    # (This file is machine-specific, which is why it lives in data/, not configs/.)
    save_yaml({"path": str(out_dir), "train": "images/train", "val": "images/val",
               "test": "images/test", "names": dict(enumerate(CLASS_NAMES))},
              out_dir / "data.yaml")

    _write_stats_csv(out_dir / "stats.csv", counts, image_counts)
    return {"images": dict(image_counts), "instances": dict(counts), "dropped_empty": dropped,
            "data_yaml": str(out_dir / "data.yaml")}


def _write_stats_csv(path: Path, counts: Counter, image_counts: Counter) -> None:
    rows = ["split,source,images," + ",".join(CLASS_NAMES)]
    for split, source in sorted(image_counts):
        per_cls = [str(counts.get((split, source, c), 0)) for c in CLASS_NAMES]
        rows.append(f"{split},{source},{image_counts[(split, source)]}," + ",".join(per_cls))
    path.write_text("\n".join(rows) + "\n")


def split_by_fraction(items: list, val_fraction: float, seed: int) -> tuple[list, list]:
    """Deterministic shuffle-split (for sources that only have one split)."""
    import random

    items = list(items)
    random.Random(seed).shuffle(items)
    n_val = int(round(len(items) * val_fraction))
    return items[n_val:], items[:n_val]
