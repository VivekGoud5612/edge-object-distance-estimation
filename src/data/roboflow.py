"""Download a Roboflow Universe dataset in YOLO format.

Needs a free Roboflow account. Put the key in the environment, never in git:

    export ROBOFLOW_API_KEY=xxxxxxxx

The export has this layout (Roboflow calls the validation split "valid"):

    <dir>/data.yaml            names: [...]
    <dir>/train/images, <dir>/train/labels
    <dir>/valid/images, ...
    <dir>/test/images, ...
"""

from __future__ import annotations

import os
from pathlib import Path

from src.config import load_yaml

# Roboflow split name -> our split name
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def download_roboflow(workspace: str, project: str, version: int, dest: str | Path,
                      fmt: str = "yolov8") -> Path:
    """Download (or reuse) a dataset version. Returns the folder it lives in."""
    dest = Path(dest)
    if (dest / "data.yaml").exists():
        print(f"[roboflow] already downloaded: {dest}")
        return dest

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Create a free account at roboflow.com, copy the "
            "private API key from Settings, then: export ROBOFLOW_API_KEY=...\n"
            "Alternative: download the zip from the dataset page by hand and use a "
            "'local_yolo' source in configs/data_sources.yaml."
        )

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version)
    ds.download(fmt, location=str(dest), overwrite=False)
    return dest


def read_roboflow_names(dataset_dir: str | Path) -> list[str]:
    """Class names in id order, from the export's data.yaml."""
    names = load_yaml(Path(dataset_dir) / "data.yaml")["names"]
    if isinstance(names, dict):  # newer exports use {0: name, ...}
        names = [names[k] for k in sorted(names)]
    return list(names)


def iter_roboflow_splits(dataset_dir: str | Path):
    """Yield (our_split, images_dir) for every split present in the export."""
    dataset_dir = Path(dataset_dir)
    for rf_split, split in SPLIT_ALIASES.items():
        img_dir = dataset_dir / rf_split / "images"
        if img_dir.is_dir():
            yield split, img_dir
