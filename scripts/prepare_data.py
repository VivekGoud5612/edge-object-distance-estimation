"""Step 1 of the offline flow: download -> remap -> merge -> split.

    python -m scripts.prepare_data                     # everything enabled in the config
    python -m scripts.prepare_data --only roadwork     # one source
    python -m scripts.prepare_data --skip-download     # re-merge what's already in data/raw

Reads configs/data_sources.yaml, writes data/merged/ (see src/data/merge.py).
Then run `python -m scripts.review_data` and LOOK at the output before training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.classes import CLASS_NAMES
from src.config import REPO_ROOT, load_yaml
from src.data import coco, roboflow
from src.data.merge import Sample, fill_missing_splits, split_by_fraction, write_merged_dataset
from src.data.yolo_labels import (build_id_map, image_to_label_path, list_images,
                                  read_yolo_labels, remap_boxes)


def samples_from_yolo_dir(source: str, split: str, images_dir: Path,
                          id_map: dict[int, int]) -> list[Sample]:
    """Read a YOLO-format folder and remap its labels to our class ids."""
    out = []
    for img in list_images(images_dir):
        boxes = remap_boxes(read_yolo_labels(image_to_label_path(img)), id_map)
        out.append(Sample(source, split, img, boxes))
    return out


def prepare_roboflow(name: str, cfg: dict, raw_dir: Path, download: bool,
                     split_cfg: dict, seed: int) -> list[Sample]:
    dest = raw_dir / name
    if download:
        roboflow.download_roboflow(cfg["workspace"], cfg["project"], int(cfg["version"]),
                                   dest, cfg.get("format", "yolov8"))
    src_names = roboflow.read_roboflow_names(dest)
    id_map = build_id_map(src_names, cfg["class_map"], CLASS_NAMES)
    _print_mapping(name, src_names, id_map)
    samples = []
    for split, img_dir in roboflow.iter_roboflow_splits(dest):
        samples += samples_from_yolo_dir(name, split, img_dir, id_map)
    _fill_splits(name, samples, cfg, split_cfg, seed)
    return samples


def _fill_splits(name: str, samples: list[Sample], cfg: dict, split_cfg: dict, seed: int) -> None:
    """Many Roboflow exports only have 'train'. Without this, the source would
    have no test images at all, i.e. no way to measure cone/barrier accuracy."""
    # Images that will be dropped anyway (none of our classes) shouldn't count.
    kept = [s for s in samples if s.boxes]
    info = fill_missing_splits(kept, float(split_cfg.get("val", 0.1)), float(split_cfg.get("test", 0.1)),
                               seed, cfg.get("group_regex"), cfg.get("group_block"))
    print(f"[{name}] splits: train {info['train']}, val {info['val']}, test {info['test']} "
          f"({info['groups']} groups; group_regex={cfg.get('group_regex')!r}, "
          f"group_block={cfg.get('group_block')})")


def prepare_coco(name: str, cfg: dict, raw_dir: Path, seed: int) -> list[Sample]:
    coco_dir = raw_dir / "coco"
    ann_dir = coco.ensure_annotations(coco_dir)
    categories = {int(k): CLASS_NAMES.index(v) for k, v in cfg["categories"].items()}

    # COCO train2017 -> our train + val
    train_pairs = coco.collect_coco_subset(
        ann_dir / "instances_train2017.json", categories, coco_dir / "train2017",
        max_images=cfg.get("max_train_images"), background_images=0, seed=seed)
    tr, va = split_by_fraction(train_pairs, float(cfg.get("val_fraction", 0.15)), seed)
    samples = [Sample(name, "train", p, b) for p, b in tr] + [Sample(name, "val", p, b) for p, b in va]

    # COCO val2017 -> our test (never seen during training or model selection)
    test_pairs = coco.collect_coco_subset(
        ann_dir / "instances_val2017.json", categories, coco_dir / "val2017", seed=seed)
    samples += [Sample(name, "test", p, b) for p, b in test_pairs]

    # Background (negative) images: teach the model that "no object" is an answer.
    n_bg = int(cfg.get("background_images", 0))
    if n_bg:
        all_pairs = coco.collect_coco_subset(
            ann_dir / "instances_train2017.json", categories, coco_dir / "train2017",
            background_images=n_bg, include_positives=False, seed=seed + 1)
        samples += [Sample(f"{name}_bg", "train", p, b) for p, b in all_pairs if not b]
    return samples


def prepare_local(name: str, cfg: dict, split_cfg: dict, seed: int) -> list[Sample]:
    root = REPO_ROOT / cfg["path"]
    id_map = build_id_map(cfg["names"], cfg["class_map"], CLASS_NAMES)
    _print_mapping(name, cfg["names"], id_map)
    samples = []
    split_dirs = {s: root / "images" / s for s in ("train", "val", "test")}
    if any(d.is_dir() for d in split_dirs.values()):
        for split, d in split_dirs.items():
            if d.is_dir():
                samples += samples_from_yolo_dir(name, split, d, id_map)
    else:
        # Unsplit folder: everything starts as train, then val/test are carved out by group.
        samples = samples_from_yolo_dir(name, "train", root / "images", id_map)
    _fill_splits(name, samples, cfg, split_cfg, seed)
    return samples


def _print_mapping(name: str, src_names, id_map: dict[int, int]) -> None:
    names = src_names if isinstance(src_names, list) else [src_names[k] for k in sorted(src_names)]
    print(f"[{name}] class mapping:")
    for i, n in enumerate(names):
        tgt = CLASS_NAMES[id_map[i]] if i in id_map else "(dropped)"
        print(f"    {i:>2} {n:<30} -> {tgt}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data_sources.yaml")
    ap.add_argument("--only", nargs="*", help="source names to include (default: all enabled)")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    raw_dir = REPO_ROOT / cfg["raw_dir"]
    out_dir = REPO_ROOT / cfg["output_dir"]
    seed = int(cfg.get("seed", 0))
    split_cfg = cfg.get("split_missing", {"val": 0.1, "test": 0.1})

    samples: list[Sample] = []
    for name, src in cfg["sources"].items():
        if args.only and name not in args.only:
            continue
        if not args.only and not src.get("enabled", True):
            continue
        kind = src["type"]
        print(f"\n=== {name} ({kind}) ===")
        if kind == "roboflow":
            samples += prepare_roboflow(name, src, raw_dir, not args.skip_download, split_cfg, seed)
        elif kind == "coco":
            samples += prepare_coco(name, src, raw_dir, seed)
        elif kind == "local_yolo":
            samples += prepare_local(name, src, split_cfg, seed)
        else:
            raise ValueError(f"unknown source type {kind!r}")

    summary = write_merged_dataset(samples, out_dir)
    print(f"\nwrote {out_dir}")
    print(f"dropped {summary['dropped_empty']} images that had none of our classes after remap")
    print((out_dir / "stats.csv").read_text())
    print(f"next: python -m scripts.review_data   (then train with data={summary['data_yaml']})")


if __name__ == "__main__":
    main()
