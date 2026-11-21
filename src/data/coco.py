"""COCO 2017 stop signs, without downloading the whole 19 GB image set.

Steps
  1. Download the annotation zip (~250 MB) once.
  2. Use pycocotools to find images that contain the categories we want
     (stop sign = category id 13 in the JSON; note it is index 11 in the
     80-class YOLO model, because COCO category ids have gaps).
  3. Download only those images by URL (in parallel), convert boxes to YOLO.

Crowd annotations (iscrowd=1, a single box around a group) are skipped.
"""

from __future__ import annotations

import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from src.data.download import download_file
from src.data.yolo_labels import YoloBox

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def ensure_annotations(raw_dir: str | Path) -> Path:
    """Download + unzip COCO annotations if needed; return the annotations/ folder.

    Safe to re-run: a complete zip already in raw_dir (e.g. fetched by hand
    with `wget -c -P data/raw/coco <ANNOTATIONS_URL>`) is reused; a truncated
    one is resumed.
    """
    raw_dir = Path(raw_dir)
    ann_dir = raw_dir / "annotations"
    if (ann_dir / "instances_train2017.json").exists() and (ann_dir / "instances_val2017.json").exists():
        return ann_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "annotations_trainval2017.zip"

    if zip_path.exists() and not _is_complete_zip(zip_path):
        # Truncated (interrupted wget or old run): turn it into a .part so it gets resumed.
        print(f"[coco] {zip_path.name} is incomplete, resuming it")
        zip_path.rename(zip_path.with_name(zip_path.name + ".part"))
    if not zip_path.exists():
        print(f"[coco] downloading annotations (~250 MB) -> {zip_path}")
        print(f"       (if this is slow, you can instead run: wget -c -P {raw_dir} {ANNOTATIONS_URL})")
        download_file(ANNOTATIONS_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        # Only the two detection files; the zip also has captions/keypoints.
        for member in ("annotations/instances_train2017.json", "annotations/instances_val2017.json"):
            zf.extract(member, raw_dir)
    return ann_dir


def _is_complete_zip(path: Path) -> bool:
    """A zip's index sits at the END of the file, so a truncated download fails this check."""
    try:
        with zipfile.ZipFile(path) as zf:
            return "annotations/instances_val2017.json" in zf.namelist()
    except zipfile.BadZipFile:
        return False


def _download(url_and_path: tuple[str, Path]) -> bool:
    url, path = url_and_path
    try:
        download_file(url, path, timeout=30, retries=3, progress=False)
        return True
    except Exception as e:  # network hiccup: skip the image, don't kill the whole run
        print(f"[coco] failed {url}: {e}")
        return False


def collect_coco_subset(ann_file: str | Path, categories: dict[int, int], image_dir: str | Path,
                        max_images: int | None = None, background_images: int = 0,
                        include_positives: bool = True, seed: int = 0, workers: int = 16):
    """Download the relevant images of one COCO split.

    Returns a list of (image_path, list[YoloBox]) pairs. Background images
    (none of our categories) get an empty box list.
    """
    from pycocotools.coco import COCO

    coco = COCO(str(ann_file))
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    positive_ids: set[int] = set()
    for cat_id in categories:
        positive_ids.update(coco.getImgIds(catIds=[cat_id]))
    positive = sorted(positive_ids) if include_positives else []
    if max_images:
        rng.shuffle(positive)
        positive = sorted(positive[:max_images])

    background: list[int] = []
    if background_images:
        pool = sorted(set(coco.getImgIds()) - positive_ids)
        background = sorted(rng.sample(pool, min(background_images, len(pool))))

    jobs, samples = [], []
    for img_id in positive + background:
        info = coco.loadImgs(img_id)[0]
        path = image_dir / info["file_name"]
        jobs.append((info["coco_url"], path))
        boxes = []
        if img_id in positive_ids:
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=list(categories), iscrowd=False)
            for ann in coco.loadAnns(ann_ids):
                x, y, w, h = ann["bbox"]
                if w < 1 or h < 1:
                    continue
                boxes.append(YoloBox.from_xywh_abs(categories[ann["category_id"]], x, y, w, h,
                                                   info["width"], info["height"]))
        samples.append((path, boxes))

    with ThreadPoolExecutor(workers) as pool:
        ok = list(tqdm(pool.map(_download, jobs), total=len(jobs), desc=f"coco {Path(ann_file).stem}"))
    return [s for s, good in zip(samples, ok) if good]
