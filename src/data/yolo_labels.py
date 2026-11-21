"""YOLO label files: read, write, remap.

YOLO format = one .txt per image, same stem, one line per object:

    <class_id> <x_center> <y_center> <width> <height>

with all four coordinates normalised to 0..1 by image width/height.
Images live in .../images/<split>/ and labels in .../labels/<split>/.

"Remap" = translate each source dataset's own class ids into our 3 ids
(cone 0, barrier 1, stop_sign 2) and drop every other class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    xc: float
    yc: float
    w: float
    h: float

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Normalised centre format -> absolute corner pixels."""
        return ((self.xc - self.w / 2) * img_w, (self.yc - self.h / 2) * img_h,
                (self.xc + self.w / 2) * img_w, (self.yc + self.h / 2) * img_h)

    @classmethod
    def from_xywh_abs(cls, class_id: int, x: float, y: float, w: float, h: float,
                      img_w: int, img_h: int) -> "YoloBox":
        """COCO-style [x_min, y_min, w, h] pixels -> normalised YOLO box (clipped to the image)."""
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(img_w), x + w), min(float(img_h), y + h)
        bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
        return cls(class_id, (x1 + bw / 2) / img_w, (y1 + bh / 2) / img_h, bw / img_w, bh / img_h)

    def to_line(self) -> str:
        return f"{self.class_id} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"


def read_yolo_labels(path: str | Path) -> list[YoloBox]:
    """Read a label file. A missing or empty file means 'no objects' (a background image)."""
    path = Path(path)
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        # Polygon (segmentation) labels have >5 numbers; we only need the box,
        # so take the polygon's bounding rectangle.
        cls_id = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
        if len(coords) == 4:
            boxes.append(YoloBox(cls_id, *coords))
        else:
            xs, ys = coords[0::2], coords[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            boxes.append(YoloBox(cls_id, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return boxes


def write_yolo_labels(path: str | Path, boxes: list[YoloBox]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(b.to_line() + "\n" for b in boxes))


def remap_boxes(boxes: list[YoloBox], id_map: dict[int, int]) -> list[YoloBox]:
    """Keep only classes in `id_map`, renumbered. Everything else is dropped."""
    return [YoloBox(id_map[b.class_id], b.xc, b.yc, b.w, b.h) for b in boxes if b.class_id in id_map]


def build_id_map(source_names: list[str] | dict[int, str],
                 name_map: dict[str, str | None],
                 target_names: list[str]) -> dict[int, int]:
    """Turn a *name*-based mapping from the config into an *id*-based one.

    We map by name (case-insensitive) rather than by id because each Roboflow
    export numbers classes alphabetically, so ids change between datasets and
    versions; names don't.

        source_names = ["Barrier", "Cone", "Drum"]
        name_map     = {"cone": "cone", "barrier": "barrier", "drum": null}
        -> {0: 1, 1: 0}      (Drum dropped)
    """
    if isinstance(source_names, dict):
        items = sorted(source_names.items())
    else:
        items = list(enumerate(source_names))
    lookup = {k.lower(): v for k, v in name_map.items()}
    target_index = {n: i for i, n in enumerate(target_names)}
    id_map: dict[int, int] = {}
    for src_id, src_name in items:
        tgt = lookup.get(str(src_name).lower())
        if tgt is not None:
            id_map[int(src_id)] = target_index[tgt]
    return id_map


def list_images(source: str | Path) -> list[Path]:
    """All images in a folder (sorted), or the paths listed in a .txt file."""
    source = Path(source)
    if source.suffix == ".txt":
        base = source.parent
        paths = [Path(line.strip()) for line in source.read_text().splitlines() if line.strip()]
        return [p if p.is_absolute() else (base / p).resolve() for p in paths]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def image_to_label_path(image_path: str | Path) -> Path:
    """.../images/<split>/x.jpg -> .../labels/<split>/x.txt (the YOLO convention)."""
    parts = list(Path(image_path).parts)
    idx = len(parts) - 1 - parts[::-1].index("images")  # last 'images' in the path
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")
