"""Generate a tiny fake dataset to test the whole pipeline without downloads.

    python -m scripts.make_synthetic_data --out data/raw/synthetic --n 60
    python -m scripts.prepare_data --config tests/data_sources_synthetic.yaml

Draws orange triangles ("cones"), yellow-black bars ("barriers") and red
octagons ("stop signs") on noisy backgrounds, in YOLO format with
train/val/test splits. A model trained on it tells you nothing about real
accuracy; it only proves every script runs end to end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.data.yolo_labels import YoloBox, write_yolo_labels


def draw_object(img, cls, x, y, s, rng):
    if cls == 0:   # cone: orange triangle with a white stripe
        pts = np.array([[x + s // 2, y], [x, y + s], [x + s, y + s]], np.int32)
        cv2.fillPoly(img, [pts], (0, 120, 255))
        cv2.line(img, (x + s // 4, y + s // 2), (x + 3 * s // 4, y + s // 2), (255, 255, 255), max(1, s // 10))
        return x, y, s, s
    if cls == 1:   # barrier: wide yellow bar with black stripes
        w, h = int(s * 2.5), s // 2
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 220, 255), -1)
        for k in range(0, w, max(4, s // 3)):
            cv2.line(img, (x + k, y), (x + k + h, y + h), (0, 0, 0), max(1, s // 12))
        return x, y, w, h
    # stop sign: red octagon
    c = (x + s // 2, y + s // 2)
    pts = np.array([[c[0] + s / 2 * np.cos(a), c[1] + s / 2 * np.sin(a)]
                    for a in np.linspace(np.pi / 8, 2 * np.pi + np.pi / 8, 8, endpoint=False)], np.int32)
    cv2.fillPoly(img, [pts], (30, 30, 210))
    return x, y, s, s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/synthetic")
    ap.add_argument("--n", type=int, default=60, help="images per split for train (val/test get n/4)")
    ap.add_argument("--size", nargs=2, type=int, default=[640, 360])
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    W, H = args.size
    for split, n in (("train", args.n), ("val", max(4, args.n // 4)), ("test", max(4, args.n // 4))):
        for i in range(n):
            img = rng.integers(60, 140, (H, W, 3), dtype=np.uint8)
            cv2.rectangle(img, (0, H // 2), (W, H), (70, 70, 70), -1)   # "road"
            boxes = []
            for _ in range(rng.integers(1, 4)):
                cls = int(rng.integers(0, 3))
                s = int(rng.integers(20, 90))
                x, y = int(rng.integers(0, W - 3 * s)), int(rng.integers(H // 3, H - s))
                bx, by, bw, bh = draw_object(img, cls, x, y, s, rng)
                boxes.append(YoloBox.from_xywh_abs(cls, bx, by, bw, bh, W, H))
            out = Path(args.out)
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "images" / split / f"syn_{i:04d}.jpg"), img)
            write_yolo_labels(out / "labels" / split / f"syn_{i:04d}.txt", boxes)
    print(f"wrote synthetic dataset to {args.out}")


if __name__ == "__main__":
    main()
