"""Step 2b: raw YOLO vs fine-tuned YOLO on the same test sets.

    python -m scripts.evaluate \
        --models raw=yolo11n.pt ft_n=runs/detect/yolo11n_ft/weights/best.pt \
                 ft_s=runs/detect/yolo11s_ft/weights/best.pt

Every model is scored by the same evaluator (src/evaluation.py) on:
    * the merged test split (all 3 classes),
    * each source's test slice (test_<source>.txt: roadwork, coco_stop, ...).

The raw COCO model knows 80 classes. "stop sign" is mapped to our class 2,
everything else is dropped. It has never seen cones or barriers, so its AP
there is 0 by construction: that's the point of the comparison.

Writes results/eval_comparison.csv and .md (paste the .md into the README).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.classes import CLASS_NAMES, COCO80_TO_TARGET
from src.config import REPO_ROOT
from src.detector import Detector, DetectorConfig
from src.evaluation import evaluate
from src.reporting import save_table


def make_detector(weights: str, device: str, imgsz: int) -> Detector:
    det = Detector(DetectorConfig(weights=weights, device=device, imgsz=imgsz, conf=0.001))
    if len(det.names) == 80:  # a raw COCO model
        det.cfg.class_map = COCO80_TO_TARGET
    return det


def test_sets(data_dir: Path) -> dict[str, Path]:
    sets = {"merged_test": data_dir / "images" / "test"}
    for txt in sorted(data_dir.glob("test_*.txt")):
        sets[txt.stem.removeprefix("test_")] = txt
    return sets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, metavar="LABEL=WEIGHTS")
    ap.add_argument("--data", default="data/merged")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default="results/eval_comparison")
    args = ap.parse_args()

    sets = test_sets(REPO_ROOT / args.data)
    rows = []
    for spec in args.models:
        label, weights = spec.split("=", 1)
        det = make_detector(weights, args.device, args.imgsz)
        for set_name, path in sets.items():
            print(f"\n[{label}] on {set_name}")
            res = evaluate(det.predict, path, max_images=args.max_images)
            row = {"model": label, "test_set": set_name, **res.as_row()}
            for c in CLASS_NAMES:
                row[f"R_{c}"] = res.per_class[c]["recall"]
            rows.append(row)
            print(f"    mAP50 {res.map50:.3f}  mAP50-95 {res.map50_95:.3f}")

    df = pd.DataFrame(rows)
    csv_path, md_path = save_table(df, REPO_ROOT / args.out)
    print("\n" + df.round(3).to_string(index=False))
    print(f"\nsaved {csv_path.relative_to(REPO_ROOT)} and {md_path.name}")


if __name__ == "__main__":
    main()
