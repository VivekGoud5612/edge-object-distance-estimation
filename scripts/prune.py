"""Optional: magnitude pruning experiment.

    python -m scripts.prune --weights runs/detect/yolo11n_ft/weights/best.pt --amounts 0.1 0.2 0.3 0.5

What it does
  For every Conv2d layer, set the smallest-magnitude X % of weights to zero
  (L1 *unstructured* pruning, torch.nn.utils.prune), then measure mAP and
  speed. Results go to results/pruning.csv/.md.

What to expect, and say honestly in the README
  * mAP drops slowly up to ~20-30 % sparsity, then falls off a cliff.
  * Speed does NOT improve. The zeros are still stored and multiplied:
    dense GPU/CPU kernels don't skip them. Real speed-ups need *structured*
    pruning (removing whole channels, so layers actually get smaller) or
    sparse-aware runtimes. That's why quantisation and a smaller input size
    are the practical levers in this project; pruning is here to show we
    tried it and understand why it doesn't pay off out of the box.
  * File size doesn't shrink either (unless you compress the file: zeros zip well).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT
from src.detector import Detector, DetectorConfig
from src.evaluation import evaluate
from src.reporting import save_table


def prune_model(weights: str, amount: float, out_path: Path) -> float:
    """Prune all Conv2d weights by `amount`, save to out_path, return achieved sparsity."""
    import torch
    import torch.nn.utils.prune as prune
    from ultralytics import YOLO

    model = YOLO(weights)
    total = zeros = 0
    for module in model.model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")         # make the zeros permanent (drop the mask)
            total += module.weight.numel()
            zeros += int((module.weight == 0).sum())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    return zeros / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--amounts", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--images", default="data/merged/images/test")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640, help="use the size the model was trained at")
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    from src.timing import time_calls
    import cv2
    from src.data.yolo_labels import list_images

    img_dir = REPO_ROOT / args.images
    sample = cv2.imread(str(list_images(img_dir)[0]))
    rows = []
    for amount in [0.0, *args.amounts]:
        if amount == 0.0:
            path, sparsity = str(REPO_ROOT / args.weights), 0.0
        else:
            out = REPO_ROOT / "models" / f"pruned_{int(amount * 100)}.pt"
            sparsity = prune_model(str(REPO_ROOT / args.weights), amount, out)
            path = str(out)
        det = Detector(DetectorConfig(weights=path, device=args.device, imgsz=args.imgsz, conf=0.001))
        res = evaluate(det.predict, img_dir, max_images=args.max_images, show_progress=False)
        det.cfg.conf = 0.25
        speed = time_calls(lambda d=det: d.predict(sample), runs=100, warmup=10)
        rows.append({"prune_amount": amount, "conv_sparsity": sparsity, "mAP50": res.map50,
                     "mAP50-95": res.map50_95, "mean_ms": speed.mean_ms, "fps": speed.fps})
        print(rows[-1])

    csv_path, _ = save_table(pd.DataFrame(rows), REPO_ROOT / "results" / "pruning")
    print(f"saved {csv_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
