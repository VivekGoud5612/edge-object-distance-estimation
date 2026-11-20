"""Measure how wrong the distance estimates are.

Mode 1: real ground truth (best evidence for the README)
    python -m scripts.validate_distance --weights best.pt --camera my_camera \
        --csv data/distance_photos/ground_truth.csv

  ground_truth.csv columns:  image,class,true_distance_m
      cone_2m.jpg,cone,2.0
      cone_4m.jpg,cone,4.0
      stop_6m.jpg,stop_sign,6.0
  Photo tips: tape-measure from the camera lens to the BASE of the object,
  camera at a fixed, measured height (put it in camera.yaml), level, one
  target object per photo, distances 2-10 m. Same camera/resolution you calibrated.

Mode 2: no ground truth, analytical sensitivity
    python -m scripts.validate_distance --sensitivity
  How the estimate moves if f or the real object height is off by +/-10 %,
  and how a 2 px box error grows with distance. Always worth a README table.

Outputs: results/distance_validation.csv/.md, results/distance_validation.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.classes import CLASS_NAMES, NAME_TO_ID
from src.config import REPO_ROOT, load_distance_config
from src.detector import Detector, DetectorConfig
from src.distance import DistanceEstimator, pinhole_distance
from src.reporting import save_table


def validate(args) -> None:
    cfg = load_distance_config(profile=args.camera)
    est = DistanceEstimator(cfg)
    det = Detector(DetectorConfig(weights=args.weights, device=args.device, conf=0.2))
    csv_path = REPO_ROOT / args.csv
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            img = cv2.imread(str(csv_path.parent / r["image"]))
            if img is None:
                print(f"missing {r['image']}")
                continue
            cls = NAME_TO_ID[r["class"]]
            cands = [d for d in det.predict(img) if d.class_id == cls]
            true = float(r["true_distance_m"])
            if not cands:
                rows.append({"image": r["image"], "class": r["class"], "true_m": true, "detected": False})
                continue
            d = max(cands, key=lambda d: d.confidence)    # the target object
            e = est.estimate(d, img.shape[1], img.shape[0])
            row = {"image": r["image"], "class": r["class"], "true_m": true, "detected": True,
                   "box_h_px": d.height, "pinhole_m": e.pinhole_m, "ground_m": e.ground_m,
                   "pred_m": e.distance_m}
            for key in ("pinhole_m", "ground_m"):
                if row[key] is not None:
                    row[key.replace("_m", "_rel_err")] = (row[key] - true) / true
            rows.append(row)

    df = pd.DataFrame(rows)
    save_table(df, REPO_ROOT / "results" / "distance_validation")
    ok = df[df.detected]
    print(df.round(3).to_string(index=False))
    for key in ("pinhole", "ground"):
        col = f"{key}_rel_err"
        if col in ok and ok[col].notna().any():
            e = ok[col].dropna()
            print(f"{key:8s}: MAE {np.mean(np.abs(ok[key + '_m'] - ok.true_m)):.2f} m, "
                  f"mean |rel err| {e.abs().mean() * 100:.1f} %, bias {e.mean() * 100:+.1f} %")
    _plot(ok, REPO_ROOT / "results" / "distance_validation.png")


def _plot(df: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    lim = [0, df.true_m.max() * 1.2]
    ax.plot(lim, lim, "k--", lw=1, label="perfect")
    ax.scatter(df.true_m, df.pinhole_m, label="pinhole")
    if df.ground_m.notna().any():
        ax.scatter(df.true_m, df.ground_m, marker="x", label="ground plane")
    ax.set_xlabel("measured distance (m)")
    ax.set_ylabel("estimated distance (m)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def sensitivity(args) -> None:
    """Analytical error table for the pinhole formula, per class."""
    cfg = load_distance_config(profile=args.camera)
    f = cfg.camera.intrinsics.fy
    rows = []
    for name in CLASS_NAMES:
        H = cfg.objects[NAME_TO_ID[name]].height_m
        for true_d in (2, 5, 10, 20, 40):
            h_px = f * H / true_d                       # box height we'd see at that distance
            rows.append({
                "class": name, "true_m": true_d, "box_h_px": round(h_px, 1),
                "f+10%": pinhole_distance(h_px, H, f * 1.1),
                "H-10%": pinhole_distance(h_px, H * 0.9, f),
                "box+2px": pinhole_distance(h_px + 2, H, f),
                "box-2px": pinhole_distance(max(h_px - 2, 1), H, f),
            })
    df = pd.DataFrame(rows)
    save_table(df, REPO_ROOT / "results" / "distance_sensitivity")
    print(f"camera {cfg.camera.name}: f = {f:.0f} px\n")
    print(df.round(2).to_string(index=False))
    print("\nReading: f and H errors scale the distance by the same %, at every range.\n"
          "A fixed 2 px box error is negligible up close but dominates far away.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights")
    ap.add_argument("--csv")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--sensitivity", action="store_true")
    args = ap.parse_args()
    if args.sensitivity:
        sensitivity(args)
    elif args.weights and args.csv:
        validate(args)
    else:
        ap.error("give --weights and --csv, or --sensitivity")


if __name__ == "__main__":
    main()
