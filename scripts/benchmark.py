"""Step 4: speed + accuracy of every exported variant, on CPU and GPU.

    python -m scripts.benchmark                        # everything in models/manifest.yaml
    python -m scripts.benchmark --devices cpu          # CPU only
    python -m scripts.benchmark --only yolo11n_ft_onnx_640 --runs 300
    python -m scripts.benchmark --no-accuracy          # speed only (much faster)

Method (same for every row, so rows are comparable)
  * images: the first --runs images of the merged test split, cycled,
  * batch 1, --warmup untimed calls first,
  * timing = end-to-end Detector.predict(): pre-process + inference + NMS +
    copying boxes back to the CPU. That's what the robot actually pays.
    Ultralytics' own per-stage breakdown is reported too (infer_ms column).
  * accuracy = mAP on the merged test split via src/evaluation.py, conf=0.001.

Output: results/benchmark.csv/.md and results/benchmark_tradeoff.png
(accuracy vs FPS: the "pick your model for your device" chart).
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_yaml
from src.data.yolo_labels import list_images
from src.detector import Detector, DetectorConfig, model_size_mb
from src.evaluation import evaluate
from src.reporting import save_table
from src.timing import time_calls


def hardware_info() -> dict[str, str]:
    info = {"cpu": platform.processor() or platform.machine(), "gpu": "none"}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def benchmark_one(entry: dict, device: str, images: list[np.ndarray], runs: int, warmup: int,
                  accuracy_dir: Path | None, max_eval: int | None) -> dict:
    weights = str(REPO_ROOT / entry["path"])
    half = entry["precision"] == "fp16" and entry["format"] == "pt"
    det = Detector(DetectorConfig(weights=weights, device=device, imgsz=entry["imgsz"], half=half))

    sync = None
    if device != "cpu":
        import torch
        sync = torch.cuda.synchronize

    # Cycle through the images so every call does real work on varied input.
    it = {"i": 0}
    infer_ms: list[float] = []

    def call():
        det.predict(images[it["i"] % len(images)])
        infer_ms.append(det.last_speed_ms.get("inference", float("nan")))
        it["i"] += 1

    stats = time_calls(call, runs=runs, warmup=warmup, sync=sync)
    row = {
        "model": entry["name"], "format": entry["format"], "precision": entry["precision"],
        "imgsz": entry["imgsz"], "device": "GPU" if device != "cpu" else "CPU",
        "size_mb": round(model_size_mb(weights), 2),
        "mean_ms": stats.mean_ms, "p95_ms": stats.p95_ms, "fps": stats.fps,
        "infer_ms": float(np.nanmean(infer_ms[warmup:])) if len(infer_ms) > warmup else float("nan"),
    }
    if accuracy_dir is not None:
        det.cfg.conf = 0.001   # full PR curve for mAP
        res = evaluate(det.predict, accuracy_dir, max_images=max_eval, show_progress=False)
        row.update({"mAP50": res.map50, "mAP50-95": res.map50_95})
    return row


def plot_tradeoff(df: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "mAP50-95" not in df:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for device, marker in (("CPU", "o"), ("GPU", "s")):
        d = df[df.device == device]
        ax.scatter(d.fps, d["mAP50-95"], marker=marker, label=device)
        for _, r in d.iterrows():
            ax.annotate(f"{r.format}-{r.precision}@{r.imgsz}", (r.fps, r["mAP50-95"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("FPS (log scale, batch 1, end-to-end)")
    ax.set_ylabel("mAP50-95 (merged test)")
    ax.set_title("Accuracy vs speed per export variant")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="models/manifest.yaml")
    ap.add_argument("--images", default="data/merged/images/test")
    ap.add_argument("--devices", nargs="+", default=["cpu", "0"])
    ap.add_argument("--only", nargs="*", help="model names from the manifest")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--no-accuracy", action="store_true")
    ap.add_argument("--max-eval-images", type=int, default=None)
    ap.add_argument("--out", default="results/benchmark")
    args = ap.parse_args()

    manifest = load_yaml(args.manifest)["models"]
    paths = list_images(REPO_ROOT / args.images)[: args.runs]
    if not paths:
        raise SystemExit(f"no images in {args.images}")
    images = [cv2.imread(str(p)) for p in paths]
    accuracy_dir = None if args.no_accuracy else REPO_ROOT / args.images

    hw = hardware_info()
    print(f"CPU: {hw['cpu']}\nGPU: {hw['gpu']}\n")

    rows = []
    for entry in manifest:
        if args.only and entry["name"] not in args.only:
            continue
        for device in args.devices:
            if device not in entry["devices"]:
                continue
            print(f"--> {entry['name']} on {device}")
            try:
                row = benchmark_one(entry, device, images, args.runs, args.warmup,
                                    accuracy_dir, args.max_eval_images)
                rows.append(row)
                print(f"    {row['mean_ms']:.1f} ms mean, {row['p95_ms']:.1f} ms p95, {row['fps']:.1f} FPS"
                      + (f", mAP50-95 {row['mAP50-95']:.3f}" if "mAP50-95" in row else ""))
            except Exception as e:
                print(f"    !! failed: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("nothing benchmarked")
    df.insert(0, "hardware", [hw["gpu"] if d == "GPU" else hw["cpu"] for d in df.device])
    csv_path, _ = save_table(df, REPO_ROOT / args.out)
    plot_tradeoff(df, (REPO_ROOT / args.out).with_name(Path(args.out).name + "_tradeoff.png"))
    print("\n" + df.drop(columns=["hardware"]).round(2).to_string(index=False))
    print(f"\nsaved {csv_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
