"""Step 2: fine-tune a COCO-pretrained YOLO on our 3 classes (transfer learning).

    python -m scripts.train --model yolo11n --smoke     # 3 epochs on 10 % (sanity check)
    python -m scripts.train --model yolo11n             # the real run
    python -m scripts.train --model yolo11s
    python -m scripts.train --model yolo11n --set imgsz=800 epochs=80   # quick experiments

What "transfer learning" means here
  The backbone already learned generic visual features (edges, textures,
  object parts) on COCO's 118k images. We keep all of that and re-train the
  network on our data; the detection head is rebuilt for 3 classes instead
  of 80. That is why a few thousand images are enough.
  Optional: --set freeze=10 keeps the first 10 layers (the backbone) fixed:
  faster and less overfitting on tiny datasets, usually a bit less accurate.

Outputs: runs/detect/<name>/weights/best.pt (+ results.csv, PR curves,
confusion matrix, train_batch*.jpg mosaics: LOOK at those once).
"""

from __future__ import annotations

import argparse
import yaml

from src.config import REPO_ROOT, load_yaml


def build_args(cfg: dict, model_key: str, smoke: bool, overrides: list[str]) -> dict:
    args = dict(cfg["common"])
    args.update(cfg["models"][model_key])
    if smoke:
        s = dict(cfg["smoke"])
        args["name"] = args["name"] + s.pop("name_suffix", "_smoke")
        args.update(s)
    for item in overrides:
        key, value = item.split("=", 1)
        args[key] = yaml.safe_load(value)   # "80" -> 80, "true" -> True
    # Paths relative to the repo root, so the script works from any directory.
    for key in ("data", "project"):
        if not str(args[key]).startswith("/"):
            args[key] = str(REPO_ROOT / args[key])
    return args


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--model", default="yolo11n", help="key under `models:` in the config")
    ap.add_argument("--smoke", action="store_true", help="3 epochs on 10%% of the data")
    ap.add_argument("--device", default="0", help="'0' = first GPU, 'cpu' = CPU")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run (last.pt)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override any training argument")
    args = ap.parse_args()

    from ultralytics import YOLO

    train_args = build_args(load_yaml(args.config), args.model, args.smoke, args.set)
    if args.resume:
        last = REPO_ROOT / "runs/detect" / train_args["name"] / "weights" / "last.pt"
        print(f"resuming from {last}")
        YOLO(str(last)).train(resume=True)
        return

    print("training with:")
    for k, v in train_args.items():
        print(f"    {k}: {v}")
    model = YOLO(train_args.pop("model"))
    results = model.train(device=args.device, **train_args)
    print(f"\nbest weights: {results.save_dir}/weights/best.pt")
    print("next: python -m scripts.evaluate --models raw=yolo11n.pt "
          f"ft={results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
