"""Step 3: export the trained model to edge-friendly formats.

    python -m scripts.export --weights runs/detect/yolo11n_ft/weights/best.pt
    python -m scripts.export --weights ... --variants onnx onnx_fp16 openvino_int8 --imgsz 640 416 320

Variants (what each one is, in one line)
  onnx            FP32 ONNX graph. Portable; runs in ONNX Runtime on CPU or CUDA.
  onnx_fp16       Same graph in half precision (16-bit floats). ~2x smaller,
                  faster on GPUs with tensor cores. Exported on GPU; GPU only.
  openvino        Intel OpenVINO FP32 IR. Usually the fastest CPU runtime on Intel.
  openvino_int8   OpenVINO with post-training INT8 quantisation (NNCF). Weights
                  AND activations as 8-bit integers, calibrated on ~300 of our
                  own images so the int ranges fit the real data. ~4x smaller
                  than FP32, faster on CPU, usually a small mAP drop.
  engine_fp16     TensorRT FP16 engine (fastest on NVIDIA). Needs TensorRT,
                  which has no Python 3.14 build yet: see docs/SETUP.md.

Every exported file goes to models/ with a descriptive name, and is listed in
models/manifest.yaml, which scripts/benchmark.py reads. So adding a variant
means adding it here; the benchmark picks it up automatically.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.config import REPO_ROOT, load_yaml, save_yaml

# variant -> (ultralytics format, export kwargs, precision label, devices it makes sense on)
# Ultralytics 8.4 uses `quantize=16` / `quantize=8` (the old half/int8 flags are deprecated).
VARIANTS: dict[str, tuple[str, dict, str, list[str]]] = {
    "onnx":          ("onnx",     {"simplify": True},                  "fp32", ["cpu", "0"]),
    "onnx_fp16":     ("onnx",     {"simplify": True, "quantize": 16},  "fp16", ["0"]),
    "openvino":      ("openvino", {},                                  "fp32", ["cpu"]),
    "openvino_int8": ("openvino", {"quantize": 8},                     "int8", ["cpu"]),
    "engine_fp16":   ("engine",   {"quantize": 16},                    "fp16", ["0"]),
}


def export_one(weights: Path, variant: str, imgsz: int, data_yaml: Path, out_dir: Path,
               calib_fraction: float) -> dict:
    from ultralytics import YOLO

    fmt, kwargs, precision, devices = VARIANTS[variant]
    kwargs = dict(kwargs)
    # FP16 export and TensorRT need the GPU; everything else exports on CPU.
    device = "0" if (kwargs.get("quantize") == 16 or fmt == "engine") else "cpu"
    if kwargs.get("quantize") == 8:
        kwargs["data"] = str(data_yaml)          # calibration images come from our dataset
        kwargs["fraction"] = calib_fraction      # how much of it to use (speed vs quality)

    print(f"\n=== export {variant} @ {imgsz} ===")
    produced = Path(YOLO(str(weights)).export(format=fmt, imgsz=imgsz, device=device,
                                              batch=1, **kwargs))

    # Ultralytics writes next to the weights with a fixed name, so the next
    # export would overwrite it. Move it to models/<stem>_<variant>_<imgsz>.
    stem = f"{weights.parent.parent.name}_{variant}_{imgsz}"
    # Ultralytics recognises an OpenVINO model by its folder name ending in
    # "_openvino_model", so folders keep that ending; files keep their extension.
    target = out_dir / (stem + (f"_{fmt}_model" if produced.is_dir() else produced.suffix))
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    shutil.move(str(produced), target)
    return {"name": stem, "path": str(target.relative_to(REPO_ROOT)), "variant": variant,
            "format": fmt, "precision": precision, "imgsz": imgsz, "devices": devices}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--variants", nargs="+", default=["onnx", "onnx_fp16", "openvino", "openvino_int8"],
                    choices=list(VARIANTS))
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640])
    ap.add_argument("--data", default="data/merged/data.yaml", help="for INT8 calibration")
    ap.add_argument("--calib-fraction", type=float, default=0.1)
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    weights = (REPO_ROOT / args.weights).resolve()
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.yaml"
    manifest = load_yaml(manifest_path) if manifest_path.exists() else {"models": []}

    # The PyTorch baseline is always in the manifest, once per input size.
    for imgsz in args.imgsz:
        _upsert(manifest, {"name": f"{weights.parent.parent.name}_pt_{imgsz}",
                           "path": str(weights.relative_to(REPO_ROOT)) if weights.is_relative_to(REPO_ROOT) else str(weights),
                           "variant": "pytorch", "format": "pt", "precision": "fp32",
                           "imgsz": imgsz, "devices": ["cpu", "0"]})

    for variant in args.variants:
        for imgsz in args.imgsz:
            try:
                _upsert(manifest, export_one(weights, variant, imgsz, REPO_ROOT / args.data,
                                             out_dir, args.calib_fraction))
            except Exception as e:  # e.g. TensorRT missing: keep going with the others
                print(f"!! {variant}@{imgsz} failed: {e}")

    save_yaml(manifest, manifest_path)
    print(f"\nmanifest: {manifest_path.relative_to(REPO_ROOT)} ({len(manifest['models'])} models)")
    print("next: python -m scripts.benchmark")


def _upsert(manifest: dict, entry: dict) -> None:
    manifest["models"] = [m for m in manifest["models"] if m["name"] != entry["name"]] + [entry]


if __name__ == "__main__":
    main()
