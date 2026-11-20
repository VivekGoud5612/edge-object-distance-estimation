"""Run the full online pipeline: frame -> detector -> distance -> annotated frame.

    python -m scripts.infer --weights runs/detect/yolo11n_ft/weights/best.pt --source path/to/image.jpg
    python -m scripts.infer --weights models/yolo11n_ft_openvino_int8_640 --source folder/ --device cpu
    python -m scripts.infer --weights best.pt --source video.mp4 --bev          # + bird's-eye view panel
    python -m scripts.infer --weights best.pt --source 0 --show --camera my_camera   # webcam

Labels look like "Cone, 1.5m". A "~" ("Cone, ~14.0m") marks an estimate we
flagged as unreliable (box cut by the image border, tiny box, or clamped).
Per-object details (both distance methods, uncertainty, flags) are written
to <out>/detections.csv so you can analyse them later.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from src.classes import CLASS_NAMES
from src.config import REPO_ROOT, load_distance_config
from src.data.yolo_labels import IMAGE_EXTS, list_images
from src.detector import Detector, DetectorConfig
from src.distance import DistanceEstimator
from src.visualize import draw_detections, draw_fps

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


class Pipeline:
    """The whole online flow in one object, so infer.py and other tools share it."""

    def __init__(self, detector: Detector, estimator: DistanceEstimator, bev: bool = False):
        self.detector, self.estimator, self.use_bev = detector, estimator, bev
        self._bev = None

    def __call__(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        dets = self.detector.predict(frame)
        ests = [self.estimator.estimate(d, w, h) for d in dets]
        out = draw_detections(frame, dets, ests)
        if self.use_bev:
            from src.bev import BirdsEyeView
            if self._bev is None:
                self._bev = BirdsEyeView(self.estimator.cfg.camera, (w, h))
            panel = self._bev.render(frame, dets, ests)
            # Put the BEV panel to the right, scaled to the frame height.
            scale = h / panel.shape[0]
            panel = cv2.resize(panel, (int(panel.shape[1] * scale), h))
            out = np.hstack([out, panel])
        return out, dets, ests


def iter_frames(source: str):
    """Yield (name, frame) from an image, a folder, a video file or a webcam index."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f"cam_{i:06d}.jpg", frame
            i += 1
        cap.release()
        return
    p = Path(source)
    if p.is_dir() or p.suffix.lower() in IMAGE_EXTS or p.suffix == ".txt":
        for img in ([p] if p.suffix.lower() in IMAGE_EXTS else list_images(p)):
            frame = cv2.imread(str(img))
            if frame is not None:
                yield img.name, frame
    elif p.suffix.lower() in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(p))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f"{p.stem}_{i:06d}.jpg", frame
            i += 1
        cap.release()
    else:
        raise ValueError(f"don't know how to read {source}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--camera", default=None, help="camera profile in configs/camera.yaml")
    ap.add_argument("--method", default=None, choices=["pinhole", "ground", "fused"],
                    help="override distance.method from the config")
    ap.add_argument("--bev", action="store_true", help="add a bird's-eye view panel")
    ap.add_argument("--show", action="store_true", help="display in a window")
    ap.add_argument("--out", default="results/infer")
    args = ap.parse_args()

    dist_cfg = load_distance_config(profile=args.camera)
    if args.method:
        from dataclasses import replace
        dist_cfg = replace(dist_cfg, settings=replace(dist_cfg.settings, method=args.method))

    detector = Detector(DetectorConfig(weights=args.weights, device=args.device,
                                       imgsz=args.imgsz, conf=args.conf))
    pipe = Pipeline(detector, DistanceEstimator(dist_cfg), bev=args.bev)

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    is_stream = Path(args.source).suffix.lower() in VIDEO_EXTS or args.source.isdigit()
    writer = None

    with open(out_dir / "detections.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["frame", "class", "conf", "x1", "y1", "x2", "y2", "distance_m", "method",
                      "pinhole_m", "ground_m", "lateral_m", "rel_uncertainty", "flags"])
        fps_smooth = None
        for name, frame in iter_frames(args.source):
            t0 = time.perf_counter()
            annotated, dets, ests = pipe(frame)
            dt = time.perf_counter() - t0
            fps_smooth = 1 / dt if fps_smooth is None else 0.9 * fps_smooth + 0.1 / dt

            for d, e in zip(dets, ests):
                log.writerow([name, CLASS_NAMES[d.class_id], f"{d.confidence:.3f}",
                              *(f"{v:.1f}" for v in (d.x1, d.y1, d.x2, d.y2)),
                              _fmt(e.distance_m), e.method, _fmt(e.pinhole_m), _fmt(e.ground_m),
                              _fmt(e.lateral_m), _fmt(e.rel_uncertainty), "|".join(e.flags)])

            if is_stream:
                draw_fps(annotated, fps_smooth, Path(args.weights).suffix or "openvino")
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(out_dir / "output.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                writer.write(annotated)
            else:
                cv2.imwrite(str(out_dir / name), annotated)
            if args.show:
                cv2.imshow("detections", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"saved results to {out_dir.relative_to(REPO_ROOT)}/ (detections.csv has every estimate)")


def _fmt(v):
    return "" if v is None else f"{v:.3f}"


if __name__ == "__main__":
    main()
