"""Extra credit: track cones across frames (detector + optical flow).

    python -m scripts.track --weights best.pt --source drive.mp4
    python -m scripts.track --weights best.pt --source frames_folder/        # sorted image sequence
    python -m scripts.track --weights best.pt --simulate path/to/still.jpg   # no video at hand

Each tracked object gets an id and a trail: "Cone #3, 12.4m". The distance
shown is the per-track smoothed value (log-space moving average). Dashed
boxes are frames where the detector missed and optical flow carried the
track. Writes <out>/tracked.mp4 and <out>/tracks.csv (one row per track per
frame), plus a summary of how many detector gaps the flow filled.

--simulate makes a fake "driving forward" clip from one still image by
zooming toward the image centre a little each frame. That is NOT real 3D
motion (everything grows at the same rate), so it only demonstrates
identity keeping and gap filling, not distance from motion.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from scripts.infer import iter_frames
from src.classes import NAME_TO_ID, display_name
from src.config import REPO_ROOT, load_distance_config
from src.detector import Detector, DetectorConfig
from src.distance import DistanceEstimator
from src.tracking import OpticalFlowTracker
from src.visualize import CLASS_COLORS


def simulated_drive(image_path: str, n_frames: int = 60, zoom_per_frame: float = 1.012):
    """Yield frames that zoom into one still image (a crude stand-in for forward motion)."""
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"cannot read {image_path}")
    h, w = img.shape[:2]
    cx, cy = w / 2, h * 0.55
    for i in range(n_frames):
        s = zoom_per_frame ** i
        M = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
        yield f"sim_{i:04d}.jpg", cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def draw_tracks(frame, tracks, dists):
    canvas = frame.copy()
    scale = max(canvas.shape[:2]) / 1280.0
    th = max(1, int(round(2 * scale)))
    for t in tracks:
        if not t.confirmed:
            continue
        d = t.det
        color = CLASS_COLORS.get(d.class_id, (255, 255, 255))
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        if t.coast == 0:
            cv2.rectangle(canvas, p1, p2, color, th)
        else:   # flow-only frame: dashed-looking box (corners only)
            for (x, y), (dx, dy) in (((p1[0], p1[1]), (1, 1)), ((p2[0], p1[1]), (-1, 1)),
                                      ((p1[0], p2[1]), (1, -1)), ((p2[0], p2[1]), (-1, -1))):
                L = max(4, int(0.25 * min(d.width, d.height)))
                cv2.line(canvas, (x, y), (x + dx * L, y), color, th)
                cv2.line(canvas, (x, y), (x, y + dy * L), color, th)
        pts = np.int32(t.history)
        if len(pts) > 1:
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, color, max(1, th - 1))
        dist = dists.get(t.track_id)
        text = f"{display_name(d.class_id)} #{t.track_id}" + (f", {dist:.1f}m" if dist else "")
        cv2.putText(canvas, text, (p1[0], max(12, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * scale, color, th, cv2.LINE_AA)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="video file, image folder or webcam index")
    src.add_argument("--simulate", help="one still image to turn into a fake forward-motion clip")
    ap.add_argument("--frames", type=int, default=60, help="--simulate only")
    ap.add_argument("--classes", nargs="+", default=["cone"], help="classes to track")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--out", default="results/track")
    args = ap.parse_args()

    det = Detector(DetectorConfig(weights=args.weights, device=args.device, conf=args.conf))
    est = DistanceEstimator(load_distance_config(profile=args.camera))
    tracker = OpticalFlowTracker(classes={NAME_TO_ID[c] for c in args.classes})
    frames = simulated_drive(args.simulate, args.frames) if args.simulate else iter_frames(args.source)

    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    writer, n_det_frames, n_flow_frames = None, 0, 0
    with open(out / "tracks.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["frame", "track_id", "class", "source", "x1", "y1", "x2", "y2", "distance_m",
                      "smoothed_m"])
        for name, frame in frames:
            h, w = frame.shape[:2]
            tracks = tracker.update(frame, det.predict(frame))
            dists = {}
            for t in tracks:
                e = est.estimate(t.det, w, h)
                dists[t.track_id] = tracker.smooth_distance(t, e)
                if t.confirmed:
                    n_det_frames += t.coast == 0
                    n_flow_frames += t.coast > 0
                    log.writerow([name, t.track_id, display_name(t.det.class_id),
                                  "detector" if t.coast == 0 else "flow",
                                  *(f"{v:.1f}" for v in (t.det.x1, t.det.y1, t.det.x2, t.det.y2)),
                                  "" if e.distance_m is None else f"{e.distance_m:.2f}",
                                  "" if dists[t.track_id] is None else f"{dists[t.track_id]:.2f}"])
            annotated = draw_tracks(frame, tracks, dists)
            if writer is None:
                writer = cv2.VideoWriter(str(out / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (w, h))
                cv2.imwrite(str(out / "first_frame.jpg"), annotated)
            writer.write(annotated)
            last = annotated
    if writer is not None:
        writer.release()
        cv2.imwrite(str(out / "last_frame.jpg"), last)
    total = n_det_frames + n_flow_frames
    print(f"track-frames: {total}, carried by optical flow (detector missed): {n_flow_frames} "
          f"({100 * n_flow_frames / max(total, 1):.1f} %), ids issued: {tracker._next_id - 1}")
    print(f"saved {out}/tracked.mp4 and tracks.csv")


if __name__ == "__main__":
    main()
