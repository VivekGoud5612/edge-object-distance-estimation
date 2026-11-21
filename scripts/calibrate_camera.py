"""Calibrate YOUR camera with a printed checkerboard (for distance validation).

    python -m scripts.calibrate_camera --images data/calib/ --pattern 9 6 --square-mm 25

How to take the photos (10 minutes)
  1. Print a checkerboard (e.g. 10x7 squares -> 9x6 *inner* corners), glue it
     flat on cardboard, measure one square with a ruler.
  2. Take 15-25 photos with the SAME camera and SAME resolution you'll use
     for the distance photos: board at different positions, angles, and
     distances, filling different parts of the frame.
  3. Run this script and paste the printed values into configs/camera.yaml
     under `my_camera`.

What it computes
  cv2.calibrateCamera finds the intrinsics K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
  (and lens distortion) that best explain where the corners appear.
  Reprojection error < ~0.5 px means a good calibration.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.config import REPO_ROOT, save_yaml
from src.data.yolo_labels import list_images


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--pattern", nargs=2, type=int, default=[9, 6], metavar=("COLS", "ROWS"),
                    help="number of INNER corners")
    ap.add_argument("--square-mm", type=float, required=True)
    ap.add_argument("--out", default="results/calibration.yaml")
    args = ap.parse_args()

    cols, rows = args.pattern
    # 3D corner positions on the board (z = 0), in metres
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * (args.square_mm / 1000.0)

    obj_points, img_points, size = [], [], None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for path in list_images(REPO_ROOT / args.images):
        gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]
        elif gray.shape[::-1] != size:
            print(f"skip {path.name}: different resolution {gray.shape[::-1]} vs {size}")
            continue
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if not found:
            print(f"no board found in {path.name}")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # sub-pixel refine
        obj_points.append(objp)
        img_points.append(corners)

    if len(obj_points) < 5:
        raise SystemExit(f"only {len(obj_points)} usable images; take more (15+)")

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    hfov = 2 * np.degrees(np.arctan(size[0] / (2 * fx)))
    result = {"image_size": list(size), "fx": float(fx), "fy": float(fy), "cx": float(cx),
              "cy": float(cy), "hfov_deg": float(hfov), "reprojection_error_px": float(rms),
              "distortion": [float(v) for v in dist.ravel()], "images_used": len(obj_points)}
    save_yaml(result, REPO_ROOT / args.out)

    print(f"\nused {len(obj_points)} images, reprojection error {rms:.3f} px")
    print("paste into configs/camera.yaml -> cameras: my_camera:")
    print(f"    image_size: [{size[0]}, {size[1]}]\n    fx: {fx:.1f}\n    fy: {fy:.1f}\n"
          f"    cx: {cx:.1f}\n    cy: {cy:.1f}")
    print(f"(horizontal FOV = {hfov:.1f} deg)")


if __name__ == "__main__":
    main()
