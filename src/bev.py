"""Bird's-eye view (BEV) via a ground-plane homography (optional extra).

Idea
  Every point on a flat road maps to exactly one pixel, and that mapping is
  a 3x3 *homography* H. With H we can "un-project" the camera image onto a
  top-down metric grid: the road looks like a map, cones appear where they
  are in metres.

How we get H without clicking points
  We already know the camera (fx, fy, cx, cy, height, pitch) from
  configs/camera.yaml. So we take 4 ground points (corners of the area we
  want to see, in metres), project them into the image with the pinhole
  model, and let cv2.getPerspectiveTransform solve for H from those 4 pairs.
  (With a real calibration you can instead click 4 road points of known
  size; the rest of the code is the same.)

Coordinates
  Ground: X = metres to the right, Z = metres forward, camera at (0, 0).
  Camera frame: x right, y DOWN, z forward. The ground is the plane y = +h_cam.
  Camera pitched down by theta: y_c = y cos(t) - z sin(t),  z_c = y sin(t) + z cos(t).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.config import CameraConfig, CameraIntrinsics
from src.types import Detection, DistanceEstimate
from src.visualize import CLASS_COLORS


def project_ground_point(X: float, Z: float, intr: CameraIntrinsics,
                         cam_height: float, pitch_deg: float) -> tuple[float, float]:
    """Ground point (X right, Z forward, metres) -> image pixel (u, v)."""
    t = math.radians(pitch_deg)
    x, y, z = X, cam_height, Z
    y_c = y * math.cos(t) - z * math.sin(t)
    z_c = y * math.sin(t) + z * math.cos(t)
    return intr.fx * x / z_c + intr.cx, intr.fy * y_c / z_c + intr.cy


class BirdsEyeView:
    """Warps camera frames to a top-down view and plots detections on it."""

    def __init__(self, camera: CameraConfig, image_size: tuple[int, int],
                 x_range_m: tuple[float, float] = (-8.0, 8.0),
                 z_range_m: tuple[float, float] = (3.0, 40.0), px_per_m: float = 15.0):
        w, h = image_size
        self.intr = camera.intrinsics.scaled_to(w, h)
        self.cam = camera
        self.x_range, self.z_range, self.ppm = x_range_m, z_range_m, px_per_m
        self.size = (int((x_range_m[1] - x_range_m[0]) * px_per_m),
                     int((z_range_m[1] - z_range_m[0]) * px_per_m))

        # 4 ground corners (metres) -> image pixels and -> BEV pixels
        corners = [(x_range_m[0], z_range_m[0]), (x_range_m[1], z_range_m[0]),
                   (x_range_m[1], z_range_m[1]), (x_range_m[0], z_range_m[1])]
        img_pts = np.float32([project_ground_point(X, Z, self.intr, camera.mount_height_m,
                                                   camera.pitch_deg) for X, Z in corners])
        bev_pts = np.float32([self.to_bev_px(X, Z) for X, Z in corners])
        self.H = cv2.getPerspectiveTransform(img_pts, bev_pts)   # image -> BEV

    def to_bev_px(self, X: float, Z: float) -> tuple[float, float]:
        """Metres -> BEV pixel (far = top of the BEV image, like a map)."""
        u = (X - self.x_range[0]) * self.ppm
        v = (self.z_range[1] - Z) * self.ppm
        return u, v

    def warp(self, image: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(image, self.H, self.size)

    def render(self, image: np.ndarray, detections: list[Detection],
               estimates: list[DistanceEstimate]) -> np.ndarray:
        """Warped road + range rings every 5 m + one dot per detection."""
        bev = self.warp(image)
        cx = int(self.to_bev_px(0, 0)[0])
        for z in range(5, int(self.z_range[1]) + 1, 5):
            _, v = self.to_bev_px(0, z)
            cv2.line(bev, (0, int(v)), (self.size[0], int(v)), (90, 90, 90), 1)
            cv2.putText(bev, f"{z}m", (2, int(v) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        cv2.line(bev, (cx, 0), (cx, self.size[1]), (90, 90, 90), 1)
        for det, est in zip(detections, estimates):
            if est.distance_m is None or est.lateral_m is None:
                continue
            u, v = self.to_bev_px(est.lateral_m, est.distance_m)
            if 0 <= u < self.size[0] and 0 <= v < self.size[1]:
                cv2.circle(bev, (int(u), int(v)), 6, CLASS_COLORS.get(det.class_id, (255, 255, 255)), -1)
                cv2.circle(bev, (int(u), int(v)), 6, (0, 0, 0), 1)
        return bev
