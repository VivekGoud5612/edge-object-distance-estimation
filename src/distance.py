"""Geometric distance estimation from a single camera.

No learning happens here: every number comes from the pinhole camera model
plus assumptions stored in configs/camera.yaml.

Two independent methods
-----------------------
1. Pinhole / similar triangles (main method, works for every class)

       An object of real height H at depth Z projects to h pixels:
           h = f * H / Z      ->      Z = f * H / h

   f is the focal length in pixels (fy, because we measure heights).
   Error sources: wrong H (cones come in sizes), wrong f, box error of a few
   pixels (matters most when h is small, i.e. far away).

2. Ground plane (cross-check, only for objects standing on the road)

       A level camera at height h_cam sees the ground point at depth Z at
       image row y, below the horizon row y0:
           y - y0 = f * h_cam / Z      ->      Z = f * h_cam / (y - y0)

   It needs no object size at all, so it fails differently from method 1.
   Agreement between the two is evidence the numbers are sensible.
   With pitch theta (camera tilted down) the horizon moves up:
       y0 = cy - f * tan(theta)

Both give Z, the depth along the optical axis. We also return the lateral
offset X = (u - cx) * Z / fx, so the bird's-eye view can place the object.
"""

from __future__ import annotations

import math

from src.config import CameraIntrinsics, DistanceConfig
from src.types import Detection, DistanceEstimate


# --------------------------------------------------------------------------
# The two formulas, as pure functions (easy to unit-test and to explain)
# --------------------------------------------------------------------------
def pinhole_distance(box_height_px: float, real_height_m: float, focal_px: float) -> float | None:
    """Z = f * H / h. Returns None for degenerate boxes."""
    if box_height_px <= 0:
        return None
    return focal_px * real_height_m / box_height_px


def horizon_row(intr: CameraIntrinsics, pitch_deg: float) -> float:
    """Image row of the horizon for a camera pitched down by `pitch_deg`."""
    return intr.cy - intr.fy * math.tan(math.radians(pitch_deg))


def ground_plane_distance(y_bottom_px: float, horizon_px: float,
                          camera_height_m: float, focal_px: float,
                          min_offset_px: float = 1.0) -> float | None:
    """Z = f * h_cam / (y_bottom - y_horizon).

    Returns None when the contact point is at or above the horizon (the ray
    never hits the ground, so the formula is meaningless).
    """
    dy = y_bottom_px - horizon_px
    if dy < min_offset_px:
        return None
    return focal_px * camera_height_m / dy


def relative_uncertainty(height_rel: float, focal_rel: float,
                         box_height_px: float, box_err_px: float = 2.0) -> float:
    """First-order error propagation for Z = f * H / h.

    For a product/quotient, relative errors add in quadrature:
        dZ/Z = sqrt( (dH/H)^2 + (df/f)^2 + (dh/h)^2 )
    `box_err_px` is a typical detector box error (~2 px).
    """
    box_rel = box_err_px / max(box_height_px, 1e-6)
    return math.sqrt(height_rel**2 + focal_rel**2 + box_rel**2)


# --------------------------------------------------------------------------
# The estimator used by the pipeline
# --------------------------------------------------------------------------
class DistanceEstimator:
    """Turns detections into distances using a DistanceConfig.

    The detector does not need to know about it, and it does not need to
    know which backend produced the boxes. It only needs a class id and a
    box in original image pixels.
    """

    def __init__(self, config: DistanceConfig):
        self.cfg = config

    def estimate(self, det: Detection, image_width: int, image_height: int) -> DistanceEstimate:
        s = self.cfg.settings
        cam = self.cfg.camera
        # Intrinsics must match the resolution the boxes are in.
        intr = cam.intrinsics.scaled_to(image_width, image_height)
        prior = self.cfg.objects.get(det.class_id)
        if prior is None:
            return DistanceEstimate(None, "none", flags=["unknown_class"])

        flags: list[str] = []
        if det.height < s.min_box_px:
            flags.append("tiny")
        m = s.border_margin_px
        if det.y1 <= m or det.y2 >= image_height - m:
            # Cut at top/bottom: the visible height is smaller than the real
            # one, so the pinhole method OVER-estimates distance.
            flags.append("truncated")

        # Method 1: pinhole
        z_pin = pinhole_distance(det.height, prior.height_m, intr.fy)

        # Method 2: ground plane (only for objects on the road, only if the
        # bottom edge is really visible)
        z_gnd = None
        if prior.on_ground and det.y2 < image_height - m:
            z_gnd = ground_plane_distance(det.y2, horizon_row(intr, cam.pitch_deg),
                                          cam.mount_height_m, intr.fy)

        z, method = self._choose(z_pin, z_gnd, s.method)
        if z is None:
            return DistanceEstimate(None, "none", z_pin, z_gnd, flags=flags + ["no_estimate"])

        # Clamp to a plausible range and say so if we had to.
        if not (s.min_distance_m <= z <= s.max_distance_m):
            flags.append("clamped")
            z = min(max(z, s.min_distance_m), s.max_distance_m)

        u, _ = det.bottom_center
        lateral = (u - intr.cx) * z / intr.fx
        unc = relative_uncertainty(prior.height_rel_uncertainty,
                                   cam.focal_rel_uncertainty, det.height)
        return DistanceEstimate(z, method, z_pin, z_gnd, lateral, unc, flags)

    def _choose(self, z_pin: float | None, z_gnd: float | None,
                method: str) -> tuple[float | None, str]:
        """Pick the final value according to `distance.method` in the config."""
        if method == "ground":
            # Fall back to pinhole when the ground method can't be applied (e.g. stop signs).
            return (z_gnd, "ground") if z_gnd is not None else (z_pin, "pinhole")
        if method == "fused" and z_pin is not None and z_gnd is not None:
            disagreement = abs(z_pin - z_gnd) / max(z_pin, z_gnd)
            if disagreement <= self.cfg.settings.fuse_max_disagreement:
                return (z_pin + z_gnd) / 2.0, "fused"
            # They disagree a lot: trust pinhole, which doesn't depend on
            # camera height/pitch (the less reliable assumptions).
            return z_pin, "pinhole"
        return z_pin, "pinhole"
