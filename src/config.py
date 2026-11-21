"""Config loading.

All numbers that are *assumptions* (focal length, camera height, real object
heights) live in YAML files under configs/, never in code. That way the
sensitivity analysis in the README is "edit one value, rerun", and a reviewer
can see every assumption in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict (relative paths resolve from the repo root)."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera model, in pixels, for a given image size.

    fx, fy : focal length in pixels (horizontal / vertical)
    cx, cy : principal point (where the optical axis hits the image)
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        """Intrinsics for the same camera at another resolution.

        Focal length in pixels scales linearly with image size. Forgetting this
        is the classic bug: calibrate at 1920x1080, run at 1280x720, and every
        distance is 1.5x too large.
        """
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return CameraIntrinsics(width, height, self.fx * sx, self.fy * sy,
                                self.cx * sx, self.cy * sy)


@dataclass(frozen=True)
class CameraConfig:
    name: str
    intrinsics: CameraIntrinsics
    mount_height_m: float      # camera height above the ground
    pitch_deg: float           # + = camera tilted down
    focal_rel_uncertainty: float  # e.g. 0.10 if f is assumed, ~0.02 if calibrated


@dataclass(frozen=True)
class ObjectPrior:
    """What we assume about a class's real-world size."""

    height_m: float
    height_rel_uncertainty: float
    on_ground: bool  # ground-plane method only makes sense for objects standing on the road


@dataclass(frozen=True)
class DistanceSettings:
    method: str                 # "pinhole" | "ground" | "fused"
    min_box_px: float           # boxes shorter than this are flagged "tiny"
    border_margin_px: float     # boxes this close to the edge are flagged "truncated"
    min_distance_m: float
    max_distance_m: float
    fuse_max_disagreement: float  # fused: average only if the two agree within this ratio


@dataclass(frozen=True)
class DistanceConfig:
    camera: CameraConfig
    objects: dict[int, ObjectPrior]   # keyed by class id
    settings: DistanceSettings


def load_distance_config(path: str | Path = CONFIG_DIR / "camera.yaml",
                         profile: str | None = None) -> DistanceConfig:
    """Build a DistanceConfig from configs/camera.yaml.

    `profile` picks a camera from the `cameras:` section; default is
    `active_camera` in the file.
    """
    from src.classes import NAME_TO_ID

    raw = load_yaml(path)
    profile = profile or raw["active_camera"]
    cam = raw["cameras"][profile]

    w, h = cam["image_size"]
    if "fx" in cam:
        fx = float(cam["fx"])
        fy = float(cam.get("fy", fx))
    else:
        # No calibration: derive f from the horizontal field of view.
        #   tan(HFOV/2) = (W/2) / f   ->   f = (W/2) / tan(HFOV/2)
        import math
        fx = fy = (w / 2.0) / math.tan(math.radians(cam["hfov_deg"]) / 2.0)
    intr = CameraIntrinsics(int(w), int(h), fx, fy,
                            float(cam.get("cx", w / 2.0)), float(cam.get("cy", h / 2.0)))
    camera = CameraConfig(
        name=profile,
        intrinsics=intr,
        mount_height_m=float(cam["mount_height_m"]),
        pitch_deg=float(cam.get("pitch_deg", 0.0)),
        focal_rel_uncertainty=float(cam.get("focal_rel_uncertainty", 0.10)),
    )

    objects = {
        NAME_TO_ID[name]: ObjectPrior(
            height_m=float(o["height_m"]),
            height_rel_uncertainty=float(o.get("height_rel_uncertainty", 0.15)),
            on_ground=bool(o.get("on_ground", True)),
        )
        for name, o in raw["objects"].items()
    }

    s = raw["distance"]
    settings = DistanceSettings(
        method=s.get("method", "pinhole"),
        min_box_px=float(s.get("min_box_px", 10)),
        border_margin_px=float(s.get("border_margin_px", 2)),
        min_distance_m=float(s.get("min_distance_m", 0.3)),
        max_distance_m=float(s.get("max_distance_m", 150)),
        fuse_max_disagreement=float(s.get("fuse_max_disagreement", 0.3)),
    )
    return DistanceConfig(camera, objects, settings)
