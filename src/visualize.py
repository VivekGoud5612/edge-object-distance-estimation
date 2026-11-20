"""Drawing: boxes with labels in the exact format the brief asks for.

    Cone, 1.5m
    Stop Sign, 3.2m

Unreliable estimates (truncated box, tiny box, clamped) get a "~" prefix,
e.g. "Cone, ~14.0m", so the viewer can see we know it's shaky.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.classes import display_name
from src.types import Detection, DistanceEstimate

# One BGR colour per class: cone orange, barrier yellow, stop sign red.
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 140, 255),
    1: (0, 215, 255),
    2: (40, 40, 220),
}


def format_label(det: Detection, est: DistanceEstimate | None) -> str:
    """'Cone, 1.5m' (or 'Cone' when no distance could be estimated)."""
    name = display_name(det.class_id)
    if est is None or est.distance_m is None:
        return name
    prefix = "" if est.reliable else "~"
    return f"{name}, {prefix}{est.distance_m:.1f}m"


def draw_detections(image: np.ndarray, detections: list[Detection],
                    estimates: list[DistanceEstimate | None] | None = None,
                    show_conf: bool = False) -> np.ndarray:
    """Return a copy of `image` with boxes and labels drawn on it."""
    canvas = image.copy()
    estimates = estimates or [None] * len(detections)
    # Scale line width / font with image size so labels stay readable on 4K and on 640px.
    scale = max(canvas.shape[:2]) / 1280.0
    thickness = max(1, int(round(2 * scale)))
    font_scale = 0.6 * scale

    for det, est in zip(detections, estimates):
        color = CLASS_COLORS.get(det.class_id, (255, 255, 255))
        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(canvas, p1, p2, color, thickness)

        text = format_label(det, est)
        if show_conf:
            text += f" ({det.confidence:.2f})"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # Put the label above the box, or inside it if there's no room above.
        y_text = p1[1] - 4 if p1[1] - th - 6 > 0 else p1[1] + th + 4
        cv2.rectangle(canvas, (p1[0], y_text - th - 4), (p1[0] + tw + 4, y_text + baseline),
                      color, -1)
        cv2.putText(canvas, text, (p1[0] + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return canvas


def draw_fps(image: np.ndarray, fps: float, backend: str = "") -> np.ndarray:
    """Small FPS overlay for the video demo (draws in place)."""
    text = f"{fps:5.1f} FPS {backend}".strip()
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return image
