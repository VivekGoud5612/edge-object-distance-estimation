"""Plain data containers passed between pipeline stages.

They are deliberately dumb (no methods with side effects) so that the
detector, the distance module and the visualiser stay independent: each one
only needs to agree on these shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
    """One detected object in *original image pixel* coordinates."""

    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Point where the object touches the ground (used by ground-plane distance)."""
        return ((self.x1 + self.x2) / 2.0, self.y2)


@dataclass
class DistanceEstimate:
    """Distance result for one detection.

    `distance_m` is the value we show on the image. The other fields explain
    where it came from, so the README can discuss reliability honestly.
    """

    distance_m: float | None           # final value (None = could not estimate)
    method: str                        # "pinhole", "ground", "fused" or "none"
    pinhole_m: float | None = None     # D = f * H / h
    ground_m: float | None = None      # D = f * h_cam / (y_bottom - y_horizon)
    lateral_m: float | None = None     # sideways offset X (+ = right of camera)
    rel_uncertainty: float | None = None  # e.g. 0.12 means +/-12 %
    flags: list[str] = field(default_factory=list)  # "truncated", "tiny", ...

    @property
    def reliable(self) -> bool:
        return self.distance_m is not None and not self.flags
