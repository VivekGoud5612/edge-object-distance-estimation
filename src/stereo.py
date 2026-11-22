"""Extra credit: the stereo disparity-depth relation as code (derivation in docs/GEOMETRY.md §6).

Not used by the monocular pipeline; it exists so the derivation can be
checked numerically (tests/test_stereo.py) and to compare error growth with
the monocular methods.
"""

from __future__ import annotations

import numpy as np


def project_rectified(X: float, Y: float, Z: float, f: float, B: float,
                      cx: float = 0.0, cy: float = 0.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """Pixel coordinates of 3-D point (X, Y, Z) (left-camera frame) in a rectified stereo pair."""
    uL, v = f * X / Z + cx, f * Y / Z + cy
    uR = f * (X - B) / Z + cx           # right camera sits B metres to the right
    return (uL, v), (uR, v)             # same row v: the epipolar line is horizontal


def depth_from_disparity(d_px: float | np.ndarray, f: float, B: float):
    """Z = f·B / d."""
    return f * B / np.asarray(d_px, dtype=float)


def depth_error_from_disparity(Z: float | np.ndarray, f: float, B: float, d_err_px: float = 1.0):
    """|dZ| ≈ Z² / (f·B) · |Δd|: grows with the SQUARE of the distance."""
    return np.asarray(Z, dtype=float) ** 2 / (f * B) * d_err_px


def essential_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """E = [t]x R, so that x_R^T E x_L = 0 for normalised image points."""
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=float)
    return tx @ R
