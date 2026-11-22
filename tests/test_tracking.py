"""Optical-flow tracker: keeps identity, fills detector gaps, follows scale."""

import cv2
import numpy as np
import pytest

from src.tracking import OpticalFlowTracker, distance_from_scale_change, propagate_box
from src.types import Detection


def textured_frame(x, y, size=60, W=320, H=240, seed=0):
    """Grey background + a random-texture square (corners for LK to grab) at (x, y)."""
    rng = np.random.default_rng(seed)
    patch = rng.integers(0, 255, (size, size), dtype=np.uint8)
    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    img = np.full((H, W), 90, np.uint8)
    img[int(y):int(y) + size, int(x):int(x) + size] = patch
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_propagate_follows_shift():
    a, b = textured_frame(50, 60), textured_frame(55, 63)
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (a, b)]
    moved = propagate_box(g[0], g[1], Detection(0, 0.9, 50, 60, 110, 120))
    assert moved is not None
    assert moved.x1 == pytest.approx(55, abs=1.0) and moved.y1 == pytest.approx(63, abs=1.0)


def test_tracker_keeps_id_through_missed_detection():
    tr = OpticalFlowTracker(classes={0})
    ids = []
    for i in range(6):
        x, y = 40 + 4 * i, 50 + 2 * i
        dets = [] if i == 3 else [Detection(0, 0.9, x, y, x + 60, y + 60)]   # detector misses frame 3
        tracks = tr.update(textured_frame(x, y), dets)
        assert len(tracks) == 1
        ids.append(tracks[0].track_id)
        if i == 3:   # flow carried the box to the right place
            assert tracks[0].coast == 1
            assert tracks[0].det.x1 == pytest.approx(x, abs=1.5)
    assert len(set(ids)) == 1


def test_scale_change_distance():
    # Cone at 10 m -> 8 m after driving 2 m: h grows by 10/8.
    f, H = 1000.0, 0.7
    assert distance_from_scale_change(f * H / 10, f * H / 8, 2.0) == pytest.approx(8.0)
    assert distance_from_scale_change(50, 49, 1.0) is None
