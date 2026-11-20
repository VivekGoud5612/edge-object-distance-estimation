"""Bird's-eye view projection must agree with the ground-plane distance formula."""

import pytest

from src.bev import project_ground_point
from src.config import CameraIntrinsics
from src.distance import ground_plane_distance, horizon_row

INTR = CameraIntrinsics(1280, 720, 1000.0, 1000.0, 640.0, 360.0)


@pytest.mark.parametrize("pitch", [0.0, 3.0])
def test_projection_matches_ground_formula(pitch):
    u, v = project_ground_point(0.0, 12.0, INTR, 1.3, pitch)
    assert u == pytest.approx(640)
    if pitch == 0.0:
        # level camera: the closed-form ground-plane formula must invert the projection
        assert ground_plane_distance(v, horizon_row(INTR, 0), 1.3, 1000) == pytest.approx(12.0)
    else:
        assert v > horizon_row(INTR, pitch)  # ground points are always below the horizon
