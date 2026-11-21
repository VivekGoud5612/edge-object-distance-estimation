"""Distance geometry: the formulas and the estimator's edge cases."""

import math

import pytest

from src.config import (CameraConfig, CameraIntrinsics, DistanceConfig, DistanceSettings,
                        ObjectPrior, load_distance_config)
from src.distance import (DistanceEstimator, ground_plane_distance, horizon_row,
                          pinhole_distance, relative_uncertainty)
from src.types import Detection

INTR = CameraIntrinsics(1280, 720, 1000.0, 1000.0, 640.0, 360.0)


def make_cfg(method="pinhole", pitch=0.0):
    cam = CameraConfig("test", INTR, mount_height_m=1.3, pitch_deg=pitch, focal_rel_uncertainty=0.1)
    objects = {0: ObjectPrior(0.7, 0.25, True), 1: ObjectPrior(0.8, 0.25, True),
               2: ObjectPrior(0.75, 0.15, False)}
    settings = DistanceSettings(method, 10, 2, 0.3, 150, 0.3)
    return DistanceConfig(cam, objects, settings)


def test_pinhole_formula():
    # A 0.7 m cone, 100 px tall, f = 1000 px -> 7 m
    assert pinhole_distance(100, 0.7, 1000) == pytest.approx(7.0)
    assert pinhole_distance(0, 0.7, 1000) is None


def test_ground_plane_formula():
    # Contact point 130 px below the horizon, camera 1.3 m high -> 10 m
    assert ground_plane_distance(360 + 130, 360, 1.3, 1000) == pytest.approx(10.0)
    assert ground_plane_distance(300, 360, 1.3, 1000) is None   # above the horizon


def test_horizon_moves_up_when_pitched_down():
    assert horizon_row(INTR, 0) == pytest.approx(360)
    assert horizon_row(INTR, 5) < 360


def test_both_methods_agree_on_a_consistent_scene():
    """Synthesize a cone exactly 10 m away and check both methods recover it."""
    Z, H, h_cam, f = 10.0, 0.7, 1.3, 1000.0
    y_bottom = 360 + f * h_cam / Z            # where the ground contact projects
    box_h = f * H / Z
    det = Detection(0, 0.9, 600, y_bottom - box_h, 640, y_bottom)
    est = DistanceEstimator(make_cfg("fused")).estimate(det, 1280, 720)
    assert est.pinhole_m == pytest.approx(10.0)
    assert est.ground_m == pytest.approx(10.0)
    assert est.distance_m == pytest.approx(10.0)
    assert est.method == "fused"
    assert est.reliable


def test_stop_sign_never_uses_ground_plane():
    det = Detection(2, 0.9, 600, 300, 650, 375)  # 75 px tall -> 10 m
    est = DistanceEstimator(make_cfg("ground")).estimate(det, 1280, 720)
    assert est.ground_m is None
    assert est.method == "pinhole"
    assert est.distance_m == pytest.approx(10.0)


def test_flags_truncated_and_tiny():
    est = DistanceEstimator(make_cfg()).estimate(Detection(0, 0.9, 100, 600, 150, 720), 1280, 720)
    assert "truncated" in est.flags and not est.reliable
    est = DistanceEstimator(make_cfg()).estimate(Detection(0, 0.9, 100, 400, 104, 405), 1280, 720)
    assert "tiny" in est.flags


def test_intrinsics_scale_with_resolution():
    """Same object at half resolution -> same distance (f halves, box halves)."""
    full = DistanceEstimator(make_cfg()).estimate(Detection(0, 0.9, 600, 400, 640, 500), 1280, 720)
    half = DistanceEstimator(make_cfg()).estimate(Detection(0, 0.9, 300, 200, 320, 250), 640, 360)
    assert half.distance_m == pytest.approx(full.distance_m)


def test_uncertainty_grows_for_small_boxes():
    assert relative_uncertainty(0.1, 0.1, 10) > relative_uncertainty(0.1, 0.1, 200)
    assert relative_uncertainty(0.0, 0.0, 100, box_err_px=2) == pytest.approx(0.02)


def test_repo_camera_config_loads():
    cfg = load_distance_config()
    # BDD assumption: HFOV 65 deg at 1280 px -> f ~ 1005 px
    assert cfg.camera.intrinsics.fx == pytest.approx(640 / math.tan(math.radians(32.5)))
    assert set(cfg.objects) == {0, 1, 2}
