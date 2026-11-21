"""The assumption tests must recover known parameters from synthetic scenes."""

import numpy as np
import pandas as pd
import pytest

from src.distance_analysis import (aspect_check, box_height_noise, ground_line_fit, huber_fit,
                                   monte_carlo_error, robust_std)


def synthetic_cones(n_img=150, per_img=6, H=0.7, h_cam=1.3, f=1000.0, W=1280, Himg=720,
                    height_sd=0.08, horizon=360.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_img):
        for _ in range(per_img):
            Z = rng.uniform(4, 40)
            Hi = H * np.exp(rng.normal(0, height_sd))
            y2 = horizon + f * h_cam / Z
            h = f * Hi / Z
            x1 = rng.uniform(0, W - 100)
            rows.append({"image": f"im{i}", "source": "syn", "img_w": W, "img_h": Himg, "cls": "cone",
                         "x1": x1, "y1": y2 - h, "x2": x1 + 0.5 * h, "y2": y2, "w": 0.5 * h, "h": h})
    df = pd.DataFrame(rows)
    return df[(df.y2 < Himg - 3) & (df.y1 > 3)]


def test_huber_ignores_outliers():
    x = np.linspace(0, 1, 200)
    y = 2 * x + 1
    y[:10] += 50
    a, b = huber_fit(np.column_stack([x, np.ones_like(x)]), y)
    assert a == pytest.approx(2, abs=0.05) and b == pytest.approx(1, abs=0.05)


def test_ground_fit_recovers_geometry():
    df = synthetic_cones()
    fit = ground_line_fit(df, n_boot=50)
    e = fit["estimates"]
    assert e["slope_H_over_hcam"].value == pytest.approx(0.7 / 1.3, rel=0.03)
    assert e["horizon_y0_norm"].value == pytest.approx(0.5, abs=0.01)
    assert abs(e["curvature"].value) < 4 * e["curvature"].se + 0.05
    assert e["resid_robust_std"].value == pytest.approx(0.08, abs=0.02)
    assert e["slope_H_over_hcam"].se > 0


def test_ground_fit_detects_pitch():
    df = synthetic_cones(horizon=300.0)   # camera tilted down -> horizon above centre
    y0 = ground_line_fit(df, n_boot=30)["estimates"]["horizon_y0_norm"]
    assert y0.value == pytest.approx(300 / 720, abs=0.01)
    assert y0.ci95()[1] < 0.5


def test_aspect_check():
    res = aspect_check(synthetic_cones(), n_boot=20)
    assert res["implied_stretch"] == pytest.approx(1.0, abs=0.01)


def test_box_noise_bias():
    df = synthetic_cones()
    df["pred_h"] = df.h * 1.10          # detector draws boxes 10 % too tall
    df["iou"] = 0.8
    out = box_height_noise(df, n_boot=20)
    # taller box -> object looks closer -> distance ~9 % too short
    assert out.dist_bias.dropna().between(-0.10, -0.08).all()


def test_monte_carlo_zero_noise_is_exact():
    b = monte_carlo_error([5, 20], 0.7, 0.0, 1000, 0.0, None)
    assert np.allclose(b.median_bias, 0) and np.allclose(b.rmse, 0)


def test_monte_carlo_height_spread_propagates():
    b = monte_carlo_error([10], 0.7, 0.10, 1000, 0.0, None, n=50_000)
    assert b.rmse.iloc[0] == pytest.approx(0.10, abs=0.01)


def test_robust_std():
    x = np.random.default_rng(0).normal(0, 2, 10_000)
    assert robust_std(x) == pytest.approx(2, rel=0.05)
