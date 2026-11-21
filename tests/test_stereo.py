import numpy as np
import pytest

from src.stereo import depth_error_from_disparity, depth_from_disparity, essential_matrix, project_rectified


def test_disparity_recovers_depth():
    f, B = 1000.0, 0.12
    for Z in (2.0, 7.5, 30.0):
        (uL, vL), (uR, vR) = project_rectified(0.4, -0.2, Z, f, B, 640, 360)
        assert vL == vR
        assert depth_from_disparity(uL - uR, f, B) == pytest.approx(Z)


def test_depth_error_is_quadratic():
    e = depth_error_from_disparity(np.array([5.0, 10.0]), 1000.0, 0.12)
    assert e[1] / e[0] == pytest.approx(4.0)


def test_epipolar_constraint_rectified():
    # Rectified pair: R = I, t = (-B, 0, 0) from left to right camera coordinates.
    B, P = 0.12, np.array([0.4, -0.2, 6.0])
    E = essential_matrix(np.eye(3), np.array([-B, 0.0, 0.0]))
    xL = P / P[2]
    PR = P - np.array([B, 0, 0])
    xR = PR / PR[2]
    assert abs(xR @ E @ xL) < 1e-12
