from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'asset_revision'))
from physicalrsi_observed_geometry import Camera, GeometryUnavailable
from physicalrsi_observed_geometry.held_circle import fit_held_circle


def test_unknown_circle_is_recovered_from_partial_views_of_moving_grasp():
    rng = np.random.default_rng(138)
    center = np.array([.11, .004, -.006])
    radius = .018
    normal = np.array([.1, 1., .08]); normal /= np.linalg.norm(normal)
    x = np.cross(normal, [0, 0, 1]); x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    views = []
    k = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]])
    for i in range(4):
        ee = np.eye(4)
        ee[:3, :3] = Rotation.from_euler('xyz', [.2 * i, -.15 * i, .1 * i]).as_matrix()
        ee[:3, 3] = [-.1 + i * .04, .02, .7]
        for side in (-1, 1):
            t = np.eye(4); t[:3, 3] = [side * .22, -.1, 1.4]
            camera = Camera(np.zeros((480, 640, 3), np.uint8), k, t)
            angle = np.linspace(.1 + .2 * i, 4.6 + .2 * i, 32)
            local = center + radius * (np.cos(angle)[:, None] * x + np.sin(angle)[:, None] * y)
            pixels = camera.project(local @ ee[:3, :3].T + ee[:3, 3])[0]
            pixels += rng.normal(0, .1, pixels.shape)
            views.append((camera, ee, pixels))
    fit = fit_held_circle(views, center + [.003, -.002, .002], radius * .8)
    np.testing.assert_allclose(fit.center, center, atol=.001)
    assert abs(fit.radius - radius) < .001
    assert abs(fit.normal @ normal) > .99
    assert fit.pixel_rms < .3


def test_missing_views_are_not_reported_as_measured_geometry():
    with pytest.raises(GeometryUnavailable):
        fit_held_circle([], [.1, 0, 0], .02)
