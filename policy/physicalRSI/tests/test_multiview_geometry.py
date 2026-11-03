from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'asset_revision'))
from physicalrsi_observed_geometry import Camera, GeometryUnavailable
from physicalrsi_observed_geometry.multiview import triangulate


def cameras(baseline=.25):
    rgb = np.zeros((480, 640, 3), np.uint8)
    k = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
    a = np.eye(4); a[2, 3] = 2.
    b = a.copy(); b[0, 3] = baseline
    return Camera(rgb, k, a), Camera(rgb, k, b)


def test_triangulation_recovers_positions_without_height_or_size_input():
    a, b = cameras()
    for point in ([.1, .05, .9], [-.1, .1, .5], [.2, -.1, 1.2]):
        uv, _ = a.project([point]); vv, _ = b.project([point])
        actual, error = triangulate(a, b, uv[0], vv[0])
        np.testing.assert_allclose(actual, point, atol=1e-8)
        assert error < 1e-8


def test_inconsistent_or_unobservable_correspondences_are_rejected():
    a, b = cameras()
    uv, _ = a.project([[.1, .05, .9]]); vv, _ = b.project([[.1, .05, .9]])
    with pytest.raises(GeometryUnavailable):
        triangulate(a, b, uv[0], vv[0] + [0, 30])
    a, b = cameras(.0001)
    uv, _ = a.project([[.1, .05, .9]]); vv, _ = b.project([[.1, .05, .9]])
    with pytest.raises(GeometryUnavailable, match='baseline'):
        triangulate(a, b, uv[0], vv[0])


@pytest.mark.parametrize('center,angles,dimensions', [
    ([.1, .05, .9], [.1, -.15, .3], [.047, .008]),
    ([-.1, .1, .5], [-.2, .08, -.4], [.083, .012]),
])
def test_rectangle_recovers_unknown_size_height_and_tilt(center, angles, dimensions):
    from scipy.spatial.transform import Rotation
    from physicalrsi_observed_geometry.multiview import fit_rectangle
    a, b = cameras()
    rotation = Rotation.from_rotvec(angles).as_matrix()
    unit = np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]])
    points = (unit * np.r_[dimensions, 1]) @ rotation.T + center
    views = [(camera, camera.project(points)[0]) for camera in (a, b)]
    actual, recovered, size, rms = fit_rectangle(views, np.asarray(center) + [.001, -.001, .002],
                                                    rotation[:, 0] + [.01, -.02, .01])
    np.testing.assert_allclose(actual, center, atol=1e-5)
    np.testing.assert_allclose(size, dimensions, atol=1e-5)
    assert abs(recovered[:, 2] @ rotation[:, 2]) > .9999
    assert rms < .001
