from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'asset_revision'))
from physicalrsi_observed_geometry import Camera, GeometryUnavailable
from physicalrsi_observed_geometry.radial_profile import observe_profile, render_profile, observed_neck_height, validate_profile


def camera(offset=0.):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('x', 30, degrees=True).as_matrix()
    transform[:3, 3] = [0, -.41, 1.308+offset]
    return Camera(np.zeros((480, 640, 3), np.uint8), [[300, 0, 320], [0, 300, 240], [0, 0, 1]], transform)


PROFILE = np.array([[0, .04], [.06, .04], [.10, .043], [.13, .015], [.19, .015], [.22, .016]])


@pytest.mark.parametrize('scale', [.7, 1., 1.2])
def test_observed_profile_tracks_unknown_bottle_size(scale):
    cam = camera(); expected = PROFILE*scale
    mask = render_profile(expected, np.eye(3), [0, -.1, .8], cam, (480, 640))
    measured = observe_profile(cam, mask, [0, 0, .8])
    assert abs(measured['height']-expected[-1, 0]) < .02*scale
    np.testing.assert_allclose(measured['base'][:2], [0, -.1], atol=.005)
    assert measured['evidence']['hidden_geometry_verified'] is False
    assert measured['evidence']['silhouette_iou'] > .8
    assert .6*measured['height'] < observed_neck_height(measured['profile']) < .95*measured['height']


def test_support_and_camera_translation_do_not_supply_an_object_size():
    cam = camera(); shifted = camera(.17)
    mask = render_profile(PROFILE, np.eye(3), [0, -.1, .8], cam, (480, 640))
    first = observe_profile(cam, mask, [0, 0, .8])
    second = observe_profile(shifted, mask, [0, 0, .97])
    np.testing.assert_allclose(first['profile'], second['profile'], atol=1e-10)
    np.testing.assert_allclose(np.asarray(second['base'])-first['base'], [0, 0, .17], atol=1e-10)


def test_incomplete_silhouette_and_unresolved_neck_have_no_cad_fallback():
    cam = camera()
    with pytest.raises(GeometryUnavailable): observe_profile(cam, np.zeros((480, 640), bool), [0, 0, .8])
    mask = np.zeros((480, 640), bool); mask[:200, 300:340] = True
    with pytest.raises(GeometryUnavailable): observe_profile(cam, mask, [0, 0, .8])
    with pytest.raises(GeometryUnavailable): observed_neck_height([[0, .04], [.1, .04], [.2, .04]])


@pytest.mark.parametrize('profile', [None, [[0, .1], [.2, .1], [.1, .1]], [[0, .1], [.1, -1], [.2, .1]]])
def test_invalid_profile_is_rejected(profile):
    with pytest.raises(GeometryUnavailable): validate_profile(profile)
