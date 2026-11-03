from pathlib import Path
import sys
import ast
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'asset_revision'))
from pour_language_camera_edits import edit_language_head_camera, edit_observer_wrist_camera
from physicalrsi_observed_geometry import Camera, GeometryUnavailable


def revised():
    source = '''"""Fixture with direct and callback projection uses."""
F = 300
def ray(u, v, z): pass
def project(xyz): pass
def ground(image, instruction, out, pair_index=0):
    return ray(320, 240, .8)
class Model:
    def reset(self): pass
    def update_obs(self, obs): pass
    def callbacks(self): return ray, project
    def locate(self): return ground(None, None, None)
class StructuredModel(Model):
    def structured_callbacks(self): return ray, project
    def structured_projection(self): return project(np.r_[ray(320, 240, .8), .8])
'''
    namespace = {'np': np}
    exec(compile(edit_language_head_camera(source), '<camera-revision>', 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('angles,origin,k', [
    ([30, 0, 0], [0, -.41, 1.308], [[640*10/22.212, 0, 320], [0, 640*10/22.212, 240], [0, 0, 1]]),
    ([20, -7, 13], [.15, -.2, 1.6], [[510, 0, 290], [0, 450, 220], [0, 0, 1]])])
def test_live_camera_roundtrip_scalar_and_batched(angles, origin, k):
    ns = revised(); transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    transform[:3, 3] = origin
    camera = Camera(np.zeros((480, 640, 3), np.uint8), k, transform)
    uv = np.array([[220, 190], [320, 240], [430, 320]])
    xy = ns['ray'](uv[:, 0], uv[:, 1], .8, camera=camera)
    for pixel, point in zip(uv, xy):
        np.testing.assert_allclose(ns['project'](np.r_[point, .8], camera=camera), pixel, atol=1e-10)
    model = ns['Model'](); model.reset()
    with pytest.raises(GeometryUnavailable): model.ray(320, 240, .8)
    model.update_obs({'vision': {'cam_head': {'color': camera.image, 'intrinsic_matrix': k, 'extrinsic_matrix': transform}}})
    ray, project = model.callbacks()
    np.testing.assert_allclose(project(np.r_[ray(320, 240, .8), .8]), [320, 240])
    np.testing.assert_allclose(model.locate(), ray(320, 240, .8))
    structured = ns['StructuredModel'](); structured.reset()
    structured.update_obs({'vision': {'cam_head': {'color': camera.image, 'intrinsic_matrix': k, 'extrinsic_matrix': transform}}})
    sr, sp = structured.structured_callbacks()
    np.testing.assert_allclose(sp(np.r_[sr(320, 240, .8), .8]), [320, 240])
    np.testing.assert_allclose(structured.structured_projection(), [320, 240])


def test_original_head_calibration_preserves_original_projection_math():
    ns = revised(); f = 640*10/22.212
    transform = np.eye(4); transform[:3, :3] = Rotation.from_euler('x', 30, degrees=True).as_matrix()
    transform[:3, 3] = [0, -.41, 1.308]
    camera = Camera(np.zeros((480, 640, 3), np.uint8), [[f, 0, 320], [0, f, 240], [0, 0, 1]], transform)
    u, v, z = np.array([100, 320, 540]), np.array([80, 240, 400]), np.array([.7, .8, .95])
    up = -(v-240)/f; distance = (z-1.308)/(.5*up-np.sqrt(3)/2)
    expected = np.stack([distance*(u-320)/f, -.41+distance*(np.sqrt(3)/2*up+.5)], axis=-1)
    np.testing.assert_allclose(ns['ray'](u, v, z, camera=camera), expected, atol=1e-12)
    with pytest.raises(GeometryUnavailable): ns['ray'](320, 240, 2., camera=camera)


@pytest.mark.parametrize('height,fy', [(480, 400), (600, 730)])
def test_wrist_projection_and_view_distance_use_current_calibration(height, fy):
    from types import SimpleNamespace
    source = '''"""Observer calibration fixture."""
FOCAL = 397
WRIST_LOCAL = None
def transform(): pass
def pixels(camera, points): pass
class BowlObserver:
    def __init__(self, model):
        self.arm = 'left'
        self.relative = WRIST_LOCAL
        self.distance = FOCAL / (240 * .75)
    def observe(self, model, ee, points):
        return pixels(ee @ WRIST_LOCAL, points)
'''
    ns = {'np': np}
    exec(compile(edit_observer_wrist_camera(source), '<wrist-revision>', 'exec'), ns)
    transform = np.eye(4); transform[:3, 3] = [.03, -.02, .05]
    k = np.array([[500, 0, 380], [0, fy, height/2-15], [0, 0, 1.]])
    rgb = np.zeros((height, 800, 3), np.uint8)
    obs = {'state': {'left_ee_pose': [0, 0, 0, 1, 0, 0, 0], 'right_ee_pose': [1, 0, 0, 1, 0, 0, 0]},
           'vision': {'cam_left_wrist': {'color': rgb, 'intrinsic_matrix': k, 'extrinsic_matrix': transform}}}
    model = SimpleNamespace(obs=obs); observer = ns['BowlObserver'](model)
    np.testing.assert_allclose(observer.relative, transform)
    assert observer.distance == pytest.approx(fy/(height/2*.75))
    pixels = np.array([[200, 100], [500, height-50]])
    camera = Camera(rgb, k, transform); points = camera.unproject(pixels, [1., 1.2])
    np.testing.assert_allclose(observer.observe(model, np.eye(4), points), pixels, atol=1e-10)
    # A later observation must supply the calibration; no module-global cache.
    changed = transform.copy(); changed[0, 3] += .1
    obs['vision']['cam_left_wrist']['extrinsic_matrix'] = changed
    expected = Camera(rgb, k, changed).project(points)[0]
    np.testing.assert_allclose(observer.observe(model, np.eye(4), points), expected, atol=1e-10)
