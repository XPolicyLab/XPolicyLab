import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

REVISION = Path(__file__).resolve().parents[1] / 'asset_revision'
sys.path.insert(0, str(REVISION))
from physicalrsi_observed_geometry import Camera, Scene, GeometryUnavailable


def scene(table, top, width=20):
    image = np.zeros((64,64,3), dtype=np.uint8)
    k = np.array([[100.,0,32],[0,100.,32],[0,0,1]])
    t = np.eye(4); t[2,3] = 2
    mask = np.zeros((64,64), bool); mask[22:42,32-width//2:32+width//2] = True
    depth = np.full((64,64), 2-table); depth[mask] = 2-top
    c = Camera(image, k, t)
    return Scene(c, predictor=lambda rgb: depth), mask


@pytest.mark.parametrize('table', [.43,.81,1.12])
def test_support_height_is_observed_not_fixed(table):
    value, mask = scene(table, table+.1)
    plane = value.support_plane(~mask)
    assert -plane['offset']/plane['normal'][2] == pytest.approx(table)
    measured = value.geometry(mask, support_mask=~mask)
    assert measured['pose'][2] == pytest.approx(table+.05)
    assert measured['dims'][2] == pytest.approx(.1)


def test_object_dimensions_change_with_observation():
    narrow, m1 = scene(.6,.8,12)
    wide, m2 = scene(.6,.8,28)
    a = narrow.geometry(m1, support_mask=~m1)
    b = wide.geometry(m2, support_mask=~m2)
    assert max(b['dims'][:2]) > max(a['dims'][:2])
    assert min(b['dims'][:2]) > min(a['dims'][:2])
    assert not b['hidden_surfaces_reconstructed']


def test_actual_camera_pose_changes_world_coordinates():
    value, _ = scene(.4,.6)
    first = value.point([32,32])
    t = value.camera.transform.copy(); t[:3,3] += [.2,-.3,.5]
    other = Scene(Camera(value.camera.image, value.camera.k, t), predictor=lambda rgb:value.depth)
    assert other.point([32,32])-first == pytest.approx([.2,-.3,.5])


def test_geometry_does_not_consume_simulator_depth_or_metadata():
    value, mask = scene(.6,.8)
    obs = {'vision':{'cam_head':{'color':value.camera.image, 'intrinsic_matrix':value.camera.k,
           'extrinsic_matrix':value.camera.transform, 'depth':np.full((64,64),999),
           'aligned_bbox':'not read'}}}
    measured = Scene.from_observation(obs, predictor=lambda rgb:value.depth)
    assert measured.point([32,32])[2] == pytest.approx(.8)
    del obs['vision']['cam_head']['intrinsic_matrix']
    with pytest.raises(GeometryUnavailable, match='calibration'):
        Scene.from_observation(obs, predictor=lambda rgb:value.depth)


def test_invalid_metric_prediction_has_no_height_fallback():
    value, _ = scene(.6,.8)
    with pytest.raises(GeometryUnavailable, match='invalid'):
        Scene(value.camera, predictor=lambda rgb:np.full((64,64),np.nan))


def test_camera_roundtrip_and_episode_independence():
    one, _ = scene(.4,.6)
    two, _ = scene(.9,1.0)
    point=one.point([32,32]); uv, d=one.camera.project([point])
    assert uv == pytest.approx(np.array([[32,32]]))
    assert one.camera.unproject(uv,d) == pytest.approx(np.array([point]))
    assert two.point([32,32])[2] == pytest.approx(1.)
    assert one.point([32,32])[2] == pytest.approx(.6)
