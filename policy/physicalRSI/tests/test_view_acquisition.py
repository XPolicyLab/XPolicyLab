from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'asset_revision'))
from physicalrsi_observed_geometry import GeometryUnavailable
from physicalrsi_observed_geometry.view_acquisition import wrist_look_at, wrist_cover_ray


def observation(transform):
    ee = np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]]
    mount = np.eye(4); mount[:3, 3] = [.03, .02, .05]
    return {'state': {'left_ee_pose': ee}, 'vision': {'cam_left_wrist': {
        'color': np.zeros((240, 320, 3), np.uint8),
        'intrinsic_matrix': np.array([[300, 0, 160], [0, 280, 120], [0, 0, 1.]]),
        'extrinsic_matrix': transform @ mount}}}


def test_observed_mount_aim_is_frame_equivariant_and_preserves_ee_position():
    base = np.eye(4); base[:3, 3] = [.1, -.2, 1.3]
    target = np.array([.25, .05, .8])
    first = wrist_look_at(observation(base), 'left', target)
    change = np.eye(4)
    change[:3, :3] = Rotation.from_euler('xyz', [.2, -.4, 1.1]).as_matrix()
    change[:3, 3] = [2, -5, .8]
    moved = change @ base
    second = wrist_look_at(observation(moved), 'left', change[:3, :3] @ target + change[:3, 3])
    np.testing.assert_allclose(first['pose'][:3], base[:3, 3])
    np.testing.assert_allclose(second['pose'][:3], moved[:3, 3])
    np.testing.assert_allclose(second['camera_transform'], change @ first['camera_transform'], atol=1e-9)
    np.testing.assert_allclose(first['proposal_pixel'], [160, 120], atol=1e-6)


def test_invalid_or_excessive_view_change_is_rejected():
    obs = observation(np.eye(4))
    for target in ([0, 0, 1], [float('nan'), 0, -1], [.03, .02, .05]):
        with pytest.raises(GeometryUnavailable):
            wrist_look_at(obs, 'left', target)
    with pytest.raises(GeometryUnavailable):
        wrist_look_at(obs, 'left', [1, 0, -.1], maximum_rotation=.2)


def test_ray_search_uses_calibrated_robot_frame_without_assigned_object_depth():
    wrist = np.eye(4); wrist[:3, 3] = [.3, -.2, 1.1]
    obs = observation(wrist)
    head = np.eye(4); head[:3, 3] = [0, 0, 1.8]
    obs['vision']['cam_head'] = dict(obs['vision']['cam_left_wrist'], extrinsic_matrix=head)
    base = np.array([0, 0, .9])
    result = wrist_cover_ray(obs, 'left', [160, 120], base, .6)
    assert result['ray_coverage'] > .5
    np.testing.assert_allclose(result['search_segment'], [[0, 0, 1.5], [0, 0, .3]])
    change = np.eye(4); change[:3, 3] = [5, -4, 2]
    change[:3, :3] = Rotation.from_euler('z', .7).as_matrix()
    moved = observation(change @ wrist)
    moved['vision']['cam_head'] = dict(obs['vision']['cam_head'], extrinsic_matrix=change @ head)
    other = wrist_cover_ray(moved, 'left', [160, 120], change[:3, :3] @ base+change[:3, 3], .6)
    np.testing.assert_allclose(other['camera_transform'], change @ result['camera_transform'], atol=1e-8)
    assert other['ray_coverage'] == result['ray_coverage']
    with pytest.raises(GeometryUnavailable):
        wrist_cover_ray(obs, 'left', [160, 120], [10, 10, 0], .1)
