from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import GeometryUnavailable
from physicalrsi_observed_geometry.view_history import SupportViewHistory


def observation(x=0):
    vision={}
    for name,offset in [('cam_head',0),('cam_left_wrist',-.2),('cam_right_wrist',.2)]:
        t=np.eye(4);t[:3,3]=[x+offset,0,2]
        vision[name]={'color':np.zeros((8,8,3),np.uint8),
                      'intrinsic_matrix':np.array([[10,0,4],[0,10,4],[0,0,1]]),
                      'extrinsic_matrix':t}
    return {'vision':vision}


def test_snapshots_survive_sensor_buffer_reuse_and_reset():
    history=SupportViewHistory();obs=observation();history.observe(obs)
    obs['vision']['cam_head']['color'][:]=255
    obs['vision']['cam_left_wrist']['extrinsic_matrix'][0,3]=9
    assert history.cameras[0].image.max()==0
    assert history.cameras[1].transform[0,3]==-.2
    with pytest.raises(ValueError):history.cameras[0].image[:]=1
    history.reset()
    with pytest.raises(GeometryUnavailable):_ = history.cameras
    history.observe(obs)
    assert history.cameras[0].image.min()==255


def test_repeated_poses_do_not_accumulate_and_recent_views_are_bounded():
    history=SupportViewHistory(max_references=4)
    assert history.observe(observation())==2
    assert history.observe(observation(.001))==0
    for x in [.03,.06,.09]:history.observe(observation(x))
    assert len(history.cameras)==5
    np.testing.assert_allclose([c.transform[0,3] for c in history.cameras[1:]],[-.2,.2,-.11,.29])


def test_incomplete_observation_does_not_partially_update_history():
    history=SupportViewHistory();obs=observation();del obs['vision']['cam_right_wrist']
    with pytest.raises(GeometryUnavailable):history.observe(obs)
    assert history.source is None
