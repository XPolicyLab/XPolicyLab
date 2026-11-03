from pathlib import Path
import sys
import importlib
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera, GeometryUnavailable
module=importlib.import_module('physicalrsi_observed_geometry.calibrated_depth_scene')

@pytest.mark.parametrize('height',[.55,1.05])
@pytest.mark.parametrize('consistent',[True,False])
def test_learned_depth_must_agree_with_measured_plane(monkeypatch,height,consistent):
    k=np.array([[160.,0,80],[0,160,60],[0,0,1]])
    transform=np.eye(4);transform[:3,3]=[0,0,2]
    transform[:3,:3]=Rotation.from_euler('x',15,degrees=True).as_matrix()
    rgb=np.full((120,160,3),[120,80,40],np.uint8)
    camera=Camera(rgb,k,transform)
    yy,xx=np.indices(rgb.shape[:2]);pixels=np.c_[xx.ravel(),yy.ravel()]
    rays=camera.unproject(pixels,np.ones(len(pixels)))-transform[:3,3]
    plane=np.array([.02,-.03,height])
    depth=((height-2)/(rays[:,2]-.02*rays[:,0]+.03*rays[:,1])).reshape(120,160)
    raw=(depth-.1)/.16 if consistent else np.random.default_rng(54).uniform(1,8,depth.shape)
    before=raw.copy();vision={}
    for name,shift in [('cam_head',0),('cam_left_wrist',.3),('cam_right_wrist',-.3)]:
        pose=transform.copy();pose[0,3]=shift
        vision[name]={'color':rgb,'intrinsic_matrix':k,'extrinsic_matrix':pose}
    # Geometry inference has separate real-image/synthetic tests. This checks
    # the depth handoff: raw model scale may never silently reach perception.
    monkeypatch.setattr(module,'initial_plane',lambda cameras,masks,**kw:(plane.copy(),masks[0]))
    monkeypatch.setattr(module,'refine_plane',lambda cameras,masks,seed,**kw:seed)
    if not consistent:
        with pytest.raises(GeometryUnavailable,match='consensus'):
            module.calibrated_depth_scene({'vision':vision},predictor=lambda image:raw)
    else:
        scene=module.calibrated_depth_scene({'vision':vision},predictor=lambda image:raw)
        np.testing.assert_allclose(scene.depth,depth,atol=1e-9)
        assert scene.depth_calibration_report['holdout']['inlier_fraction']==1
    np.testing.assert_array_equal(raw,before)
