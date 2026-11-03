from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.partial_rectangle import fit_partial_rectangle


@pytest.mark.parametrize('height,dimensions,tilt',[(.63,[.27,.14],[.05,-.04,.2]),(1.12,[.19,.085],[-.03,.07,-.15])])
@pytest.mark.parametrize('provide_depth_seed',[False,True])
def test_calibrated_clipped_views_recover_unknown_plane(height,dimensions,tilt,provide_depth_seed):
    center=np.array([.02,0,height]);rotation=Rotation.from_euler('xyz',tilt).as_matrix()
    points=(np.array([[-.5,-.5,0],[.5,-.5,0],[.5,.5,0],[-.5,.5,0]])*np.r_[dimensions,1])@rotation.T+center
    views=[];clipped_count=0
    for shift in (0,-.48,.48):
        transform=np.eye(4);transform[:3,3]=[shift,0,height+.8]
        camera=Camera(np.zeros((480,640,3),np.uint8),np.array([[600,0,320],[0,600,240],[0,0,1]]),transform)
        poly=camera.project(points)[0].astype('float32')
        border=np.array([[0,0],[639,0],[639,479],[0,479]],dtype='float32')
        area,clipped=cv2.intersectConvexConvex(poly,border)
        assert area>0
        clipped_count+=int(np.any(poly[:,0]<0) or np.any(poly[:,0]>639))
        views.append((camera,clipped.reshape(-1,2)))
    assert clipped_count==2
    observed,rot,dims,rms=fit_partial_rectangle(views,center+[.01,-.01,-.02] if provide_depth_seed else None)
    np.testing.assert_allclose(observed,center,atol=.002)
    np.testing.assert_allclose(rot[:,2],rotation[:,2],atol=.02)
    np.testing.assert_allclose(np.sort(dims),np.sort(dimensions),atol=.002)
    assert rms<.15
    with pytest.raises(GeometryUnavailable):fit_partial_rectangle(views[:1],center)
    with pytest.raises(GeometryUnavailable):fit_partial_rectangle([views[0],views[0]],center)
