from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.supported_outline import fit_supported_outline


@pytest.mark.parametrize('level,height,scale',[(.62,.029,1.),(.98,.047,1.4)])
def test_observed_outline_measures_unknown_height_without_template(level,height,scale):
    angles=np.linspace(0,2*np.pi,16,endpoint=False)
    radius=.03*scale*(1+.12*np.cos(angles*3))
    xy=np.c_[np.cos(angles),np.sin(angles)]*radius[:,None]
    center=np.array([.02,.01,level+height/2])
    points=np.r_[np.c_[xy,np.full(16,-height/2)],np.c_[xy,np.full(16,height/2)]]+center
    views=[]
    for shift in [-.3,0,.3]:
        transform=np.eye(4);transform[:3,3]=[shift,-.25,level+.9]
        transform[:3,:3]=Rotation.from_euler('x',18,degrees=True).as_matrix()
        camera=Camera(np.zeros((480,640,3),np.uint8),[[600,0,320],[0,600,240],[0,0,1]],transform)
        mask=np.zeros((480,640),np.uint8)
        poly=cv2.convexHull(np.rint(camera.project(points)[0]).astype('int32'))
        cv2.fillConvexPoly(mask,poly,1);views.append((camera,mask.astype(bool)))
    result=fit_supported_outline(views,[0,0,level])
    np.testing.assert_allclose(result['center'],center,atol=.003)
    assert abs(result['height']-height)<.003
    assert result['rms_px']<1.
    with pytest.raises(GeometryUnavailable):fit_supported_outline(views[:1],[0,0,level])
    with pytest.raises(GeometryUnavailable):fit_supported_outline([views[0],views[0]],[0,0,level])
