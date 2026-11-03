from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.support_projection import support_point


@pytest.mark.parametrize('plane',[[0.,0.,.62],[.08,-.05,.94]])
def test_current_camera_intersects_translated_or_sloped_support(plane):
    plane=np.asarray(plane)
    transform=np.eye(4);transform[:3,3]=[.2,-.1,1.6]
    camera=Camera(np.zeros((480,640,3),np.uint8),[[410.,0,301],[0,365.,219],[0,0,1]],transform)
    for xy in ([.1,-.2],[.3,.05]):
        point=np.r_[xy,plane[:2]@xy+plane[2]+.07]
        pixel=camera.project(point)[0][0]
        np.testing.assert_allclose(support_point(pixel,.07,camera=camera,support_plane=plane),point,atol=1e-12)


def test_support_projection_rejects_missing_or_behind_camera_plane():
    transform=np.eye(4);transform[2,3]=1.
    camera=Camera(np.zeros((10,10,3),np.uint8),[[10.,0,5],[0,10.,5],[0,0,1]],transform)
    for plane in ([0,0,np.nan],[0,0,2.],None):
        with pytest.raises(GeometryUnavailable):
            support_point([5,5],0,camera=camera,support_plane=plane)
