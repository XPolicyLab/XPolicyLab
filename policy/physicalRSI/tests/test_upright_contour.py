from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.upright_contour import refine_mask,pixel_scale


@pytest.mark.parametrize('table,height,x,focal',[(.43,.04,-.08,500),(1.2,.19,.15,680)])
def test_contour_height_uses_current_camera_and_support(table,height,x,focal):
    pose=np.eye(4);pose[:3,3]=[.02,-.45,table+.7]
    pose[:3,:3]=Rotation.from_euler('x',35,degrees=True).as_matrix()
    camera=Camera(np.zeros((480,640,3),np.uint8),[[focal,0,320],[0,focal,240],[0,0,1]],pose)
    # An upright planar face satisfies the explicit midpoint-axis hypothesis.
    vertices=np.array([[x-.02,0,table],[x+.02,0,table],[x+.02,0,table+height],[x-.02,0,table+height]])
    mask=np.zeros((480,640),np.uint8)
    cv2.fillConvexPoly(mask,np.rint(camera.project(vertices)[0]).astype('int32'),1)
    result=refine_mask(mask,{},camera=camera,plane=[0,0,table])
    assert result['geometric_valid']
    assert result['height_m']==pytest.approx(height,abs=.004)
    np.testing.assert_allclose(result['xy'],[x,0],atol=.004)
    assert result['height_source']=='current_calibrated_RGB_upright_contour'
    assert pixel_scale(camera,[0,0,table],[x,0])==pytest.approx(camera.project([[x,0,table]])[1][0]/focal)


def test_missing_observed_support_has_no_fixed_height_fallback():
    camera=Camera(np.zeros((100,100,3),np.uint8),[[100,0,50],[0,100,50],[0,0,1]],np.eye(4))
    with pytest.raises(GeometryUnavailable,match='observed support'):
        refine_mask(np.ones((100,100),np.uint8),{},camera=camera,plane=None)


def test_clipped_contour_is_not_given_a_height():
    camera=Camera(np.zeros((100,100,3),np.uint8),[[100,0,50],[0,100,50],[0,0,1]],np.eye(4))
    mask=np.zeros((100,100),np.uint8);mask[:50,30:60]=1
    assert refine_mask(mask,{},camera=camera,plane=[0,0,-1]) is None
