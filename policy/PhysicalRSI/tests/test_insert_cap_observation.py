import ast
from pathlib import Path
import sys
import numpy as np
import cv2
import pytest
from scipy.spatial.transform import Rotation
r=Path(__file__).resolve().parents[1]/'asset_revision';sys.path.insert(0,str(r))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
source=ast.parse((r/'overlays/insert_rgb_perception.py').read_text());ns={'np':np,'cv2':cv2,'GeometryUnavailable':GeometryUnavailable}
exec(compile(ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='cap_xy_from_rgb'],type_ignores=[]),'<cap>','exec'),ns)


@pytest.mark.parametrize('height',[.67,1.13])
def test_cap_servo_uses_current_calibration_and_predicted_height(height):
    image=np.full((480,640,3),[150,100,60],np.uint8);image[400:420,30:60]=245
    transform=np.eye(4);transform[:3,3]=[.1,-.2,1.8];transform[:3,:3]=Rotation.from_euler('x',15,degrees=True).as_matrix()
    camera=Camera(image,[[600,0,320],[0,600,240],[0,0,1]],transform)
    expected=np.array([.02,.03,height]);pixel=np.rint(camera.project([expected])[0][0]).astype(int)
    # Table shares hue with the cap but is not a localized bright observation.
    with pytest.raises(GeometryUnavailable):ns['cap_xy_from_rgb'](camera,expected)
    cv2.circle(image,tuple(pixel),5,(255,210,130),-1)
    np.testing.assert_allclose(ns['cap_xy_from_rgb'](camera,expected),expected[:2],atol=.002)
    with pytest.raises(GeometryUnavailable):ns['cap_xy_from_rgb'](camera,expected+[.2,0,0])


def test_large_bright_orange_region_is_not_a_cap():
    image=np.full((480,640,3),[255,210,130],np.uint8);image[:20,:20]=245
    transform=np.eye(4);transform[2,3]=2
    camera=Camera(image,[[600,0,320],[0,600,240],[0,0,1]],transform)
    with pytest.raises(GeometryUnavailable):ns['cap_xy_from_rgb'](camera,[0,0,1])


def test_shadowed_robot_pixels_do_not_merge_cap_with_wood():
    image=np.full((480,640,3),[168,115,75],np.uint8)
    image[350:390,30:130]=160
    image[390:420,30:130]=220
    transform=np.eye(4);transform[2,3]=2
    camera=Camera(image,[[600,0,320],[0,600,240],[0,0,1]],transform)
    expected=np.array([0.,0.,1.])
    with pytest.raises(GeometryUnavailable):ns['cap_xy_from_rgb'](camera,expected)
    cv2.circle(image,(320,240),9,(245,210,172),-1)
    np.testing.assert_allclose(ns['cap_xy_from_rgb'](camera,expected),expected[:2],atol=1e-6)
