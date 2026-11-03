from pathlib import Path
import sys
from itertools import product
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.upright_row import observe_upright_row


def scene(height,thickness,width,support,angle,focal):
    t=np.eye(4);t[:3,:3]=Rotation.from_euler('x',angle,degrees=True).as_matrix();t[:3,3]=[0.,-.5,support+.6]
    c=Camera(np.zeros((480,640,3),np.uint8),[[focal,0,320],[0,focal,240],[0,0,1]],t)
    xs=np.linspace(-3*width,3*width,7)
    corners=np.array(list(product((xs[0],xs[-1]),(-.18,-.18+thickness),(support,support+height))))
    pixels=c.project(corners)[0];mask=np.zeros((480,640),np.uint8)
    cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(pixels).astype('int32')),1)
    bbox=cv2.boundingRect(cv2.findNonZero(mask))
    front=c.project(np.c_[xs,np.full(7,-.18),np.full(7,support+height)])[0]
    return c,mask,bbox,float(front[0,1]),front[:,0],[0,0,support]


@pytest.mark.parametrize('values',[(.05,.025,.035,.6,30,420),(.095,.05,.065,.91,35,320),(.075,.038,.045,.78,40,400)])
def test_dimensions_follow_visible_geometry_and_calibration(values):
    result=observe_upright_row(*scene(*values))
    height,thickness,width,*_=values
    assert abs(result['height_m']-height)<result['height_uncertainty_m']
    assert abs(result['thickness_m']-thickness)<result['thickness_uncertainty_m']
    assert abs(result['width_m']-width)<2*result['pixel_metric_m']
    assert not result['evidence']['uses_object_assets']


def test_missing_or_clipped_row_has_no_size_fallback():
    args=list(scene(.075,.038,.045,.78,40,400))
    empty=args.copy();empty[1]=np.zeros_like(args[1])
    with pytest.raises(GeometryUnavailable):observe_upright_row(*empty)
    clipped=args.copy();clipped[2]=(0,args[2][1],args[2][2],args[2][3])
    with pytest.raises(GeometryUnavailable):observe_upright_row(*clipped)
