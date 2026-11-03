from pathlib import Path
import sys
import numpy as np
import cv2
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.supported_frustum import fit_supported_frustum

@pytest.mark.parametrize('height,bottom,top,xy',[(.10,.04,.05,[0.,0.]),(.16,.03,.045,[.12,.08]),(.075,.035,.028,[-.1,.02])])
def test_frustum_sizes_are_recovered_from_current_projection(height,bottom,top,xy):
    transform=np.eye(4);transform[:3,3]=[0.,-.4,1.5];transform[:3,:3]=Rotation.from_euler('x',30,degrees=True).as_matrix()
    camera=Camera(np.zeros((480,640,3),np.uint8),[[600,0,320],[0,600,240],[0,0,1]],transform)
    angle=np.linspace(0,2*np.pi,300,endpoint=False);circle=np.c_[np.cos(angle),np.sin(angle)]
    base=np.r_[xy,.7];points=np.r_[base+np.c_[circle*bottom,np.zeros(len(circle))],base+np.c_[circle*top,np.full(len(circle),height)]]
    pixels=camera.project(points)[0];mask=np.zeros((480,640),np.uint8);cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(pixels).astype('int32')),1)
    fit=fit_supported_frustum([(camera,mask)],np.array([0.,0.,.7]))
    np.testing.assert_allclose(fit['base'],base,atol=.004)
    np.testing.assert_allclose([fit['height'],fit['bottom_radius'],fit['top_radius']],[height,bottom,top],atol=.004)
    assert fit['hidden_surfaces_reconstructed']


def test_clipped_frustum_is_not_given_invented_dimensions():
    camera=Camera(np.zeros((100,100,3),np.uint8),[[100,0,50],[0,100,50],[0,0,1]],np.eye(4))
    mask=np.zeros((100,100),bool);mask[:50,30:70]=True
    with pytest.raises(GeometryUnavailable,match='unclipped'):fit_supported_frustum([(camera,mask)],[0,0,-1])


def test_overhead_circle_does_not_determine_hidden_height():
    transform=np.eye(4);transform[2,3]=1.5
    camera=Camera(np.zeros((160,160,3),np.uint8),[[200,0,80],[0,200,80],[0,0,1]],transform)
    mask=np.zeros((160,160),np.uint8);cv2.circle(mask,(80,80),20,1,-1)
    with pytest.raises(GeometryUnavailable,match='ambiguous'):fit_supported_frustum([(camera,mask)],[0,0,.7])
