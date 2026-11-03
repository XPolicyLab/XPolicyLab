"""Metric tube fitting varies support height, length and diameter independently."""
from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.tubes import fit_resting_tube


@pytest.mark.parametrize('height,length,radius,yaw',[(.52,.07,.008,.4),(.94,.13,.016,-.3)])
def test_dimensions_and_root_follow_current_calibrated_outline(height,length,radius,yaw):
    transform=np.eye(4);transform[:3,3]=[0.,-.35,height+.7]
    transform[:3,:3]=Rotation.from_euler('x',25,degrees=True).as_matrix()
    camera=Camera(np.zeros((480,640,3),np.uint8),[[430,0,320],[0,430,240],[0,0,1]],transform)
    axis=np.array([np.sin(yaw),np.cos(yaw),0.]);side=np.cross(axis,[0,0,1]);root=np.array([-.03,.0,height+radius])
    points=[]
    for distance in [0.,length]:
        for angle in np.linspace(0,2*np.pi,96,endpoint=False):
            points.append(root+axis*distance+radius*(side*np.cos(angle)+np.array([0,0,np.sin(angle)])))
    uv=camera.project(points)[0];mask=np.zeros((480,640),np.uint8);cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(uv).astype('int32')),1)
    cap=camera.project([root+axis*length])[0][0]
    measured=fit_resting_tube(camera,mask,cap,[0,0,height])
    np.testing.assert_allclose(measured['pose'][:3],root,atol=.004)
    assert abs(measured['length']-length)<.004
    assert abs(measured['radius']-radius)<.003
    rotation=Rotation.from_quat(np.asarray(measured['pose'])[[4,5,6,3]]).as_matrix()
    assert rotation[:,1]@axis>.99


def test_unobserved_tube_has_no_dimensions_fallback():
    camera=Camera(np.zeros((80,100,3),np.uint8),[[100,0,50],[0,100,40],[0,0,1]],np.eye(4))
    with pytest.raises(GeometryUnavailable):fit_resting_tube(camera,np.zeros((80,100),bool),[50,40],[0,0,.7])
