from pathlib import Path
from itertools import product
import sys
import cv2
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera, GeometryUnavailable
from physicalrsi_observed_geometry.occlusion import verify_partial_box


def scene():
    camera=Camera(np.zeros((240,320,3),np.uint8),[[300,0,160],[0,300,120],[0,0,1]],np.eye(4))
    center=np.array([0.,0.,-1.]);dimensions=np.array([.2,.16,.12])
    pixels=camera.project(np.asarray(list(product([-.5,.5],repeat=3)))*dimensions+center)[0]
    mask=np.zeros((240,320),np.uint8)
    cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(pixels).astype(np.int32)),1)
    occluder=np.zeros_like(mask,bool);occluder[:120]=True
    return camera,center,dimensions,mask.astype(bool),occluder


def test_partial_observation_retains_supported_pose():
    c,p,d,mask,occ=scene()
    result=verify_partial_box(c,p,np.eye(3),d,mask&~occ,occ)
    assert result['recall']==1 and result['precision']==1


@pytest.mark.parametrize('failure',['moved','missing','unexplained_occlusion','almost_hidden'])
def test_partial_observation_rejects_contradiction_or_insufficient_evidence(failure):
    c,p,d,mask,occ=scene();observed=mask&~occ
    if failure=='moved':observed=np.roll(observed,14,axis=1)
    elif failure=='missing':observed[:]=False
    elif failure=='unexplained_occlusion':occ[:]=False
    else:occ[:140]=True;observed=mask&~occ
    with pytest.raises(GeometryUnavailable):verify_partial_box(c,p,np.eye(3),d,observed,occ)
