"""Synthetic silhouette inputs; segmentation quality is not tested here."""
from pathlib import Path
import sys
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.pad_observation import observe_pad_surface


def scene(level,height):
    angles=np.linspace(0,2*np.pi,64,endpoint=False)
    xy=np.c_[.035*np.cos(angles),.05*np.sin(angles)]+[.02,.01]
    points=np.r_[np.c_[xy,np.full(len(xy),level)],np.c_[xy,np.full(len(xy),level+height)]]
    observation={'vision':{}}
    masks={}
    for index,shift in enumerate((-.3,0.,.3)):
        transform=np.eye(4);transform[:3,3]=[shift,-.25,level+.9]
        transform[:3,:3]=Rotation.from_euler('x',18,degrees=True).as_matrix()
        image=np.zeros((480,640,3),np.uint8)
        camera=Camera(image,[[600,0,320],[0,600,240],[0,0,1]],transform)
        mask=np.zeros((480,640),np.uint8)
        cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(camera.project(points)[0]).astype('int32')),1)
        observation['vision'][str(index)]={'color':image,'intrinsic_matrix':camera.k,'extrinsic_matrix':transform}
        masks[id(image)]=mask.astype(bool)
    def segment(image,prompts):
        return [SimpleNamespace(masks=[masks[id(image)]],scores=[.9])]
    return observation,segment


@pytest.mark.parametrize('level,height',[(.61,.009),(.93,.031),(1.08,.054)])
def test_pad_thickness_comes_from_views(level,height):
    observation,segment=scene(level,height)
    result=observe_pad_surface(observation,segment,[0,0,level],[.02,.01])
    assert result['views']==3
    assert result['surface_z']==pytest.approx(level+height,abs=.003)
    assert result['height']==pytest.approx(height,abs=.003)


def test_missing_or_ambiguous_pad_has_no_fixed_height_fallback():
    observation,segment=scene(.82,.021)
    with pytest.raises(GeometryUnavailable,match='multiple calibrated silhouettes'):
        observe_pad_surface(observation,lambda *args:[],[0,0,.82],[.02,.01])
    def ambiguous(*args):
        group=segment(*args)[0]
        return [SimpleNamespace(masks=group.masks*2,scores=group.scores*2)]
    with pytest.raises(GeometryUnavailable,match='Ambiguous'):
        observe_pad_surface(observation,ambiguous,[0,0,.82],[.02,.01])
