from pathlib import Path
import sys
import numpy as np
import cv2
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera, GeometryUnavailable
from physicalrsi_observed_geometry.multiview_circle import fit_circle


def views_for(center, radius):
    result=[]
    t=np.linspace(0,2*np.pi,720,endpoint=False)
    points=np.array(center)+radius*np.c_[np.cos(t),np.sin(t),np.zeros(len(t))]
    for location,angle in [([0,-.4,1.5],30),([0,.3,1.4],-25)]:
        transform=np.eye(4);transform[:3,3]=location;transform[:3,:3]=Rotation.from_euler('x',angle,degrees=True).as_matrix()
        camera=Camera(np.zeros((480,640,3),np.uint8),[[600,0,320],[0,600,240],[0,0,1]],transform)
        pixels=camera.project(points)[0];mask=np.zeros((480,640),np.uint8)
        cv2.fillConvexPoly(mask,np.rint(pixels).astype('int32'),1)
        result.append((camera,mask))
    return result


@pytest.mark.parametrize('center,radius',[([0,0,.8],.025),([.1,.04,.9],.045),([-.1,-.03,.7],.018)])
def test_unknown_center_and_radius(center,radius):
    fit=fit_circle(views_for(center,radius),[0,0,1])
    np.testing.assert_allclose(fit['center'],center,atol=.002)
    assert abs(fit['radius']-radius)<.0015


def test_no_stereo_baseline():
    view=views_for([0,0,.8],.03)[0]
    with pytest.raises(GeometryUnavailable,match='ambiguous'):
        fit_circle([view,view],[0,0,1])


def test_clipped_mask():
    views=views_for([0,0,.8],.03);views[0][1][:100,:10]=1
    with pytest.raises(GeometryUnavailable,match='unclipped'):
        fit_circle(views,[0,0,1])


def test_inconsistent_radius_between_views():
    views=views_for([0,0,.8],.03);views[1]=views_for([0,0,.8],.07)[1]
    with pytest.raises(GeometryUnavailable,match='inconsistent|observable geometry'):
        fit_circle(views,[0,0,1])


def test_multiple_valid_openings_are_not_silently_selected():
    from physicalrsi_observed_geometry.multiview_circle import select_circle
    a=views_for([-.08,0,.8],.03);b=views_for([.08,0,.8],.03)
    groups=[(a[i][0],[a[i][1],b[i][1]]) for i in range(2)]
    with pytest.raises(GeometryUnavailable,match='Multiple plausible'):
        select_circle(groups,[0,0,1])


def test_duplicate_proposals_are_not_ambiguity():
    from physicalrsi_observed_geometry.multiview_circle import select_circle
    views=views_for([0,0,.8],.03)
    fit=select_circle([(c,[m,m.copy()]) for c,m in views],[0,0,1])
    np.testing.assert_allclose(fit['center'],[0,0,.8],atol=.002)
