from pathlib import Path
import sys,itertools
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.boxes import fit_box_views


def views(center,rotation,dimensions):
    result=[]
    points=(np.asarray(list(itertools.product([-.5,.5],repeat=3)))*dimensions)@rotation.T+center
    for shift in (-.3,0,.3):
        transform=np.eye(4);transform[:3,3]=[shift,-.25,2]
        transform[:3,:3]=Rotation.from_euler('x',18,degrees=True).as_matrix()
        camera=Camera(np.zeros((480,640,3),np.uint8),np.array([[600,0,320],[0,600,240],[0,0,1]]),transform)
        result.append((camera,camera.project(points)[0]))
    return result


@pytest.mark.parametrize('center,dimensions,yaw', [([.1,0,.61],[.03,.05,.09],.2),([-.05,.03,1.13],[.07,.04,.055],-.4)])
def test_unknown_box_height_and_dimensions_are_fitted(center,dimensions,yaw):
    rotation=Rotation.from_euler('z',yaw).as_matrix()
    result=fit_box_views(views(center,rotation,dimensions))
    np.testing.assert_allclose(result['center'],center,atol=.001)
    np.testing.assert_allclose(np.sort(result['dimensions']),np.sort(dimensions),atol=.001)
    assert result['rms_px']<.1


def test_rotated_box_tracks_with_only_previously_measured_size():
    center=[.03,.02,.95];dimensions=[.04,.07,.08]
    rotation=Rotation.from_euler('xyz',[.2,-.15,.4]).as_matrix()
    prior=Rotation.from_euler('xyz',[.1,-.1,.3]).as_matrix()
    result=fit_box_views(views(center,rotation,dimensions),previous_rotation=prior,measured_dimensions=dimensions)
    np.testing.assert_allclose(result['center'],center,atol=.001)
    assert Rotation.from_matrix(result['rotation']@rotation.T).magnitude()<.01


def test_box_fit_does_not_fallback_to_asset_geometry():
    with pytest.raises(GeometryUnavailable):fit_box_views([])


def test_single_view_tracking_requires_previously_observed_shape_and_pose():
    center=np.array([.03,.02,.95]);dimensions=np.array([.04,.07,.08])
    rotation=Rotation.from_euler('xyz',[.2,-.15,.4]).as_matrix()
    prior=Rotation.from_euler('xyz',[.15,-.1,.35]).as_matrix()
    single=views(center,rotation,dimensions)[:1]
    with pytest.raises(GeometryUnavailable):fit_box_views(single)
    result=fit_box_views(single,previous_rotation=prior,measured_dimensions=dimensions,
                         previous_center=center+[.01,-.01,.01])
    np.testing.assert_allclose(result['center'],center,atol=.002)
    assert result['rms_px']<.1


@pytest.mark.parametrize('plane,dimensions', [([.04,-.03,.59],[.03,.05,.06]),([-.02,.05,1.03],[.07,.04,.05])])
def test_supported_tracking_uses_measured_tilt_height_and_dimensions(plane,dimensions):
    from physicalrsi_observed_geometry.boxes import fit_supported_box_pose
    plane=np.asarray(plane);dimensions=np.asarray(dimensions)
    normal=np.r_[-plane[:2],1.];normal/=np.linalg.norm(normal)
    axis=np.array([1.,0.,0.]);axis-=normal*(axis@normal);axis/=np.linalg.norm(axis)
    basis=np.column_stack([axis,np.cross(normal,axis),normal])
    center=basis[:,:2]@np.array([.03,.02])+normal*(plane[2]/np.linalg.norm(np.r_[-plane[:2],1.])+dimensions[2]/2)
    rotation=basis@Rotation.from_euler('z',.3).as_matrix()
    camera,contour=views(center,rotation,dimensions)[0]
    result=fit_supported_box_pose(camera,contour,plane,dimensions,center+[.01,-.01,.02],rotation)
    np.testing.assert_allclose(result['center'],center,atol=.001)
    np.testing.assert_allclose(result['dimensions'],dimensions)
    assert result['rms_px']<.1


def test_tracking_handles_box_clipped_by_image_boundary():
    from physicalrsi_observed_geometry.boxes import image_clipped_polygon
    center=np.array([.03,.02,.95]);dimensions=np.array([.04,.07,.08])
    rotation=Rotation.from_euler('xyz',[.2,-.15,.4]).as_matrix()
    camera,points=views(center,rotation,dimensions)[0]
    camera=Camera(np.zeros((480,515,3),np.uint8),camera.k,camera.transform)
    import cv2
    hull=cv2.convexHull(points.astype('float32')).reshape(-1,2)
    visible=image_clipped_polygon(hull,camera.width,camera.height)
    result=fit_box_views([(camera,visible)],previous_rotation=rotation,
                        measured_dimensions=dimensions,previous_center=center+[.002,-.002,.003])
    np.testing.assert_allclose(result['center'],center,atol=.003)
    assert result['rms_px']<.1
