from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.object_points import observed_object_points,initial_size_order


def scene(table,height,width):
    observation={'vision':{}}
    for name,x in [('cam_head',0.),('cam_left_wrist',.3)]:
        pose=np.eye(4);pose[:3,3]=[x,-.4,table+.65];pose[:3,:3]=Rotation.from_euler('x',32,degrees=True).as_matrix()
        observation['vision'][name]={'color':np.zeros((480,640,3),np.uint8),'intrinsic_matrix':np.array([[600,0,320],[0,600,240],[0,0,1]]),'extrinsic_matrix':pose}
    points=np.array([[x,0,table+z] for x in np.linspace(-width/2,width/2,8) for z in np.linspace(0,height,8)])
    head=Camera.from_observation(observation);uv=head.project(points)[0]
    contour=cv2.convexHull(np.rint(uv).astype('int32'))[:,0]
    doll={'geometric_valid':True,'xy':[0.,0.],'height_m':height,'contour':contour.tolist()}
    def matcher(a,b):return a.project(points)[0],b.project(points)[0],np.ones(len(points))
    return observation,doll,points,matcher


@pytest.mark.parametrize('table,height,width',[(.51,.06,.04),(1.15,.17,.08)])
def test_surface_points_follow_current_geometry(table,height,width):
    observation,doll,points,matcher=scene(table,height,width)
    profiles,evidence=observed_object_points(observation,[doll],[0,0,table],matcher=matcher)
    assert len(profiles[0])>=8
    np.testing.assert_allclose(np.ptp(profiles[0],axis=0),[width,0,height],atol=1e-6)
    assert evidence[0]['head_silhouette_iou']>=.65
    assert not evidence[0]['uses_asset_geometry'] and not evidence[0]['hidden_surfaces_reconstructed']


def test_missing_correspondences_do_not_load_a_profile():
    observation,doll,_,_=scene(.7,.1,.05)
    with pytest.raises(GeometryUnavailable,match='Insufficient'):
        observed_object_points(observation,[doll],[0,0,.7],matcher=lambda *args:([],[],[]))


def test_points_on_small_patch_cannot_stand_in_for_whole_object():
    observation,doll,points,_=scene(.7,.1,.05)
    center=points[(abs(points[:,0])<.015)&(points[:,2]>.73)&(points[:,2]<.78)]
    def matcher(a,b):return a.project(center)[0],b.project(center)[0],np.ones(len(center))
    with pytest.raises(GeometryUnavailable,match='cover'):
        observed_object_points(observation,[doll],[0,0,.7],matcher=matcher)


def test_initial_order_uses_observed_size_and_preserves_observed_heights():
    dolls=[dict(geometric_valid=True,size_metric=i*.02,size_pixel_quantum=.001,height_m=.035+i*.009) for i in (4,2,0,3,1)]
    ordered,report=initial_size_order(dolls)
    assert report['valid'] and [d['height_m'] for d in ordered]==sorted(d['height_m'] for d in dolls)
    for d in dolls:d['size_metric']=.1
    assert initial_size_order(dolls)[0] is None
