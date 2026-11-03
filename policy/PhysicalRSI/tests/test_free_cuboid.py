from pathlib import Path
import sys,itertools
import cv2,numpy as np,pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.free_cuboid import fit_upright_cuboid


def views_for(center,dimensions,yaw):
    vertices=np.array(list(itertools.product([-.5,.5],repeat=3)))*dimensions
    vertices=vertices@Rotation.from_euler('z',yaw).as_matrix().T+center
    views=[]
    for offset in ([.32,-.28,.55],[-.28,.12,.45],[.1,.3,.7]):
        origin=np.asarray(center)+offset
        backward=np.asarray(offset)/np.linalg.norm(offset)
        right=np.cross([0,1.,0],backward);right/=np.linalg.norm(right)
        up=np.cross(backward,right)
        transform=np.eye(4);transform[:3,:3]=np.column_stack([right,up,backward]);transform[:3,3]=origin
        camera=Camera(np.zeros((480,640,3),np.uint8),[[600,0,320],[0,600,240],[0,0,1]],transform)
        projected=camera.project(vertices)[0]
        mask=np.zeros((480,640),'uint8');cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(projected).astype('int32')),1)
        views.append((camera,mask.astype(bool)))
    return views


@pytest.mark.parametrize('center,dimensions,yaw',[
    ([.0,.1,.51],[.035,.055,.08],.3),
    ([-.2,-.1,1.13],[.09,.04,.03],-.5),
    ([.25,.3,.86],[.022,.027,.025],.9)])
def test_unknown_box_size_and_height_from_moved_cameras(center,dimensions,yaw):
    result=fit_upright_cuboid(views_for(center,dimensions,yaw))
    np.testing.assert_allclose(result['center'],center,atol=.002)
    np.testing.assert_allclose(np.sort(result['dimensions'][:2]),np.sort(dimensions[:2]),atol=.002)
    assert abs(result['dimensions'][2]-dimensions[2])<.002
    assert abs(result['bottom_z']-(center[2]-dimensions[2]/2))<.002
    assert result['rms_px']<.8


def test_cuboid_rejects_missing_baseline_and_clipped_views():
    views=views_for([0,0,.9],[.05,.04,.03],.2)
    with pytest.raises(GeometryUnavailable):fit_upright_cuboid(views[:1])
    with pytest.raises(GeometryUnavailable):fit_upright_cuboid([views[0],views[0]])
    camera,mask=views[0];mask=mask.copy();mask[0,100:120]=True
    with pytest.raises(GeometryUnavailable,match='unclipped'):fit_upright_cuboid([(camera,mask),views[1]])


def test_inconsistent_views_do_not_supply_geometry():
    views=views_for([0,0,.8],[.06,.03,.04],.1)
    camera,mask=views[1];views[1]=(camera,np.roll(mask,35,axis=1))
    with pytest.raises(GeometryUnavailable):fit_upright_cuboid(views)
