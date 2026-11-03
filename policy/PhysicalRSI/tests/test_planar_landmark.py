from pathlib import Path
import sys
import numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.planar_landmark import observe_planar_landmark


def scene(height):
    obs={'vision':{}};cameras={}
    for name,offset in [('cam_head',0),('cam_left_wrist',-.25),('cam_right_wrist',.3)]:
        transform=np.eye(4);transform[:3,3]=[offset,0,height+.8]
        camera=Camera(np.zeros((480,640,3),np.uint8),[[500,0,320],[0,500,240],[0,0,1]],transform)
        cameras[name]=camera
        obs['vision'][name]={'color':camera.image,'intrinsic_matrix':camera.k,'extrinsic_matrix':camera.transform}
    points=np.array([[x,y,height+.03*x] for x in [-.12,-.04,.04,.12] for y in [-.1,0,.1]])
    def matcher(a,b):return a.project(points)[0],b.project(points)[0],np.ones(len(points))
    return obs,cameras,matcher


@pytest.mark.parametrize('height',[.52,1.17])
def test_planar_landmark_scale_and_height_come_from_calibrated_views(height):
    obs,cameras,matcher=scene(height);target=np.array([.02,-.03,height+.0006])
    pixel=cameras['cam_head'].project([target])[0][0]
    mask=np.ones((480,640),bool)
    # A visible recess can lie between the texture samples, not on their mask.
    u,v=np.rint(pixel).astype(int);mask[v-2:v+3,u-2:u+3]=False
    actual,evidence=observe_planar_landmark(obs,mask,pixel,matcher=matcher)
    np.testing.assert_allclose(actual,target,atol=1e-6)
    assert len(evidence['views'])==2 and not evidence['uses_object_geometry']


def test_no_matches_and_extrapolated_targets_are_rejected():
    obs,cameras,matcher=scene(.8);mask=np.ones((480,640),bool)
    with pytest.raises(GeometryUnavailable):
        observe_planar_landmark(obs,mask,[320,240],matcher=lambda a,b:(np.empty((0,2)),np.empty((0,2)),np.empty(0)))
    with pytest.raises(GeometryUnavailable):observe_planar_landmark(obs,mask,[10,10],matcher=matcher)
