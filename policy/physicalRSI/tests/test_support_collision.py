from pathlib import Path
import sys,itertools
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.support_collision import support_patches,patches_in_robot_frame


@pytest.mark.parametrize('plane',[[0,0,.53],[.04,-.03,1.06]])
def test_collision_patch_tops_follow_observed_plane_and_transform_with_robot(plane):
    transform=np.eye(4);transform[2,3]=2.
    camera=Camera(np.zeros((120,160,3),np.uint8),[[160,0,80],[0,160,60],[0,0,1]],transform)
    mask=np.ones((120,160),bool);mask[40:80,60:100]=False
    patches=support_patches(camera,mask,plane,cell_size=.08)
    for patch in patches:
        r=Rotation.from_quat(np.asarray(patch['pose'])[ [4,5,6,3] ]).as_matrix()
        top=np.asarray(patch['pose'][:3])+r[:,2]*patch['dims'][2]/2
        assert abs(top[2]-(plane[0]*top[0]+plane[1]*top[1]+plane[2]))<1e-10
    q=Rotation.from_euler('xyz',[.2,-.1,.4]).as_quat()[[3,0,1,2]]
    base=np.r_[.3,-.4,.8,q];local=patches_in_robot_frame(patches,base)['cuboid']
    rotation=Rotation.from_quat(base[[4,5,6,3]])
    for original,converted in zip(patches,local.values()):
        recovered=rotation.apply(converted['pose'][:3])+base[:3]
        np.testing.assert_allclose(recovered,original['pose'][:3],atol=1e-12)


def test_collision_geometry_requires_observed_support():
    camera=Camera(np.zeros((20,20,3),np.uint8),[[20,0,10],[0,20,10],[0,0,1]],np.eye(4))
    with pytest.raises(GeometryUnavailable):support_patches(camera,np.zeros((20,20),bool),[0,0,.7])


def test_narrow_visible_surface_refines_grid_without_filling_unseen_pixels():
    transform=np.eye(4);transform[2,3]=1.4
    camera=Camera(np.zeros((120,160,3),np.uint8),[[160,0,80],[0,160,60],[0,0,1]],transform)
    mask=np.zeros((120,160),bool);mask[20:100,78:85]=True
    patches=support_patches(camera,mask,[0,0,.8])
    assert patches and all(p['dims'][2]<.05 for p in patches)
    for patch in patches:
        rotation=Rotation.from_quat(np.asarray(patch['pose'])[[4,5,6,3]]).as_matrix()
        # Dense sampling of each top face verifies that refinement preserves
        # the original full-footprint visibility rule.
        local=np.array([[x,y,.5] for x in np.linspace(-.5,.5,9) for y in np.linspace(-.5,.5,9)])*patch['dims']
        world=local@rotation.T+np.asarray(patch['pose'][:3])
        uv=np.rint(camera.project(world)[0]).astype(int)
        assert np.all(mask[uv[:,1],uv[:,0]])
