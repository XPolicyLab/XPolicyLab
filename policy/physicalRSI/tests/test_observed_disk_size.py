from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.disk_size import observed_disk

@pytest.mark.parametrize('radius,height',[(.013,.55),(.034,1.1)])
@pytest.mark.parametrize('tilt',[0,35])
def test_disk_size_tracks_observed_scale_and_plane_without_radius_constant(radius,height,tilt):
    transform=np.eye(4);transform[:3,3]=[.01,0,height+.3]
    camera=Camera(np.zeros((480,640,3),np.uint8),[[700,0,320],[0,700,240],[0,0,1]],transform)
    rotation=Rotation.from_euler('x',tilt,degrees=True).as_matrix()
    theta=np.linspace(0,2*np.pi,90,endpoint=False)
    center=np.array([0,0,height]);points=center+np.c_[radius*np.cos(theta),radius*np.sin(theta),np.zeros(len(theta))]@rotation.T
    pixels=camera.project(points)[0];normal=rotation[:,2].copy()*4;unchanged=normal.copy()
    result=observed_disk(camera,pixels,center,normal)
    assert result['radius']==pytest.approx(radius,abs=1e-6)
    np.testing.assert_allclose(result['center'],center,atol=1e-6)
    np.testing.assert_array_equal(normal,unchanged)


def test_clipped_silhouette_does_not_silently_use_a_nominal_coin_size():
    camera=Camera(np.zeros((40,60,3),np.uint8),[[60,0,30],[0,60,20],[0,0,1]],np.eye(4))
    with pytest.raises(GeometryUnavailable,match='clipped'):
        observed_disk(camera,[[0,i] for i in range(12)],[0,0,-1],[0,0,1])


def test_occluded_half_disk_is_not_reported_as_a_complete_measurement():
    transform=np.eye(4);transform[2,3]=1.
    camera=Camera(np.zeros((480,640,3),np.uint8),[[700,0,320],[0,700,240],[0,0,1]],transform)
    theta=np.linspace(0,np.pi,60)
    points=np.c_[.025*np.cos(theta),.025*np.sin(theta),np.full(len(theta),.7)]
    with pytest.raises(GeometryUnavailable,match='complete disk'):
        observed_disk(camera,camera.project(points)[0],[0,0,.7],[0,0,1])
