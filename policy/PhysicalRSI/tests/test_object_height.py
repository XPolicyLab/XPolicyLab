from pathlib import Path
import sys
import numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.object_height import measured_height


def setup_points(table,height):
    cameras=[]
    for offset in (0.,-.3,.3):
        transform=np.eye(4);transform[:3,3]=[offset,0,table+.65]
        cameras.append(Camera(np.zeros((480,640,3),np.uint8),[[700,0,320],[0,700,240],[0,0,1]],transform))
    points=np.array([[x,y,table+height] for x in [-.04,-.02,.02,.04] for y in [-.03,.03]])
    def matcher(a,b):return a.project(points)[0],b.project(points)[0],np.ones(len(points))
    return cameras,matcher


@pytest.mark.parametrize('table,height',[(.51,.009),(1.12,.026),(.83,.047)])
def test_glyph_height_tracks_observations_not_identity(table,height):
    cameras,matcher=setup_points(table,height)
    observed,evidence=measured_height(cameras,np.ones((480,640),bool),[0,0,table],matcher=matcher)
    assert abs(observed-height)<1e-8
    assert evidence['points']>=4 and not evidence['uses_asset_dimensions']


def test_support_pixels_are_not_accepted_as_object_thickness():
    cameras,matcher=setup_points(.7,0.)
    with pytest.raises(GeometryUnavailable):
        measured_height(cameras,np.ones((480,640),bool),[0,0,.7],matcher=matcher)


def test_unknown_height_does_not_fall_back_to_a_constant():
    cameras,_=setup_points(.7,.02)
    with pytest.raises(GeometryUnavailable):
        measured_height(cameras,np.ones((480,640),bool),[0,0,.7],matcher=lambda a,b:([],[],[]))
