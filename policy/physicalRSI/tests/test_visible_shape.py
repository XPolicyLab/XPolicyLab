from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.visible_shape import measure_visible_shape

@pytest.mark.parametrize('support,height',[(.4,.06),(1.1,.14)])
def test_visible_measurements_follow_observed_geometry(support,height):
    t=np.eye(4);t[:3,3]=[.3,-.2,2]
    camera=Camera(np.zeros((40,40,3),np.uint8),np.array([[80,0,20],[0,80,20],[0,0,1]]),t)
    mask=np.zeros((40,40),bool);mask[12:28,12:28]=True
    depth=np.full(mask.shape,2-support-height)
    result=measure_visible_shape(camera,depth,mask,[0,0,support])
    np.testing.assert_allclose(result['height'],height,atol=1e-10)
    np.testing.assert_allclose(result['base'][2],support,atol=1e-10)
    assert result['visible_only'] and not result['hidden_surfaces_reconstructed']
    assert not result['mask_touches_border']


def test_unobserved_geometry_is_not_filled_from_an_asset():
    camera=Camera(np.zeros((8,8,3),np.uint8),np.eye(3),np.eye(4))
    with pytest.raises(GeometryUnavailable,match='Too few'):
        measure_visible_shape(camera,np.full((8,8),np.nan),np.ones((8,8),bool),[0,0,0])
