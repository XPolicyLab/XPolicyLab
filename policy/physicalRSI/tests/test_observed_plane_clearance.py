from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import GeometryUnavailable
from physicalrsi_observed_geometry.support_collision import plane_clearance


@pytest.mark.parametrize('angles',[(0.,0.),(.2,-.15),(-.3,.25)])
def test_clearance_tracks_rotated_translated_surface(angles):
    from scipy.spatial.transform import Rotation
    rotation=Rotation.from_euler('xy',angles).as_matrix()
    normal=rotation[:,2]
    origin=np.array([.17,-.23,1.07])
    # Construct points at known signed distances, independently of plane equation.
    on_plane=np.array([[0.,0.,0.],[.1,.2,0.],[-.2,.1,0.]])@rotation.T+origin
    vertices=on_plane+np.array([.02,.04,.03])[:,None]*normal
    plane=np.r_[-normal[:2]/normal[2],normal@origin/normal[2]]
    assert plane_clearance(vertices,plane)==pytest.approx(.02)
    assert plane_clearance(vertices-.05*normal,plane)==pytest.approx(-.03)


def test_horizontal_plane_preserves_old_distance_definition():
    vertices=np.array([[.3,-.2,.9],[-.1,.4,.83]])
    assert plane_clearance(vertices,[0.,0.,.71])==pytest.approx(.12)
    assert plane_clearance(vertices,[0.,0.,.81])==pytest.approx(.02)


@pytest.mark.parametrize('plane',[None,[0.,0.],[0.,float('nan'),1.]])
def test_no_default_support_height(plane):
    with pytest.raises(GeometryUnavailable):
        plane_clearance(np.zeros((3,3)),plane)
