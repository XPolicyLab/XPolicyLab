from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import GeometryUnavailable
from physicalrsi_observed_geometry.key_outline import observed_key_tip


@pytest.mark.parametrize('scale',[.4,1.,2.3])
def test_tip_uses_visible_endpoint_and_changes_with_observed_size(scale):
    outline=scale*np.array([[-.01,0,.01],[.012,0,.01],[.008,0,-.03],[.003,0,-.07],[-.004,0,-.06]])
    tip=observed_key_tip(outline)
    np.testing.assert_array_equal(tip,outline[3])
    tip[:]=0
    assert outline[3,2]!=0


@pytest.mark.parametrize('outline',[[],[[0,0,0]]*3,[[0,0,0],[1,0,0],[0,0,np.nan]]])
def test_tip_has_no_fallback_for_unresolved_outline(outline):
    with pytest.raises(GeometryUnavailable):observed_key_tip(outline)
