from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from source_edits import STAGING_DEFINITION


def staging(translation):
    ns={'np':np,'StackBlocksEnvelopeError':ValueError};exec(STAGING_DEFINITION,ns)
    shift=np.asarray(translation);plane=np.array([.03,-.02,.61+shift[2]-.03*shift[0]+.02*shift[1]])
    geometry={label:NS(center=np.array(center)+shift,half_extents=np.array(size)) for label,center,size in
              [('base',[.4,.1,.65],[.03,.025,.04]),('target',[-.3,-.1,.65],[.02,.02,.04]),('other',[0,.3,.65],[.025,.025,.04])]}
    policy=NS(rgb_observed_shapes={0:{'target':{'support_plane':plane.tolist()}}})
    point=ns['_staging_point'](policy,0,geometry,['base','target','other'],1)
    assert point is not None
    assert abs(point[2]-(plane[:2]@point[:2]+plane[2]+np.linalg.norm(np.r_[-plane[:2],1.])*.04))<1e-12
    return point


def test_staging_moves_with_observed_scene_and_uses_measured_height():
    origin=staging([0,0,0]);shift=np.array([.8,-.5,.42]);translated=staging(shift)
    np.testing.assert_allclose(translated,origin+shift,atol=1e-12)


def test_staging_has_no_fixed_height_fallback():
    ns={'np':np,'StackBlocksEnvelopeError':ValueError};exec(STAGING_DEFINITION,ns)
    geometry={'a':NS(center=np.zeros(3)),'b':NS(center=np.ones(3))}
    with pytest.raises(ValueError,match='Observed support'):
        ns['_staging_point'](NS(),0,geometry,['a','b'],1)
