import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import GeometryUnavailable
from physicalrsi_observed_geometry.pickup_target import select_target_region
from pickup_geometry_edits import replace_pickup_depth_scale,replace_pickup_target_region


def test_target_scope_excludes_neighbor_and_rejects_ambiguity():
    target=np.zeros((50,80),bool);target[10:30,15:25]=True
    neighbor=np.zeros_like(target);neighbor[10:30,60:70]=True
    result=SimpleNamespace(masks=[neighbor,target],scores=[.99,.8])
    np.testing.assert_array_equal(select_target_region([result],[14,9,26,31],target.shape),target)
    competing=target.copy();competing[15:25,18:25]=False
    with pytest.raises(GeometryUnavailable,match='Ambiguous'):
        select_target_region([SimpleNamespace(masks=[target,competing],scores=[.9,.85])],[14,9,26,31],target.shape)
    with pytest.raises(GeometryUnavailable,match='No segmented'):
        select_target_region([SimpleNamespace(masks=[neighbor],scores=[.99])],[14,9,26,31],target.shape)


def test_release_affordance_requires_target_mask_and_has_no_box_center_fallback():
    root=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_master')
    if not root.exists():pytest.skip('Release fixture unavailable')
    paths=list(root.glob('*/rdj_master/tasks/general_pickup/rgb_perception.py'));assert paths
    for path in paths:
        text=replace_pickup_target_region(replace_pickup_depth_scale(path.read_text()))
        assert 'crop_height >' not in text and 'round((x0 + x1)' not in text
        tree=ast.parse(text)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_depth_affordance')
        assert [a.arg for a in fn.args.kwonlyargs]==['observation','target_mask']
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_depth_affordance']
        assert len(calls)==1
        maskarg=next(k.value for k in calls[0].keywords if k.arg=='target_mask')
        assert isinstance(maskarg,ast.Call) and maskarg.func.id=='observe_pickup_target'
