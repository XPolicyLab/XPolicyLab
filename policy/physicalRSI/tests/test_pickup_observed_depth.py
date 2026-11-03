"""Actual release affordance must follow observed geometry across world frames."""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from pickup_geometry_edits import replace_pickup_depth_scale
from physicalrsi_observed_geometry import Camera, GeometryUnavailable
import physicalrsi_observed_geometry.calibrated_depth_scene as calibrated
SOURCE=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_master')

@pytest.mark.skipif(not SOURCE.exists(),reason='Release fixture not installed')
def test_all_variants_follow_observed_support_and_fail_without_it(monkeypatch):
    paths=list(SOURCE.glob('*/rdj_master/tasks/general_pickup/rgb_perception.py'))
    assert paths
    for path in paths:
        text=replace_pickup_depth_scale(path.read_text())
        assert '_TABLE_FORWARD_DISTANCE' not in text and '0.778' not in text
        tree=ast.parse(text)
        original=ast.parse(path.read_text())
        original_detect=next(n for n in ast.walk(original) if isinstance(n,ast.FunctionDef) and n.name=='detect_batch')
        revised_detect=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='detect_batch')
        calls=[n for n in ast.walk(revised_detect) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_depth_affordance']
        assert len(calls)==1 and [k.arg for k in calls[0].keywords]==['observation']
        calls[0].keywords=[]
        assert ast.dump(original_detect)==ast.dump(revised_detect), 'Grounding and target selection must remain unchanged'
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_depth_affordance')
        scope={'np':np}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'release affordance','exec'),scope)
        observation={'identity':object()}
        depth=np.ones((80,100));depth[25:55,30:70]=.93
        for support_z in [.45,1.25]:
            transform=np.eye(4);transform[:3,3]=[.2,-.3,support_z+1.]
            camera=Camera(np.zeros((80,100,3),dtype=np.uint8),np.array([[100.,0,50],[0,100,40],[0,0,1.]]),transform)
            def scene(obs,*,predictor):
                assert obs is observation
                np.testing.assert_array_equal(predictor(camera.image),depth)
                return SimpleNamespace(camera=camera,depth=depth,depth_calibration_report={'observed_plane':[0,0,support_z]})
            monkeypatch.setattr(calibrated,'calibrated_depth_scene',scene)
            pixel,z,axis,elongation=scope['_depth_affordance'](depth,(20,15,80,65),observation=observation)
            assert 30 <= pixel[0] <70 and 25 <= pixel[1] <55
            assert z==pytest.approx(support_z+.07)
        def unavailable(*args,**kwargs):raise GeometryUnavailable('No measured support')
        monkeypatch.setattr(calibrated,'calibrated_depth_scene',unavailable)
        with pytest.raises(GeometryUnavailable,match='No measured support'):
            scope['_depth_affordance'](depth,(20,15,80,65),observation=observation)
