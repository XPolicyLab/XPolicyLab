import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from classify_language_depth import edit_language_depth_perception,bind_language_depth_observation,remove_language_surface_height_clamp
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry import calibrated_depth_scene as geometry


def test_height_map_uses_observed_camera_and_has_no_absolute_clip(monkeypatch):
    for samples,expected in [([.43,.45,.47],.45),([1.2,1.4,1.6],1.4),([-.4,-.2,0],-.2)]:
        assert geometry.observed_surface_height(samples)==pytest.approx(expected)
    for bad in [[],[np.nan],[.7,np.inf]]:
        with pytest.raises(GeometryUnavailable):geometry.observed_surface_height(bad)
    prediction=np.ones((12,16));obs=object()
    for height in [.4,1.4]:
        transform=np.eye(4);transform[2,3]=height+1
        camera=Camera(np.zeros((12,16,3),np.uint8),np.array([[20.,0,8],[0,20.,6],[0,0,1.]]),transform)
        depth=np.ones((12,16));depth[4:8,6:10]=.9
        def scene(observation,*,predictor):
            assert observation is obs
            np.testing.assert_array_equal(predictor(camera.image),prediction)
            return SimpleNamespace(camera=camera,depth=depth)
        monkeypatch.setattr(geometry,'calibrated_depth_scene',scene)
        world=geometry.observed_world_height(prediction,obs)
        assert world[0,0]==pytest.approx(height)
        assert world[5,7]==pytest.approx(height+.1)
    def unavailable(*a,**kw):raise GeometryUnavailable('Missing measured support')
    monkeypatch.setattr(geometry,'calibrated_depth_scene',unavailable)
    with pytest.raises(GeometryUnavailable,match='Missing measured support'):
        geometry.observed_world_height(prediction,obs)


def test_release_fork_preserves_planning_and_binds_calibration():
    root=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_v1_classify_objects_by_language')
    if not root.exists():pytest.skip('Release fixture unavailable')
    paths=list(root.glob('*/rdj_v1_classify_objects_by_language/rgb_perception.py'));assert paths
    for p in paths:
        after=edit_language_depth_perception(p.read_text())
        assert '_TABLE_FORWARD_DISTANCE' not in after and 'np.clip(world_z' not in after
        tree=ast.parse(after)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='calibrated_world_height')
        assert [a.arg for a in fn.args.kwonlyargs]==['observation','segment_fn','support_callback','prior_plane']
        before=ast.parse(p.with_name('model.py').read_text())
        bound=ast.parse(bind_language_depth_observation(p.with_name('model.py').read_text()))
        def capture(tree):
            return next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr=='obs' for t in n.targets))
        new_capture=capture(bound)
        assert 'intrinsic_matrix' in ast.unparse(new_capture) and 'extrinsic_matrix' in ast.unparse(new_capture)
        new_capture.value=capture(before).value
        calls=[n for n in ast.walk(bound) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='calibrated_world_height']
        assert len(calls)==1 and ast.unparse(calls[0].keywords[0].value)=='self.obs'
        calls[0].keywords=[]
        assert ast.dump(bound)==ast.dump(before), 'Only camera capture and height-call binding may change'


def test_final_language_model_uses_unclamped_measurement_without_old_plane_probe():
    from classify_language_camera import bind_language_camera_calls
    root=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_v1_classify_objects_by_language')
    if not root.exists():pytest.skip('Release fixture unavailable')
    paths=list(root.glob('*/rdj_v1_classify_objects_by_language/model.py'));assert paths
    for path in paths:
        text=remove_language_surface_height_clamp(bind_language_camera_calls(bind_language_depth_observation(path.read_text()),'model'))
        tree=ast.parse(text)
        samples=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='observed_surface_height']
        assert len(samples)==1 and ast.unparse(samples[0].args[0])=='values_z'
        assert 'old_xy' not in text and 'np.clip(np.median(values_z)' not in text
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'pixel_to_world_on_height','world_to_pixel_on_height'}]
        assert len(calls)==5 and all(any(k.arg=='camera' for k in n.keywords) for n in calls)
