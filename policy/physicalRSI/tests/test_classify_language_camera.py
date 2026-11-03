import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import cv2
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from classify_language_depth_edits import edit_language_depth_perception,bind_language_depth_observation
from classify_language_camera_edits import replace_language_camera_helpers,bind_language_camera_calls,PROJECTIONS
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.camera_projection import pixel_to_world_on_height,world_to_pixel_on_height


def test_current_camera_projection_roundtrip_and_invalid_geometry():
    transform=np.eye(4);transform[:3,3]=[.15,-.2,1.5]
    camera=Camera(np.zeros((100,120,3),np.uint8),np.array([[100.,0,60],[0,90.,50],[0,0,1.]]),transform)
    xy=(.25,-.1);uv=world_to_pixel_on_height(xy,world_z=.65,camera=camera)
    np.testing.assert_allclose(pixel_to_world_on_height(uv,world_z=.65,camera=camera),xy)
    with pytest.raises(GeometryUnavailable):world_to_pixel_on_height([np.nan,0],world_z=.65,camera=camera)
    with pytest.raises(GeometryUnavailable):world_to_pixel_on_height(xy,world_z=2,camera=camera)


def test_all_release_calls_are_bound_and_relay_executes_with_current_camera():
    root=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_v1_classify_objects_by_language')
    if not root.exists():pytest.skip('Release fixture unavailable')
    paths=list(root.glob('*/rdj_v1_classify_objects_by_language/model.py'));assert paths
    for p in paths:
        perception=replace_language_camera_helpers(edit_language_depth_perception(p.with_name('rgb_perception.py').read_text()))
        assert '_FOCAL_PIXELS' not in perception and '1.308' not in perception
        for kind,name,count in [('model','model.py',6),('relay','relay.py',5)]:
            before=p.with_name(name).read_text()
            if kind=='model':before=bind_language_depth_observation(before)
            after=bind_language_camera_calls(before,kind);tree=ast.parse(after)
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in PROJECTIONS]
            assert len(calls)==count
            for call in calls:assert any(k.arg=='camera' for k in call.keywords)
            if kind!='relay':continue
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='free_relay_sites')
            transform=np.eye(4);transform[:3,3]=[0,-.2,1.5]
            camera=Camera(np.zeros((100,120,3),np.uint8),np.array([[100.,0,60],[0,100.,50],[0,0,1.]]),transform)
            seen=[]
            def to_pixel(xy,*,world_z,camera):seen.append(camera);return world_to_pixel_on_height(xy,world_z=world_z,camera=camera)
            scope={'np':np,'cv2':cv2,'FIXED_SITES':[],'world_to_pixel_on_height':to_pixel,'pixel_to_world_on_height':pixel_to_world_on_height}
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual relay','exec'),scope)
            scope['free_relay_sites'](np.zeros((100,120),bool),[],3,camera=camera)
            assert len(seen)>=4 and all(c is camera for c in seen)
