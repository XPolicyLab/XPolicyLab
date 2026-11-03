import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry import pickup_target
from physicalrsi_observed_geometry.camera_projection import pixel_to_world_on_height
from pickup_geometry_edits import remove_pickup_geometry_overrides,replace_pickup_projection,replace_pickup_skill_offsets


def setup_scene(monkeypatch):
    mask=np.zeros((80,100),bool);mask[25:55,35:65]=True
    monkeypatch.setattr(pickup_target,'observe_pickup_target',lambda image,description,box:mask)
    transform=np.eye(4);transform[:3,3]=[.3,.1,1.5]
    camera=Camera(np.zeros((80,100,3),np.uint8),np.array([[100.,0,50],[0,100.,40],[0,0,1.]]),transform)
    detection=SimpleNamespace(description='unseen object',box_xyxy=(34,24,66,56),solid_xy=(50,40),surface_z=.8)
    return camera,detection


def test_visual_contact_is_retained_only_inside_current_target(monkeypatch):
    camera,detection=setup_scene(monkeypatch)
    for preferred,expected in [(None,(50,40)),((3,4),(50,40)),((np.nan,40),(50,40)),((55,35),(55,35))]:
        actual=pickup_target.observed_pickup_contact(detection,camera,preferred_pixel=preferred)
        np.testing.assert_allclose(actual,pixel_to_world_on_height(expected,world_z=.8,camera=camera))
    detection.solid_xy=(2,2)
    with pytest.raises(GeometryUnavailable,match='inside target'):
        pickup_target.observed_pickup_contact(detection,camera)


def test_actual_release_plan_ignores_training_world_offsets(monkeypatch):
    root=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_master')
    if not root.exists():pytest.skip('Release fixture unavailable')
    paths=list(root.glob('*/rdj_master/tasks/general_pickup/rgb_cap.py'));assert paths
    camera,detection=setup_scene(monkeypatch)
    for path in paths:
        text=replace_pickup_skill_offsets(replace_pickup_projection(remove_pickup_geometry_overrides(path.read_text())))
        assert 'skill.delta_xy' not in text
        fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_plan_from_skill')
        fn.returns=None
        for arg in fn.args.args:arg.annotation=None
        scope={'__name__':__name__,'dataclass':dataclass,'_EXACT_REPLAY_MATCH_INLIERS':8,'_adapt_quaternion':lambda q,*a:q}
        plan_class=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='_GraspPlan')
        exec(compile(ast.Module(body=[plan_class],type_ignores=[]),'actual grasp plan','exec'),scope)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual current-contact planner','exec'),scope)
        skill=SimpleNamespace(contact_pixel=None,delta_xy=(100,-100),quaternion=(1,0,0,0),source_arm='right',close_z=.84,lift_height=.24,match_inliers=0,trajectory_absolute=None,trajectory_residual=None)
        result=scope['_plan_from_skill'](detection,-20,30,skill,camera=camera)
        np.testing.assert_allclose(result.contact_pose[:2],(.3,.1))
        assert result.arm=='right' and result.contact_pose[2:]==(.84,1,0,0,0)
        original_text=replace_pickup_projection(remove_pickup_geometry_overrides(path.read_text()))
        original_fn=next(n for n in ast.parse(original_text).body if isinstance(n,ast.FunctionDef) and n.name=='_plan_from_skill')
        assert [ast.dump(n) for n in fn.body[2:]]==[ast.dump(n) for n in original_fn.body[1:]], 'Non-position planning logic must remain unchanged'
