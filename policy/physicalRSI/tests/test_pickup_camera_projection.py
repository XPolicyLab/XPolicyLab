import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from pickup_geometry_edits import remove_pickup_geometry_overrides, replace_pickup_projection
from physicalrsi_observed_geometry import Camera, GeometryUnavailable
from physicalrsi_observed_geometry.camera_projection import pixel_to_world_on_height
SOURCE=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_master')


def test_projection_uses_each_camera_and_measured_height():
    for yaw,x,height in [(0.,.2,1.7),(.6,-.3,2.1)]:
        transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('z',yaw).as_matrix();transform[:3,3]=[x,.1,height]
        camera=Camera(np.zeros((80,100,3),np.uint8),np.array([[90.,0,49.],[0,110.,37.],[0,0,1.]]),transform)
        point=transform[:3,3]+transform[:3,:3]@np.array([.12,.09,-.8])
        pixel,_=camera.project([point])
        result=pixel_to_world_on_height(pixel[0],world_z=point[2],camera=camera)
        np.testing.assert_allclose(result,point[:2],atol=1e-12)
        with pytest.raises(GeometryUnavailable,match='behind camera'):
            pixel_to_world_on_height(pixel[0],world_z=height+1,camera=camera)


@pytest.mark.skipif(not SOURCE.exists(),reason='Release fixture unavailable')
def test_release_variants_pass_observation_camera_to_all_pickup_calls():
    paths=list(SOURCE.glob('*/rdj_master/tasks/general_pickup/rgb_cap.py'));assert paths
    for path in paths:
        text=replace_pickup_projection(remove_pickup_geometry_overrides(path.read_text()))
        assert '_TARGET_CENTER_PLANE_Z_M' not in text
        tree=ast.parse(text)
        ingest=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_ingest')
        assert 'camera = Camera.from_observation(observation)' in ast.unparse(ingest)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='pixel_to_world_on_height']
        assert len(calls)==5
        for call in calls:assert any(k.arg=='camera' and isinstance(k.value,ast.Name) and k.value.id=='camera' for k in call.keywords)
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_plans')
        method.returns=None
        for arg in method.args.args:arg.annotation=None
        camera=object();seen=[]
        candidates=[SimpleNamespace(arm='left'),SimpleNamespace(arm='right')]
        def plan(d,x,y,e,*,camera):seen.append(camera);return e
        def principal(d,a,*,camera):seen.append(camera);return None
        scope={'_SAME_ARM_SKILL_DESCRIPTIONS':set(),'_PRINCIPAL_AXIS_DESCRIPTIONS':set(),'_plan_from_skill':plan,'_principal_axis_quaternion':principal,'_uses_complementary_retry':lambda d:False}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'actual camera planner','exec'),scope)
        owner=SimpleNamespace(_skills=SimpleNamespace(select=lambda *a,**kw:candidates))
        target=SimpleNamespace(description='unknown object',appearance=(),box_xyxy=(0,0,10,10),keypoints_xy=(),descriptors=())
        result=scope['_plans'](owner,target,.2,.1,camera=camera)
        assert result==(candidates[0],candidates[1]) and len(seen)==3 and all(c is camera for c in seen)
