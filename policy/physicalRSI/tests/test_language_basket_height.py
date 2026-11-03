from pathlib import Path
import ast
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera, GeometryUnavailable
from physicalrsi_observed_geometry.calibrated_depth_scene import observed_pixel_height
from physicalrsi_observed_geometry.camera_projection import pixel_to_world_on_height
from classify_language_camera import bind_language_camera_calls
from classify_language_depth import replace_language_basket_projection_height


def test_basket_ray_follows_observed_surface_under_world_translation():
    k=np.array([[100.,0,10],[0,100.,10],[0,0,1]])
    uv=[14.2,11.3]
    xy=[]
    for height in [-.3,.43,1.6]:
        transform=np.eye(4);transform[:3,3]=[.2,-.1,height+1.]
        camera=Camera(np.zeros((20,20,3),np.uint8),k,transform)
        heights=np.full((20,20),height)
        heights[11,14]=height+2 # A single outlier must not determine the target ray.
        measured=observed_pixel_height(heights,uv)
        assert measured==pytest.approx(height)
        xy.append(pixel_to_world_on_height(uv,world_z=measured,camera=camera))
    np.testing.assert_allclose(xy,np.tile(xy[0],(3,1)),atol=1e-12)


def test_missing_or_invalid_basket_height_never_defaults_to_fixed_plane():
    for height_map,pixel in [(np.zeros((0,0)),[0,0]),(np.zeros((10,10)),[-5,0]),(np.zeros((10,10)),[10,5]),(np.zeros((10,10)),[np.nan,5]),(np.full((10,10),np.nan),[5,5]),(np.zeros(10),[5,5])]:
        with pytest.raises(GeometryUnavailable):observed_pixel_height(height_map,pixel)


def test_release_model_only_changes_basket_projection_geometry():
    root=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_v1_classify_objects_by_language')
    paths=list(root.glob('*/rdj_v1_classify_objects_by_language/model.py'))
    if not paths:pytest.skip('Release fixture unavailable')
    for path in paths:
        before=ast.parse(bind_language_camera_calls(path.read_text(),'model'))
        after=ast.parse(replace_language_basket_projection_height(ast.unparse(before)))
        calls=[n for n in ast.walk(after) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='pixel_to_world_on_height' and n.args and ast.unparse(n.args[0])=='bc']
        assert len(calls)==1
        height=next(k for k in calls[0].keywords if k.arg=='world_z')
        assert ast.unparse(height.value)=='basket_surface_z'
        height.value=ast.Constant(.8)
        for node in ast.walk(after):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                node.body[:]=[statement for statement in node.body if not (
                    isinstance(statement,ast.ImportFrom) and statement.module=='physicalrsi_observed_geometry.calibrated_depth_scene'
                    or isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='basket_surface_z' for t in statement.targets))]
        assert ast.dump(after)==ast.dump(before)
