"""Exercise the actual planner geometry functions without starting model workers."""
import ast
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]/'asset_revision'


def functions():
    tree=ast.parse((ROOT/'overlays/pour_balls_model.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['aperture_geometry','wall_cloud']]
    scope={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<planner-functions>','exec'),scope)
    return scope


def test_observed_opening_overrides_depth_top_and_radius_floor():
    scope=functions();frame=np.eye(4);frame[:3,3]=[.2,-.1,.6]
    frame[:3,:3]=Rotation.from_euler('z',.4).as_matrix()
    opening={'center':np.array([.02,.03,.1]),'radius':.009,'normal':np.array([0,0,1.])}
    body=np.array([[0,0,.7],[.1,.2,.72]])
    center,radius=scope['aperture_geometry'](body,frame,opening)
    np.testing.assert_allclose(center,frame[:3,:3]@opening['center']+frame[:3,3])
    assert radius==.009


def test_collision_keeps_body_and_adds_observed_rim_without_filling_aperture():
    scope=functions();frame=np.eye(4)
    opening={'center':np.array([0.,0.,1.]),'radius':.03,'normal':np.array([0,0,1.])}
    body=np.array([[0,0,.8],[.1,0,.9],[0,0,1.]])
    walls=scope['wall_cloud'](body,frame,.03,opening)
    np.testing.assert_allclose(walls[:2],body[:2])
    assert not np.any(np.linalg.norm(walls-[0,0,1],axis=1)<.029)
    np.testing.assert_allclose(walls[2:,2],1.)
    np.testing.assert_allclose(np.linalg.norm(walls[2:,:2],axis=1),.03)
