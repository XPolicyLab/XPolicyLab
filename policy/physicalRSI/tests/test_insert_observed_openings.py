"""Recognize aperture size groups without a stored rack grid or metric sizes."""
import ast
from pathlib import Path
import cv2
import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'asset_revision/overlays/insert_rgb_perception.py'
tree=ast.parse(SOURCE.read_text());ns={'cv2':cv2,'np':np}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_components','_rack_openings')],type_ignores=[]),'<observed-openings>','exec'),ns)


@pytest.mark.parametrize('scale',[1,2,3])
def test_larger_apertures_are_separated_from_auxiliary_perforations(scale):
    mask=np.zeros((100*scale,180*scale),np.uint8)
    expected=[(30*scale,35*scale),(80*scale,40*scale),(130*scale,45*scale)]
    for x,y in expected:cv2.circle(mask,(x,y),10*scale,1,-1)
    for x in range(20,170,20):cv2.circle(mask,(x*scale,80*scale),4*scale,1,-1)
    found=ns['_rack_openings'](mask)
    assert len(found)==3
    np.testing.assert_allclose(sorted(tuple(center) for _,center in found),expected)


def test_equal_apertures_are_not_forced_into_a_stored_count():
    mask=np.zeros((60,200),np.uint8)
    for x in [20,60,100,140]:cv2.circle(mask,(x,30),8,1,-1)
    assert len(ns['_rack_openings'](mask))==4
    assert ns['_rack_openings'](np.zeros_like(mask))==[]


def test_outer_edge_notch_is_not_a_closed_aperture():
    outer=np.zeros((100,200),np.uint8);outer[10:91,10:191]=1
    holes=np.zeros_like(outer)
    for x in [30,70,110,150]:cv2.circle(holes,(x,40),8,1,-1)
    holes[80:91,80:97]=1
    assert len(ns['_rack_openings'](holes))==5
    found=ns['_rack_openings'](holes,outer)
    assert len(found)==4
    np.testing.assert_allclose(sorted(tuple(c) for _,c in found),[(30,40),(70,40),(110,40),(150,40)])


@pytest.mark.parametrize('cap_visible',[True,False])
def test_cap_separates_from_same_hue_table_using_current_body_brightness(cap_visible):
    from scipy.spatial.transform import Rotation
    from types import SimpleNamespace
    namespace={'cv2':cv2,'np':np,'_quaternion':lambda m:Rotation.from_matrix(m).as_quat()[[3,0,1,2]].tolist()}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_components','_colors','_tubes')]
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<observed-tubes>','exec'),namespace)
    image=np.full((160,200,3),[150,110,70],np.uint8)
    image[50:60,30:90]=[245,245,245]
    if cap_visible:image[48:62,90:98]=[255,210,130]
    def surface(mask):
        y,x=np.nonzero(mask)
        return np.c_[x*.001,y*.001,np.full(len(x),.8)]
    scene=SimpleNamespace(camera=SimpleNamespace(image=image,transform=np.eye(4)),surface=surface)
    tubes=namespace['_tubes'](scene)
    assert len(tubes)==int(cap_visible)
    assert namespace['_tubes'](scene,instance_mask=np.zeros(image.shape[:2],bool))==[]
    assert namespace['_tubes'](scene,instance_mask=np.ones(image.shape[:2],bool))==tubes
    if cap_visible:
        assert .05<tubes[0][1]<.08
        assert tubes[0][0][0]<.04


@pytest.mark.parametrize('body,cap',[((30,50,10,170),(33,37,170,173)),((30,32,20,80),(29,33,80,83))])
def test_white_wall_and_unresolved_reflection_are_not_tubes(body,cap):
    from scipy.spatial.transform import Rotation
    from types import SimpleNamespace
    namespace={'cv2':cv2,'np':np,'_quaternion':lambda m:Rotation.from_matrix(m).as_quat()[[3,0,1,2]].tolist()}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_components','_colors','_tubes')]
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<observed-tubes>','exec'),namespace)
    image=np.full((160,200,3),[150,110,70],np.uint8)
    y0,y1,x0,x1=body;image[y0:y1,x0:x1]=245
    y0,y1,x0,x1=cap;image[y0:y1,x0:x1]=[255,210,130]
    def surface(mask):
        y,x=np.nonzero(mask)
        return np.c_[x*.001,y*.001,np.full(len(x),.8)]
    scene=SimpleNamespace(camera=SimpleNamespace(image=image,transform=np.eye(4)),surface=surface)
    assert namespace['_tubes'](scene)==[]
