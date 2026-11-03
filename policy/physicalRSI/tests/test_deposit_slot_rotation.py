"""The same visible slot must remain detectable as a wrist rotates."""
from pathlib import Path
import ast
import math
import cv2
import numpy as np
import pytest


def detector():
    p=Path(__file__).resolve().parents[1]/'asset_revision/overlays/deposit_rgb_perception.py'
    tree=ast.parse(p.read_text())
    selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_bank_mask','_slot_candidates')],type_ignores=[])
    scope={'np':np,'cv2':cv2,'atan2':math.atan2}
    exec(compile(selected,str(p),'exec'),scope)
    return scope

@pytest.mark.parametrize('color',[[255,180,190],[248,211,203],[247,197,185]])
@pytest.mark.parametrize('angle',[0,30,60,90,130])
def test_slot_detection_is_invariant_to_image_rotation(angle,color):
    scope=detector();rgb=np.full((180,240,3),color,np.uint8)
    center=np.array([120.,90.]);corners=cv2.boxPoints((tuple(center),(80.,10.),float(angle)))
    cv2.fillConvexPoly(rgb,np.rint(corners).astype(np.int32),(10,10,10))
    candidates=scope['_slot_candidates'](rgb,scope['_bank_mask'](rgb),'right')
    assert len(candidates)==1
    _,observed,orientation,_=candidates[0]
    np.testing.assert_allclose(observed,center,atol=1)
    direction=np.array([math.cos(orientation),math.sin(orientation)])
    expected=np.array([math.cos(math.radians(angle)),math.sin(math.radians(angle))])
    assert abs(direction@expected)>.99


def test_dark_gripper_edge_next_to_pink_surface_is_not_a_slot():
    scope=detector();rgb=np.full((180,240,3),60,np.uint8)
    rgb[:90]=[255,180,190]
    cv2.rectangle(rgb,(80,85),(160,95),(10,10,10),-1)
    assert not scope['_slot_candidates'](rgb,scope['_bank_mask'](rgb),'right')
