from pathlib import Path
from types import SimpleNamespace as NS
import ast
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]/'asset_revision/overlays'


def detector():
    source=ast.parse((ROOT/'contact_press_model_v2.py').read_text())
    function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='plateau_candidate')
    def reached(actual,target,**kwargs):return np.linalg.norm(np.asarray(actual)[:3]-np.asarray(target)[:3])<kwargs['position_tolerance_m']
    ns={'np':np,'baseline':NS(pose_reached=reached),'PROBE_BUDGET_M':.04}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<endpoint>','exec'),ns)
    return ns['plateau_candidate']


@pytest.mark.parametrize('travel',[.006,.017,.029])
@pytest.mark.parametrize('offset',[0.,.3])
def test_endpoint_is_observed_pose_without_known_button_travel(travel,offset):
    height=.9+offset-travel
    actual=np.array([.1,.2,height,1.,0.,0.,0.])
    target=actual.copy();target[2]=.9+offset-.04
    anchor=NS(retry_down_pose=target) # No native joint travel field needed.
    history=[actual[:3]+[0,0,j] for j in [.0002,.0001,0.]]
    assert detector()(history,actual,anchor)==height


def test_moving_tip_and_xy_slip_do_not_confirm_endpoint():
    actual=np.array([.1,.2,.89,1.,0.,0.,0.]);target=actual.copy();target[2]=.86;anchor=NS(retry_down_pose=target)
    for history in [[],[actual[:3]+[0,0,z] for z in [.004,.002,0]],
                    [actual[:3]+[x,0,0] for x in [.004,.002,0]]]:
        assert detector()(history,actual,anchor) is None


def test_probe_and_transit_targets_translate_with_observed_buttons():
    source=ast.parse((ROOT/'observed_press_model.py').read_text())
    function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='observed_anchors')
    def shifted(p,z):
        p=list(p);p[2]+=z;return tuple(p)
    baseline=NS(BUTTON_LABELS=['button0','button1','button2'],PRESS_TOOL_TIP_OFFSET_M=.1,
        RELEASE_CLEARANCE_M=.05,ButtonAnchor=lambda **kw:NS(**kw),
        make_up_tip_pose=lambda p,axis:shifted(p,.05),tool_tip_pose_to_ee_pose=lambda p:shifted(p,.1),
        make_high_transit_ee_pose=lambda p,minimum_z_m:tuple([p[0],p[1],max(p[2]+.05,minimum_z_m),*p[3:]]))
    ns={'np':np,'baseline':baseline,'PROBE_BUDGET_M':.04}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<anchors>','exec'),ns)
    positions=np.array([[-.2,.1,.7],[.1,.2,.73],[.3,-.1,.76]]);shift=np.array([.4,-.3,.25])
    original=ns['observed_anchors'](positions);moved=ns['observed_anchors'](positions+shift)
    for name in baseline.BUTTON_LABELS:
        for field in ['down_pose','retry_down_pose','high_pose','up_pose']:
            np.testing.assert_allclose(np.asarray(getattr(moved[name],field))[:3]-np.asarray(getattr(original[name],field))[:3],shift)
        assert abs(original[name].down_pose[2]-original[name].tip_pose[2]-(.1-.04))<1e-12
