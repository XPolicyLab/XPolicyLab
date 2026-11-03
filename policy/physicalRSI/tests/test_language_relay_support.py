import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import cv2
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry import language_support as module
from classify_language_camera import bind_language_camera_calls
from classify_language_relay import replace_language_relay_plane


def support(height):
    pose=np.eye(4);pose[:3,3]=[0,-.2,height+1]
    camera=Camera(np.zeros((240,320,3),np.uint8),np.array([[300.,0,160],[0,300.,120],[0,0,1]]),pose)
    return module.SupportProjection(camera,[.03,-.02,height],np.ones((240,320),bool))


@pytest.mark.parametrize('height',[-.2,.4,1.3])
def test_observed_sloped_plane_roundtrip_and_extent(height):
    measured=support(height)
    for xy in [[0,-.2],[.05,-.25]]:
        uv=measured.project(xy)
        np.testing.assert_allclose(measured.unproject(uv),xy,atol=1e-12)
        measured.visible[:]=False
        with pytest.raises(GeometryUnavailable,match='outside observed support'):measured.unproject(uv)
        measured.visible[:]=True
    with pytest.raises(GeometryUnavailable):measured.unproject([np.nan,0])


def test_support_cache_is_bound_to_current_observation(monkeypatch):
    measured=support(.4);calls=[]
    ep=SimpleNamespace(obs=object(),inventory=SimpleNamespace(sam=SimpleNamespace(segment_many=object())))
    def observe(obs,segment,**kw):calls.append(obs);return measured.plane,measured.visible
    monkeypatch.setattr(module,'observe_semantic_plane',observe)
    monkeypatch.setattr(module.Camera,'from_observation',lambda obs:measured.camera)
    first=module.observed_language_support(ep)
    assert module.observed_language_support(ep) is first and len(calls)==1
    ep.obs=object()
    assert module.observed_language_support(ep) is not first and len(calls)==2


def test_release_relay_uses_current_support_for_sites_and_release_height():
    root=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_v1_classify_objects_by_language')
    paths=list(root.glob('*/rdj_v1_classify_objects_by_language/relay.py'))
    if not paths:pytest.skip('Release fixture unavailable')
    for path in paths:
        text=replace_language_relay_plane(bind_language_camera_calls(path.read_text(),'relay'))
        tree=ast.parse(text)
        assert 'world_z=0.78' not in text
        keep=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Assign))]
        for height in [.4,1.3]:
            measured=support(height)
            scope={'np':np,'cv2':cv2,'observed_language_support':lambda ep:measured,'time':SimpleNamespace(perf_counter=lambda:0.)}
            exec(compile(ast.Module(body=keep,type_ignores=[]),'actual relay','exec'),scope)
            occupied=np.zeros((240,320),bool);occupied[[0,-1],:]=True;occupied[:,[0,-1]]=True
            sites=scope['free_relay_sites'](occupied,[],3,support=measured)
            assert sites
            for x,y,_ in sites:
                np.testing.assert_allclose(measured.unproject(measured.project([x,y])),[x,y],atol=1e-12)
            # Unknown support extent must not offer an apparently free site.
            measured.visible[:]=False
            assert scope['free_relay_sites'](np.zeros((240,320),bool),[],3,support=measured)==[]
            measured.visible[:]=True
            chosen={'symmetry':'identity'}
            ep=SimpleNamespace(relay_count=0,relay_sites=[],used_objects=0,step=0,scene={'proposals':[{'accepted':True,'best_positive':'toy'}],'baskets':[]},inventory=SimpleNamespace(last_robot=np.zeros((240,320),bool)),masks=[],record=lambda **kw:None,obs={'state':{}},branch=SimpleNamespace(select=lambda *args:(chosen,[])))
            plan=scope['plan_relay'](ep,{'box':[1,1,8,8],'best_positive':'toy'},.05,-.15,height+.15,[1,0,0,0])
            assert plan is not None
            x,y=plan['xy']
            assert plan['lower_z']==pytest.approx(height+.15+measured.height(x,y)-measured.height(.05,-.15)+.015)
