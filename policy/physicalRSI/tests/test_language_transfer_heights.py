import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import GeometryUnavailable
from physicalrsi_observed_geometry.language_transfer import observed_basket_top,observed_transfer_heights
from classify_language_transfer_edits import replace_language_transfer_heights,replace_language_relay_transfer


def test_basket_upper_surface_comes_only_from_current_matched_mask(monkeypatch):
    mask=np.zeros((32,40),bool);mask[5:25,5:25]=True
    from physicalrsi_observed_geometry import language_transfer
    monkeypatch.setattr(language_transfer,'observed_basket_region',lambda episode,box:mask)
    for base in [-.3,.5,1.4]:
        heights=np.full((32,40),9.)
        heights[mask]=base+.1
        heights[10:20,10:20]=base+.02
        heights[7,7]=9. # Isolated depth outlier, not the measured upper surface.
        ep=SimpleNamespace(rgb=np.zeros((32,40,3),np.uint8),inventory_height=heights,inventory=SimpleNamespace(sam=SimpleNamespace(segment_many=lambda *args:[SimpleNamespace(masks=[mask],scores=[.95])])))
        top=observed_basket_top(ep,[5,5,25,25])
        assert top==pytest.approx(base+.1)
        carry,lower=observed_transfer_heights(base+.16,top,50)
        relative=observed_transfer_heights(.16,.1,50)
        np.testing.assert_allclose([carry-base,lower-base],relative)
        assert carry>=base+.26 and lower>top
        ep.inventory_height[mask]=np.nan
        with pytest.raises(GeometryUnavailable):observed_basket_top(ep,[5,5,25,25])
    with pytest.raises(GeometryUnavailable):observed_transfer_heights(.8,.7,90)


def test_release_motion_executes_the_planned_heights(tmp_path):
    root=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_v1_classify_objects_by_language')
    paths=list(root.glob('*/rdj_v1_classify_objects_by_language/model.py'))
    if not paths:pytest.skip('Release fixture unavailable')
    for path in paths:
        text=replace_language_transfer_heights(path.read_text())
        assert '1.02' not in text and "a.get('lower_z'" not in text
        tree=ast.parse(text)
        functions=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in ('_memory_advance_2','_memory_advance_3')]
        scope={};exec(compile(ast.Module(body=functions,type_ignores=[]),'actual motion methods','exec'),scope)
        for height in [.6,1.6]:
            for stage,method,arrived,expected in [('close','_memory_advance_2',True,height),('grasp_confirm','_memory_advance_2',True,height),('transport','_memory_advance_3',False,height),('transport','_memory_advance_3',True,height-.1),('place_confirm','_memory_advance_3',True,height)]:
                calls=[]
                ep=SimpleNamespace(stage=stage,operation_trace=[],settle_used=set(),arrived=lambda *a:arrived,motion=lambda *a:calls.append(a),grasp_confirm=lambda:True,start_track=lambda:None,out=tmp_path,case='height',step=0,obs={'vision':{}})
                frame={'locals':{'a':{'carry_z':height},'arm':'left','x':.1,'y':-.2,'z':height-.2,'bx':.2,'by':-.1,'q':[1,0,0,0],'cq':[1,0,0,0],'lower_z':height-.1}}
                scope[method](ep,frame,None)
                assert len(calls)==1 and calls[0][1][2]==pytest.approx(expected)
        active=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='active' for t in n.targets) and isinstance(n.value,ast.Dict)]
        assert len(active)==1
        assert any(isinstance(k,ast.Constant) and k.value=='carry_z' and ast.unparse(v)=='carry_z' for k,v in zip(active[0].value.keys,active[0].value.values))


def test_relay_exports_the_same_carry_height_it_qualifies():
    from classify_language_camera_edits import bind_language_camera_calls
    from classify_language_relay_edits import replace_language_relay_plane
    root=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_v1_classify_objects_by_language')
    paths=list(root.glob('*/rdj_v1_classify_objects_by_language/relay.py'))
    if not paths:pytest.skip('Release fixture unavailable')
    for path in paths:
        text=replace_language_relay_transfer(replace_language_relay_plane(bind_language_camera_calls(path.read_text(),'relay')))
        assert '1.02' not in text
        tree=ast.parse(text)
        phases=[n for n in ast.walk(tree) if isinstance(n,ast.Tuple) and n.elts and isinstance(n.elts[0],ast.Constant) and n.elts[0].value in ('lift','transport')]
        assert len(phases)==2 and all(ast.unparse(n.elts[-1])=='carry_z' for n in phases)
        assert "'carry_z': carry_z" in text


def test_basket_box_prompt_uses_current_pixels_and_same_instance_gate():
    from contextlib import nullcontext
    from physicalrsi_observed_geometry.language_transfer import observed_basket_region
    image=np.zeros((32,40,3),np.uint8);image[:,:,0]=150
    mask=np.zeros((1,32,40),bool);mask[:,5:25,5:25]=True
    calls=[]
    class Tensor:
        def __init__(self,value):self.value=np.asarray(value)
        def cpu(self):return self
        def float(self):return self
        def numpy(self):return self.value
    def set_image(value):np.testing.assert_array_equal(np.asarray(value),image);return object()
    def prompt(box,positive,state):
        calls.append((box,positive))
        return {'masks':Tensor(mask),'scores':Tensor([.95])}
    sam=SimpleNamespace(device='cpu',_torch=SimpleNamespace(inference_mode=nullcontext),processor=SimpleNamespace(set_image=set_image,reset_all_prompts=lambda state:None,add_geometric_prompt=prompt))
    episode=SimpleNamespace(rgb=image,inventory=SimpleNamespace(sam=sam))
    result=observed_basket_region(episode,[5,5,25,25])
    np.testing.assert_array_equal(result,mask[0])
    np.testing.assert_allclose(calls[0][0],[15/40,15/32,20/40,20/32])
    assert calls[0][1] is True
    with pytest.raises(GeometryUnavailable):observed_basket_region(episode,[30,1,39,10])
