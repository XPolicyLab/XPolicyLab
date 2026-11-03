from pathlib import Path
import importlib.util,sys
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]/'asset_revision'
sys.path.insert(0,str(ROOT))


def module(monkeypatch):
    spec=importlib.util.spec_from_file_location('align_prompt_retry',ROOT/'overlays/align_perception.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    monkeypatch.setattr(m,'initial_plane',lambda cameras,masks,**kw:(np.array([0.,0.,.8]),masks[0]))
    monkeypatch.setattr(m,'refine_plane',lambda cameras,masks,start,**kw:start)
    monkeypatch.setattr(m,'support_patches',lambda *args:[])
    monkeypatch.setattr(m,'fit_supported_primitive',lambda *args:{'pose':[0,0,.9,1,0,0,0],'bbox_local':[[0,0,0],[1,1,1]],'source':'test','hidden_surfaces_reconstructed':True,'model_family':'test'})
    return m


def scene(m,primary_scores,retry_scores):
    calls=[];image=np.zeros((40,60,3),np.uint8);mask=np.zeros((40,60),bool);mask[10:14,20:45]=True
    def segment(image,prompts):
        calls.append(prompts)
        return [SimpleNamespace(prompt=p,masks=[np.ones_like(mask) if p in m.SUPPORT_PROMPTS else mask for _ in scores],scores=scores)
                for p,threshold in prompts
                for scores in [([.8] if p in m.SUPPORT_PROMPTS else primary_scores if p=='ruler' else retry_scores if p=='straight ruler' else [])]]
    transform=np.eye(4);transform[2,3]=1.4
    obs={'vision':{'cam_head':{'color':image,'intrinsic_matrix':np.array([[60,0,30],[0,60,20],[0,0,1]]),'extrinsic_matrix':transform}}}
    task=m.TaskPerceptionSpec('align_blocks',(m.LabelSpec('target','ruler','target'),),{'target':1})
    return lambda:m.estimate_scene(obs,'align_blocks',segment_fn=segment,spec=task),calls


def test_retry_missing_ruler_uses_same_confidence_and_recovers_one_instance(monkeypatch):
    m=module(monkeypatch);run,calls=scene(m,[],[.4]);result=run()
    assert list(result.objects)==['target']
    assert calls[-1]==[('straight ruler',.3)]


def test_existing_primary_instance_does_not_trigger_or_merge_alias(monkeypatch):
    m=module(monkeypatch);run,calls=scene(m,[.4],[.7,.8]);assert len(run().objects)==1
    assert len(calls)==1

@pytest.mark.parametrize('scores',[[],[.2],[.4,.5]])
def test_retry_does_not_accept_missing_low_confidence_or_ambiguous_instances(monkeypatch,scores):
    m=module(monkeypatch);run,calls=scene(m,[],scores)
    with pytest.raises(m.SceneNotObserved,match='instance count'):run()


def test_reference_masks_use_each_camera_and_exclude_robot_occluders(monkeypatch):
    m=module(monkeypatch)
    from physicalrsi_observed_geometry import Camera
    cameras=[Camera(np.full((40,60,3),v,np.uint8),np.array([[60,0,30],[0,60,20],[0,0,1]]),np.eye(4)) for v in [10,20]]
    seen=[]
    def segment(image,prompts):
        seen.append(int(image[0,0,0]));surface=np.zeros((20,30),bool);surface[4:18]=True
        arm=np.zeros_like(surface);arm[:,10:15]=True
        return [SimpleNamespace(prompt='wooden surface',masks=[surface],scores=[.7]),
                SimpleNamespace(prompt='robot arm',masks=[arm],scores=[.6])]
    masks=m.reference_support_masks(cameras,segment)
    assert seen==[10,20]
    for mask in masks:
        assert mask.shape==(40,60) and mask[10,10]
        assert not mask[:8].any() and not mask[:,20:30].any()


def test_missing_reference_support_is_not_replaced_by_whole_image(monkeypatch):
    m=module(monkeypatch)
    from physicalrsi_observed_geometry import Camera,GeometryUnavailable
    camera=Camera(np.zeros((40,60,3),np.uint8),np.array([[60,0,30],[0,60,20],[0,0,1]]),np.eye(4))
    with pytest.raises(GeometryUnavailable,match='reference support'):
        m.reference_support_masks([camera],lambda image,prompts:[])
