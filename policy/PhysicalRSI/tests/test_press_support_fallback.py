import ast
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import pytest

class MissingGeometry(ValueError):pass


def helpers():
    source=Path(__file__).parents[1]/'asset_revision/overlays/press_perception.py'
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)]
    ns={'np':np,'time':time,'GeometryUnavailable':MissingGeometry,
        'refine_plane':lambda cameras,masks,plane:plane}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    return ns


def camera():return SimpleNamespace(image=np.zeros((40,50,3),np.uint8))
def detections():return [{'query':'table','box_xyxy':[1,1,48,38],'score':.9},
                         {'query':'cards','box_xyxy':[18,18,22,22],'score':.7}]


def test_reference_mask_excludes_current_foreground_and_background():
    ns=helpers();mask=ns['observed_reference_support'](camera(),detections())
    assert mask[5,5] and not mask[20,20] and not mask[0,0]

@pytest.mark.parametrize('rows',[[],[{'query':'table','box_xyxy':[1,1,48,38],'score':.4}],
    detections()+[{'query':'table','box_xyxy':[2,2,47,37],'score':.85}],
    detections()+[{'query':'robot','box_xyxy':[1,1,48,38],'score':.8}]])
def test_unobserved_or_ambiguous_support_is_rejected(rows):
    ns=helpers()
    with pytest.raises(MissingGeometry):ns['observed_reference_support'](camera(),rows)


def test_successful_primary_geometry_does_not_run_fallback():
    ns=helpers();mask=np.ones((40,50),bool);plane=np.array([0.,0.,1.2])
    ns['initial_plane']=lambda *args:plane;ns['refine_plane']=lambda *args:plane
    def forbidden(*args,**kw):raise AssertionError('Fallback must not replace accepted primary geometry')
    ns['photometric_initial_plane']=forbidden
    p,visible,method=ns['measured_support']([camera(),camera()],mask,SimpleNamespace(predict=forbidden))
    assert p is plane and visible is mask and method=='sparse_rgb_and_photometric_refinement'


def test_unresolved_primary_requires_masked_independent_validation():
    ns=helpers();mask=np.ones((40,50),bool);validated=mask.copy();validated[:10]=False;calls=[]
    def unavailable(*args,**kwargs):raise MissingGeometry('No sparse correspondence consensus')
    def validate(cameras,masks,**kwargs):
        assert kwargs=={'proposal_error_px':3.}
        assert masks[0] is mask
        assert all(not m[20,20] and m[5,5] for m in masks[1:])
        calls.append('validation');return np.array([.01,0.,1.3]),validated
    ns['initial_plane']=unavailable;ns['photometric_initial_plane']=validate
    detector=SimpleNamespace(predict=lambda *a,**kw:detections())
    _,visible,method=ns['measured_support']([camera(),camera(),camera()],mask,detector)
    assert calls==['validation'] and visible is validated and method.startswith('heldout_')
    ns['photometric_initial_plane']=unavailable
    with pytest.raises(MissingGeometry):ns['measured_support']([camera(),camera()],mask,detector)


def test_programming_errors_do_not_trigger_geometric_fallback():
    ns=helpers()
    def broken(*args):raise TypeError('Wrong reconstruction API')
    ns['initial_plane']=broken
    with pytest.raises(TypeError):ns['measured_support']([camera(),camera()],np.ones((40,50),bool),None)


def test_confident_duplicate_label_does_not_erase_reference_surface():
    ns=helpers()
    rows=detections()+[{'query':'cards','box_xyxy':[1,1,48,38],'score':.3}]
    mask=ns['observed_reference_support'](camera(),rows)
    assert mask[5,5] and not mask[20,20]


def test_smaller_occluder_retained_even_with_large_confidence_gap():
    ns=helpers()
    rows=detections()+[{'query':'robot','box_xyxy':[3,3,10,10],'score':.3}]
    mask=ns['observed_reference_support'](camera(),rows)
    assert not mask[5,5] and mask[30,35]


def test_learned_initializations_follow_observation_scale_and_camera_translation():
    ns=helpers();mask=np.ones((40,50),bool);depth=np.full((40,50),2.)
    def cam(offset):
        return SimpleNamespace(image=np.zeros((40,50,3),np.uint8),
            unproject=lambda uv,d:np.c_[uv[:,0]*d,uv[:,1]*d,d]+offset)
    original=ns['learned_depth_plane_proposals'](cam(np.zeros(3)),depth,mask)
    scaled=ns['learned_depth_plane_proposals'](cam(np.zeros(3)),depth*3,mask)
    shifted=ns['learned_depth_plane_proposals'](cam(np.array([4.,-3.,.8])),depth,mask)
    np.testing.assert_allclose(scaled,original*3)
    np.testing.assert_allclose(shifted,original+[4.,-3.,.8])
    with pytest.raises(MissingGeometry):ns['learned_depth_plane_proposals'](cam(np.zeros(3)),depth,np.zeros_like(mask))


def test_semantic_and_learned_fallback_cannot_skip_independent_validation(monkeypatch):
    import sys
    ns=helpers();cameras=[camera(),camera(),camera()];mask=np.ones((40,50),bool);raw=np.ones((40,50))
    def unavailable(*args,**kwargs):raise MissingGeometry('Independent image validation failed')
    ns['initial_plane']=unavailable;ns['photometric_initial_plane']=unavailable
    monkeypatch.setitem(sys.modules,'PourBallsRGB.sam.sam3_http',SimpleNamespace(segment_many_http=lambda *args:None))
    ns['observe_support_masks']=lambda *args,**kwargs:(cameras,[mask]*3)
    ns['learned_depth_plane_proposals']=lambda cam,prediction,visible:np.array([[0.,0.,1.]])
    ns['photometric_plane_from_proposals']=unavailable
    with pytest.raises(MissingGeometry,match='Independent image validation failed'):
        ns['measured_support'](cameras,mask,SimpleNamespace(predict=lambda *args,**kwargs:detections()),observation={},depth_prediction=raw)
