import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.language_support import observed_contact_height,cache_language_support,observed_language_support
from physicalrsi_observed_geometry import calibrated_depth_scene as geometry


def test_contact_tracks_support_translation_and_preserves_retry_clearance():
    for height in [-.2,.4,.765,1.3]:
        for object_height in [.002,.06,.3]:
            for retry in [-.03,0,.02]:
                value=observed_contact_height(height+object_height,height,retry)
                assert value>=height+.15757000232458113+.003
                assert value-height==pytest.approx(observed_contact_height(object_height,0,retry))
    assert observed_contact_height(.763542,.765)==pytest.approx(.9255700023245811)
    # Elevated observed targets must not be clipped at the old absolute 0.98 m.
    assert observed_contact_height(1.5,1.3)>1.5
    with pytest.raises(GeometryUnavailable):observed_contact_height(np.nan,.7)


def test_depth_support_handoff_reuses_exact_observation_geometry(monkeypatch):
    pose=np.eye(4);pose[2,3]=1.4
    camera=Camera(np.zeros((12,16,3),np.uint8),np.array([[20.,0,8],[0,20.,6],[0,0,1]]),pose)
    scene=SimpleNamespace(camera=camera,depth=np.ones((12,16)),observed_support_plane=np.array([0,0,.4]),observed_support_mask=np.ones((12,16),bool))
    monkeypatch.setattr(geometry,'calibrated_depth_scene',lambda *a,**kw:scene)
    episode=SimpleNamespace(obs=object())
    heights=geometry.observed_world_height(np.ones((12,16)),episode.obs,support_callback=lambda c,p,m:cache_language_support(episode,c,p,m))
    np.testing.assert_allclose(heights,.4)
    support=observed_language_support(episode) # No resident segmenter needed: use the validated current cache.
    assert support.height(.1,-.1)==pytest.approx(.4)
    np.testing.assert_array_equal(support.visible,scene.observed_support_mask)


def test_release_model_contact_floor_and_cache_are_wired():
    from classify_language_depth import bind_language_depth_observation,bind_language_support_segmentation,replace_language_contact_floor
    from classify_language_camera import bind_language_camera_calls
    root=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_v1_classify_objects_by_language')
    paths=list(root.glob('*/rdj_v1_classify_objects_by_language/model.py'))
    if not paths:pytest.skip('Release fixture unavailable')
    for path in paths:
        text=replace_language_contact_floor(bind_language_support_segmentation(bind_language_camera_calls(bind_language_depth_observation(path.read_text()),'model')))
        assert '0.765' not in text and '0.9255700023245811' not in text
        tree=ast.parse(text)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        assert any(c.func.id=='observed_contact_height' and [ast.unparse(a) for a in c.args]==['z','support_z','retry_adjustment'] for c in calls)
        depth=next(c for c in calls if c.func.id=='calibrated_world_height')
        callback=next(k.value for k in depth.keywords if k.arg=='support_callback')
        assert ast.unparse(callback.body)=='cache_language_support(self, camera, plane, visible)'
