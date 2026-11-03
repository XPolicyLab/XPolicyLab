import ast
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import pytest

class MissingGeometry(RuntimeError):
    pass

class Base:
    def get_action_batch(self, indices):
        return [[{'joint':np.array([index+10.])}] for index in indices]
    def reset(self):
        self._core._slots.clear()


def candidate(tmp_path):
    source=Path(__file__).parents[1]/'asset_revision/overlays/observed_press_model.py'
    tree=ast.parse(source.read_text());node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Model')
    namespace={'IncumbentModel':Base,'np':np,'time':time,'GeometryUnavailable':MissingGeometry,
               'baseline':SimpleNamespace(_robot_state=lambda obs:{'joint':np.asarray(obs['state']['joint']).copy()})}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    model=namespace['Model'].__new__(namespace['Model']);calls=[];unavailable={1}
    def observe(observations):
        index=observations[0]['env_idx'];calls.append(index)
        if index in unavailable:raise MissingGeometry('unobserved support')
        model._core._slots[index]=object()
    model._core=SimpleNamespace(_slots={},output=tmp_path,update_obs_batch=observe)
    return model,calls,unavailable


def observation(index):
    return {'env_idx':index,'state':{'joint':[float(index)],'privileged_object_pose':[999.]},
            'vision':{'cam_head':{'color':np.zeros((3,4,3),np.uint8),'intrinsic_matrix':np.eye(3),'extrinsic_matrix':np.eye(4)}}}


def test_failed_environment_holds_without_aborting_peers(tmp_path):
    model,calls,_=candidate(tmp_path)
    model.update_obs_batch([observation(0),observation(1),observation(2)])
    actions=model.get_action_batch([2,1,0])
    assert calls==[0,1,2]
    assert [float(a[0]['joint'][0]) for a in actions]==[12.,1.,10.]
    files=list(tmp_path.glob('unobserved-*.npz'));assert len(files)==1
    with np.load(files[0],allow_pickle=False) as data:
        assert 'state/privileged_object_pose' not in data.files
        assert data['vision/cam_head/color'].shape==(3,4,3)


def test_retry_uses_current_robot_state_and_recovers(tmp_path):
    model,calls,unavailable=candidate(tmp_path)
    model.update_obs_batch([observation(1)])
    unavailable.clear()
    for step in range(1,16):
        obs=observation(1);obs['state']['joint']=[float(step)]
        model.update_obs_batch([obs]);assert model.get_action_batch([1])[0][0]['joint'][0]==step
    assert calls==[1]
    assert model.get_action_batch([1])[0][0]['joint'][0]==15
    model.update_obs_batch([observation(1)])
    assert calls==[1,1] and not model._localization_wait
    assert model.get_action_batch([1])[0][0]['joint'][0]==11
    assert len(list(tmp_path.glob('unobserved-*.npz')))==1


def test_reset_discards_waiting_state(tmp_path):
    model,_,_=candidate(tmp_path);model.update_obs_batch([observation(1)])
    model.reset();assert model._localization_wait=={} and model._observed_indices==()


def test_unexpected_controller_error_is_not_hidden(tmp_path):
    model,_,_=candidate(tmp_path)
    def broken(obs):raise ValueError('program contract error')
    model._core.update_obs_batch=broken
    with pytest.raises(ValueError,match='program contract error'):
        model.update_obs_batch([observation(0)])


@pytest.mark.parametrize('observations',[[],[observation(0),observation(0)]])
def test_invalid_batch_still_rejected(tmp_path,observations):
    model,calls,_=candidate(tmp_path)
    with pytest.raises(ValueError,match='unique env_idx'):
        model.update_obs_batch(observations)
    assert calls==[]


def test_outer_clearance_filters_before_inner_wrappers(tmp_path):
    import copy
    model, calls, _ = candidate(tmp_path)
    inherited = model.__class__
    inner_calls = []
    class RequiresSlot(inherited):
        def get_action_batch(self, indices=None):
            inner_calls.append(tuple(indices))
            for index in indices:
                assert index in self._core._slots
            return super().get_action_batch(indices)
    source = Path(__file__).parents[1]/'asset_revision/overlays/efficient_clearance_model.py'
    node = next(x for x in ast.parse(source.read_text()).body
                if isinstance(x, ast.ClassDef) and x.name == 'Model')
    namespace = {'ClearanceModel':RequiresSlot, 'copy':copy}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    model.__class__ = namespace['Model']
    model.update_obs_batch([observation(1)])
    assert model.get_action_batch()[0][0]['joint'][0] == 1
    assert inner_calls == []
    model.update_obs_batch([observation(0), observation(1), observation(2)])
    for index in (0, 2):
        model._core._slots[index] = SimpleNamespace(session=SimpleNamespace(current_label=None))
    actions = model.get_action_batch([2, 1, 0])
    assert inner_calls == [(2, 0)]
    assert [float(a[0]['joint'][0]) for a in actions] == [12., 1., 10.]
