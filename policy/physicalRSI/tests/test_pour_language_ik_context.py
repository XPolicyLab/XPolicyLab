import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).parents[1]/'asset_revision'


def load_context():
    import sys
    sys.path.insert(0,str(ROOT))
    spec=importlib.util.spec_from_file_location('pour_context_test',ROOT/'overlays/pour_language_ik_context.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_shared_solver_receives_each_episode_current_state_and_support():
    module=load_context();calls=[]
    def solve(*args,**kwargs):calls.append((args,kwargs));return [1,2,3]
    solver=SimpleNamespace(solve=solve)
    first=SimpleNamespace(obs={'state':{'observed':'first'}},observed_support=[{'visible':'one'}],env_idx=3)
    second=SimpleNamespace(obs={'state':{'observed':'second'}},observed_support=[{'visible':'two'}],env_idx=9)
    a,b=[module.ObservedIKContext(solver,owner) for owner in (first,second)]
    assert a.solve('left',[0],[1])==[1,2,3]
    b.solve('right',[2],[3])
    first.obs={'state':{'observed':'updated'}};a.solve('left',[4],[5])
    assert [item[1]['env_idx'] for item in calls]==[3,9,3]
    assert [item[1]['observed_state']['observed'] for item in calls]==['first','second','updated']
    assert calls[0][1]['support'] is first.observed_support
    assert calls[1][1]['support'] is second.observed_support


def test_unobserved_support_cannot_fall_back_to_fixed_collision_table():
    module=load_context();calls=[]
    owner=SimpleNamespace(obs={'state':{}},observed_support=None,env_idx=0)
    context=module.ObservedIKContext(SimpleNamespace(solve=lambda *a,**k:calls.append(k)),owner)
    with pytest.raises(module.GeometryUnavailable,match='Observe support'):context.solve('left',[],[])
    assert calls==[]


def test_solver_errors_are_not_hidden():
    module=load_context()
    def solve(*a,**k):raise TypeError('broken solver API')
    owner=SimpleNamespace(obs={'state':{}},observed_support=[{}],env_idx=0)
    with pytest.raises(TypeError,match='broken solver API'):
        module.ObservedIKContext(SimpleNamespace(solve=solve),owner).solve('left',[],[])
