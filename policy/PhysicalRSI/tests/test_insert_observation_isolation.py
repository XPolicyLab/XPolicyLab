"""One missing geometric observation must not abort an otherwise valid batch."""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import os
import numpy as np
import pytest


def perception_class():
    source=Path(__file__).resolve().parents[1]/'asset_revision/overlays/insert_rgb_perception.py'
    tree=ast.parse(source.read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='RgbInsertTubesPerception')
    class Unavailable(ValueError):pass
    calls=[]
    def initial(obs):
        marker=int(obs['vision']['cam_head']['color'][0,0,0]);calls.append(marker)
        if marker==0:raise Unavailable('no stereo correspondence')
        return SimpleNamespace(tube_poses=((0,0,1,1,0,0,0),),slot_poses=((0,0,1,1,0,0,0),),rack_pose=[0,0,1,1,0,0,0],tube_lengths=(.1,),support=[])
    ns={'np':np,'os':os,'Path':Path,'hashlib':hashlib,'GeometryUnavailable':Unavailable,
        'InsertTubesEnvelopeError':Unavailable,'_initial':initial}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),ns)
    return ns['RgbInsertTubesPerception'],calls,Unavailable


def observation(marker):
    return {'vision':{'cam_head':{'color':np.full((8,8,3),marker,np.uint8),
                                'intrinsic_matrix':np.eye(3),'extrinsic_matrix':np.eye(4)}},'state':{}}


def test_failed_slot_does_not_abort_other_observations_and_new_evidence_can_recover(monkeypatch):
    monkeypatch.delenv('PHYSICALRSI_INSERT_TRACE',raising=False)
    cls,calls,error=perception_class();perception=cls()
    for index,marker in [(0,0),(1,1)]:perception.observe(index,observation(marker))
    assert not perception.is_ready(0) and perception.is_ready(1)
    assert perception.initial_estimate(1)['env_idx']==1
    with pytest.raises(error,match='no stereo'):perception.initial_estimate(0)
    perception.observe(0,observation(0))
    assert calls==[0,1]  # Identical evidence cannot make a failed fit informative.
    changed=observation(0);changed['vision']['cam_head']['extrinsic_matrix'][0,3]=.1
    perception.observe(0,changed);assert calls==[0,1,0]
    perception.observe(0,observation(2))
    assert perception.is_ready(0) and perception.initial_estimate(0)['env_idx']==0
    assert perception.is_ready(1)
    perception.reset()
    assert not perception.is_ready(0) and not perception.is_ready(1)
    perception.observe(0,observation(0));assert calls==[0,1,0,2,0]
