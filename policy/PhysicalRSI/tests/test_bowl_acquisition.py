from pathlib import Path
import ast
import json
import os
import sys
import time
import types
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REVISION = Path(__file__).resolve().parents[1]/'asset_revision'
sys.path.insert(0, str(REVISION))
from physicalrsi_observed_geometry import GeometryUnavailable


def controller(tmp_path, monkeypatch):
    tree = ast.parse((REVISION/'overlays/pour_language_bowl_acquisition.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BowlAcquisition')
    monkeypatch.setenv('PBL_DIAGNOSTICS', str(tmp_path))
    controls = ['left_arm_joint_state', 'right_arm_joint_state', 'left_ee_joint_state', 'right_ee_joint_state']
    namespace = dict(np=np, Path=Path, os=os, time=time, json=json, Rotation=Rotation,
                     GeometryUnavailable=GeometryUnavailable, CONTROLS=controls)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<actual acquisition controller>', 'exec'), namespace)
    acquisition = namespace['BowlAcquisition']()
    state = {'left_arm_joint_state': np.zeros(6), 'right_arm_joint_state': np.zeros(6),
             'left_ee_joint_state': [1.], 'right_ee_joint_state': [1.],
             'left_ee_pose': [0, 0, 1, 1, 0, 0, 0]}
    owner = types.SimpleNamespace(obs={'state': state}, bowl_observations=None)
    acquisition.plan = {'arm': 'left', 'pose': state['left_ee_pose']}
    acquisition.path = [[.1]*6, [.2]*6]
    acquisition.phase = 'view'; acquisition.index = 0; acquisition.settle = 0
    return acquisition, owner


def test_measurement_finishes_before_return_and_task_handoff(tmp_path, monkeypatch):
    acquisition, owner = controller(tmp_path, monkeypatch)
    measured = []
    def measure(current):
        measured.append(np.asarray(current.obs['state']['left_arm_joint_state']).copy())
        acquisition.path = [[.1]*6, [0.]*6]; acquisition.index = 0
        acquisition.phase = 'return'; acquisition.station_steps = 0
    def complete(current):
        np.testing.assert_allclose(current.obs['state']['left_arm_joint_state'], 0., atol=.008)
        current.bowl_observations = {'black': 'observed'}; acquisition.phase = 'complete'
    acquisition.measure = measure; acquisition.next_view = complete
    for _ in range(80):
        before = np.asarray(owner.obs['state']['left_arm_joint_state']).copy()
        action = acquisition.step(owner)
        if action is None: break
        assert owner.bowl_observations is None
        command = np.asarray(action[0]['left_arm_joint_state'])
        assert np.max(np.abs(command-before)) <= .0600001
        np.testing.assert_allclose(action[0]['right_arm_joint_state'], 0.)
        owner.obs['state'].update(action[0])
    else: pytest.fail('Inspection never handed back to the task')
    assert len(measured) == 1 and owner.bowl_observations is not None
    np.testing.assert_allclose(measured[0], .2, atol=.008)


def test_stalled_waypoint_holds_without_inventing_bowl_geometry(tmp_path, monkeypatch):
    acquisition, owner = controller(tmp_path, monkeypatch)
    for _ in range(25): acquisition.step(owner)
    assert acquisition.phase == 'failed' and owner.bowl_observations is None
    action = acquisition.step(owner)
    np.testing.assert_allclose(action[0]['left_arm_joint_state'], 0.)
    assert 'did not converge' in (acquisition.out/'trace.jsonl').read_text()


def test_joint_reached_but_observed_ee_wrong_does_not_trigger_measurement(tmp_path, monkeypatch):
    acquisition, owner = controller(tmp_path, monkeypatch)
    acquisition.phase = 'settle'; acquisition.settle = 2
    owner.obs['state']['left_ee_pose'] = [0, 0, 1.05, 1, 0, 0, 0]
    acquisition.measure = lambda owner: pytest.fail('Unreached pose used for geometry')
    acquisition.step(owner)
    assert acquisition.phase == 'failed' and owner.bowl_observations is None
