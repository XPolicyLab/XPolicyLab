from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any
import numpy as np
from .geometry import _numbers, _rotation, _quaternion, _offset_local_x, semantic_pose_to_ee
from .contract import CLOSED_GRIPPER,OPEN_GRIPPER,InsertTubesEnvelopeError
_TUBE_GRASP_LOCAL = np.asarray([0.0, 0.07, 0.0])

_PICK_OBJECT_IN_GRIPPER = np.asarray([0.132, 0.0, 0.0])

_EXPECTED_ATTACHMENT_X = (0.1195, 0.15)

_PICKUP_HORIZONTAL = 0.22

_MIN_SLOT_DISTANCE = 0.06

_UP = np.asarray([0.0, 0.0, 1.0])

_TUBE_UPRIGHT_IN_RACK = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

_STEPS = {'pregrasp': 8, 'contact': 12, 'close': 12, 'lift': 15, 'above_slot': 18, 'preinsert': 8, 'align': 12, 'realign': 8, 'settle_align': 5, 'guide': 18, 'release': 10, 'retreat': 12, 'park': 16, 'home': 15, 'settle_home': 13}

@dataclass(frozen=True, slots=True)
class _TubeCandidate:
    label: str
    arm: str
    slot_index: int
    slot_position: tuple[float, float, float]
    pickup_index: int
    cost: float
    joints: dict[str, list[float]]

def _robot_state(observation: Mapping[str, object]) -> dict[str, list[float]]:
    state = observation.get('state')
    if not isinstance(state, Mapping):
        raise InsertTubesEnvelopeError('official observation has no state')
    return {'left_arm_joint_state': _numbers(state['left_arm_joint_state'], 6, 'left arm'), 'right_arm_joint_state': _numbers(state['right_arm_joint_state'], 6, 'right arm'), 'left_ee_joint_state': _numbers(state['left_ee_joint_state'], 1, 'left gripper'), 'right_ee_joint_state': _numbers(state['right_ee_joint_state'], 1, 'right gripper')}

def _pickup_contacts(tube_pose: Sequence[float]) -> tuple[list[float], ...]:
    object_rotation = _rotation(tube_pose[3:])
    object_position = np.asarray(tube_pose[:3], dtype=float)
    tube_axis = object_rotation[:, 1]
    tube_axis /= np.linalg.norm(tube_axis)
    closing = np.cross(_UP, tube_axis)
    closing /= np.linalg.norm(closing)
    grip_position = object_position + object_rotation @ _TUBE_GRASP_LOCAL
    contacts = []
    vertical = np.sqrt(1.0 - _PICKUP_HORIZONTAL ** 2)
    for closing_sign, axial_sign in product((1.0, -1.0), repeat=2):
        candidate_closing = closing * closing_sign
        approach = tube_axis * axial_sign * _PICKUP_HORIZONTAL - _UP * vertical
        approach /= np.linalg.norm(approach)
        gripper_rotation = np.column_stack((approach, candidate_closing, np.cross(approach, candidate_closing)))
        gripper_position = grip_position - gripper_rotation @ _PICK_OBJECT_IN_GRIPPER
        contacts.append(gripper_position.tolist() + _quaternion(gripper_rotation))
    return tuple(contacts)

def _delivery_poses(tube_pose: Sequence[float], pickup_contact: Sequence[float], slot_pose: Sequence[float], expected_attachment_x: float | None=None) -> dict[str, list[float]]:
    initial_object_rotation = _rotation(tube_pose[3:])
    pickup_rotation = _rotation(pickup_contact[3:])
    object_in_gripper = pickup_rotation.T @ (np.asarray(tube_pose[:3], dtype=float) - np.asarray(pickup_contact[:3], dtype=float))
    if expected_attachment_x is not None:
        object_in_gripper[0] = expected_attachment_x
    relative_rotation = pickup_rotation.T @ initial_object_rotation
    slot_rotation = _rotation(slot_pose[3:])
    desired_object_rotation = slot_rotation @ _TUBE_UPRIGHT_IN_RACK
    desired_gripper_rotation = desired_object_rotation @ relative_rotation.T

    def pose(height: float) -> list[float]:
        object_position = np.asarray(slot_pose[:3], dtype=float) + slot_rotation[:, 2] * height
        gripper_position = object_position - desired_gripper_rotation @ object_in_gripper
        return gripper_position.tolist() + _quaternion(desired_gripper_rotation)
    return {'above_slot': pose(0.105), 'preinsert': pose(0.04), 'align': pose(0.015), 'realign': pose(0.01), 'guide': pose(-0.015), 'retreat': pose(0.11)}

def _solve(runtime: Any, env_idx: int, arm: str, poses: Mapping[str, Sequence[float]]) -> dict[str, list[float]] | None:
    joints = {}
    for name, pose in poses.items():
        target = _joint_target(runtime, env_idx, arm, pose)
        if target is None:
            return None
        joints[name] = target
    return joints

def _tube_candidates(runtime: Any, env_idx: int, label: str, tube_pose: Sequence[float], slot_poses: Sequence[Sequence[float]]) -> tuple[_TubeCandidate, ...]:
    natural_arm = 'left' if tube_pose[0] < 0.0 else 'right'
    candidates = []
    for arm in (natural_arm, 'right' if natural_arm == 'left' else 'left'):
        for pickup_index, contact in enumerate(_pickup_contacts(tube_pose)):
            pickup = {'pregrasp': _offset_local_x(contact, -0.085), 'contact': contact, 'lift': list(contact[:2]) + [float(contact[2]) + 0.12] + list(contact[3:])}
            pickup_joints = _solve(runtime, env_idx, arm, pickup)
            if pickup_joints is None:
                continue
            for slot_index, slot_pose in enumerate(slot_poses):
                delivery = _delivery_poses(tube_pose, contact, slot_pose, _EXPECTED_ATTACHMENT_X[pickup_index % 2])
                delivery_joints = _solve(runtime, env_idx, arm, delivery)
                if delivery_joints is None:
                    continue
                slot_x = float(slot_pose[0])
                opposite_side = max(0.0, slot_x) if arm == 'left' else max(0.0, -slot_x)
                row_penalty = 0.04 if slot_index >= 5 else 0.0
                wrist_penalty = 3.0 if pickup_index >= 2 else 0.0
                long_attachment_penalty = 3.0 if pickup_index % 2 == 1 else 0.0
                cross_arm_penalty = 6.0 if arm != natural_arm else 0.0
                cost = float(np.linalg.norm(np.asarray(delivery_joints['above_slot']) - np.asarray(pickup_joints['lift'])) + 20.0 * opposite_side + row_penalty + wrist_penalty + long_attachment_penalty + cross_arm_penalty)
                candidates.append(_TubeCandidate(label=label, arm=arm, slot_index=slot_index, slot_position=tuple((float(item) for item in slot_pose[:3])), pickup_index=pickup_index, cost=cost, joints={**pickup_joints, **delivery_joints}))
    candidates.sort(key=lambda item: item.cost)
    return tuple(candidates[:12])

def _best_assignment(candidate_groups: Sequence[Sequence[_TubeCandidate]]) -> tuple[_TubeCandidate, ...]:
    best: tuple[tuple[int, float], tuple[_TubeCandidate, ...]] | None = None
    choices = [tuple(group) + (None,) for group in candidate_groups]
    for selection in product(*choices):
        assigned = tuple((item for item in selection if item is not None))
        slots = [item.slot_index for item in assigned]
        if len(slots) != len(set(slots)):
            continue
        if any((np.linalg.norm(np.asarray(left.slot_position) - np.asarray(right.slot_position)) < _MIN_SLOT_DISTANCE for index, left in enumerate(assigned) for right in assigned[index + 1:])):
            continue
        rank = (-len(assigned), sum((item.cost for item in assigned)))
        if best is None or rank < best[0]:
            best = (rank, assigned)
    if best is None:
        return ()
    return tuple(sorted(best[1], key=lambda item: (-abs(item.slot_position[0]), item.arm, item.label)))

def _target(current: Mapping[str, Sequence[float]], *, arm: str | None=None, joints: Sequence[float] | None=None, gripper: float | None=None) -> dict[str, list[float]]:
    result = {key: [float(item) for item in current[key]] for key in current}
    if arm is not None and joints is not None:
        result[f'{arm}_arm_joint_state'] = [float(item) for item in joints]
    if arm is not None and gripper is not None:
        result[f'{arm}_ee_joint_state'] = [float(gripper)]
    return result

def _stages(robot: Mapping[str, Sequence[float]], assignment: Sequence[_TubeCandidate]) -> list[dict[str, object]]:
    current = _target(robot)
    stages = []

    def add(candidate: _TubeCandidate, phase: str, joint_name: str | None=None, gripper: float | None=None, steps: int | None=None) -> None:
        nonlocal current
        current = _target(current, arm=candidate.arm, joints=None if joint_name is None else candidate.joints[joint_name], gripper=gripper)
        stages.append({'name': f'{candidate.label}:{phase}', 'steps': _STEPS[phase] if steps is None else steps, 'target': current})
    for candidate_index, candidate in enumerate(assignment):
        add(candidate, 'pregrasp', 'pregrasp', OPEN_GRIPPER)
        add(candidate, 'contact', 'contact', OPEN_GRIPPER)
        add(candidate, 'close', 'contact', CLOSED_GRIPPER)
        add(candidate, 'lift', 'lift', CLOSED_GRIPPER)
        add(candidate, 'above_slot', 'above_slot', CLOSED_GRIPPER)
        add(candidate, 'preinsert', 'preinsert', CLOSED_GRIPPER)
        add(candidate, 'align', 'align', CLOSED_GRIPPER)
        add(candidate, 'realign', 'realign', CLOSED_GRIPPER)
        add(candidate, 'settle_align', 'align', CLOSED_GRIPPER)
        add(candidate, 'guide', 'guide', CLOSED_GRIPPER)
        add(candidate, 'release', 'guide', OPEN_GRIPPER, steps=15 if candidate_index == len(assignment) - 1 else None)
        add(candidate, 'retreat', 'retreat', OPEN_GRIPPER)
        current = _target(current, arm=candidate.arm, joints=robot[f'{candidate.arm}_arm_joint_state'], gripper=OPEN_GRIPPER)
        stages.append({'name': f'{candidate.label}:park', 'steps': _STEPS['park'], 'target': current})
    current = {'left_arm_joint_state': [float(item) for item in robot['left_arm_joint_state']], 'right_arm_joint_state': [float(item) for item in robot['right_arm_joint_state']], 'left_ee_joint_state': [OPEN_GRIPPER], 'right_ee_joint_state': [OPEN_GRIPPER]}
    stages.append({'name': 'home', 'steps': _STEPS['home'], 'target': current})
    stages.append({'name': 'settle_home', 'steps': _STEPS['settle_home'], 'target': current})
    return stages

__all__ = ['build_insert_tubes_envelope', 'eval_one_episode', 'eval_one_episode_batch']

def _joint_target(runtime, env_idx, arm, semantic_pose):
    seed = runtime.observations[env_idx]["state"][f"{arm}_arm_joint_state"]
    return _joint_target_seeded(runtime, env_idx, arm, semantic_pose, seed)

def _joint_target_seeded(runtime, env_idx, arm, semantic_pose, seed_joints):
    pose = semantic_pose_to_ee(runtime.robot_geometry, semantic_pose)
    return runtime.ik.solve(arm, seed_joints, pose)
