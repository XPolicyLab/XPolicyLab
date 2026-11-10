from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
from .contract import InsertTubesEnvelopeError
from .geometry import _numbers,_rotation,_quaternion,_ee_pose_to_semantic
from .planning import _tube_candidates,_best_assignment,_stages,_pickup_contacts,_delivery_poses,_solve,_joint_target_seeded
_HELD_ROOT_IN_GRIPPER = np.asarray([0.1194, -0.0004, -0.06885])
_CALIBRATED_ROOT_RESIDUAL = {'left': np.asarray([-0.00172, 0.00158]), 'right': np.asarray([0.00158, 0.00165])}

def _initial(value: object, env_idx: int) -> tuple[tuple[list[float], ...], tuple[list[float], ...]]:
    if not isinstance(value, Mapping) or value.get('env_idx') != env_idx:
        raise InsertTubesEnvelopeError('RGB estimate does not match its environment')
    if value.get('provenance') != 'rgb':
        raise InsertTubesEnvelopeError('insert_tubes estimate provenance must be rgb')
    raw_tubes = value.get('tube_poses')
    raw_slots = value.get('slot_poses')
    if not isinstance(raw_tubes, Sequence) or isinstance(raw_tubes, (str, bytes, bytearray, Mapping)) or (not isinstance(raw_slots, Sequence)) or isinstance(raw_slots, (str, bytes, bytearray, Mapping)):
        raise InsertTubesEnvelopeError('RGB geometry must contain pose sequences')
    tubes = tuple((_numbers(pose, 7, f'RGB tube pose[{index}]') for index, pose in enumerate(raw_tubes)))
    slots = tuple((_numbers(pose, 7, f'RGB slot pose[{index}]') for index, pose in enumerate(raw_slots)))
    if not 1 <= len(tubes) <= 3:
        raise InsertTubesEnvelopeError('RGB estimate must contain one to three tubes')
    if len(slots) != 10:
        raise InsertTubesEnvelopeError('RGB estimate must contain ten slot poses')
    return (tubes, slots)

def _resolve_plan(runtime: Any, env_idx: int, robot: Mapping[str, Sequence[float]], value: object) -> dict[str, object]:
    tube_poses, slot_poses = _initial(value, env_idx)
    groups = [_tube_candidates(runtime, env_idx, f'tube{index}', pose, slot_poses) for index, pose in enumerate(tube_poses)]
    assignment = _best_assignment(groups)
    if not assignment:
        print(f'[insert_tubes rgb] env={env_idx} skill_choice=infeasible', flush=True)
        return {'skill_choice': 'infeasible', 'stages': []}
    skill_choice = 'direct' if len(assignment) == 3 else 'partial'
    stages = _stages(robot, assignment)
    contexts = runtime.contexts
    if not isinstance(contexts, dict):
        contexts = {}
        runtime.contexts = contexts
    contexts[env_idx] = {item.label: {'arm': item.arm, 'slot_index': item.slot_index, 'tube_pose': list(tube_poses[int(item.label.removeprefix('tube'))]), 'pickup_contact': list(_pickup_contacts(tube_poses[int(item.label.removeprefix('tube'))])[item.pickup_index]), 'slot_pose': list(slot_poses[item.slot_index]), 'refreshed': False, 'realigned': False, 'guided': False} for item in assignment}
    for context in contexts[env_idx].values():
        context['axis_in_gripper'] = (_rotation(context['pickup_contact'][3:]).T @ _rotation(context['tube_pose'][3:])[:, 1]).tolist()
    planned = ','.join((f'{item.label}:{item.arm}:p{item.pickup_index}:s{item.slot_index}' for item in assignment))
    print(f"[insert_tubes rgb] env={env_idx} skill_choice={skill_choice} detected={len(tube_poses)} planned={len(assignment)} actions={sum((int(stage['steps']) for stage in stages))} plan={planned}", flush=True)
    return {'skill_choice': skill_choice, 'stages': stages}

def _refresh_delivery(runtime: Any, env_idx: int, plan: dict[str, object], label: str, official_observation: Mapping[str, object], axis_estimate: object) -> None:
    contexts = runtime.contexts
    if not isinstance(contexts, Mapping):
        raise InsertTubesEnvelopeError('RGB replan contexts are unavailable')
    env_context = contexts.get(env_idx)
    if not isinstance(env_context, Mapping):
        raise InsertTubesEnvelopeError('RGB environment replan context is unavailable')
    context = env_context.get(label)
    if not isinstance(context, dict) or context.get('refreshed') is True:
        return
    if not isinstance(axis_estimate, Mapping) or axis_estimate.get('env_idx') != env_idx or axis_estimate.get('side') != context['arm'] or (axis_estimate.get('provenance') != 'rgb'):
        raise InsertTubesEnvelopeError('held-tube axis is not RGB-derived')
    if axis_estimate.get('visible') is not True:
        context['refreshed'] = True
        print(f'[insert_tubes rgb] env={env_idx} tube={label} live_replan=not_visible keep_planned_delivery continue_head_realign', flush=True)
        return
    axis_in_gripper = np.asarray(_numbers(axis_estimate.get('axis_in_gripper'), 3, 'held tube axis in gripper'))
    axis_in_gripper /= np.linalg.norm(axis_in_gripper)
    context['axis_in_gripper'] = axis_in_gripper.tolist()
    state = official_observation.get('state')
    if not isinstance(state, Mapping):
        raise InsertTubesEnvelopeError('official observation has no state')
    arm = str(context['arm'])
    robot = runtime.robot_geometry
    ee_pose = _numbers(state[f'{arm}_ee_pose'], 7, f'robot.{arm}_ee_pose')
    semantic_pose = _ee_pose_to_semantic(robot, ee_pose)
    semantic_rotation = _rotation(semantic_pose[3:])
    pickup_rotation = _rotation(context['pickup_contact'][3:])
    nominal_relative_rotation = pickup_rotation.T @ _rotation(context['tube_pose'][3:])
    relative_x = nominal_relative_rotation[:, 0]
    relative_x -= axis_in_gripper * float(relative_x @ axis_in_gripper)
    relative_x /= np.linalg.norm(relative_x)
    relative_rotation = np.column_stack((relative_x, axis_in_gripper, np.cross(relative_x, axis_in_gripper)))
    tube_rotation = semantic_rotation @ relative_rotation
    tube_position = np.asarray(semantic_pose[:3], dtype=float) + semantic_rotation @ _HELD_ROOT_IN_GRIPPER
    tube_pose = tube_position.tolist() + _quaternion(tube_rotation)
    poses = _delivery_poses(tube_pose, semantic_pose, context['slot_pose'])
    joints = _solve(runtime, env_idx, arm, poses)
    if joints is None:
        context['refreshed'] = True
        print(f'[insert_tubes rgb] env={env_idx} tube={label} live_replan=unreachable keep_planned_delivery continue_head_realign', flush=True)
        return
    stages = plan.get('stages')
    if not isinstance(stages, list):
        raise InsertTubesEnvelopeError('cached RGB stages are unavailable')
    for stage in stages:
        stage_name = str(stage['name'])
        if not stage_name.startswith(f'{label}:'):
            continue
        phase = stage_name.split(':', 1)[1]
        joint_name = 'align' if phase == 'settle_align' else phase
        if joint_name in joints:
            target = stage['target']
            if not isinstance(target, Mapping):
                raise InsertTubesEnvelopeError('RGB stage target is invalid')
            stage['target'] = {key: list(value) for key, value in target.items()}
            stage['target'][f'{arm}_arm_joint_state'] = list(joints[joint_name])
    context['refreshed'] = True
    print(f'[insert_tubes rgb] env={env_idx} tube={label} live_replan=ready axis_in_gripper={[round(float(value), 5) for value in axis_in_gripper]} attachment={_HELD_ROOT_IN_GRIPPER.tolist()}', flush=True)

def _refresh_after_lift(runtime: Any, env_idx: int, plan: dict[str, object], official_observation: Mapping[str, object]) -> None:
    action_count = int(runtime.action_counts[env_idx])
    elapsed = 0
    for stage in plan['stages']:
        elapsed += int(stage['steps'])
        if action_count != elapsed or not str(stage['name']).endswith(':lift'):
            continue
        label = str(stage['name']).split(':', 1)[0]
        contexts = runtime.contexts[env_idx]
        side = str(contexts[label]['arm'])
        tube_rotation = _rotation(contexts[label]['tube_pose'][3:])
        tube_axis = tube_rotation[:, 1]
        estimated_yaw = float(np.arctan2(tube_axis[1], tube_axis[0]))
        axis_estimate = [runtime.perception.held_axis(env_idx, side, estimated_yaw)][0]
        _refresh_delivery(runtime, env_idx, plan, label, official_observation, axis_estimate)
        return

def _refresh_realign(runtime: Any, env_idx: int, plan: dict[str, object], label: str, official_observation: Mapping[str, object], cap_estimate: object) -> None:
    contexts = runtime.contexts
    if not isinstance(contexts, Mapping):
        raise InsertTubesEnvelopeError('RGB replan contexts are unavailable')
    env_context = contexts.get(env_idx)
    if not isinstance(env_context, Mapping):
        raise InsertTubesEnvelopeError('RGB environment replan context is unavailable')
    context = env_context.get(label)
    if not isinstance(context, dict) or context.get('realigned') is True:
        return
    if not isinstance(cap_estimate, Mapping) or cap_estimate.get('env_idx') != env_idx or cap_estimate.get('provenance') != 'rgb':
        raise InsertTubesEnvelopeError('aligned cap is not RGB-derived')
    cap_xy = None
    if cap_estimate.get('visible') is True:
        cap_xy = np.asarray(_numbers(cap_estimate.get('cap_xy'), 2, 'aligned cap xy'), dtype=float)
    axis_in_gripper = np.asarray(_numbers(context.get('axis_in_gripper'), 3, 'held tube axis in gripper'), dtype=float)
    state = official_observation.get('state')
    if not isinstance(state, Mapping):
        raise InsertTubesEnvelopeError('official observation has no state')
    arm = str(context['arm'])
    robot = runtime.robot_geometry
    ee_pose = _numbers(state[f'{arm}_ee_pose'], 7, f'robot.{arm}_ee_pose')
    semantic_pose = _ee_pose_to_semantic(robot, ee_pose)
    semantic_rotation = _rotation(semantic_pose[3:])
    axis_world = semantic_rotation @ axis_in_gripper
    semantic_position = np.asarray(semantic_pose[:3], dtype=float)
    if cap_xy is None:
        root_residual = _CALIBRATED_ROOT_RESIDUAL[arm].copy()
    else:
        observed_root_xy = cap_xy - axis_world[:2] * 0.1404
        slot_xy = np.asarray(context['slot_pose'][:2], dtype=float)
        root_residual = observed_root_xy - slot_xy
    residual_norm = float(np.linalg.norm(root_residual))
    if residual_norm > 0.0035:
        root_residual *= 0.0035 / residual_norm
    correction = np.asarray([-root_residual[0], -root_residual[1], 0.0])
    slot_axis = _rotation(context['slot_pose'][3:])[:, 2]
    poses = {'realign': (semantic_position + correction - slot_axis * 0.005).tolist() + list(semantic_pose[3:]), 'guide': (semantic_position + correction - slot_axis * 0.03).tolist() + list(semantic_pose[3:])}
    current_joints = _numbers(state[f'{arm}_arm_joint_state'], 6, f'robot.{arm}_arm_joint_state')
    realign_joints = _joint_target_seeded(runtime, env_idx, arm, poses['realign'], current_joints)
    if realign_joints is None:
        realign_joints = current_joints
        print(f'[insert_tubes rgb] env={env_idx} tube={label} visual_realign=unreachable guided_drop_from=align', flush=True)
    guide_joints = _joint_target_seeded(runtime, env_idx, arm, poses['guide'], realign_joints)
    if guide_joints is None:
        print(f'[insert_tubes rgb] env={env_idx} tube={label} visual_guide=unreachable keep_planned_guide', flush=True)
        context['realigned'] = True
        return
    stages = plan.get('stages')
    if not isinstance(stages, list):
        raise InsertTubesEnvelopeError('cached RGB stages are unavailable')
    for stage in stages:
        name = str(stage['name'])
        if name in {f'{label}:realign', f'{label}:settle_align'}:
            joints = realign_joints
        elif name in {f'{label}:guide', f'{label}:release'}:
            joints = guide_joints
        else:
            continue
        target = stage['target']
        if not isinstance(target, Mapping):
            raise InsertTubesEnvelopeError('RGB stage target is invalid')
        stage['target'] = {key: list(value) for key, value in target.items()}
        stage['target'][f'{arm}_arm_joint_state'] = list(joints)
    context['realigned'] = True
    print(f"[insert_tubes rgb] env={env_idx} tube={label} visual_realign={('calibrated' if cap_xy is None else 'ready')} cap_xy={(None if cap_xy is None else [round(float(value), 5) for value in cap_xy])} root_in_slot_xy={[round(float(value), 5) for value in root_residual]}", flush=True)

def _refresh_after_align(runtime: Any, env_idx: int, plan: dict[str, object], official_observation: Mapping[str, object]) -> None:
    action_count = int(runtime.action_counts[env_idx])
    elapsed = 0
    for stage in plan['stages']:
        elapsed += int(stage['steps'])
        stage_name = str(stage['name'])
        if action_count != elapsed or not stage_name.endswith(':align') or stage_name.endswith(':settle_align'):
            continue
        label = stage_name.split(':', 1)[0]
        context = runtime.contexts[env_idx][label]
        if context.get('realigned') is True:
            return
        axis_in_gripper = np.asarray(context['axis_in_gripper'], dtype=float)
        state = official_observation.get('state')
        if not isinstance(state, Mapping):
            raise InsertTubesEnvelopeError('official observation has no state')
        arm = str(context['arm'])
        robot = runtime.robot_geometry
        ee_pose = _numbers(state[f'{arm}_ee_pose'], 7, f'robot.{arm}_ee_pose')
        semantic_pose = _ee_pose_to_semantic(robot, ee_pose)
        axis_world = _rotation(semantic_pose[3:]) @ axis_in_gripper
        slot_pose = np.asarray(context['slot_pose'][:3], dtype=float)
        expected_cap = slot_pose + np.asarray([0.0, 0.0, 0.015])
        expected_cap += axis_world * 0.1404
        cap_estimate = [runtime.perception.aligned_cap(env_idx, expected_cap.tolist())][0]
        _refresh_realign(runtime, env_idx, plan, label, official_observation, cap_estimate)
        return

def _refresh_guide(runtime: Any, env_idx: int, plan: dict[str, object], label: str, official_observation: Mapping[str, object], cap_estimate: object) -> None:
    context = runtime.contexts[env_idx][label]
    if context.get('guided') is True:
        return
    context['guided'] = True
    if not isinstance(cap_estimate, Mapping) or cap_estimate.get('env_idx') != env_idx or cap_estimate.get('provenance') != 'rgb':
        raise InsertTubesEnvelopeError('guided cap is not RGB-derived')
    if cap_estimate.get('visible') is not True:
        print(f'[insert_tubes rgb] env={env_idx} tube={label} visual_guide=not_visible keep_realign_guide', flush=True)
        return
    cap_xy = np.asarray(_numbers(cap_estimate.get('cap_xy'), 2, 'guided cap xy'), dtype=float)
    state = official_observation.get('state')
    if not isinstance(state, Mapping):
        raise InsertTubesEnvelopeError('official observation has no state')
    arm = str(context['arm'])
    robot = runtime.robot_geometry
    semantic_pose = _ee_pose_to_semantic(robot, _numbers(state[f'{arm}_ee_pose'], 7, f'robot.{arm}_ee_pose'))
    semantic_rotation = _rotation(semantic_pose[3:])
    axis_world = semantic_rotation @ np.asarray(_numbers(context['axis_in_gripper'], 3, 'held tube axis in gripper'))
    root_residual = cap_xy - axis_world[:2] * 0.1404 - np.asarray(context['slot_pose'][:2], dtype=float)
    residual_norm = float(np.linalg.norm(root_residual))
    if residual_norm > 0.0025:
        root_residual *= 0.0025 / residual_norm
    slot_axis = _rotation(context['slot_pose'][3:])[:, 2]
    correction = np.asarray([-root_residual[0], -root_residual[1], 0.0])
    guide_pose = (np.asarray(semantic_pose[:3], dtype=float) + correction - slot_axis * 0.025).tolist() + list(semantic_pose[3:])
    current_joints = _numbers(state[f'{arm}_arm_joint_state'], 6, f'robot.{arm}_arm_joint_state')
    guide_joints = _joint_target_seeded(runtime, env_idx, arm, guide_pose, current_joints)
    if guide_joints is None:
        print(f'[insert_tubes rgb] env={env_idx} tube={label} second_visual_guide=unreachable keep_realign_guide', flush=True)
        return
    for stage in plan['stages']:
        if stage['name'] not in {f'{label}:guide', f'{label}:release'}:
            continue
        target = stage['target']
        stage['target'] = {key: list(value) for key, value in target.items()}
        stage['target'][f'{arm}_arm_joint_state'] = list(guide_joints)
    print(f'[insert_tubes rgb] env={env_idx} tube={label} second_visual_guide=ready root_in_slot_xy={[round(float(value), 5) for value in root_residual]}', flush=True)

def _refresh_before_guide(runtime: Any, env_idx: int, plan: dict[str, object], official_observation: Mapping[str, object]) -> None:
    action_count = int(runtime.action_counts[env_idx])
    elapsed = 0
    for stage in plan['stages']:
        elapsed += int(stage['steps'])
        stage_name = str(stage['name'])
        if action_count != elapsed or not stage_name.endswith(':settle_align'):
            continue
        label = stage_name.split(':', 1)[0]
        context = runtime.contexts[env_idx][label]
        if context.get('guided') is True:
            return
        state = official_observation.get('state')
        if not isinstance(state, Mapping):
            raise InsertTubesEnvelopeError('official observation has no state')
        arm = str(context['arm'])
        robot = runtime.robot_geometry
        semantic_pose = _ee_pose_to_semantic(robot, _numbers(state[f'{arm}_ee_pose'], 7, f'robot.{arm}_ee_pose'))
        axis_world = _rotation(semantic_pose[3:]) @ np.asarray(context['axis_in_gripper'], dtype=float)
        slot_pose = np.asarray(context['slot_pose'][:3], dtype=float)
        slot_axis = _rotation(context['slot_pose'][3:])[:, 2]
        expected_cap = slot_pose + slot_axis * 0.01 + axis_world * 0.1404
        cap_estimate = [runtime.perception.aligned_cap(env_idx, expected_cap.tolist())][0]
        _refresh_guide(runtime, env_idx, plan, label, official_observation, cap_estimate)
        return
__all__ = ['eval_one_episode', 'eval_one_episode_batch']
