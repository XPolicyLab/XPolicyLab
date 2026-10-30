"""Measure real replay action costs and preflight optional direct-left pickups."""
import copy
import math
import numpy as np


def timed_actions(ctx, program, label, nominal_frames=None, arm=None):
    """Count every emitted action, including retiming and measured arrival waits."""
    record = {'label': label, 'stage': ctx.stage, 'arm': arm,
              'start_step': int(ctx.steps), 'nominal_frames': nominal_frames,
              'actual_frames': 0, 'completed': False, 'motion_phases': {}}
    ctx.memory.execution_segments.append(record)
    try:
        for action in program:
            record['actual_frames'] += 1
            phase = getattr(ctx, 'motion_plan', {}).get('phase', 'unreported')
            record['motion_phases'][phase] = record['motion_phases'].get(phase, 0) + 1
            yield action
        record['completed'] = True
    except Exception as error:
        record['error'] = str(error)
        raise
    finally:
        record['end_step'] = int(ctx.steps)
        if nominal_frames is not None:
            record['extra_frames'] = max(0, record['actual_frames']-nominal_frames)


def observed_motion_extra(memory, remaining_objects, profile_stability_floor=0):
    """Keep a conservative historical reserve and increase it from measured costs."""
    completed = [record for record in memory.execution_segments
                 if record['completed'] and record['nominal_frames'] is not None
                 and record['label'] != 'final_home']
    completed_objects = len(memory.replay_completions)
    extra = sum(record.get('extra_frames', 0) for record in completed)
    measured_per_object = extra/max(1, completed_objects)
    reserve = max(20*remaining_objects+profile_stability_floor,
                  int(math.ceil(measured_per_object*remaining_objects)))
    return reserve, {'historical_extra_per_object': 20, 'completed_objects': completed_objects,
        'measured_extra_frames': extra, 'profile_stability_floor':profile_stability_floor,
        'profile_stability_reserve_basis':'conservative_prior_separate_joint_and_cartesian_settles',
        'profile_stability_reserve_is_actual_frame_floor':False,
        'selected_profile_command_frames_measured_per_segment':True,
        'reserved_extra_per_object': reserve/max(1,remaining_objects),
        'forecast_is_estimate':True,'future_retiming_is_not_guaranteed':True}


def direct_left_preflight(ctx, position, quaternion_candidates, destination, sequence_position=0):
    """Reuse the public motion planner before replacing a central table handoff."""
    if not ctx.numeric_ik or not 0 <= position[0] <= .28:
        return None
    from coap.tasks.memory import reachable_lift
    rejected = []
    current = np.asarray(ctx.state['left_ee_pose'][3:], float)
    choices = sorted(quaternion_candidates,
                     key=lambda quat: abs(float(np.dot(quat, current))), reverse=True)
    for quat in choices:
        preview = copy.copy(ctx)
        preview.memory = copy.deepcopy(getattr(ctx, "memory", {}))
        try:
            lift = reachable_lift(preview, 'left', position, quat)
            frames = 0
            state = {key: np.asarray(value).copy() for key, value in ctx.state.items()}
            segments = []
            for label, point, grip, duration in (('approach',position+[0.,0.,lift],1.,18),
                                                ('descent',position,1.,12),
                                                ('lift',position+[0.,0.,lift],0.,18),
                                                ('delivery',destination,0.,24)):
                preview.stage = f'deliver_{sequence_position}' if label=='delivery' else f'replay_{sequence_position}'
                record = {'label':label,'stage':preview.stage,'nominal_frames':duration,'ideal_command_frames':0}
                segments.append(record)
                for action in preview.move('left', point, quat, grip=grip, duration=duration):
                    record['ideal_command_frames'] += 1
                    frames += 1
                    for arm in ('left', 'right'):
                        joints = np.asarray(action[arm+'_arm_joint_state']).copy()
                        state[arm+'_arm_joint_state'] = joints
                        state[arm+'_ee_pose'] = preview.kinematics.forward(arm, joints)
                        state[arm+'_ee_joint_state'] = np.asarray(action[arm+'_ee_joint_state']).copy()
                    preview.update_obs({'state': state.copy(), 'vision': ctx.vision})
                record['motion_profile'] = getattr(preview,'motion_plan',{}).get('motion_profile')
            return {'arm': 'left', 'quaternion': np.asarray(quat).copy(),
                    'lift_height': lift, 'preflight_command_frames': frames,
                    'preflight_motion_segments':segments, 'preflight_motion_nominal_frames':72,
                    'unpreviewed_gripper_and_settle_frames':30,
                    'ideal_complete_route_frames':frames+30,
                    'forecast_uses_actual_selected_profile':True,
                    'source': 'public_robot_pickup_and_delivery_preflight',
                    'physics_verified': False, 'rejected': rejected}
        except RuntimeError as error:
            rejected.append(str(error))
    return None
