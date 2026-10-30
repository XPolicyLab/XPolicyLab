"""Grasp four queued eggs vertically across their long axes with one arm."""
import math

import cv2
import numpy as np

from b88r42wq.motion import down_quat, quat_matrix, tcp_to_link, time_parameterize_joint_path
from b88r42wq.tasks.egg_motion import enable, endpoint_solution
from b88r42wq.tasks.egg_single_arm import (EGG_EMPTY_PROFILE, EGG_CARRY_PROFILE, EGG_PLACE_PROFILE,
    select_visible_holder, initialize_target_queue, close_lid, finish_egg_arm)

from b88r42wq.tasks.egg_carry import carry_orientations


# Leave distal pad travel for source-height error and gravity-driven shell sliding.
EGG_PAD_CENTER_OFFSET_M=.0025


def _chain_cost(ctx,arm,goals):
    """Rank only the current motion using public kinematics, without rejection gates."""
    if not ctx.numeric_ik:
        return sum(goal[2] for goal in goals)
    joints=np.asarray(ctx.state[arm+'_arm_joint_state'],float).copy()
    cost=0.
    for position,quat,duration,velocity in goals:
        result=endpoint_solution(ctx,arm,tcp_to_link(position,quat),joints)
        target=np.asarray(result['joint_value'],float)
        frames=len(time_parameterize_joint_path(np.stack([joints,target]),duration,velocity_limit=velocity))
        yaw=float(target[0]-ctx.initial_state[arm+'_arm_joint_state'][0])
        cost+=frames+10000.*result['position_error']+200.*result['rotation_error']+20.*(1.-math.cos(yaw))
        joints=target
    return float(cost)


def choose_pick(ctx,arm,center,yaw):
    """Keep both equivalent jaw headings exactly downward and across the shell."""
    position=np.asarray(center,float)+[0.,0.,EGG_PAD_CENTER_OFFSET_M]
    candidates=[]
    for offset in (0.,math.pi):
        quat=down_quat(float(yaw)+offset)
        above=position+[0.,0.,.09]
        cost=_chain_cost(ctx,arm,[(above,quat,10,3.),(position,quat,8,1.6)])
        candidates.append((cost,position,above,quat))
    return min(candidates,key=lambda item:item[0])[1:]


def choose_placement(ctx,arm,slot,grasp_quat,yaw,pitch,held_offset):
    """Stand the measured shell axis and compensate its known grasp-frame offset."""
    choices=[]
    for quat in carry_orientations(ctx,grasp_quat,yaw,pitch,slot,arm=arm):
        offset=quat_matrix(quat)@held_offset
        target=np.asarray(slot,float).copy()
        target[:2]-=offset[:2]
        target[2]=max(float(slot[2])+.023,ctx.table_z+.065)-offset[2]
        above=target+[0.,0.,.07]
        retreat=target.copy();retreat[2]=max(target[2]+.06,ctx.table_z+.125)
        cost=_chain_cost(ctx,arm,[(above,quat,18,1.6),(target,quat,6,1.6),
                                (retreat,quat,6,3.)])
        choices.append((cost,quat,target,above,retreat))
    return min(choices,key=lambda item:item[0])[1:]


def _grasp_observation(ctx,arm,center,position,quat,yaw,pitch):
    """Record direction and position errors without waiting, retrying, or stopping."""
    measured=np.asarray(ctx.state[arm+'_ee_pose'],float)
    matrix=quat_matrix(measured[3:]);command=quat_matrix(quat)
    pitch=0. if pitch is None else float(pitch)
    axis=np.asarray([math.cos(yaw)*math.cos(pitch),math.sin(yaw)*math.cos(pitch),math.sin(pitch)])
    actual=ctx.tcp(arm) if ctx.numeric_ik else measured[:3]
    return dict(source_center=np.asarray(center).tolist(),grasp_position=np.asarray(position).tolist(),
        source_yaw=float(yaw),source_pitch=pitch,command_quaternion=np.asarray(quat).tolist(),
        measured_tcp=np.asarray(actual).tolist(),position_error_m=float(np.linalg.norm(actual-position)),
        commanded_tool_down_cosine=float(-command[2,0]),
        commanded_closing_long_axis_dot=float(np.dot(command[:,1],axis)),
        measured_tool_down_cosine=float(-matrix[2,0]),
        measured_closing_long_axis_dot=float(np.dot(matrix[:,1],axis)),
        diagnostic_only=True,grasp_retention_asserted=False)


def run(ctx):
    """Keep a fixed four-source queue and execute every pick-place before the cover."""
    from b88r42wq.tasks.long_horizon import _egg_slots
    from b88r42wq.tasks.egg_source import measure_source
    enable(ctx)
    ctx.stage='measure'
    selected=select_visible_holder(ctx,_egg_slots)
    for attempt in range(2):
        if selected is not None:
            break
        ctx.stage=f'remeasure_egg_holder_{attempt}'
        yield from ctx.hold(3)
        selected=select_visible_holder(ctx,_egg_slots)
    if selected is None:
        raise RuntimeError('Initial four egg wells were not observed')
    holder,slots,root,rotation=selected
    arm,queue,head_reference=yield from initialize_target_queue(ctx,holder,slots)
    ctx.egg_active_arm=arm
    ctx.memory['egg_execution']=dict(active_arm=arm,passive_arm='left' if arm=='right' else 'right',
        placed=0,queued_sources=4,passive_arm_support=False,placement_counter_is_action_sequences_only=True)
    ctx.memory['egg_grasps']=[]
    for index,entry in enumerate(queue):
        egg=entry['source'];slot=entry['well_position']
        record=ctx.memory['egg_target_queue']['entries'][index]
        record.update(status='executing',start_step=int(ctx.steps))
        center=np.asarray(egg['position'],float).copy()
        center[2]=max(center[2],ctx.table_z+.034)
        yaw=float(egg['yaw']);pitch=None
        position,above,quat=choose_pick(ctx,arm,center,yaw)
        ctx.stage=f'pick_egg_{index}'
        ctx.memory['egg_pick_plan']=dict(source_index=index,source_id=egg['source_id'],
            position=position.tolist(),quat=quat.tolist(),hover=.09,active_arm=arm,
            safety_checks_enabled=False,approach='directly_above_source')
        yield from ctx.move(arm,above,quat,grip=.65,duration=10,motion_profile=EGG_EMPTY_PROFILE)
        ctx.stage=f'measure_egg_{index}'
        try:
            fitted=measure_source(ctx,egg,arm,index,head_reference=head_reference)
        except (cv2.error,ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
            fitted=None
            ctx.memory.setdefault('egg_source_fit_fallbacks',[]).append(dict(source_index=index,reason=str(error)))
        if fitted is not None:
            center=np.asarray(fitted['position'],float).copy()
            yaw=float(fitted['yaw']);pitch=float(fitted['pitch'])
            position,above,quat=choose_pick(ctx,arm,center,yaw)
            ctx.stage=f'align_above_egg_{index}'
            yield from ctx.move(arm,above,quat,grip=.65,duration=6,motion_profile=EGG_EMPTY_PROFILE)
        ctx.stage=f'descend_egg_{index}'
        yield from ctx.move(arm,position,quat,grip=.65,duration=8)
        yield from ctx.hold(2)
        grasp=dict(source_index=index,source_id=egg['source_id'],measurement_step=int(ctx.steps),
                   above=above.tolist(),**_grasp_observation(ctx,arm,center,position,quat,yaw,pitch))
        ctx.memory['egg_grasps'].append(grasp)
        ctx.stage=f'clamp_egg_{index}'
        yield from ctx.grip(arm,0.,duration=10)
        yield from ctx.hold(4)
        grasp['after_closure']=_grasp_observation(ctx,arm,center,position,quat,yaw,pitch)
        held_offset=quat_matrix(quat).T@(center-position)
        ctx.stage=f'lift_egg_{index}'
        yield from ctx.move(arm,position+[0.,0.,.08],quat,grip=0.,duration=12,
                            motion_profile=EGG_CARRY_PROFILE)
        place_quat,target,place_above,retreat=choose_placement(ctx,arm,slot,quat,yaw,pitch,held_offset)
        ctx.memory['egg_release_geometry']=dict(source_index=index,source_id=egg['source_id'],
            offset_source='source_rgb_grasp_frame',shell_center=(target+quat_matrix(place_quat)@held_offset).tolist(),
            tcp_target=target.tolist(),well_position=np.asarray(slot).tolist(),retention_verified=False)
        ctx.stage=f'transport_egg_{index}'
        yield from ctx.move(arm,place_above,place_quat,grip=0.,duration=18,motion_profile=EGG_CARRY_PROFILE)
        ctx.stage=f'place_egg_{index}'
        yield from ctx.move(arm,target,place_quat,grip=0.,duration=6,motion_profile=EGG_PLACE_PROFILE)
        ctx.stage=f'release_egg_{index}'
        yield from ctx.grip(arm,1.,duration=8)
        yield from ctx.hold(2)
        ctx.stage=f'retreat_egg_{index}'
        yield from ctx.move(arm,retreat,place_quat,grip=1.,duration=6,motion_profile=EGG_EMPTY_PROFILE)
        record.update(status='action_sequence_complete',end_step=int(ctx.steps))
        ctx.memory['egg_execution']['placed']+=1
        ctx.memory['egg_target_queue']['completed_command_sequences']+=1
        ctx.memory['egg_target_queue']['pending_count']-=1
    yield from close_lid(ctx,arm,root,rotation)
    yield from finish_egg_arm(ctx,arm)
