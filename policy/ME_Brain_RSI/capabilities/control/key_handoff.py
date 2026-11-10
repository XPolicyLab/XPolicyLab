"""Transfer a visually located key from its blade to the flat bow faces."""
import copy
import math
import numpy as np
from coap.public_mesh import PublicRobotConvexGeometry
from coap.tasks.mesh_kinematics import MeshFilteredKinematics


def make_plan(ctx, key, slot, transfer_y=-.30):
    """Use observed bow direction and public dimensions to preserve directed entry."""
    from coap.tasks import precision as p
    measured = p._key_bow(ctx, key)
    if measured is None:
        raise RuntimeError('Key bow and blade direction are not visually resolved')
    bow, _, yaw = measured
    donor = 'left' if bow[0] < 0 else 'right'
    receiver = 'right' if donor == 'left' else 'left'
    sign = 1. if receiver == 'right' else -1.
    heading = 0. if sign > 0 else math.pi
    axis = p._direction(yaw)
    blade_grasp = np.asarray(bow) - .025*axis
    blade_grasp[2] = ctx.table_z+.014
    donor_quat = ctx.down_quat(yaw)
    transfer = np.array([-.04*sign, transfer_y, ctx.table_z+.135])
    transfer_quat = p._multiply(p._yaw_quaternion(heading-yaw), donor_quat)
    flat_axis = p._direction(heading)
    transferred_bow = transfer+.025*flat_axis
    transferred_bow[2] -= .009
    face_quat = p._multiply(p._yaw_quaternion(heading+math.pi/2),
                            np.array([.5,.5,.5,.5]))
    face_tcp = transferred_bow-p._rotate(face_quat,[.00565,0,0])
    slot_yaw = float(slot.get('slit_yaw',p._observed_yaw(slot)))
    slot_yaw = (slot_yaw+math.pi/2)%math.pi-math.pi/2
    upright = p._axis_rotation(np.cross(flat_axis,[0,0,1]),math.pi/2)
    correction = p._yaw_quaternion(slot_yaw-heading-math.pi/2)
    rotation = p._multiply(correction,upright)
    insert_quat = p._multiply(rotation,face_quat)
    bow_offset = p._rotate(rotation,transferred_bow-face_tcp)
    hole = np.asarray(slot.get('keyhole_position',p._position(slot))).copy()
    hole[2] = ctx.table_z+.055
    touch = hole+[0,0,.0528467+.004]-bow_offset
    return {'donor':donor,'receiver':receiver,'blade_grasp':blade_grasp,
            'donor_quat':donor_quat,'transfer':transfer,'transfer_quat':transfer_quat,
            'face_tcp':face_tcp,'face_quat':face_quat,'insert_quat':insert_quat,
            'touch':touch,'inserted':touch-[0,0,.030],
            'observed_bow':np.asarray(bow),'transferred_bow':transferred_bow,
            'source_yaw':yaw,'slot_yaw':slot_yaw}


def source_program(ctx, plan):
    """Acquire the blade and expose the bow before measuring the handoff."""
    from coap.tasks import precision as p
    donor, receiver = plan['donor'], plan['receiver']
    source, source_quat = plan['blade_grasp'], plan['donor_quat']
    transfer, transfer_quat = plan['transfer'], plan['transfer_quat']
    face, face_quat = plan['face_tcp'], plan['face_quat']
    insertion, insertion_quat = plan['inserted'], plan['insert_quat']
    ctx.stage = 'key_blade_grasp'
    yield from p._grasp(ctx,donor,source,source_quat,lift=.13,fast=True)
    ctx.stage = 'key_blade_transfer'
    yield from ctx.move(donor,transfer,transfer_quat,0.,18)


def handoff_program(ctx, plan, observe_slot=False):
    """Keep the donor closed until the receiver acquires the observed bow faces."""
    from coap.tasks import precision as p
    donor, receiver = plan['donor'], plan['receiver']
    transfer, transfer_quat = plan['transfer'], plan['transfer_quat']
    face, face_quat = plan['face_tcp'], plan['face_quat']
    insertion, insertion_quat = plan['inserted'], plan['insert_quat']
    ctx.stage = 'key_face_approach'
    approach = face-.055*p._rotate(face_quat,[1,0,0])
    yield from p._keep_closed(ctx.move(receiver,approach,face_quat,.5,14),donor)
    ctx.stage = 'key_face_contact'
    yield from p._keep_closed(ctx.move(receiver,face,face_quat,.5,12),donor)
    ctx.stage = 'key_face_close'
    yield from p._keep_closed(ctx.move(receiver,face,face_quat,0.,10),donor)
    ctx.stage = 'key_donor_release'
    yield from p._keep_closed(ctx.move(donor,transfer,transfer_quat,1.,8),receiver)
    ctx.stage = 'key_donor_retract'
    yield from p._keep_closed(ctx.move(donor,transfer+[0,0,.08],transfer_quat,1.,10),receiver)
    yield from p._keep_closed(ctx.home(donor,duration=12),receiver)
    ctx.stage = 'key_upright'
    yield from ctx.move(receiver,face+[0,0,.02],insertion_quat,0.,20)
    ctx.stage = 'key_transport'
    yield from ctx.move(receiver,plan['touch']+[0,0,.025],insertion_quat,0.,12)
    if observe_slot:
        from coap.tasks.key_collision import audit_key_frames
        from coap.tasks.key_aperture import measured_aperture
        ctx.stage = 'key_slot_measure'
        yield from ctx.hold(2)
        rotation = p._multiply(plan['insert_quat'],p._inverse(plan['face_quat']))
        offset = p._rotate(rotation,np.asarray(plan['transferred_bow'])-plan['face_tcp'])
        expected = np.asarray(plan['touch'])+offset-[0,0,.0528467+.004]
        if measured_aperture(ctx,receiver,expected) is None:
            selected = None
            for sign in (1.,-1.):
                lateral = sign*.014*p._direction(plan['slot_yaw']+np.pi/2)
                scout = dict(plan,scout_target=ctx.tcp(receiver)+lateral,
                             scout_quat=np.asarray(ctx.state[receiver+'_ee_pose'][3:]))
                try:
                    preview,frames = preview_frames(ctx,scout,slot_scout_program)
                    audit = audit_key_frames(preview,frames,plan['measured_slot'])
                    if audit['accepted']:
                        selected = scout
                        ctx.memory['keyhole_visibility_scout'] = {'offset':lateral,
                                                                 'actions':len(frames),
                                                                 'public_collision_audit':audit}
                        break
                except RuntimeError:
                    continue
            if selected is None:
                raise RuntimeError('No complete collision-checked keyhole visibility motion found')
            yield from slot_scout_program(ctx,selected)
        plan = aperture_plan(ctx,plan)
        preview,frames = preview_frames(ctx,plan,insertion_program)
        slot = plan['measured_slot']
        audit = audit_key_frames(preview,frames,slot)
        if not audit['accepted']:
            raise RuntimeError('The current RGB keyhole correction failed complete public collision checks')
        ctx.memory['key_handoff_plan'].update(plan,keyhole_suffix_audit=audit,
                                              keyhole_remaining_actions=len(frames))
    yield from insertion_program(ctx,plan)


def slot_scout_program(ctx, plan):
    """Uncover an aperture hidden behind the held blade using a small lateral view."""
    ctx.stage = 'key_slot_visibility_scout'
    yield from ctx.move(plan['receiver'],plan['scout_target'],plan['scout_quat'],0.,6)
    yield from ctx.hold(2)


def insertion_program(ctx, plan):
    """Align, insert and turn using the registered circle center and directed slit."""
    from coap.tasks import precision as p
    receiver = plan['receiver']
    insertion,insertion_quat = plan['inserted'],plan['insert_quat']
    ctx.stage = 'key_align'
    yield from ctx.move(receiver,plan['touch'],insertion_quat,0.,12)
    ctx.stage = 'key_insert'
    yield from ctx.move(receiver,insertion,insertion_quat,0.,20)
    ctx.stage = 'key_turn'
    turned = p._multiply(p._yaw_quaternion(-math.pi/3),insertion_quat)
    yield from ctx.move(receiver,insertion,turned,0.,18)
    yield from ctx.hold(4)
    ctx.stage = 'key_release'
    yield from ctx.move(receiver,insertion,turned,1.,6)
    yield from ctx.move(receiver,insertion+[0,0,.03],turned,1.,10)
    yield from ctx.home(receiver,duration=12)


def aperture_plan(ctx, plan):
    """Replace a coarse head contour centroid with the public wrist aperture fit."""
    from coap.tasks import precision as p
    from coap.tasks.key_aperture import measured_aperture
    rotation = p._multiply(plan['insert_quat'],p._inverse(plan['face_quat']))
    offset = p._rotate(rotation,np.asarray(plan['transferred_bow'])-plan['face_tcp'])
    expected = np.asarray(plan['touch'])+offset-[0,0,.0528467+.004]
    measurement = measured_aperture(ctx,plan['receiver'],expected)
    if measurement is None:
        raise RuntimeError('The current asymmetric keyhole outline is not uniquely resolved in wrist RGB')
    correction = p._yaw_quaternion(measurement['slit_yaw']-plan['slot_yaw'])
    quaternion = p._multiply(correction,plan['insert_quat'])
    rotation = p._multiply(quaternion,p._inverse(plan['face_quat']))
    offset = p._rotate(rotation,np.asarray(plan['transferred_bow'])-plan['face_tcp'])
    touch = measurement['position']+[0,0,.0528467+.004]-offset
    slot = dict(plan.get('measured_slot',{}))
    if not slot:
        raise RuntimeError('The observed slot outer body is required for collision checks')
    slot.update(keyhole_position=measurement['position'],slit_yaw=measurement['slit_yaw'],
                position=measurement['position']+.002*p._direction(measurement['slit_yaw'])-[0,0,.0275])
    return dict(plan,touch=touch,inserted=touch-[0,0,.030],insert_quat=quaternion,
                slot_yaw=measurement['slit_yaw'],keyhole_measurement=measurement,
                measured_slot=slot)


def program(ctx, plan):
    """Replay the complete nominal path for public collision preflight."""
    yield from source_program(ctx,plan)
    yield from handoff_program(ctx,plan)


def observed_plan(ctx, plan, slot):
    """Correct bow sag from the current RGB circle while preserving the public entry."""
    from coap.tasks import precision as p
    from coap.tasks.key_visual import measured_bow
    heading = 0. if plan['receiver']=='right' else math.pi
    direction = p._direction(heading)
    expected = ctx.tcp(plan['donor'])+.025*direction-[0,0,.009]
    measurement = measured_bow(ctx,plan['donor'],expected,direction)
    if measurement is None:
        raise RuntimeError('The exposed key bow is not independently resolved in current RGB')
    adjusted = dict(plan)
    bow = measurement['position']
    face = bow-p._rotate(plan['face_quat'],[.00565,0,0])
    rotation = p._multiply(plan['insert_quat'],p._inverse(plan['face_quat']))
    offset = p._rotate(rotation,bow-face)
    hole = np.asarray(slot.get('keyhole_position',p._position(slot))).copy()
    hole[2] = ctx.table_z+.055
    touch = hole+[0,0,.0528467+.004]-offset
    adjusted.update(face_tcp=face,transferred_bow=bow,touch=touch,
                    inserted=touch-[0,0,.030],bow_measurement=measurement)
    return adjusted


def preview_frames(ctx, plan, program_function):
    """Clone controller targets and advance public FK without altering live observations."""
    from coap.tasks.key_collision import PublicKeyPlanningGeometry
    model = PublicRobotConvexGeometry()
    base = getattr(ctx.kinematics,'base',ctx.kinematics)
    preview = copy.copy(ctx)
    preview.obs = {'state':{name:np.asarray(value).copy() for name,value in ctx.state.items()},
                   'vision':ctx.vision}
    preview.memory = copy.deepcopy(ctx.memory)
    preview.gripper_targets = dict(ctx.gripper_targets)
    preview.joint_targets = {name:np.asarray(value).copy() for name,value in ctx.joint_targets.items()}
    preview.kinematics = MeshFilteredKinematics(base,PublicKeyPlanningGeometry(preview,model))
    frames = []
    for action in program_function(preview,plan):
        state = {}
        for arm in ('left','right'):
            joints = np.asarray(action[arm+'_arm_joint_state'])
            state[arm+'_arm_joint_state'] = joints
            state[arm+'_ee_joint_state'] = np.asarray(action[arm+'_ee_joint_state'])
            state[arm+'_ee_pose'] = base.forward(arm,joints)
        frames.append({'index':len(frames),'stage':preview.stage,'state':state})
        preview.update_obs({'state':state,'vision':ctx.vision})
        if len(frames)>300-ctx.steps:
            raise RuntimeError('Public key action limit exceeded during preflight')
    return preview,frames


def run(ctx, key, slot):
    """Preflight the complete observed layout before committing to a handoff."""
    from coap.tasks.key_collision import PublicKeyPlanningGeometry, audit_key_frames
    if not ctx.numeric_ik:
        raise RuntimeError('The key handoff requires public numerical kinematics')
    model = PublicRobotConvexGeometry()
    base = getattr(ctx.kinematics,'base',ctx.kinematics)
    rejected = []
    selected = None
    for transfer_y in (-.30,-.33,-.28,-.35):
        plan = make_plan(ctx,key,slot,transfer_y)
        try:
            preview, frames = preview_frames(ctx,plan,program)
            audit = audit_key_frames(preview,frames,slot)
            if not audit['accepted']:
                rejected.append({'transfer_y':transfer_y,'audit':audit})
                continue
            selected = plan
            ctx.memory['key_handoff_plan'] = dict(plan,planned_actions=len(frames),
                                                  public_collision_audit=audit,
                                                  rejected_candidates=rejected)
            break
        except RuntimeError as exc:
            rejected.append({'transfer_y':transfer_y,'reason':str(exc)})
    if selected is None:
        ctx.memory['key_handoff_plan'] = {'accepted':False,'rejected_candidates':rejected}
        raise RuntimeError('No complete collision-checked key blade-to-bow handoff found')
    ctx.kinematics = MeshFilteredKinematics(base,PublicKeyPlanningGeometry(ctx,model))
    yield from source_program(ctx,selected)
    ctx.stage = 'key_bow_measure'
    yield from ctx.hold(3)
    corrected = observed_plan(ctx,selected,slot)
    preview, frames = preview_frames(ctx,corrected,handoff_program)
    audit = audit_key_frames(preview,frames,slot)
    if not audit['accepted']:
        ctx.memory['key_handoff_plan']['observed_handoff_audit'] = audit
        raise RuntimeError('The RGB-corrected key handoff failed complete public collision checks')
    ctx.memory['key_handoff_plan'].update(corrected,observed_handoff_audit=audit,
                                          remaining_actions=len(frames))
    corrected['measured_slot'] = {name:value for name,value in slot.items()
                                if name not in ('crop','contour')}
    yield from handoff_program(ctx,corrected,observe_slot=True)
