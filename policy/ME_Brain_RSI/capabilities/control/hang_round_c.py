"""Compare fixed rim-pinch mechanics using current RGB and public cup geometry."""
import math

import numpy as np

from coap.tasks.hang_candidate_motion import HangCurveContext
from coap.tasks.hang_candidate_v import (DURATIONS, holder_geometry, quaternion,
    source_geometry as standard_geometry, tool_frame, ik_cost, peg_geometry,
    placement_rotation)
from coap.tasks.hang_candidate_d import (source_geometry as random_geometry,
    refresh_rim as random_refresh)
from coap.tasks.hang_candidate_an_aperture import refresh_aperture
from coap.tasks.hang_candidate_ag_prefix import approach


MODE = 'tangent'
STANDARD_DEPTH = 0.02
RANDOM_DEPTH = 0.035
CANDIDATE = 'r01-hang-c'


def plan_transfer(ctx, source, holder, available, arm=None):
    """Rank every available construction continuously and execute the minimum."""
    arm = arm or ('left' if source['grasp'][0] < 0 else 'right')
    reference = np.asarray(ctx.state[arm+'_arm_joint_state'], float)
    if MODE == 'vertical':
        direction = np.array([0., 0., -1.])
    elif MODE == 'tangent':
        radial = np.asarray(source['radial'], float)
        tangent = np.array([-radial[1], radial[0], 0.])
        tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
        if tangent[1] < 0.:
            tangent = -tangent
        direction = math.cos(math.radians(65.))*tangent
        direction[2] = -math.sin(math.radians(65.))
    else:
        direction = np.array([0., math.cos(math.radians(65.)), -math.sin(math.radians(65.))])
    plans = []
    for opening_sign in (1., -1.):
        pick = tool_frame(direction, opening_sign*source['radial'])
        pickup_cost, pickup_joints = ik_cost(ctx, arm, source['grasp'], pick, reference)
        hole_tool = pick.T @ (source['hole']-source['grasp'])
        for peg in available:
            center, axis = peg_geometry(holder, peg)
            for sign in (1., -1.):
                final = placement_rotation(pick, source.get('hole_radial', source['radial']), axis, sign)
                hole = center+axis*.020-np.array([0., 0., .005])
                target = hole-final @ hole_tool
                final_cost, _ = ik_cost(ctx, arm, target, final, pickup_joints)
                side = -1. if arm == 'left' else 1.
                side_preference = .15*max(0., -side*axis[0])
                front_preference = .06*max(0., -final[1, 0])
                plans.append(dict(cost=pickup_cost+final_cost+side_preference+front_preference,
                    arm=arm, peg=peg, pick=pick, final=final, center=center, axis=axis,
                    hole_tool=hole_tool, target=target))
    return min(plans, key=lambda plan: plan['cost'])


def execute(ctx, randomized):
    """Run the entire fixed action program without runtime rejection gates."""
    from coap.tasks.open_motion import enable
    enable(ctx)
    if ctx.numeric_ik:
        ctx.__class__ = HangCurveContext
    ctx.hang_candidate_name = CANDIDATE
    ctx.stage = 'hang_r01_measure'
    holder = holder_geometry(ctx)
    if holder is None:
        yield from ctx.hold(4)
        return
    if randomized:
        from coap.tasks.hang_candidate_d_sources import associate_sources
        cups = associate_sources(ctx, holder)
        geometry, refresh = random_geometry, random_refresh
        depth, available = RANDOM_DEPTH, [4, 5, 2]
    else:
        from coap.tasks.hang_candidate_s_sources import associate_standard_sources
        cups = associate_standard_sources(ctx, holder)
        geometry, refresh = standard_geometry, refresh_aperture
        depth, available = STANDARD_DEPTH, [0, 1, 2, 3, 4, 5]
    sources = []
    for cup in cups[:3]:
        source = geometry(ctx, cup)
        source = dict(source, grasp=np.asarray(source['grasp'], float).copy())
        source['grasp'][2] = ctx.table_z+float(source['height'])-depth
        sources.append(source)
    ctx.memory['hang_r01'] = dict(candidate=CANDIDATE, randomized=bool(randomized),
        pinch_depth_m=depth, approach_mode=MODE, source_geometry=sources,
        finite_hover_branch_selection=True, safety_checks_enabled=False,
        fixed_frame_budget=3*(sum(DURATIONS.values())+3)+41)
    yield from ctx.hold(4)
    for index, source in enumerate(sources):
        plan = plan_transfer(ctx, source, holder, available)
        arm = plan['arm']
        ctx.stage = 'hang_r01_approach_'+str(index)
        yield from approach(ctx, source, plan)
        source = refresh(ctx, source, arm)
        plan = plan_transfer(ctx, source, holder, available, arm=arm)
        available.remove(plan['peg'])
        ctx.memory.setdefault('hang_r01_plans', []).append(plan)
        ctx.memory.setdefault('hang_r01_refreshed_sources', []).append(source)
        arm, grasp = plan['arm'], source['grasp']
        pick, final = quaternion(plan['pick']), quaternion(plan['final'])
        stage = np.array([-.28 if arm == 'left' else .28, -.24, 1.025])
        prehang = plan['center']+plan['axis']*.090-plan['final'] @ plan['hole_tool']
        thread = plan['center']+plan['axis']*.020-plan['final'] @ plan['hole_tool']
        transfer = prehang+plan['axis']*.040
        yield from ctx.move(arm, grasp, pick, 1., DURATIONS['descend'])
        ctx.stage = 'hang_r01_close_'+str(index)
        yield from ctx.grip(arm, 0., duration=DURATIONS['close'])
        yield from ctx.move(arm, grasp+[0., 0., .120], pick, 0., DURATIONS['lift'])
        ctx.stage = 'hang_r01_carry_'+str(index)
        yield from ctx.move(arm, stage, pick, 0., DURATIONS['stage'])
        yield from ctx.move(arm, stage, final, 0., DURATIONS['rotate'])
        yield from ctx.move(arm, transfer, final, 0., DURATIONS['transfer'])
        yield from ctx.move(arm, prehang, final, 0., DURATIONS['prehang'])
        ctx.stage = 'hang_r01_insert_'+str(index)
        yield from ctx.move(arm, thread, final, 0., DURATIONS['thread'])
        yield from ctx.move(arm, plan['target'], final, 0., DURATIONS['seat'])
        ctx.stage = 'hang_r01_release_'+str(index)
        yield from ctx.grip(arm, 1., duration=DURATIONS['release'])
        retreat = plan['target']-plan['final'][:, 0]*.070
        yield from ctx.move(arm, retreat, final, 1., DURATIONS['retreat'])
        yield from ctx.home(arm, duration=DURATIONS['home'])
    ctx.stage = 'done'
    yield from ctx.home(duration=26)
    yield from ctx.hold(8)


def run_standard(ctx):
    """Use the standard task's native RGB source association."""
    yield from execute(ctx, False)


def run_random(ctx):
    """Use the random task's native RGB source association."""
    yield from execute(ctx, True)
