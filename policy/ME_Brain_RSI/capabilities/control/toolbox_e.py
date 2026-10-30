"""Use balanced public-mesh grasps and a continuous equivalent wrist frame."""
import math

import numpy as np

from coap.motion import down_quat, tcp_to_link
from coap.tasks.toolbox_support import measure, _lift
from coap.tasks.toolbox_d import ToolboxDirectContext, tape_frame

BALANCED_SHIFT = {'hammer': -.015, 'wrench': -.015, 'pliers': .000, 'tape_measure': 0.}


def _top_quaternion(ctx, arm, grasp, target, yaw):
    """Rank equivalent pinch frames by total public-IK joint travel."""
    waypoints = [_lift(grasp, .085), grasp, _lift(grasp, .10),
                 _lift(target, .020), target, _lift(target, .055)]
    candidates = []
    for half_turn in (0., math.pi):
        quat = down_quat(math.pi/2+yaw+half_turn)
        previous = np.asarray(ctx.state[arm+'_arm_joint_state'], float).copy()
        travel = []
        residual = []
        for position in waypoints:
            result = ctx.kinematics.solve(arm, tcp_to_link(position, quat),
                current_joints=previous, position_tol=.002, rotation_tol=.02,
                max_starts=16, max_nfev=180)
            joints = np.asarray(result['joint_value'], float)
            travel.append(float(np.max(np.abs(joints-previous))))
            residual.append(float(result['position_error']*10+result['rotation_error']))
            previous = joints
        cost = sum(travel)+.5*max(travel)+sum(residual)
        candidates.append((cost, quat, dict(half_turn_rad=half_turn, travel_rad=travel,
                                             residual_cost=residual, cost=cost)))
    _, quat, choice = min(candidates, key=lambda item: item[0])
    ctx.memory.setdefault('toolbox_wrist_choices', []).append(dict(arm=arm, selected=choice,
        candidates=[item[2] for item in candidates]))
    return quat


def run(ctx):
    """Balance rigid tools around the pinch before carrying them to the cavity."""
    from coap.tasks.open_motion import enable
    ctx.stage = 'toolbox_measure'
    moves = measure(ctx)
    enable(ctx)
    ctx.__class__ = ToolboxDirectContext
    ctx.memory['toolbox_strategy'] = dict(candidate='E', planned_tools=len(moves),
        safety_checks_enabled=False, execution='fixed_duration_public_numeric_joint_chords',
        balanced_local_x_m=BALANCED_SHIFT, tape_tilt_degrees=35.)
    priority = {'tape_measure': 0, 'wrench': 1, 'pliers': 2, 'hammer': 3}
    moves.sort(key=lambda item: priority[item['label']])
    for item in moves:
        name = item['label']
        source = np.asarray(item['source'])
        arm = 'left' if source[0] < 0 else 'right'
        if name == 'tape_measure':
            grasp, target, quat = tape_frame(ctx, source, item['destination'], arm)
        else:
            grasp = source+item['source_rotation'] @ [BALANCED_SHIFT[name], 0., 0.]
            grasp[2] = ctx.table_z+.015
            target = np.asarray(item['destination'])+(grasp-source)
            target[2] += .040
            quat = _top_quaternion(ctx, arm, grasp, target, float(item['source_yaw']))
        ctx.stage = 'toolbox_approach_'+name
        yield from ctx.move(arm, _lift(grasp, .085), quat, 1., 18)
        yield from ctx.move(arm, grasp, quat, 1., 18)
        ctx.stage = 'toolbox_close_'+name
        yield from ctx.grip(arm, 0., duration=26)
        yield from ctx.hold(3)
        yield from ctx.move(arm, _lift(grasp, .100), quat, 0., 26)
        ctx.stage = 'toolbox_carry_'+name
        yield from ctx.move(arm, _lift(target, .020), quat, 0., 30)
        yield from ctx.move(arm, target, quat, 0., 20)
        ctx.stage = 'toolbox_release_'+name
        yield from ctx.grip(arm, 1., duration=12)
        yield from ctx.move(arm, _lift(target, .055), quat, 1., 18)
        yield from ctx.home(arm, duration=20)
    ctx.stage = 'done'
    yield from ctx.home(duration=28)
    yield from ctx.hold(10)
