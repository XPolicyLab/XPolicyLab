"""Retract each emptied-bottle arm before the next instructed pour."""
import numpy as np

from coap.tasks.pour_support import GRASP_HEIGHT, _pivot_pose, _quat, ground


def _pair(ctx, source, target, index):
    arm = ctx.arm_for(source['position'])
    # A forward-facing pinch keeps the pour sweep parallel to the bowl row.
    rotation = np.column_stack(([0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]))
    quat = _quat(rotation)
    contact = np.asarray(source['position'], float).copy()
    contact[2] = ctx.table_z + GRASP_HEIGHT
    contact += rotation[:, 0] * .015
    approach = contact - rotation[:, 0] * .10 + [0., 0., .07]
    lifted = contact + [0., 0., .25]
    pivot = lambda angle: _pivot_pose(ctx, target['position'], rotation, rotation,
                                    target['radius'], angle)
    above, above_quat = pivot(0.)
    transfer = above.copy()
    transfer[2] = lifted[2]

    ctx.stage = 'pour_%d_approach' % index
    yield from ctx.move(arm, approach, quat, 1., duration=18)
    yield from ctx.move(arm, contact, quat, 1., duration=12)
    ctx.stage = 'pour_%d_close' % index
    yield from ctx.grip(arm, 0., duration=12)
    ctx.stage = 'pour_%d_lift' % index
    yield from ctx.move(arm, lifted, quat, 0., duration=14)
    ctx.stage = 'pour_%d_transfer' % index
    yield from ctx.move(arm, transfer, quat, 0., duration=16)
    ctx.stage = 'pour_%d_above' % index
    yield from ctx.move(arm, above, above_quat, 0., duration=12)

    angles = np.linspace(0., 2.75, 9)
    for step, angle in enumerate(angles[1:]):
        ctx.stage = 'pour_%d_tilt_%d' % (index, step)
        position, orientation = pivot(float(angle))
        yield from ctx.move(arm, position, orientation, 0., duration=4)
    ctx.stage = 'pour_%d_drain' % index
    yield from ctx.hold(32)
    for step, angle in enumerate(reversed(angles[:-1])):
        ctx.stage = 'pour_%d_upright_%d' % (index, step)
        position, orientation = pivot(float(angle))
        yield from ctx.move(arm, position, orientation, 0., duration=4)

    ctx.stage = 'pour_%d_raise_return' % index
    yield from ctx.move(arm, transfer, quat, 0., duration=12)
    ctx.stage = 'pour_%d_return' % index
    yield from ctx.move(arm, lifted, quat, 0., duration=16)
    yield from ctx.move(arm, contact, quat, 0., duration=18)
    yield from ctx.grip(arm, 1., duration=8)
    yield from ctx.move(arm, approach, quat, 1., duration=10)


def run(ctx):
    from coap.context import Context
    ctx.numeric_ik = False
    ctx.__class__ = Context
    pairs = ground(ctx)
    ctx.memory['pour_strategy'] = dict(name='sequential_empty_arm_home', candidate='20261022-r01-a',
        reference='v265 pour C geometry; RGB/proprioception/instruction only',
        planned_pairs=len(pairs), tilt_axis_world=[0., 1., 0.],
        transfer_lift_m=.25, angle_rad=2.75, dwell_frames=32, empty_arm_home_frames=12,
        safety_checks_enabled=False)
    for index, (source, target) in enumerate(pairs):
        yield from _pair(ctx, source, target, index)
        if index + 1 < len(pairs):
            # Keep the empty hand out of the next bottle's working area.
            ctx.stage = 'pour_%d_empty_arm_home' % index
            yield from ctx.home(arm=ctx.arm_for(source['position']), duration=12)
    ctx.stage = 'done'
    yield from ctx.home(duration=30)
    yield from ctx.hold(12)
