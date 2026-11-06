import numpy as np

from skills.shared.motion.garment_joint_sampling.implementation import GarmentOwnModel, sampled_wrist_path, sampled_joint_path
from skills.shared.motion.garment_command_execution.implementation import GarmentProgram, allowed, unavailable
from skills.shared.motion.urdf_pose_kinematics.implementation import pose_matrix
from skills.shared.verification.observed_garment_mount_state.implementation import recorded


class ClearGarment(GarmentProgram):
    def review_completion(self):
        self.current.update(part=self.current['part'] + 1, chunk=0, attempts=0, route=None, start_wrists=None)
        self.current['phase'] = 'view' if self.current['part'] < 3 else 'prehome_settle'


def _vertical_goals(api, model, roots, starts, sides):
    goals = {}
    for side in sides:
        seed = api.own_joint_states()[side].copy()
        selected = None
        for height in (.050, .035, .020, .010):
            previous = seed.copy()
            valid = True
            for index in range(16):
                fraction = (index + 1) / 16.
                smooth = fraction * fraction * (3 - 2 * fraction)
                target = starts[side].copy()
                target[2, 3] += height * smooth
                try:
                    solved = model.solve(target, previous, roots[side])
                except ValueError:
                    valid = False
                    break
                if np.max(np.abs(solved - previous)) > .30:
                    valid = False
                    break
                previous = solved
            if valid:
                selected = starts[side].copy()
                selected[2, 3] += height
                break
        if selected is None:
            return None
        goals[side] = selected
    return goals


def run(api, request):
    current = recorded(api.command_memory(), api.instruction())
    if not allowed(current, 'clear'):
        return unavailable(api, request, current)
    model = GarmentOwnModel()
    roots = model.roots(api)
    sides = tuple(current['route']['arms'])
    starts = {side: pose_matrix(api.targets()[side + '_ee_pose']) for side in sides}

    candidates = []
    if current['part'] in (0, 1):
        # The packaged rule parked the finished arm at its own root's outer
        # side and kept the released wrist rotation:
        #   retreat = place.copy()
        #   retreat[:3,3] = roots[side][:3,3] + offset
        # The offset belongs to the root, not to the released wrist.  Adding it
        # to the wrist put the park pose 0.72 m from the shoulder, past the
        # 0.70 m envelope, so the outward candidate was rejected and the arm
        # fell back to parking above the shirt.  r75 measured the cost: the
        # left arm sat at x=-0.01 through the right arm's cross-body carry, the
        # forearms collided around frame 246, and the right wrist was shoved
        # from y=-0.17 to y=-0.05, 19 cm short of its destination with the
        # cloth no longer under the pad.  Keep the acknowledged rotation and
        # take the position from the root, as the packaged rule did.
        outward = {}
        for side in sides:
            goal = starts[side].copy()
            goal[:3, 3] = roots[side][:3, 3] + np.asarray([
                -.06 if side == 'left' else .06, .15, .275])
            outward[side] = goal
        candidates.append(outward)
        # A lower park on the same outboard ray, for the case where the high
        # one sits past the envelope.  It must stay root-relative too: the
        # packaged rule's only retreat was outboard, and a fallback that parks
        # over the shirt re-creates the collision this stage exists to avoid.
        camera_clear = {}
        for side in sides:
            goal = starts[side].copy()
            goal[:3, 3] = roots[side][:3, 3] + np.asarray([
                -.06 if side == 'left' else .06, .15, .18])
            camera_clear[side] = goal
        candidates.append(camera_clear)
    vertical = _vertical_goals(api, model, roots, starts, sides)
    if vertical is not None:
        candidates.append(vertical)

    for index, goals in enumerate(candidates):
        try:
            controls = sampled_wrist_path(api, model, goals, {side: 1. for side in sides})
            return ClearGarment(api, request, current, controls)
        except ValueError:
            # Preserve the outward reset endpoint even when its Cartesian
            # interpolation crosses an IK branch.  Solve that same public
            # endpoint once, then use the native sixteen-command joint clock.
            if current['part'] in (0, 1) and index == 0:
                try:
                    joints = {side: model.solve(goals[side], api.own_joint_states()[side].copy(), roots[side])
                              for side in sides}
                    controls = sampled_joint_path(api, joints)
                    return ClearGarment(api, request, current, controls)
                except ValueError:
                    pass
            continue
    return unavailable(api, request, current)
