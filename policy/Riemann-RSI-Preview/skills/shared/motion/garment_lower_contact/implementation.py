from skills.shared.motion.garment_joint_sampling.implementation import (GarmentOwnModel, sampled_joint_path, sampled_wrist_path)
from skills.shared.motion.garment_command_execution.implementation import GarmentProgram, allowed, unavailable
from skills.shared.motion.urdf_pose_kinematics.implementation import pose_matrix
from skills.shared.perception.current_garment_landmarks.implementation import read
from skills.shared.planning.current_garment_contacts.implementation import refine, run as contact_route
from skills.shared.verification.observed_garment_mount_state.implementation import recorded


def run(api, request):
    current = recorded(api.command_memory(), api.instruction())
    if not allowed(current, 'contact'):
        return unavailable(api, request, current)
    # Approach and lower are one non-mutating contact transition in the old
    # rule. Re-reading the head image here lets the wrists occlude the cloth
    # and changes the cuff assignment even though the garment did not move.
    # Reuse the qualified scene-view anchor/route; recovery performs a fresh
    # view only after an actual lower failure.
    if current.get('anchor') is not None and current.get('route') is not None:
        # The old rule carries the scene-view contact through approach and
        # lower. Re-segmenting here lets occluding wrists change the cuff
        # assignment even though the cloth has not moved. Reuse the qualified
        # public anchor/route for every part; recovery performs a fresh view.
        measured_anchor = current['anchor']
        planned = current['route']
    else:
        measured = read(api, current['anchor'])
        measured_anchor = measured['anchor'] if measured else None
        try:
            planned = contact_route(api, measured_anchor, current['part']) if measured_anchor else None
        except ValueError:
            # Same packaged refusal as the scene view: an unplannable route is
            # a missing plan, which the recover phase already handles, not an
            # exception that should abandon the shard.
            planned = None
    if planned is None or planned['arms'] != current['route']['arms']:
        current['phase'] = 'recover'
        return unavailable(api, request, current)
    current.update(anchor=measured_anchor, route=planned)
    controls = None
    if current['part'] == 2:
        # The packaged fold_body() re-measured both hem contacts from the
        # current wrist RGB at the open approach, before the first squeeze.
        # The head silhouette can put a nominal pad on a curled hem edge, and
        # the refinement reuses this pending lower chunk instead of spending a
        # whole part retry on the failed grasp.
        refined = refine(api, planned, measured_anchor)
        if refined is not None:
            planned, controls = refined
            current['route'] = planned
    goals = {side: pose_matrix(value) for side, value in planned['contact_poses'].items()}
    try:
        if controls is None:
            controls = sampled_wrist_path(api, GarmentOwnModel(), goals,
                                          {side: 1. for side in planned['arms']}, contact=True)
    except ValueError:
        # Preserve the same public contact goals when a Cartesian sample
        # crosses an IK branch. The fallback only interpolates to those
        # solved own-joint endpoints and remains a bounded 16-command chunk.
        model = GarmentOwnModel()
        roots = model.roots(api)
        seeds = api.own_joint_states()
        fallback_goals = {}
        try:
            for side, goal in goals.items():
                fallback_goals[side] = model.solve(
                    goal, seeds[side].copy(), roots[side], position_tolerance=.006)
            controls = sampled_joint_path(api, fallback_goals)
        except ValueError:
            current['phase'] = 'recover'
            return unavailable(api, request, current)
    return GarmentProgram(api, request, current, controls, following='close')
