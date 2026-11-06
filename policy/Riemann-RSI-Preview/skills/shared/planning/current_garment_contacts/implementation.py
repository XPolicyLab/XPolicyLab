import itertools

import numpy as np

from skills.shared.motion.garment_joint_sampling.implementation import (GarmentOwnModel, downward_wrist,
                                                                     sampled_wrist_path)
from skills.shared.geometry.measured_cloth_arc.implementation import fold_arc_wrist
from skills.shared.motion.urdf_pose_kinematics.implementation import matrix_pose, pose_matrix
from skills.shared.perception.current_garment_landmarks.implementation import (interior_contact,
                                                                                    supported_contact,
                                                                                    wrist_contact)
from skills.shared.verification.observed_garment_state.implementation import route as validate_route


# The packaged contact_frame() swept these five wrist tilts (controller.py:547).
# A previous revision carried the nine closing-axis rotations here instead,
# which is a different search dimension entirely; that one now lives in
# ROTATIONS below, where plan_body_contacts() actually used it.
TILTS = (0., 20., -20., 40., -40.)
CLEARANCE = .105
# The packaged plan_body_contacts() also searched the closing axis itself: it
# rotated that axis through these angles and re-planned both arms for each one,
# accepting the first pair that held the inter-wrist clearance and the rigid
# body path.  Only planning the straight cross-body axis dropped that search.
ROTATIONS = (0., 10., -10., 20., -20., 45., -45., 90., -90.)


def rotated_axis(axis, degrees):
    """The packaged closing axis after a rotation about the table normal."""
    angle = np.radians(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([axis[0] * cosine - axis[1] * sine,
                       axis[0] * sine + axis[1] * cosine])


def contact_choices(api, model, side, contact, closing, destination, body):
    """Every contact frame the packaged contact_frame() offered, cheapest first.

    The packaged planner asked contact_frame(..., return_choices=True) for the
    whole list and then searched the left x right product for a pair that also
    kept the inter-arm clearance along the fold arc.  Keeping only the cheapest
    frame per arm dropped that search: the cheapest left frame and the cheapest
    right frame are each reachable on their own and can still be an unusable
    pair, which is what r79 and r80 ended on at the torso stage.
    """
    roots = model.roots(api)
    seed = api.own_joint_states()[side]
    choices = []
    for tilt in TILTS:
        for sign in (-1., 1.):
            wrist = downward_wrist(contact, closing * sign, tilt)
            high = wrist.copy()
            high[2, 3] += .085
            arc = []
            try:
                high_joints = model.solve(high, seed, roots[side])
                low_joints = model.solve(wrist, high_joints, roots[side])
                if body:
                    # Match the packaged body contact preflight: validate the
                    # complete 0->pi fold arc before accepting this grasp, not
                    # only a direct endpoint solve. This keeps the public route
                    # equivalent to the old three-segment cuRobo path, and the
                    # sampled joints are reused for the paired clearance test.
                    displacement = np.asarray(destination)[:2] - contact[:2]
                    previous = low_joints
                    maximum_delta = 0.
                    for start_angle, end_angle in ((0., np.pi / 2.),
                                                   (np.pi / 2., 3. * np.pi / 4.),
                                                   (3. * np.pi / 4., np.pi)):
                        for index in range(16):
                            fraction = (index + 1) / 16.
                            smooth = fraction * fraction * (3. - 2. * fraction)
                            target = fold_arc_wrist(wrist, displacement,
                                                    start_angle + smooth * (end_angle - start_angle))
                            solved = model.solve(target, previous, roots[side])
                            maximum_delta = max(maximum_delta,
                                                float(np.max(np.abs(solved - previous))))
                            arc.append(solved)
                            previous = solved
                    placed = fold_arc_wrist(wrist, displacement, np.pi)
                    placed[2, 3] -= .003
                    model.solve(placed, previous, roots[side])
                    if maximum_delta > .22:
                        raise ValueError('Original paired body arc has a discontinuous joint branch')
                else:
                    lift = wrist.copy()
                    lift[2, 3] += .075
                    lifted_joints = model.solve(lift, low_joints, roots[side])
                    carry = lift.copy()
                    carry[:2, 3] += destination[:2] - contact[:2]
                    carried_joints = model.solve(carry, lifted_joints, roots[side])
                    # Admit only routes with the old withdrawal branch as well
                    # as the lift and carry endpoints.
                    retreat = carry.copy()
                    retreat[2, 3] += .06
                    model.solve(retreat, carried_joints, roots[side])
            except ValueError:
                continue
            cost = np.linalg.norm(high_joints - seed) + .7 * np.linalg.norm(low_joints - high_joints) + abs(tilt) * .025
            if abs(high_joints[4]) > 2.4:
                continue
            choices.append({'cost': float(cost), 'tilt': float(tilt), 'sign': float(sign),
                            'wrist': wrist, 'approach_joints': high_joints,
                            'contact_joints': low_joints,
                            'arc': np.asarray(arc, dtype=float) if arc else None})
    choices.sort(key=lambda item: item['cost'])
    return choices


def contact_choice(api, model, side, contact, closing, destination, body):
    choices = contact_choices(api, model, side, contact, closing, destination, body)
    if not choices:
        return None
    best = choices[0]
    return best['cost'], best['wrist'], best['approach_joints'], best['contact_joints']


def separated_arc(model, roots, left, right):
    """Both fold arcs keep the packaged wrist clearance at every sampled point.

    The packaged plan_body_contacts() rejected a pair through
    preflight_body_rigid_path(), which sampled the same 0->pi arc on both arms.
    Every sampled joint vector here was already solved by contact_choices(), so
    this is forward kinematics only: no extra inverse solve per pair.
    """
    positions = {}
    for side, choice in (('left', left), ('right', right)):
        arc = choice['arc']
        if not isinstance(arc, np.ndarray) or arc.ndim != 2 or len(arc) < 16 or not np.isfinite(arc).all():
            return False
        positions[side] = np.asarray([(roots[side] @ model.eef(row))[:3, 3] for row in arc])
    return bool(float(np.min(np.linalg.norm(positions['left'] - positions['right'], axis=1))) >= CLEARANCE)


RETRY_ROTATIONS = (None, 10., -10., 30., -30., 60., -60.)


def retry_contacts(api, model, route, points):
    """Re-plan the paired hem frames on the refined current-RGB points.

    Ported from the packaged plan_retry_contacts(..., phase='hem_open_refine'):
    the first strategy keeps each wrist frame and translates it by the measured
    correction; every later one sweeps the wrist's own closing axis and
    re-plans both arms, keeping the packaged first four frames per arm.
    """
    wrists = {side: pose_matrix(route['contact_poses'][side]) for side in route['arms']}
    failures = []
    for degrees in RETRY_ROTATIONS:
        choices = {}
        try:
            for side in route['arms']:
                if degrees is None:
                    goal = wrists[side].copy()
                    goal[:3, 3] += points[side] - np.asarray(route['contacts'][side], dtype=float)
                    choices[side] = [goal]
                else:
                    closing = rotated_axis(wrists[side][:3, 1][:2], degrees)
                    options = contact_choices(api, model, side, points[side], closing,
                                              np.asarray(route['destinations'][side], dtype=float), True)
                    choices[side] = [option['wrist'] for option in options[:4]]
                if not choices[side]:
                    raise ValueError('no refined contact frame for %s at %s degrees' % (side, degrees))
            for pair in itertools.product(*(choices[side] for side in route['arms'])):
                goals = dict(zip(route['arms'], pair))
                try:
                    # The packaged plan_retry_actions() preflighted the whole
                    # lower chunk before committing to it; the same sixteen
                    # sample chain is that preflight here, and its result is the
                    # chunk the caller executes.
                    sampled_wrist_path(api, model, goals, {side: 1. for side in goals}, contact=True)
                    return goals
                except ValueError as error:
                    failures.append(str(error))
        except ValueError as error:
            failures.append(str(error))
    raise ValueError('no continuous separated RGB regrasp: %s' % failures[:4])


def refine(api, route, anchor):
    """Refine the paired hem contacts at the measured open approach.

    The packaged fold_body() ran
        wrists,contacts,opening=self.refine_open_hem_contacts(wrists,contacts)
    between hem_approach and hem_lower, which is exactly where this is called.
    It reuses the pending lower chunk and adds no native action: a pad that
    landed on a curled hem edge is corrected before the first squeeze instead
    of spending a whole part retry on the failed grasp.  Both packaged
    strategies are tried in the packaged order and the planned route stands
    when neither produces a continuous separated regrasp.
    """
    lateral = np.asarray(anchor['lateral_axis'], dtype=float)
    # The packaged interior axis is [-sin(angle), cos(angle)] for its
    # closing = [cos(angle), sin(angle)], which is the retained lateral axis.
    interior = np.asarray([-lateral[1], lateral[0]])
    model = GarmentOwnModel()
    for unused_strategy, axis in (('longitudinal', interior), ('radial', None)):
        try:
            points = {}
            for side in route['arms']:
                points[side], unused_evidence = wrist_contact(
                    api, anchor, side, np.asarray(route['contacts'][side], dtype=float),
                    interior_axis=axis)
            goals = retry_contacts(api, model, route, points)
            controls = sampled_wrist_path(api, model, goals,
                                          {side: 1. for side in goals}, contact=True)
        except (ValueError, RuntimeError):
            continue
        refined = dict(route, contacts={side: points[side].tolist() for side in route['arms']},
                       contact_poses={side: matrix_pose(goals[side]).tolist() for side in route['arms']})
        return validate_route(refined), controls
    return None


def run(api, anchor, part):
    if type(part) is not int or not 0 <= part <= 2:
        raise ValueError('A garment route names a bounded cuff or hem stage')
    body = part == 2
    names = ('left_hem', 'right_hem') if body else (('left_cuff',) if part == 0 else ('right_cuff',))
    contacts = {name: supported_contact(api, anchor, name) for name in names}
    if any(value is None for value in contacts.values()):
        # The packaged rule raised rather than quietly declining to plan.
        raise ValueError('no current outline support for ' + ','.join(
            name for name, value in contacts.items() if value is None))
    axis = np.asarray(anchor['lateral_axis'])
    longitudinal = np.asarray(anchor['longitudinal_axis'])
    center = np.asarray(anchor['center'])
    model = GarmentOwnModel()
    destinations, selected, frames = {}, {}, {}
    ordered = list(names) if body else sorted(contacts, key=lambda name: contacts[name][0])
    for index, name in enumerate(ordered):
        contact = contacts[name].copy()
        contact[2] = anchor['height_m'] + .003
        if body:
            # Preserve the packaged Rule semantics exactly: fold_body() grasps
            # landmarks[side + "_hem"] and lands on
            # landmarks[side + "_shoulder"], not on the chest landmark.  The
            # chest landmark sits half way up the torso, so landing there
            # shortened the fold to roughly a third of the garment and the
            # official hem-versus-shoulder check could never pass.
            side = 'left' if name == 'left_hem' else 'right'
            shoulder = name.replace('_hem', '_shoulder')
            # The packaged fold_body() takes its closing frame from the
            # rectangle angle itself, closing = [cos(angle), sin(angle)],
            # which is the cross-body axis, not the hem-to-collar one:
            #   destination[:2] += (-.04 if side == 'left' else .04) * closing
            # Running the 40 mm landing margin along the longitudinal axis
            # instead pushed the destination toward the collar and left the
            # paired arc without a shared rigid branch.
            offset = (-.04 if side == 'left' else .04) * axis
            destination = np.r_[np.asarray(anchor[shoulder]) + offset, anchor['height_m'] + .012]
            closing = axis
        else:
            # Match the packaged sleeve semantics: cuff identity selects its
            # corresponding own arm.  The observed contact point may cross the
            # image center after a fold, but it must not swap left/right arms.
            side = 'left' if name == 'left_cuff' else 'right'
            inward = center - contact[:2]
            inward /= max(np.linalg.norm(inward), 1e-8)
            contact[:2] += .022 * inward
            # The packaged rule keeps the pinch inside the current RGB edge;
            # the retained landmark is the outer silhouette vertex, so slide
            # the contact along the sleeve to the deepest current-mask point.
            contact = interior_contact(api, anchor, contact, inward)
            chest = 'right_chest' if name == 'left_cuff' else 'left_chest'
            destination = np.r_[.8 * np.asarray(anchor[chest]) + .2 * center, anchor['height_m'] + .015]
            if name == 'right_cuff':
                destination[:2] -= .035 * axis
            closing = np.asarray([-axis[1], axis[0]])
        options = contact_choices(api, model, side, contact, closing, destination, body)
        # The packaged plan_body_contacts() re-planned an arm whose first
        # closing axis offered nothing; the sleeve stage kept the single
        # cross-body axis and so refused the contact outright.  The closing axis
        # is the one candidate dimension the sleeve stage never searched, and it
        # is what solves the far-table layouts: the cuff is graspable, but only
        # from an approach that turns the wrist off the cross-body axis.  Each
        # rotated axis still goes through the same five-segment inverse solve,
        # the same joint limit and the same lift/carry/retreat branch test, so
        # this widens the candidates without skipping a constraint.
        if not body and not options:
            for rotation_deg in ROTATIONS:
                if rotation_deg == 0.:
                    continue
                options = contact_choices(api, model, side, contact,
                                          rotated_axis(closing, rotation_deg),
                                          destination, body)
                if options:
                    break
        # The packaged plan_body_contacts() re-planned an arm whose first
        # closing axis offered nothing, so an empty list here is not fatal yet:
        # the rotation search below retries the other eight axes.
        if not body and (not options or side in selected):
            raise ValueError('no reachable contact frame for %s at part %d: contact=%s destination=%s closing=%s seed=%s'
                             % (side, part, np.round(contact, 3).tolist(),
                                np.round(destination, 3).tolist(),
                                np.round(closing, 3).tolist(),
                                np.round(api.own_joint_states()[side], 3).tolist()))
        if body:
            frames[side] = {'contact': contact, 'destination': destination, 'options': options}
            continue
        unused_cost, wrist, joints, low_joints = (options[0]['cost'], options[0]['wrist'],
                                                  options[0]['approach_joints'],
                                                  options[0]['contact_joints'])
        selected[side] = {'contact': contact.tolist(), 'contact_pose': matrix_pose(wrist).tolist(),
                          'approach_joints': joints.tolist(), 'contact_joints': low_joints.tolist()}
        destinations[side] = destination.tolist()
    if body:
        # The packaged plan_body_contacts() admitted the torso fold as one
        # object: for each closing-axis rotation it took the full candidate
        # list from both arms, sorted the left x right product by cost, and
        # accepted the first pair that also held the inter-wrist clearance and
        # the rigid body path.  Planning each arm on its own and then testing
        # that single pair turned a solved problem into "No reachable separated
        # body grasp orientation"; so did searching only the first axis.
        roots = model.roots(api)
        accepted = None
        rejected_axes = []
        for rotation_deg in ROTATIONS:
            if rotation_deg == 0.:
                options = {side: frames[side]['options'] for side in ('left', 'right')}
            else:
                rotated = rotated_axis(axis, rotation_deg)
                options = {side: contact_choices(api, model, side, frames[side]['contact'],
                                                 rotated, frames[side]['destination'], True)
                           for side in ('left', 'right')}
            pairs = sorted(((left['cost'] + right['cost'], left, right)
                            for left in options['left']
                            for right in options['right']),
                           key=lambda item: item[0])
            for unused_cost, left, right in pairs:
                if np.linalg.norm(left['wrist'][:3, 3] - right['wrist'][:3, 3]) < CLEARANCE:
                    continue
                if not separated_arc(model, roots, left, right):
                    continue
                accepted = (left, right)
                break
            if accepted is not None:
                break
            rejected_axes.append({'rotation_degrees': rotation_deg,
                                  'left': len(options['left']),
                                  'right': len(options['right'])})
        if accepted is None:
            raise ValueError('no reachable separated body grasp orientation: %s' % rejected_axes)
        for side, choice in (('left', accepted[0]), ('right', accepted[1])):
            selected[side] = {'contact': frames[side]['contact'].tolist(),
                              'contact_pose': matrix_pose(choice['wrist']).tolist(),
                              'approach_joints': choice['approach_joints'].tolist(),
                              'contact_joints': choice['contact_joints'].tolist()}
            destinations[side] = frames[side]['destination'].tolist()
    arms = sorted(selected)
    result = {'arms': arms, 'contacts': {side: selected[side]['contact'] for side in arms},
              'destinations': destinations, 'contact_poses': {side: selected[side]['contact_pose'] for side in arms},
              'approach_joints': {side: selected[side]['approach_joints'] for side in arms},
              'carry_chunks': 2, 'body_chunks': 2}
    return validate_route(result)
