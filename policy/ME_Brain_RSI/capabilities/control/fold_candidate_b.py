"""Acquire thin cloth at the table and create the second hem edge by lifting."""
import math

import cv2
import numpy as np

from coap.motion import ARMS, down_quat, quat_matrix, quat_multiply
from coap.tasks.fold_candidate_a import (enable, lift, snap_edge, body_frame,
    paired_targets, paired_arc, contour_world, _grip_both)
from coap.tasks.garment import _garment_points, _carry_quat


CONTACT_HEIGHT = .0128


def landmarks(ctx, item):
    """Resolve neck/hem ambiguity with the public initial rotation range."""
    points = _garment_points(ctx, item) if item else None
    if points is None:
        return None
    upper = .5*(points['left_shoulder']+points['right_shoulder'])
    lower = .5*(points['left_hem']+points['right_hem'])
    if upper[1] < lower[1]:
        # The public task rotates an initially neck-forward top by at most 25 degrees.
        # Smooth pale necklines can otherwise look straighter than the curved hem.
        for side in ('left', 'right'):
            points[side+'_shoulder'], points[side+'_hem'] = (
                points[side+'_hem'].copy(), points[side+'_shoulder'].copy())
        upper, lower = lower, upper
        direction = upper[:2]-lower[:2]
        direction /= max(np.linalg.norm(direction), 1e-9)
        for side in ('left', 'right'):
            points[side+'_chest'] = upper+.65*(points[side+'_shoulder']-upper)
            points[side+'_chest'][:2] -= .065*direction
        ctx.memory['fold_neck_hem_orientation_corrected'] = True
    return points


def measure(ctx):
    """Segment current RGB with a public task prior and episode-local tracking."""
    camera = 'cam_head'
    image = np.asarray(ctx.vision[camera]['color'], np.uint8)[..., :3]
    seed_world = np.asarray(ctx.memory.get('fold_rgb_segmentation_seed', [0., -.18, ctx.table_z+.006]), float)
    seed = np.rint(ctx.perception.world_to_pixel(seed_world, camera, ctx.state, image.shape)).astype(int)
    if 'fold_rgb_segmentation_seed' in ctx.memory:
        center = seed_world[:2]
        corners = [[center[0]+dx, center[1]+dy, ctx.table_z+.006]
                   for dx in (-.40, .40) for dy in (-.26, .28)]
    else:
        # The public task config places a long-sleeved top near the table center.
        # This prior seeds segmentation and never supplies a grasp coordinate.
        corners = [[x, y, ctx.table_z+.006] for x in (-.38, .38) for y in (-.39, .09)]
    pixels = np.asarray([ctx.perception.world_to_pixel(p, camera, ctx.state, image.shape)
                         for p in corners])
    lower = np.maximum(np.floor(pixels.min(axis=0)).astype(int), 0)
    upper = np.minimum(np.ceil(pixels.max(axis=0)).astype(int), [image.shape[1]-1, image.shape[0]-1])
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[lower[1]:upper[1], lower[0]:upper[0]] = cv2.GC_PR_BGD
    cv2.circle(mask, tuple(seed), 15, cv2.GC_FGD, -1)
    cv2.setRNGSeed(0)
    cv2.grabCut(image, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    material = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)*255
    material = cv2.morphologyEx(material, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(material, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    pixel = [moments['m10']/max(moments['m00'], 1.), moments['m01']/max(moments['m00'], 1.)]
    world = ctx.project(pixel, ctx.table_z+.006, camera)
    ctx.memory['fold_rgb_segmentation_seed'] = world.tolist()
    ctx.memory.setdefault('fold_rgb_segmentations', []).append({
        'step': int(ctx.steps), 'bbox': [x, y, width, height],
        'area': float(cv2.contourArea(contour)), 'source': 'current_RGB_seeded_GrabCut',
        'initial_seed_source': 'public_task_placement_prior', 'pixel_seed': seed.tolist()})
    return {'bbox': (x, y, width, height), 'contour': contour[:, 0, :].astype(float),
            'camera': camera, 'position': world, 'confidence': 1., 'pixel': pixel}


def entry_pose(source, inward, quat):
    """Put the outward pad slightly lower before raking a loose material edge."""
    direction = np.asarray(inward, float)[:2]
    direction /= max(np.linalg.norm(direction), 1e-9)
    closing = quat_matrix(quat)[:2, 1]
    angle = math.radians(2. if np.dot(direction, closing) >= 0 else -2.)
    tilted = quat_multiply(quat, [math.cos(angle/2.), 0., 0., math.sin(angle/2.)])
    start = np.asarray(source, float)-np.r_[.016*direction, 0.]
    gathered = np.asarray(source, float)+np.r_[.006*direction, 0.]
    return start, gathered, tilted


def sleeve(ctx, side, points):
    """Cross the cuff edge with both pads before closing at cloth height."""
    other = 'right' if side == 'left' else 'left'
    center = .5*(points['left_chest']+points['right_chest'])
    target = center+.5*(points[other+'_chest']-center)
    contour = contour_world(ctx, measure(ctx))
    inward = target-points[side+'_sleeve']
    source = snap_edge(points[side+'_sleeve'], contour, inward)
    source[2] = ctx.table_z+CONTACT_HEIGHT
    target[2] = ctx.table_z+CONTACT_HEIGHT+.006
    arm = 'left' if source[0] < 0 else 'right'
    # The jaw-closing axis crosses the free cuff edge along its inward direction.
    yaw = math.atan2(inward[1], inward[0])-math.pi/2.
    quat = down_quat(yaw)
    entry, gathered, tilted = entry_pose(source, inward, quat)
    ctx.memory.setdefault('fold_sleeve_targets', {})[side] = {
        'source': source.tolist(), 'gathered': gathered.tolist(),
        'destination': target.tolist(), 'arm': arm, 'tcp_contact_height_m': CONTACT_HEIGHT}
    ctx.stage = 'fold_b_'+side+'_sleeve_approach'
    yield from ctx.move(arm, lift(entry, .065), quat, .45, 16)
    yield from ctx.move(arm, entry, tilted, .45, 12)
    ctx.stage = 'fold_b_'+side+'_sleeve_open_rake'
    yield from ctx.move(arm, gathered, tilted, .45, 8)
    ctx.stage = 'fold_b_'+side+'_sleeve_close'
    yield from ctx.grip(arm, 0., 12)
    yield from ctx.move(arm, lift(gathered, .035), quat, 0., 8)
    for index, fraction in enumerate((1./3., 2./3., 1.)):
        point = (1.-fraction)*gathered+fraction*target
        point[2] += .060*math.sin(math.pi*fraction)
        carry = _carry_quat(ctx.kinematics, arm, target, quat, 20., fraction)
        ctx.stage = 'fold_b_'+side+'_sleeve_arc_'+str(index)
        yield from ctx.move(arm, point, carry, 0., 10)
    ctx.stage = 'fold_b_'+side+'_sleeve_release'
    yield from ctx.grip(arm, 1., 12)
    carry = _carry_quat(ctx.kinematics, arm, target, quat, 20., 1.)
    yield from ctx.move(arm, lift(target, .035), carry, 1., 8)
    ctx.stage = 'fold_b_'+side+'_sleeve_park'
    yield from ctx.move(arm, lift(source, .080), quat, 1., 10)


def run(ctx):
    """Run a 476-frame thin-edge strategy without retention or arrival gates."""
    enable(ctx)
    ctx.stage = 'fold_b_measure'
    item = measure(ctx)
    points = landmarks(ctx, item)
    if points is None:
        ctx.memory['fold_missing_rgb_landmarks'] = True
        yield from ctx.hold(1)
        return
    ctx.memory['fold_strategy'] = {'candidate': 'B', 'reference': 'WorldScape-R2SI',
        'method': 'lower_edge_rake_then_sequential_paired_hem',
        'native_frame_budget': 476, 'safety_checks_enabled': False,
        'contact_height_m': CONTACT_HEIGHT, 'entry_roll_degrees': 2.}
    ctx.memory['fold_initial_landmarks'] = {key: value.tolist() for key, value in points.items()}
    for side in ('left', 'right'):
        yield from sleeve(ctx, side, points)
    width, inward = body_frame(points)
    contour = contour_world(ctx, measure(ctx))
    sources = [snap_edge(points[side+'_hem'], contour, inward) for side in ('left', 'right')]
    sources.sort(key=lambda point: point[0])
    for source in sources:
        source[2] = ctx.table_z+CONTACT_HEIGHT
    targets = paired_targets(points, sources)
    for target in targets:
        target[2] = ctx.table_z+CONTACT_HEIGHT+.008
    yaw = math.atan2(width[1], width[0])
    quats = [down_quat(yaw), down_quat(yaw+math.pi)]
    entries = [entry_pose(source, inward, quat) for source, quat in zip(sources, quats)]
    gathered = [entry[1].copy() for entry in entries]
    ctx.stage = 'fold_b_paired_hem_approach'
    yield from ctx.move_both(*[lift(entry[0], .065) for entry in entries], *quats, .45, .45, 18)
    for index, arm in enumerate(ARMS):
        if index == 1:
            # A lifted first hem corner makes a loose vertical edge for the other hand.
            current = contour_world(ctx, measure(ctx))
            sources[index] = snap_edge(sources[index], current, inward, search_radius=.055)
            sources[index][2] = ctx.table_z+CONTACT_HEIGHT
            entries[index] = entry_pose(sources[index], inward, quats[index])
            gathered[index] = entries[index][1].copy()
        entry, point, tilted = entries[index]
        ctx.stage = 'fold_b_'+arm+'_hem_entry'
        yield from ctx.move(arm, entry, tilted, .45, 14)
        ctx.stage = 'fold_b_'+arm+'_hem_open_rake'
        yield from ctx.move(arm, point, tilted, .45, 8)
        ctx.stage = 'fold_b_'+arm+'_hem_close'
        yield from ctx.grip(arm, 0., 12)
        if index == 0:
            ctx.stage = 'fold_b_first_hem_make_edge'
            yield from ctx.move(arm, lift(point, .035), quats[index], 0., 10)
    ctx.stage = 'fold_b_paired_hem_lift'
    yield from ctx.move_both(*[lift(point, .035) for point in gathered], *quats, 0., 0., 10)
    targets = paired_targets(points, gathered)
    for target in targets:
        target[2] = ctx.table_z+CONTACT_HEIGHT+.008
    ctx.memory['fold_current_sources'] = [point.tolist() for point in gathered]
    ctx.memory['fold_destinations'] = [point.tolist() for point in targets]
    travel = np.linalg.norm((.5*(targets[0]+targets[1])-.5*(gathered[0]+gathered[1]))[:2])
    arc_height = float(np.clip(.5*travel, .095, .140))
    for index, fraction in enumerate(np.linspace(1./6., 1., 6)):
        goals = paired_arc(gathered, targets, float(fraction), arc_height)
        carry = [_carry_quat(ctx.kinematics, arm, target, quat, 20., float(fraction), True)
                 for arm, target, quat in zip(ARMS, targets, quats)]
        ctx.stage = 'fold_b_paired_hem_arc_'+str(index)
        yield from ctx.move_both(*goals, *carry, 0., 0., 12)
    ctx.stage = 'fold_b_paired_hem_release'
    yield from _grip_both(ctx, 1., 14)
    carry = [_carry_quat(ctx.kinematics, arm, target, quat, 20., 1., True)
             for arm, target, quat in zip(ARMS, targets, quats)]
    ctx.stage = 'fold_b_paired_hem_vertical_separation'
    yield from ctx.move_both(*[lift(target, .055) for target in targets], *carry, 1., 1., 12)
    ctx.stage = 'done'
    yield from ctx.home(duration=32)
    yield from ctx.hold(8)


fold_clothes = run
fold_clothes_random = run
