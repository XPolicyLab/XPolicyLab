"""RGB language grounding with a neck grasp and a mouth-centered pour.

The attachment and upstream outlet geometry follows the Apache-2.0
PhysicalRSI pour_by_language release. Execution uses the task-local COAP
executor and does not use upstream rejection or perception services.
"""
import itertools
import math
import os
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

BOTTLE_HEIGHT = .2181
GRASP_HEIGHT = .1675
SPOUT_RADIUS = .0154353


def _quat(rotation):
    return np.roll(Rotation.from_matrix(rotation).as_quat(), 1)


def _image(ctx):
    return np.asarray(ctx.vision['cam_head']['color'], np.uint8)[..., :3]


def _project(ctx, pixel, height, shape):
    return ctx.perception.project(pixel, ctx.table_z + height, 'cam_head', ctx.state, shape)


def _colors(image):
    hue, saturation, value = np.moveaxis(cv2.cvtColor(image, cv2.COLOR_RGB2HSV), -1, 0)
    return {
        'red': ((hue < 7) | (hue > 170)) & (saturation > 145) & (value > 45),
        'turquoise': (hue > 70) & (hue < 105) & (saturation > 65) & (value > 45),
        'violet': (hue > 120) & (hue < 169) & (saturation > 28) & (value > 30),
    }


def _bottles(ctx, image):
    found = {}
    for color, selected in _colors(image).items():
        mask = cv2.morphologyEx(selected.astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if not (100 < area < 14000 and height > 1.15 * width and width > 8):
                continue
            yy, xx = np.where(labels == index)
            mouth = [float(np.median(xx[yy <= yy.min() + 4])), float(yy.min() + 6.5)]
            position = _project(ctx, mouth, BOTTLE_HEIGHT, image.shape)
            if abs(position[0]) < .6 and -.35 < position[1] < .4:
                candidates.append((area, dict(color=color, position=position,
                    mouth_pixel=mouth, bbox=[int(x), int(y), int(width), int(height)],
                    area=int(area), source='head_rgb_color_neck_top')))
        if candidates:
            found[color] = max(candidates, key=lambda item: item[0])[1]
    return found


def _bowls(ctx, image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 25, 70)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for selection in (((hsv[..., 2] < 55) & (hsv[..., 1] < 125)),
                      ((hsv[..., 2] < 90) & (hsv[..., 1] < 125)),
                      ((hsv[..., 2] < 115) & (hsv[..., 1] < 125)),
                      ((hsv[..., 2] > 155) & (hsv[..., 1] < 65))):
        mask = cv2.morphologyEx(selection.astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8))
        extra, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = list(contours) + list(extra)
    candidates = []
    for contour in contours:
        if len(contour) < 20:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if not (220 < area < 10000 and .65 < width / height < 1.8):
            continue
        ellipse = cv2.fitEllipse(contour)
        center = np.asarray(ellipse[0], float)
        position = _project(ctx, center, .04, image.shape)
        if not (abs(position[0]) < .45 and -.30 < position[1] < .08):
            continue
        left = _project(ctx, [center[0] - width / 2, center[1]], .04, image.shape)
        right = _project(ctx, [center[0] + width / 2, center[1]], .04, image.shape)
        diameter = float(np.linalg.norm((right - left)[:2]))
        if not .080 < diameter < .140:
            continue
        full = np.zeros(image.shape[:2], np.uint8)
        cv2.ellipse(full, ellipse, 1, -1)
        filling = float(area / max(np.count_nonzero(full), 1))
        score = 1. - abs(diameter - .11) * 25. - abs(filling - .96)
        value = float(np.median(hsv[..., 2][full > 0]))
        candidates.append(dict(position=position, center_pixel=center, radius=diameter / 2,
                               value=value, score=score, bbox=[x, y, width, height]))
    candidates.sort(key=lambda item: item['score'], reverse=True)
    distinct = []
    for candidate in candidates:
        if all(np.linalg.norm(candidate['position'][:2] - item['position'][:2]) > .045
               for item in distinct):
            distinct.append(candidate)
    best = None
    best_score = -float('inf')
    for group in itertools.combinations(distinct[:15], 3):
        positions = np.array([item['position'] for item in group])
        if min(np.linalg.norm(a[:2] - b[:2]) for a, b in itertools.combinations(positions, 2)) < .08:
            continue
        score = sum(item['score'] for item in group) - 25 * np.std(positions[:, 1])
        if score > best_score:
            best, best_score = group, score
    if best is None:
        return {}
    ordered = sorted(best, key=lambda item: item['value'])
    return {color: dict(item, color=color, source='head_rgb_ellipse_brightness')
            for color, item in zip(('black', 'brown', 'white'), ordered)}


def ground(ctx):
    image = _image(ctx)
    bottles, bowls = _bottles(ctx, image), _bowls(ctx, image)
    instruction = str(ctx.obs.get('instruction', ''))
    pairs = re.findall(r'(red|turquoise|violet) bottle into the (?:first|second|third) '
                       r'(black|white|brown) bowl', instruction.lower())
    ctx.memory['pour_grounding'] = dict(bottles=bottles, bowls=bowls, pairs=pairs,
                                       online_inputs='RGB_proprioception_instruction')
    return [(bottles[source], bowls[target]) for source, target in pairs
            if source in bottles and target in bowls]


def _frame(ctx, source, target, arm):
    base = np.array([-.3 if arm == 'left' else .3, -.45])
    forward = np.r_[(source['position'][:2] + target['position'][:2]) / 2 - base, 0.]
    forward /= np.linalg.norm(forward)
    closing = np.cross([0., 0., 1.], forward)
    return np.column_stack([forward, closing, [0., 0., 1.]])


def _pivot_pose(ctx, center, rotation, grasp_rotation, radius, angle):
    axis = grasp_rotation[:, 0]
    turn = Rotation.from_rotvec(axis * angle).as_matrix()
    discharge = np.cross(axis, [0., 0., 1.])
    offset = max(0., radius - SPOUT_RADIUS - .004)
    mouth = np.array([*center[:2], ctx.table_z]) - offset * discharge
    mouth[2] += max(.145, BOTTLE_HEIGHT * math.cos(angle) + .045 * abs(math.sin(angle)) + .075)
    neck_to_mouth = np.array([0., 0., BOTTLE_HEIGHT - GRASP_HEIGHT])
    tcp_offset = grasp_rotation[:, 0] * .015 - neck_to_mouth
    return mouth + turn @ tcp_offset, _quat(turn @ rotation)


def _pour_pair(ctx, source, target, index):
    arm = ctx.arm_for(source['position'])
    rotation = _frame(ctx, source, target, arm)
    quat = _quat(rotation)
    contact = source['position'].copy()
    contact[2] = ctx.table_z + GRASP_HEIGHT
    contact += rotation[:, 0] * .015
    approach = contact - rotation[:, 0] * .10 + [0., 0., .07]
    lifted = contact + [0., 0., .09]
    ctx.stage = 'pour_%d_approach' % index
    yield from ctx.move(arm, approach, quat, 1., duration=18)
    yield from ctx.move(arm, contact, quat, 1., duration=12)
    ctx.stage = 'pour_%d_close' % index
    yield from ctx.grip(arm, 0., duration=12)
    yield from ctx.move(arm, lifted, quat, 0., duration=12)
    pivot = lambda angle: _pivot_pose(ctx, target['position'], rotation, rotation,
                                    target['radius'], angle)
    above, above_quat = pivot(0.)
    ctx.stage = 'pour_%d_above' % index
    yield from ctx.move(arm, above, above_quat, 0., duration=18)
    final_angle = float(os.environ.get('COAP_POUR_ANGLE', '2.75'))
    count = int(os.environ.get('COAP_POUR_TILT_SEGMENTS', '8'))
    angles = np.linspace(0., final_angle, count + 1)
    for step, angle in enumerate(angles[1:]):
        ctx.stage = 'pour_%d_tilt_%d' % (index, step)
        position, orientation = pivot(float(angle))
        yield from ctx.move(arm, position, orientation, 0., duration=4)
    ctx.stage = 'pour_%d_drain' % index
    yield from ctx.hold(int(os.environ.get('COAP_POUR_DWELL', '32')))
    for step, angle in enumerate(reversed(angles[:-1])):
        ctx.stage = 'pour_%d_upright_%d' % (index, step)
        position, orientation = pivot(float(angle))
        yield from ctx.move(arm, position, orientation, 0., duration=4)
    ctx.stage = 'pour_%d_return' % index
    yield from ctx.move(arm, lifted, quat, 0., duration=18)
    yield from ctx.move(arm, contact, quat, 0., duration=12)
    yield from ctx.grip(arm, 1., duration=8)
    yield from ctx.move(arm, approach, quat, 1., duration=10)


def run(ctx):
    from coap.context import Context
    ctx.numeric_ik = False
    ctx.__class__ = Context
    pairs = ground(ctx)
    ctx.memory['pour_strategy'] = dict(name='side_neck_fixed_spout',
        reference='PhysicalRSI pour_by_language b50c0200475f3dc6 pure geometry',
        candidate=os.environ.get('COAP_POUR_CANDIDATE', 'A'),
        planned_pairs=len(pairs), safety_checks_enabled=False,
        angle_rad=float(os.environ.get('COAP_POUR_ANGLE', '2.75')),
        dwell_frames=int(os.environ.get('COAP_POUR_DWELL', '32')))
    for index, (source, target) in enumerate(pairs):
        yield from _pour_pair(ctx, source, target, index)
    ctx.stage = 'done'
    yield from ctx.home(duration=30)
    yield from ctx.hold(12)
