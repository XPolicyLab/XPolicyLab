"""Classify RGB objects with compact direct-arm tilted pick-and-drop motions."""
import itertools
import math
from copy import deepcopy
import numpy as np
import cv2
from coap.motion import ARMS, down_quat, link_to_tcp, quat_multiply, tcp_to_link
from coap.tasks.memory_unchecked import SortingContext

class ClassifyDirectContext(SortingContext):
    """Use direct source chords and smooth Cartesian loaded transport paths."""

    def _move_targets(self, targets, duration, tcp=True, motion_profile=None):
        starts = {arm: np.asarray(self.state[arm + '_arm_joint_state'], float).copy() for arm in ARMS}
        self.joint_targets = {arm: joints.copy() for arm, joints in starts.items()}
        endpoints = {arm: joints.copy() for arm, joints in starts.items()}
        diagnostics = []
        for arm, goal in targets.items():
            pose = tcp_to_link(goal[:3], goal[3:]) if tcp else goal
            solver = self.kinematics
            source_stage = str(getattr(self, 'stage', '')).startswith(('classify_b_approach_', 'classify_b_pick_'))
            if source_stage:
                from coap.tasks.classify_r07_periodic_ik import PeriodicPostureKinematics
                if not hasattr(self, '_classify_source_periodic_ik'):
                    self._classify_source_periodic_ik = PeriodicPostureKinematics(self.kinematics)
                solver = self._classify_source_periodic_ik
            result = solver.solve(arm, pose, current_joints=starts[arm], position_tol=0.002, rotation_tol=0.02, max_starts=8, max_nfev=140)
            endpoints[arm] = np.asarray(result['joint_value'], float)
            diagnostics.append(dict(arm=arm, position_error_m=result['position_error'], rotation_error_rad=result['rotation_error']))
        path = np.stack([np.concatenate([starts[arm] for arm in ARMS]), np.concatenate([endpoints[arm] for arm in ARMS])])
        carrying = any((self.gripper_targets[arm] < 0.8 for arm in targets))
        velocity = 1.7 if getattr(self, 'sorting_heavy_grasp', False) else 2.0 if carrying else 2.4
        if str(getattr(self, 'stage', '')).startswith('classify_b_drop_') and carrying:
            from coap.tasks.classify_r28_normal_camera_transport import transport_path
            path, transport_diagnostics = transport_path(self, targets, starts, endpoints, tcp=tcp)
            diagnostics.extend(transport_diagnostics)
        yield from self._emit_path(path, tuple(targets), duration, velocity, {arm: goal.tolist() for arm, goal in targets.items()})
        self.motion_plan.update(endpoint_ik_diagnostics=diagnostics, strategy='r27_soft_cartesian_loaded_transport')

def _category(obj):
    return str(obj.get('category', obj.get('appearance_family', 'unknown')))

def _pinch(ctx, obj):
    """Reuse v286 low body heights while honoring measured solid-part grasps."""
    if obj.get('grasp_position') is not None:
        return np.asarray(obj['grasp_position'], float).copy()
    point = np.asarray(obj['position'], float).copy()
    category = _category(obj)
    family = str(obj.get('appearance_family', category))
    height = 0.029
    if family in ('phone', 'chocolate_bar') or category == 'chocolate_bar':
        height = 0.02
    elif family in ('toy_car', 'car') or category in ('car', 'toy_car'):
        height = 0.021
    elif family in ('pen', 'pepper') or category in ('pen', 'pepper'):
        height = 0.017 if category == 'pen' else 0.022
    elif family == 'garage' or category == 'garage':
        height = 0.039
    elif category == 'wooden_toy':
        height = 0.029
    elif family == 'action_camera':
        height = 0.03
    point[2] = ctx.table_z + float(obj.get('grasp_height', height))
    return point

def _tilted_quat(yaw, arm, point, degrees):
    """Tilt the tool toward the source while preserving its closing heading."""
    quat = down_quat(yaw)
    if degrees == 0:
        return quat
    base = np.array([-0.3 if arm == 'left' else 0.3, -0.45])
    direction = np.asarray(point[:2], float) - base
    direction /= max(float(np.linalg.norm(direction)), 1e-08)
    angle = math.radians(float(degrees))
    tilt = np.r_[math.cos(angle / 2), np.array([direction[1], -direction[0], 0.0]) * math.sin(angle / 2)]
    return quat_multiply(tilt, quat)

def _drop_points(ctx, basket, arm, ordinal):
    """Spread releases inside the basket, with a small near-side reach option."""
    center = np.asarray(basket['position'], float).copy()
    center[2] = float(basket.get('rim_z', ctx.table_z + 0.07)) + 0.065
    center[2] = max(center[2], ctx.table_z + 0.135)
    center[1] += (0.0, -0.025, 0.025)[ordinal % 3]
    near = center.copy()
    near[0] += -0.045 if arm == 'left' else 0.045
    near[1] -= 0.015
    return (center, near)

def _joint_proposal(ctx, arm, point, quat, seed):
    result = ctx.kinematics.solve(arm, tcp_to_link(point, quat), current_joints=seed, position_tol=0.002, rotation_tol=0.02, max_starts=3, max_nfev=90)
    joints = np.asarray(result['joint_value'], float)
    residual = 1500.0 * float(result['position_error']) + 12.0 * float(result['rotation_error'])
    travel = float(np.max(np.abs(joints - seed)))
    return (joints, residual, travel)

def choose_route(ctx, obj, basket, ordinal=0):
    """Soft-rank both arms, equivalent headings, tilts and release postures."""
    point = _pinch(ctx, obj)
    yaw = float(obj.get('grasp_yaw', obj.get('yaw', 0.0)))
    lift = max(0.095, float(obj.get('height', 0.04)) + 0.045)
    lift = min(lift, 0.145)
    if not ctx.numeric_ik:
        arm = 'left' if float(basket['position'][0]) < 0 else 'right'
        quat = down_quat(yaw)
        return dict(arm=arm, point=point, quaternion=quat, release_quaternion=quat, release=_drop_points(ctx, basket, arm, ordinal)[0], lift_height=lift, cost=0.0, candidates=[])
    proposals = []
    tall = _category(obj) in ('garage', 'wooden_toy', 'action_camera')
    for arm in ARMS:
        initial = np.asarray(ctx.state[arm + '_arm_joint_state'], float)
        for half_turn in (0.0, math.pi):
            for tilt_degrees in (0.0, 22.0 if tall else 16.0):
                quat = _tilted_quat(yaw + half_turn, arm, point, tilt_degrees)
                pinch_joints, pinch_error, pinch_travel = _joint_proposal(ctx, arm, point, quat, initial)
                lift_point = point + [0.0, 0.0, lift]
                lift_joints, lift_error, lift_travel = _joint_proposal(ctx, arm, lift_point, quat, pinch_joints)
                for release_index, release in enumerate(_drop_points(ctx, basket, arm, ordinal)):
                    release_yaw = 0.0 if arm == 'left' else math.pi
                    release_options = [quat, down_quat(release_yaw)]
                    release_options.extend((_tilted_quat(release_yaw, arm, release, angle) for angle in (35.0, 55.0, 70.0)))
                    for release_quat in release_options:
                        _, release_error, release_travel = _joint_proposal(ctx, arm, release, release_quat, lift_joints)
                        cost = pinch_error + lift_error + release_error + pinch_travel + 0.5 * lift_travel + release_travel + 0.05 * tilt_degrees / 16.0 + 0.08 * release_index
                        proposals.append(dict(arm=arm, point=point.copy(), quaternion=quat.copy(), release_quaternion=release_quat.copy(), release=release.copy(), lift_height=lift, cost=float(cost), tilt_degrees=tilt_degrees, residual_cost=float(pinch_error + lift_error + release_error)))
    from coap.tasks.classify_r30_normal_pen_source_objective_source_objective import rank_source_objective
    selected = rank_source_objective(ctx, proposals)
    selected['candidates'] = [dict(arm=item['arm'], cost=item['cost'], tilt_degrees=item['tilt_degrees'], residual_cost=item['residual_cost']) for item in proposals]
    return selected

def _ordinary_mapping(objects, baskets):
    """Choose the free category assignment by source-to-basket travel distance."""
    categories = sorted(set(map(_category, objects)))
    if len(categories) > len(baskets):
        categories.sort(key=lambda category: (-sum((_category(obj) == category for obj in objects)), category))
        categories = categories[:len(baskets)]
    costs = []
    for assignment in itertools.permutations(range(len(baskets)), len(categories)):
        cost = 0.0
        for category, index in zip(categories, assignment):
            destination = np.asarray(baskets[index]['position'], float)
            for obj in objects:
                if _category(obj) == category:
                    cost += float(np.linalg.norm(np.asarray(obj['position'])[:2] - destination[:2]))
        costs.append((cost, assignment))
    _, assignment = min(costs, key=lambda item: item[0])
    return dict(zip(categories, assignment))

def _ordered_objects(objects, baskets, mapping):
    """Complete whole categories before starting another basket."""
    categories = sorted(mapping, key=lambda category: (sum((_category(obj) == category for obj in objects)), sum((np.linalg.norm(np.asarray(obj['position'])[:2] - np.asarray(baskets[mapping[category]]['position'])[:2]) for obj in objects if _category(obj) == category))))
    ordered = []
    previous = np.array([0.0, -0.3])
    for category in categories:
        remaining = [dict(obj) for obj in objects if _category(obj) == category]
        while remaining:
            selected_index = min(range(len(remaining)), key=lambda item_index: np.linalg.norm(np.asarray(remaining[item_index]['position'])[:2] - previous))
            obj = remaining.pop(selected_index)
            ordered.append(obj)
            previous = np.asarray(baskets[mapping[category]]['position'])[:2]
    return ordered

def _head_watch_case(ctx, obj, fallback_point, fallback_quat):
    """Measure neutral case material inside the selected green watch region."""
    frame = np.asarray(ctx.vision['cam_head']['color'], np.uint8)[..., :3]
    view = obj.get('views', {}).get('cam_head', obj)
    x, y, width, height = map(int, view['bbox'])
    crop = frame[y:y + height, x:x + width]
    if not crop.size:
        return (fallback_point, fallback_quat, None)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    balanced = (crop[..., 1] >= crop[..., 0] * 0.94) & (crop[..., 1] >= crop[..., 2] * 0.94)
    neutral = ((hsv[..., 1] < 85) & (hsv[..., 2] > 90) & balanced).astype(np.uint8)
    _, _, stats, centers = cv2.connectedComponentsWithStats(neutral)
    candidates = []
    for stat, center in zip(stats[1:], centers[1:]):
        if 4 <= stat[4] < 0.55 * width * height and min(stat[2:4]) >= 2:
            candidates.append((int(stat[4]), center + [x, y]))
    if not candidates:
        return (fallback_point, fallback_quat, None)
    _, pixel = max(candidates, key=lambda item: item[0])
    point = ctx.project(pixel, ctx.table_z + 0.035, 'cam_head')
    point[2] = ctx.table_z + 0.027
    radial = point[:2] - np.asarray(obj['position'])[:2]
    length = float(np.linalg.norm(radial))
    quat = fallback_quat
    if length > 0.008:
        radial /= length
        point[:2] -= 0.004 * radial
        yaw = float(np.arctan2(radial[1], radial[0]) - math.pi / 2.0)
        quaternions = [down_quat(yaw), down_quat(yaw + math.pi)]
        quat = max(quaternions, key=lambda value: abs(float(np.dot(value, fallback_quat))))
    return (point, quat, dict(method='selected_green_watch_head_case', pixel=pixel.tolist(), point=point.tolist(), same_source_region=True))

def _close_view_refinement(ctx, obj, arm, point, quat):
    """Reuse v286 watch and toy solid-body measurements from the current view."""
    if _category(obj) == 'watch':
        from coap.tasks.classify_r32_normal_watch_material_material import watch_pose
        return watch_pose(obj, point, quat)
    category = _category(obj)
    source_point, source_quat = (np.asarray(point, float).copy(), np.asarray(quat, float).copy())
    evidence = dict(method='initial_rgb_solid_part')
    if category == 'watch':
        from coap.tasks.memory import images_of, visible_green_strap, watch_face, dark_watch_display, green_watch_grasp, dark_watch_grasp
        green = visible_green_strap(images_of(ctx), obj)
        ctx.dark_watch_triangulated = False
        face = watch_face(ctx, arm, point) if green else dark_watch_display(ctx, arm, np.asarray(obj['position'], float), neutral='ring_pixel' in obj)
        if face is not None:
            anchor = np.asarray(obj['position'], float)
            if green:
                point, quat = green_watch_grasp(ctx, arm, face, anchor, quat)
            else:
                point, measured_quat = dark_watch_grasp(ctx, arm, face, anchor)
                if measured_quat is not None:
                    quat = measured_quat
            evidence = dict(method='v286_watch_solid_case', camera=face[2], pixel=np.asarray(face[1]).tolist(), point=point.tolist())
    elif category in ('garage', 'wooden_toy'):
        from coap.tasks.memory import images_of
        from coap.tasks.memory_toy import toy_face_grasp
        body = toy_face_grasp(ctx, obj, images_of(ctx), eye_centers=True)
        if body is not None:
            point = np.asarray(body['pinch'], float).copy()
            evidence = dict(method='v286_toy_eye_stereo', point=point.tolist(), ray_gap=body['ray_gap'])
    if category == 'watch' and evidence['method'] != 'initial_rgb_solid_part':
        head = np.asarray(ctx.vision['cam_head']['color'])
        pixel = ctx.perception.world_to_pixel(point, 'cam_head', ctx.state, head.shape)
        view = obj.get('views', {}).get('cam_head', obj)
        x, y, width, height = map(float, view['bbox'])
        same_region = x - 5 <= pixel[0] <= x + width + 5 and y - 5 <= pixel[1] <= y + height + 5
        evidence['head_association_pixel'] = pixel.tolist()
        evidence['same_source_region'] = bool(same_region)
        if not same_region:
            evidence['unused_proposal'] = evidence.get('point')
            point, quat = (source_point, source_quat)
            evidence['method'] = 'retain_same_source_head_geometry'
    if category == 'watch' and green and (evidence['method'] in ('initial_rgb_solid_part', 'retain_same_source_head_geometry')):
        point, quat, head_evidence = _head_watch_case(ctx, obj, point, quat)
        if head_evidence is not None:
            evidence = head_evidence
    return (point, quat, evidence)

def _park_goal(ctx, arm):
    """Move the previous releaser toward its own foreground without a full home."""
    initial_pose = np.asarray(ctx.initial_state[arm + '_ee_pose'], float)
    nominal = link_to_tcp(initial_pose)
    points = [nominal, np.array([-0.34 if arm == 'left' else 0.34, -0.245, ctx.table_z + 0.18])]
    quaternions = [np.asarray(ctx.state[arm + '_ee_pose'][3:], float), down_quat(0.0 if arm == 'left' else math.pi), initial_pose[3:]]
    if not ctx.numeric_ik:
        return (points[0], quaternions[0])
    seed = np.asarray(ctx.state[arm + '_arm_joint_state'], float)
    choices = []
    for point in points:
        for quat in quaternions:
            _, residual, travel = _joint_proposal(ctx, arm, point, quat, seed)
            choices.append((residual + travel, point, quat))
    _, point, quat = min(choices, key=lambda item: item[0])
    return (point, quat)

def _observation_park(ctx):
    """Expose tabletop material behind the initial horizontal gripper fingers."""
    goals = {}
    for arm in ARMS:
        side = -1.0 if arm == 'left' else 1.0
        points = [np.array([side * 0.42, -0.275, ctx.table_z + 0.18]), np.array([side * 0.4, -0.285, ctx.table_z + 0.165])]
        quaternions = [down_quat(0.0), down_quat(math.pi)]
        if not ctx.numeric_ik:
            goals[arm] = (points[0], quaternions[0])
            continue
        seed = np.asarray(ctx.state[arm + '_arm_joint_state'], float)
        proposals = []
        for point in points:
            for quat in quaternions:
                _, residual, travel = _joint_proposal(ctx, arm, point, quat, seed)
                proposals.append((residual + travel, point, quat))
        _, point, quat = min(proposals, key=lambda item: item[0])
        goals[arm] = (point, quat)
    yield from ctx.move_both(goals['left'][0], goals['right'][0], goals['left'][1], goals['right'][1], left_grip=1.0, right_grip=1.0, duration=18)
    yield from ctx.hold(3)

def _overlap(first, second):
    a, b, w, h = map(float, first['bbox'])
    x, y, width, height = map(float, second['bbox'])
    intersection = max(0.0, min(a + w, x + width) - max(a, x)) * max(0.0, min(b + h, y + height) - max(b, y))
    return intersection / max(1.0, min(w * h, width * height))

def _legacy_complete_regions(objects, baskets=()):
    """Prefer complete current source regions over nested color fragments."""
    sources = []
    for obj in objects:
        x, y, width, height = map(float, obj['bbox'])
        basket_shadow = False
        if _category(obj) == 'pen':
            for basket in baskets:
                if 'bbox' not in basket:
                    continue
                bx, by, bw, bh = map(float, basket['bbox'])
                overlap = max(0.0, min(x + width, bx + bw) - max(x, bx))
                below = y - (by + bh)
                if height < 0.075 * bh and width > 5.0 * height and (-0.03 * bh <= below < 0.25 * bh) and (overlap > 0.3 * width):
                    basket_shadow = True
                    break
        if not basket_shadow:
            sources.append(obj)
    selected = []
    ordered = sorted(sources, key=lambda item: -float(item.get('area', np.prod(item.get('size', (0.05, 0.05))[:2]))))
    for obj in ordered:
        duplicate = False
        for other in selected:
            distance = np.linalg.norm(np.asarray(obj['position'])[:2] - np.asarray(other['position'])[:2])
            if distance < 0.04 and _overlap(obj, other) > 0.35:
                duplicate = True
                break
        if not duplicate:
            selected.append(dict(obj))
    complete = []
    toys = [obj for obj in selected if _category(obj) in ('garage', 'wooden_toy')]
    for obj in selected:
        fragment = False
        if _category(obj) in ('watch', 'action_camera') and 'ring_pixel' not in obj:
            x, y, width, height = map(float, obj['bbox'])
            for toy in toys:
                distance = np.linalg.norm(np.asarray(obj['position'])[:2] - np.asarray(toy['position'])[:2])
                a, b, w, h = map(float, toy['bbox'])
                pixel_gap = max(a - (x + width), x - (a + w), 0.0) + max(b - (y + height), y - (b + h), 0.0)
                if distance < 0.055 and pixel_gap < 0.6 * max(w, h) and (width * height < 0.8 * w * h):
                    fragment = True
                    break
        if not fragment:
            complete.append(obj)
    return complete

def _complete_regions(objects, baskets=()):
    """Keep grounded complete regions and retain legacy normal-task grouping."""
    if any(('dino_proposal' in obj for obj in objects)):
        return [dict(obj) for obj in objects]
    return _legacy_complete_regions(objects, baskets)

def _associate_sources(tracks, objects, refresh_index, baskets=()):
    """Associate current table material with stable category and source history."""
    for track in tracks:
        track['visible'] = False
    used = set()
    new_ids = []
    for obj in _complete_regions(objects, baskets):
        category = _category(obj)
        point = np.asarray(obj['position'], float)
        matches = []
        for track in tracks:
            if track['id'] in used:
                continue
            distance = float(np.linalg.norm(point[:2] - np.asarray(track['object']['position'])[:2]))
            overlap = _overlap(obj, track['object'])
            rgb = np.asarray(obj.get('mean_rgb', (128.0, 128.0, 128.0)), float)
            prior_rgb = np.asarray(track['object'].get('mean_rgb', (128.0, 128.0, 128.0)), float)
            appearance = float(np.linalg.norm(rgb - prior_rgb) / 255.0)
            same_category = track['category'] == category
            same_region = distance < 0.045 and overlap > 0.5 and (appearance < 0.4)
            if same_category and (distance < 0.085 or overlap > 0.35) or same_region:
                cost = distance + 0.025 * appearance + (0.0 if same_category else 0.025)
                matches.append((cost, track))
        if matches:
            _, track = min(matches, key=lambda item: item[0])
            if category != track['category']:
                obj['raw_refreshed_category'] = category
                obj['category'] = track['category']
                obj['appearance_family'] = track['object'].get('appearance_family', track['category'])
                prior = track['object']
                if 'grasp_position' in obj and 'grasp_position' in prior:
                    obj['grasp_position'] = np.asarray(obj['grasp_position'], float).copy()
                    obj['grasp_position'][2] = np.asarray(prior['grasp_position'])[2]
                obj['position'] = np.asarray(obj['position'], float).copy()
                obj['position'][2] = np.asarray(prior['position'])[2]
                for field in ('grasp_height', 'surface_z'):
                    if field in prior:
                        obj[field] = prior[field]
        else:
            track = dict(id=len(tracks), category=category, attempts=0, object=obj, first_seen=refresh_index)
            tracks.append(track)
            new_ids.append(track['id'])
        track.update(object=obj, visible=True, last_seen=refresh_index)
        used.add(track['id'])
    return new_ids

def _next_source(ctx, tracks, mapping, active_category):
    """Spend remaining actions on whole categories and untried visible bodies."""
    visible = [track for track in tracks if track.get('visible') and track['category'] in mapping and (track['attempts'] < 2)]
    if not visible:
        return None
    category_priority = {'garage': 0, 'phone': 1, 'wooden_toy': 1, 'chocolate_bar': 2, 'action_camera': 2, 'pen': 3, 'pepper': 3, 'toy_car': 4, 'car': 4, 'watch': 5}
    untouched = [track for track in visible if track['attempts'] == 0]
    pool = untouched if untouched else visible
    active = [track for track in pool if track['category'] == active_category]
    if active:
        pool = active
    else:
        category = min({track['category'] for track in pool}, key=lambda value: (category_priority.get(value, 3), sum((track['category'] == value for track in pool))))
        pool = [track for track in pool if track['category'] == category]

    def distance(track):
        point = np.asarray(track['object']['position'], float)
        return min((float(np.linalg.norm(ctx.tcp(arm)[:2] - point[:2])) for arm in ARMS))
    return min(pool, key=lambda track: (track['attempts'], distance(track)))

def _policy_run(ctx, scene=None):
    """Refresh table sources while retaining the first basket and category mapping."""
    from coap.tasks.classify_r30_normal_pen_source_objective_observer import observe
    from coap.tasks.open_motion import enable
    enable(ctx)
    if ctx.numeric_ik:
        ctx.__class__ = ClassifyDirectContext
    ctx._classify_skip_count_prior = True
    ctx.stage = 'classify_b_observation_park'
    if scene is None:
        yield from _observation_park(ctx)
        scene = observe(ctx)
    baskets = deepcopy(scene['baskets'])
    objects = _complete_regions(scene['objects'], baskets)
    mapping = dict(scene.get('instruction_mapping', scene.get('mapping', {})))
    if ctx.task_name == 'classify_objects':
        mapping = _ordinary_mapping(objects, baskets)
    ctx._classify_basket_cache = deepcopy(baskets)
    ctx._classify_free_mapping = dict(mapping)
    ctx._classify_normal_categories = tuple(mapping)
    ctx.memory['classify_b'] = dict(candidate='B_r02', mapping=mapping, planned_objects=len(objects), source='closed_rgb_source_queue_and_fixed_baskets', execution='direct_arm_joint_chords', records=[], refreshes=[], initial_baskets=[np.asarray(basket['position']).tolist() for basket in baskets])
    memory = ctx.memory['classify_b']
    tracks = []
    new_ids = _associate_sources(tracks, objects, 0, baskets)
    memory['refreshes'].append(dict(index=0, new_source_ids=new_ids, visible_sources=len(objects), step=ctx.steps))
    ordinals = {key: 0 for key in mapping}
    previous_arm = None
    active_category = None
    review_done = False
    while True:
        track = _next_source(ctx, tracks, mapping, active_category)
        completed_costs = [record['end_step'] - record['start_step'] for record in memory['records'] if 'end_step' in record]
        estimated_cost = min(150.0, max(95.0, float(np.mean(completed_costs[-5:])))) if completed_costs else 120.0
        if ctx.steps + estimated_cost > 1040:
            memory['finish_reason'] = 'remaining_actions_reserved_for_origin'
            break
        if track is None:
            if review_done:
                memory['finish_reason'] = 'no_remaining_untried_visible_sources'
                break
            ctx.stage = 'classify_b_final_table_review'
            yield from _observation_park(ctx)
            fresh = observe(ctx)
            review_done = True
            new_ids = _associate_sources(tracks, fresh['objects'], len(memory['refreshes']), baskets)
            memory['refreshes'].append(dict(index=len(memory['refreshes']), step=ctx.steps, new_source_ids=new_ids, visible_sources=sum((item['visible'] for item in tracks)), final_review=True))
            previous_arm = None
            continue
        index = len(memory['records'])
        obj = dict(track['object'])
        category = track['category']
        basket = baskets[mapping[category]]
        track['attempts'] += 1
        if category == 'watch':
            from coap.tasks.classify_r32_normal_watch_material_material import prepare_grasp
            prepare_grasp(ctx, obj, track['attempts'], track.get('last_grasp'))
            track['last_grasp'] = obj.get('grasp_signature')
        active_category = category
        ctx.sorting_heavy_grasp = category in ('toy_car', 'car') and float(np.asarray(obj.get('mean_rgb', [128.0, 128.0, 128.0]))[0]) > 130.0
        route = choose_route(ctx, obj, basket, ordinals[category])
        arm = route['arm']
        point = route['point'].copy()
        quat = route['quaternion'].copy()
        lift = route['lift_height']
        record = dict(index=index, source_id=track['id'], attempt=track['attempts'], category=category, arm=arm, basket_index=mapping[category], raw_refreshed_category=obj.get('raw_refreshed_category'), start_step=ctx.steps, initial_pinch=point.tolist(), cost=route['cost'], tilt_degrees=route.get('tilt_degrees', 0.0), candidates=route['candidates'])
        if category == 'watch':
            record.update(material_grasp=obj.get('material_grasp_evidence'),
                          grasp_yaw=float(obj.get('grasp_yaw', obj.get('yaw', 0.))),
                          grasp_opening=float(obj.get('grasp_opening', 1.)))
        memory['records'].append(record)
        ctx.stage = f'classify_b_approach_{index}'
        if previous_arm is not None and previous_arm != arm:
            parked, parked_quat = _park_goal(ctx, previous_arm)
            goals = {arm: (point + [0.0, 0.0, lift], quat), previous_arm: (parked, parked_quat)}
            yield from ctx.move_both(goals['left'][0], goals['right'][0], goals['left'][1], goals['right'][1], left_grip=1.0, right_grip=1.0, duration=16)
            record['previous_releaser_parked'] = previous_arm
        else:
            yield from ctx.move(arm, point + [0.0, 0.0, lift], quat, grip=1.0, duration=16)
        if category == 'action_camera':
            from coap.tasks.classify_r28_normal_camera_camera import refine_camera_point
            old_point = point.copy()
            point, camera_evidence = refine_camera_point(ctx, arm, point, quat)
            record['camera_material_grasp'] = camera_evidence
            if np.linalg.norm(point - old_point) > 0.003:
                yield from ctx.move(arm, point + [0.0, 0.0, lift], quat, grip=1.0, duration=8)
        if category in ('watch', 'garage', 'wooden_toy'):
            yield from ctx.hold(3)
            old_point, old_quat = (point.copy(), quat.copy())
            point, quat, evidence = _close_view_refinement(ctx, obj, arm, point, quat)
            if np.linalg.norm(point - old_point) > 0.003 or abs(np.dot(quat, old_quat)) < 0.999:
                from coap.tasks.classify_r07_wrist_pose import select_refined_pose
                point, quat, pose_evidence = select_refined_pose(ctx, obj, arm, old_point, old_quat, point, quat, lift)
                record['close_view_pose_ranking'] = pose_evidence
            record['close_view'] = evidence
            if np.linalg.norm(point - old_point) > 0.003 or abs(np.dot(quat, old_quat)) < 0.999:
                yield from ctx.move(arm, point + [0.0, 0.0, lift], quat, grip=1.0, duration=8)
        ctx.stage = f'classify_b_pick_{index}'
        heavy = bool(ctx.sorting_heavy_grasp)
        yield from ctx.move(arm, point + [0.0, 0.0, 0.025], quat, grip=float(obj.get('grasp_opening', 1.)) if category == 'watch' else 1.0, duration=8)
        yield from ctx.move(arm, point, quat, grip=float(obj.get('grasp_opening', 1.)) if category == 'watch' else 1.0, duration=8 if heavy else 6)
        yield from ctx.hold(2)
        yield from ctx.grip(arm, 0.0, duration=10 if heavy else 8)
        yield from ctx.hold(6 if heavy else 4)
        yield from ctx.move(arm, point + [0.0, 0.0, lift], quat, grip=0.0, duration=18 if heavy else 12)
        ctx.stage = f'classify_b_drop_{index}'
        yield from ctx.move(arm, route['release'], route['release_quaternion'], grip=0.0, duration=28 if heavy else 20)
        yield from ctx.hold(2)
        yield from ctx.grip(arm, 1.0, duration=8)
        yield from ctx.hold(3)
        record.update(end_step=ctx.steps, final_pinch=point.tolist(), release=route['release'].tolist(), completed_commands=True)
        ordinals[category] += 1
        previous_arm = arm
        ctx.stage = f'classify_b_refresh_{index}'
        fresh = observe(ctx)
        new_ids = _associate_sources(tracks, fresh['objects'], len(memory['refreshes']), baskets)
        record['same_source_visible_after_drop'] = bool(track['visible'])
        memory['refreshes'].append(dict(index=len(memory['refreshes']), step=ctx.steps, new_source_ids=new_ids, visible_sources=sum((item['visible'] for item in tracks)), after_source_id=track['id'], source_still_visible=bool(track['visible'])))
    memory['source_attempts'] = [dict(id=track['id'], category=track['category'], attempts=track['attempts'], visible=track.get('visible'), first_seen=track['first_seen']) for track in tracks]
    ctx.stage = 'done'
    yield from ctx.home(duration=12)
    yield from ctx.hold(5)

def run(ctx, scene=None):
    """Bind resident FP32 vision devices before the unchanged policy generator."""
    from coap.tasks.classify_r28_cuda_observers import enable_for_observers
    ctx.memory['classify_device'] = enable_for_observers([])
    return (yield from _policy_run(ctx, scene))
