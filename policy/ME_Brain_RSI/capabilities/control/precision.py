"""RGB-driven programs for insertion and mug hanging.

Only observed detections, robot proprioception and fixed asset dimensions are
used. Asset dimensions are design constants, never episode state.
"""

import math
import numpy as np
from coap.manipulation import visual_evidence, verification


TUBE_LENGTH = 0.1404
MUG_HANDLES = (
    (0.075, 0.0, 0.015), (0.063, 0.0, 0.005),
    (0.048, 0.0, 0.0), (0.050, 0.0, 0.010),
    (0.042, 0.0, 0.0), (0.047, 0.0, 0.0),
    (0.047, 0.005, 0.015), (0.045, 0.0, 0.0),
    (0.055, 0.0, 0.005), (0.038, 0.0, 0.015),
)
MUG_WIDTHS = (0.1457, 0.1290, 0.1015, 0.1027, 0.0909,
              0.0985, 0.1052, 0.1150, 0.1205, 0.0823)
PEG_FRAMES = (
    ((0.045, 0.0, 0.150), (0.965925826, 0.0, -0.258819045, 0.0)),
    ((-0.045, 0.0, 0.155), (0.0, 0.190808995, 0.0, 0.981627183)),
    ((0.0, -0.055, 0.075), (0.694115238, -0.134922335, -0.134922335, -0.694115238)),
    ((0.0, 0.050, 0.070), (0.686102688, 0.171064613, -0.171064613, 0.686102688)),
)
LOWER_PEG_FRAMES = (
    ((0.055, 0.0, -0.010), (0.981793322, 0.0, -0.189952292, 0.0)),
    ((-0.050, -0.002, -0.012), (0.0, 0.207058017, 0.0, 0.978328665)),
)


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _multiply(first, second):
    w, x, y, z = first
    v, a, b, c = second
    return _unit([w*v-x*a-y*b-z*c, w*a+x*v+y*c-z*b,
                  w*b-x*c+y*v+z*a, w*c+x*b-y*a+z*v])


def _inverse(quaternion):
    w, x, y, z = _unit(quaternion)
    return np.array([w, -x, -y, -z])


def _axis_rotation(axis, angle):
    return np.r_[math.cos(angle / 2), _unit(axis) * math.sin(angle / 2)]


def _yaw_quaternion(yaw):
    return _axis_rotation([0, 0, 1], yaw)


def _rotate(quaternion, vector):
    w = float(quaternion[0])
    xyz = np.asarray(quaternion[1:], dtype=float)
    vector = np.asarray(vector, dtype=float)
    return vector + 2 * np.cross(xyz, np.cross(xyz, vector) + w * vector)


def _down(yaw):
    # Local +X is the X5 tool approach axis, and local Y is the jaw axis.
    return _multiply(_yaw_quaternion(yaw), [math.sqrt(0.5), 0, math.sqrt(0.5), 0])


def _reachable_down(ctx, arm, yaw):
    current = np.asarray(ctx.state[f"{arm}_ee_pose"], dtype=float)[3:]
    candidates = [_down(yaw), _down(yaw + math.pi)]
    return max(candidates, key=lambda quaternion: abs(float(np.dot(quaternion, current))))


def _has_contact(ctx, arm, minimum=0.08):
    opening = np.asarray(ctx.state.get(f"{arm}_ee_joint_state", [0.0]), dtype=float)
    return bool(opening.size and opening[0] > minimum)


def _keep_closed(actions, arm):
    for action in actions:
        action = dict(action)
        action[f"{arm}_ee_joint_state"] = np.array([0.0])
        yield action


def _direction(yaw):
    return np.array([math.cos(yaw), math.sin(yaw), 0.0])


def _observed_yaw(item, directed=False):
    keys = ("direction_yaw", "axis_yaw", "yaw") if directed else ("yaw", "axis_yaw")
    for key in keys:
        value = item.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _position(item):
    return np.asarray(item["position"], dtype=float).copy()


def _arm(ctx, position):
    chooser = getattr(ctx, "arm_for", None)
    return chooser(position) if chooser else ("left" if position[0] < 0 else "right")


def _best(ctx, kind, **kwargs):
    detections = ctx.detect(kind, **kwargs)
    return max(detections, key=lambda item: item.get("confidence", 0.0)) if detections else None


def _nearby(ctx, kind, anchor, arm, radius=0.08):
    # Reject wrist false positives that disagree strongly with the head anchor.
    candidates = ctx.detect(kind, camera=f"cam_{arm}_wrist")
    candidates = [item for item in candidates
                  if np.linalg.norm(_position(item)[:2] - np.asarray(anchor)[:2]) < radius]
    return min(candidates, key=lambda item: np.linalg.norm(_position(item)[:2] - np.asarray(anchor)[:2])) if candidates else None


def _hole(rack, expected):
    observed_holes = rack.get("hole_positions", [])
    if not observed_holes:
        return np.asarray(expected).copy()
    points = [np.asarray(point.get("position", point), dtype=float)
              if isinstance(point, dict) else np.asarray(point, dtype=float)
              for point in observed_holes]
    return min(points, key=lambda point: np.linalg.norm(point[:2] - expected[:2])).copy()


def _public_self_margin(ctx, arm, joints):
    """Check task candidates against the public arm collision geometry."""
    from coap.public_mesh import PublicRobotConvexGeometry
    model = getattr(ctx, "precision_collision_geometry", None)
    if model is None:
        model = PublicRobotConvexGeometry()
        ctx.precision_collision_geometry = model
    kinematics = getattr(ctx, "kinematics", None) or getattr(ctx, "precision_kinematics", None)
    if kinematics is None:
        from coap.kinematics import X5Kinematics
        kinematics = X5Kinematics(ctx.initial_state)
        ctx.precision_kinematics = kinematics
    margins = model.check(kinematics, arm, joints, grip=1.0,
                          pairs=[("link3", "link5"), ("link4", "link6")])
    return min(margins.values())


def _tube_targets(ctx, arm, grasp, quaternion, expected, rack, pinch_height, occupied):
    """Choose a reachable cylindrical heading and an unoccupied observed hole."""
    from coap.kinematics import X5Kinematics
    from coap.public_mesh import PublicRobotConvexGeometry
    from .precision_geometry import occupied_tube_margin
    kinematics = getattr(ctx, "kinematics", None) or X5Kinematics(ctx.initial_state)
    model = getattr(ctx, "precision_collision_geometry", None)
    if model is None:
        model = PublicRobotConvexGeometry()
        ctx.precision_collision_geometry = model
    points = rack.get("hole_positions", [])
    holes = [np.asarray(point.get("position", point) if isinstance(point, dict) else point,
                        dtype=float) for point in points]
    if not holes:
        holes = [np.asarray(expected)]
    holes = [point for point in holes
             if all(np.linalg.norm(point[:2]-old[:2]) > 0.055 for old in occupied)]
    holes.sort(key=lambda point: float(np.linalg.norm(point[:2]-expected[:2])))
    reference = ctx.state[f"{arm}_arm_joint_state"]
    rotation_point = grasp + [0, 0, 0.300]
    for point in holes:
        for yaw in (0.0, math.pi, math.pi/2, -math.pi/2,
                    -math.pi/6, math.pi/6, -math.pi/4, math.pi/4, -math.pi/3, math.pi/3):
            # Rotation about the upright tube axis preserves tip and cap heights.
            candidate = _multiply(_yaw_quaternion(yaw), quaternion)
            surface = point + [0, 0, pinch_height + 0.003]
            seated = surface - [0, 0, 0.051]
            contact_results = [kinematics.solve(arm, np.r_[position, candidate], tcp=True,
                                                current_joints=reference, max_starts=6)
                               for position in (rotation_point, surface, seated)]
            if any(result["status"] != "Success" or
                   _public_self_margin(ctx, arm, result["joint_value"]) < 0.001 or
                   occupied_tube_margin(kinematics, model, arm, result["joint_value"], occupied) < 0.004
                   for result in contact_results):
                continue
            for clearance in (0.160, 0.100, 0.080):
                above = point + [0, 0, pinch_height + clearance]
                result = kinematics.solve(arm, np.r_[above, candidate], tcp=True,
                                          current_joints=reference, max_starts=6)
                if result["status"] == "Success" and _public_self_margin(ctx, arm, result["joint_value"]) > 0.001:
                    ctx.memory[f"tube_occupied_choice_{len(occupied)}"] = {
                        "yaw": yaw, "hole": point.tolist(),
                        "occupied_count": len(occupied)}
                    return point.copy(), candidate, above
    # Preserve the measured target if no public-kinematics candidate is exact.
    point = _hole(rack, expected)
    return point, quaternion, point + [0, 0, pinch_height + 0.160]


def _tube_retry(ctx, arm, hole, seated, quaternion, disk, measured_tcp, index):
    """Retry a proprioceptively blocked descent using the observed tube tip."""
    actual = np.asarray(ctx.tcp(arm))
    diagnostic = {"blocked_height": float(actual[2]-seated[2]), "attempted": False}
    ctx.memory[f"tube_retry_{index}"] = diagnostic
    # Reserve the optional retry's 12 frames and the final release, home and settle path.
    if index == 2 and ctx.steps > 452:
        diagnostic["reason"] = "final_frame_reserve"
        return seated
    if actual[2]-seated[2] < .015 or disk is None:
        return seated
    normal = np.asarray(disk["normal"])
    tip_offset = np.asarray(disk["center"])-TUBE_LENGTH*normal-measured_tcp
    surface = np.asarray(hole)-tip_offset+[0, 0, .003]
    corrected = surface-[0, 0, .051]
    delta = corrected-seated
    diagnostic["tip_offset"] = tip_offset.tolist()
    diagnostic["correction"] = delta.tolist()
    # Limit the retry to a small registration error, retaining the original heading.
    if np.linalg.norm(delta[:2]) > .008 or abs(delta[2]) > .008:
        diagnostic["reason"] = "large_correction"
        return seated
    kinematics = getattr(ctx, "kinematics", None) or getattr(ctx, "precision_kinematics", None)
    if kinematics is not None:
        reference = ctx.state[f"{arm}_arm_joint_state"]
        candidates = [kinematics.solve(arm, np.r_[position, quaternion], tcp=True,
                                       current_joints=reference, max_starts=8)
                      for position in (surface, corrected)]
        if any(result["status"] != "Success" or
               _public_self_margin(ctx, arm, result["joint_value"]) < .001
               for result in candidates):
            diagnostic["reason"] = "unreachable"
            return seated
    diagnostic["attempted"] = True
    ctx.stage = f"tube_retry_lift_{index}"
    yield from ctx.move(arm, actual+[0, 0, .010], quaternion, 0.0, 3)
    ctx.stage = f"tube_retry_align_{index}"
    yield from ctx.move(arm, surface, quaternion, 0.0, 3)
    ctx.stage = f"tube_retry_insert_{index}"
    yield from ctx.move(arm, corrected, quaternion, 0.0, 6)
    diagnostic["remaining_height"] = float(ctx.tcp(arm)[2]-corrected[2])
    return corrected


def _held_tube_cap_impl(ctx, arm, expected, index=None):
    """Triangulate the same cap top from simultaneous head and wrist RGB."""
    import cv2
    rays = []
    diagnostic = {"expected": np.asarray(expected).tolist(), "pixels": {},
                  "examined_cameras": [], "reason": "pending"}
    ctx.memory[f"tube_cap_measure_{index if index is not None else arm}"] = diagnostic
    for camera in ("cam_head", f"cam_{arm}_wrist"):
        if camera not in ctx.obs["vision"]:
            diagnostic["reason"] = f"missing_{camera}"
            return None
        image = np.asarray(ctx.obs["vision"][camera]["color"])[..., :3]
        diagnostic['examined_cameras'].append(camera)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        red, green, blue = np.moveaxis(image.astype(float), -1, 0)
        mask = (((hsv[..., 0] > 10) & (hsv[..., 0] < 35) &
                 (hsv[..., 1] > 30) & (hsv[..., 1] < 190) &
                 (red > 200) & (green > 130) & (blue > 65) &
                 (red > green*1.04) & (green > blue*1.06)).astype(np.uint8) * 255)
        # Preserve pale cap pixels whose warm color is washed out by lighting.
        warm_top = ((hsv[..., 0] > 5) & (hsv[..., 0] < 40) &
                    (hsv[..., 1] > 18) & (hsv[..., 1] < 120) &
                    (red > 200) & (green > 180) & (blue > 140) &
                    (red > green*1.03) & (green > blue*1.025))
        mask[warm_top] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        prediction = ctx.perception.world_to_pixel(expected, camera, ctx.state, image.shape)
        choices = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 12:
                continue
            moments = cv2.moments(contour)
            pixel = np.array([moments["m10"]/moments["m00"], moments["m01"]/moments["m00"]])
            distance = float(np.linalg.norm(pixel-prediction))
            if camera == "cam_head" and distance > 45:
                continue
            # The held cap dominates the near-field wrist image.
            score = distance if camera == "cam_head" else -area
            choices.append((score, contour))
        if not choices:
            diagnostic["reason"] = f"no_cap_{camera}"
            return None
        contour = min(choices, key=lambda pair: pair[0])[1]
        component = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(component, [contour], -1, 255, -1)
        # The pale top is less saturated than the orange ribbed side wall.
        # A blue-channel quantile can mix the side wall with a partly masked top.
        saturation = hsv[..., 1]
        threshold, _ = cv2.threshold(saturation[component > 0], 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        top = (component > 0) & (saturation < threshold)
        diagnostic.setdefault("top_saturation_threshold", {})[camera] = float(threshold)
        rows, columns = np.nonzero(top)
        if len(rows) < 8:
            diagnostic["reason"] = f"no_top_{camera}"
            return None
        pixel = np.array([columns.mean(), rows.mean()])
        diagnostic["pixels"][camera] = pixel.tolist()
        origin, rotation, fx, fy = ctx.perception.camera_geometry(camera, ctx.state, image.shape)
        ray = rotation @ np.array([(pixel[0]-image.shape[1]/2)/fx,
                                   -(pixel[1]-image.shape[0]/2)/fy, -1.0])
        rays.append((origin, _unit(ray), pixel.tolist()))
    first, first_ray, first_pixel = rays[0]
    second, second_ray, second_pixel = rays[1]
    if abs(float(np.dot(first_ray, second_ray))) > 0.98:
        diagnostic["reason"] = "parallel_rays"
        return None
    distances = np.linalg.lstsq(np.column_stack([first_ray, -second_ray]), second-first, rcond=None)[0]
    if np.min(distances) <= 0:
        diagnostic["reason"] = "negative_depth"
        return None
    first_point = first + distances[0]*first_ray
    second_point = second + distances[1]*second_ray
    gap = float(np.linalg.norm(first_point-second_point))
    diagnostic["ray_gap"] = gap
    diagnostic["reason"] = "matched" if gap <= 0.007 else "ray_gap"
    if gap > 0.007:
        return None
    return (first_point+second_point)/2, gap, first_pixel, second_pixel


def _held_tube_cap(ctx, arm, expected, index=None):
    """Keep existing cap triangulation and record its limited geometric evidence."""
    result = _held_tube_cap_impl(ctx, arm, expected, index)
    diagnostic = ctx.memory.get(f'tube_cap_measure_{index if index is not None else arm}', {})
    cameras = tuple(diagnostic.get('examined_cameras', diagnostic.get('pixels', {})))
    if hasattr(ctx, 'steps') and cameras:
        object_id = 'tube_geometry_'+str(index if index is not None else arm)
        evidence = visual_evidence(ctx, object_id, 'geometry', accepted=result is not None,
            source='rgb_tube_multiview_cap_geometry', cameras=cameras,
            metrics=dict(diagnostic=diagnostic,
                         cap_position_m=None if result is None else np.asarray(result[0]).tolist(),
                         ray_gap_m=diagnostic.get('ray_gap'), ray_gap_limit_m=.007,
                         ray_gap_is_position_error_bound=False,
                         position_error_m=None, retention_verified=False),
            reason='' if result is not None else diagnostic.get('reason', 'Cap geometry was not resolved'),
            limits='Ray agreement supports cap association; it is not a calibrated position bound or retention proof.')
        verification(ctx, 'tube_cap_geometry', object_id, evidence, 'geometry')
    return result


def _tube_low_recheck(ctx, arm, hole, surface, quaternion, disk, measured_tcp, occupied, index):
    """Refresh the held transform after transport, before touching the rack."""
    from .precision_geometry import tube_cap_disk_pose, tube_tip_registration, occupied_tube_margin
    ctx.stage = f"tube_low_measure_{index}"
    yield from ctx.hold(3)
    actual = np.asarray(ctx.tcp(arm))
    expected = actual + (np.asarray(disk["center"])-measured_tcp) if disk is not None else \
        actual + [0, 0, TUBE_LENGTH-(surface[2]-hole[2]-.003)]
    cap = _held_tube_cap(ctx, arm, expected, f"low_{index}")
    diagnostic = {"accepted": False, "tcp": actual.tolist()}
    ctx.memory[f"tube_low_registration_{index}"] = diagnostic
    if cap is None:
        diagnostic["reason"] = "missing_cap"
        return surface, quaternion, disk, measured_tcp
    observed_disk = tube_cap_disk_pose(ctx, arm, cap[0])
    ctx.memory[f"tube_low_disk_{index}"] = observed_disk or {"matched": False}
    nominal = np.asarray(cap[0])-actual-[0, 0, TUBE_LENGTH]
    registration = tube_tip_registration(observed_disk, actual, nominal, upright=True)
    if registration is None:
        diagnostic["reason"] = "invalid_registration"
        return surface, quaternion, disk, measured_tcp
    adjusted = np.asarray(hole)-np.asarray(registration["tip_offset"])+[0, 0, .003]
    adjusted_quaternion = _multiply(registration["rotation"], quaternion)
    kinematics = getattr(ctx, "kinematics", None) or getattr(ctx, "precision_kinematics", None)
    if kinematics is None:
        from coap.kinematics import X5Kinematics
        kinematics = X5Kinematics(ctx.initial_state)
    # Correct above the surface so the observed tilted tip cannot scrape the rim.
    goals = (adjusted+[0, 0, .012], adjusted, adjusted-[0, 0, .051])
    results = [kinematics.solve(arm, np.r_[goal, adjusted_quaternion], tcp=True,
                                current_joints=ctx.state[f"{arm}_arm_joint_state"], max_starts=8)
               for goal in goals]
    accepted = all(result["status"] == "Success" and
                   _public_self_margin(ctx, arm, result["joint_value"]) >= .001 and
                   occupied_tube_margin(kinematics, ctx.precision_collision_geometry, arm,
                                        result["joint_value"], occupied) >= .004 for result in results)
    diagnostic.update(registration, accepted=accepted)
    if not accepted:
        diagnostic["reason"] = "unreachable_or_collision"
        return surface, quaternion, disk, measured_tcp
    ctx.stage = f"tube_low_rotate_{index}"
    yield from ctx.move(arm, goals[0], adjusted_quaternion, 0.0, 2)
    ctx.stage = f"tube_low_align_{index}"
    yield from ctx.move(arm, adjusted, adjusted_quaternion, 0.0, 2)
    rotated_disk = dict(observed_disk,
                        center=(actual+_rotate(registration["rotation"],
                                np.asarray(observed_disk["center"])-actual)).tolist(),
                        normal=_rotate(registration["rotation"], observed_disk["normal"]).tolist())
    return adjusted, adjusted_quaternion, rotated_disk, actual


def _home(ctx, arm, duration=20):
    yield from ctx.home(arm, duration=duration)


def _grasp(ctx, arm, position, quaternion, lift=0.15, fast=False, timings=None):
    above = position + [0, 0, lift]
    timings = timings or ((14, 10, 9, 15) if fast else (18, 14, 12, 20))
    yield from ctx.move(arm, above, quaternion, 1.0, timings[0])
    yield from ctx.move(arm, position, quaternion, 1.0, timings[1])
    yield from ctx.move(arm, position, quaternion, 0.0, timings[2])
    yield from ctx.move(arm, above, quaternion, 0.0, timings[3])


def _key_bow(ctx, key):
    """Use the widest white region to distinguish the bow from the blade."""
    contour = key.get("contour")
    if contour is None or len(contour) < 6:
        return None
    import cv2
    camera = key.get("camera", "cam_head")
    shape = np.asarray(ctx.obs["vision"][camera]["color"]).shape[:2]
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.asarray(contour, np.int32).reshape(-1, 1, 2)
    cv2.drawContours(mask, [points], -1, 255, -1)
    _, radius, _, bow_pixel = cv2.minMaxLoc(cv2.distanceTransform(mask, cv2.DIST_L2, 5))
    if radius < 2:
        return None
    bow = np.asarray(ctx.project(bow_pixel, z=ctx.table_z + 0.010, camera=camera))
    world = np.array([ctx.project(point, z=ctx.table_z + 0.010, camera=camera)
                      for point in np.asarray(contour)])
    tip = world[np.argmax(np.linalg.norm(world[:, :2] - bow[:2], axis=1))]
    delta = bow - tip
    if not 0.035 < np.linalg.norm(delta[:2]) < 0.080:
        return None
    return bow, tip, math.atan2(delta[1], delta[0])


def insert_tubes(ctx):
    """Pick horizontal tubes, rotate the narrow end down, and insert separately."""
    ctx.stage = "tube_measure"
    rack = _best(ctx, "tube_rack")
    if rack is None:
        yield from ctx.hold(12)
        rack = _best(ctx, "tube_rack")
    if rack is None:
        return
    rack_yaw = _observed_yaw(rack)
    rack_center = _position(rack)
    rack_center[2] = ctx.table_z + 0.030
    rack_rotation = _yaw_quaternion(rack_yaw)
    # Seat the center first so two occupied outer holes cannot trap the fingers.
    holes = ((0.0, -0.018), (-0.072, -0.018), (0.072, -0.018))
    used_positions = []
    occupied_holes = []
    def visible_sources(index, phase):
        accepted, review = [], []
        for item in ctx.detect("tube"):
            point = _position(item)
            reasons = []
            if any(np.linalg.norm(point[:2]-old[:2]) <= .045 for old in used_positions):
                reasons.append("used_source")
            if abs(point[0]) >= .435 or not -.285 < point[1] < .025:
                reasons.append("outside_source_region")
            if np.linalg.norm(point[:2]-rack_center[:2]) <= .10:
                reasons.append("rack_region")
            review.append({"position": point.tolist(), "bbox": list(item.get("bbox", [])),
                           "rejected": reasons})
            if not reasons:
                accepted.append(item)
        ctx.memory[f"tube_sources_{index}_{phase}"] = review
        return accepted
    for hole_index, hole in enumerate(holes):
        ctx.stage = f"tube_measure_{hole_index}"
        tubes = visible_sources(hole_index, "initial")
        if not tubes and hole_index == 2:
            # Home fingers can merge with the remaining tube's white silhouette.
            from .precision_planning import tube_visibility_plan
            plan, diagnostic = tube_visibility_plan(ctx)
            ctx.memory["tube_visibility_scout"] = diagnostic
            if plan is not None:
                positions, quaternions = plan
                ctx.stage = "tube_visibility_scout"
                yield from ctx.move_both(positions["left"], positions["right"],
                                         quaternions["left"], quaternions["right"], 1., 1., 8)
                ctx.stage = "tube_visibility_refresh"
                yield from ctx.hold(3)
                tubes = visible_sources(hole_index, "refreshed")
        if not tubes:
            break
        expected_hole = rack_center + _rotate(rack_rotation, [hole[0], hole[1], 0.028])
        tube = min(tubes, key=lambda item: abs(_position(item)[0] - expected_hole[0]))
        center = _position(tube)
        center[2] = ctx.table_z + 0.0167
        tube_yaw = _observed_yaw(tube, directed=True)
        tube_axis = _direction(tube_yaw)
        # Pinch the upper straight section while keeping the cap in the wrist view.
        grasp_offset = 0.040
        grasp = center + tube_axis * grasp_offset
        grasp[2] += 0.002
        arm = _arm(ctx, grasp)
        initial_gripper = _reachable_down(ctx, arm, tube_yaw)
        pinch_height = TUBE_LENGTH / 2 + grasp_offset
        if "bottom_position" in tube:
            measured = float(np.dot(grasp - np.asarray(tube["bottom_position"]), tube_axis))
            if 0.055 < measured < 0.13:
                pinch_height = measured
        from .precision_planning import tube_grasp_plan
        ctx.stage = f"tube_grasp_preflight_{hole_index}"
        initial_gripper, preflight = tube_grasp_plan(
            ctx, arm, grasp, tube_axis, initial_gripper, _hole(rack, expected_hole),
            rack, pinch_height, occupied_holes)
        ctx.memory[f"tube_grasp_preflight_{hole_index}"] = preflight
        ctx.stage = f"tube_grasp_{hole_index}"
        yield from _grasp(ctx, arm, grasp, initial_gripper, lift=0.09, fast=True,
                          timings=(10, 10, 7, 12))
        used_positions.append(center.copy())
        upright_rotation = _axis_rotation(np.cross(tube_axis, [0, 0, 1]), math.pi / 2)
        final_gripper = _multiply(upright_rotation, initial_gripper)
        hole_position = rack_center + _rotate(rack_rotation, [hole[0], hole[1], 0.028])
        hole_position = _hole(rack, hole_position)
        hole_position, final_gripper, above = _tube_targets(
            ctx, arm, grasp, final_gripper, hole_position, rack, pinch_height, occupied_holes)
        occupied_holes.append(hole_position.copy())
        ctx.memory[f"tube_target_{hole_index}"] = {
            "hole": hole_position.tolist(), "quaternion": final_gripper.tolist(),
            "transport": above.tolist(), "pinch_height": pinch_height}
        ctx.stage = f"tube_rotate_{hole_index}"
        # Rotate at the source before transport; TCP interpolation can dip 10 cm.
        rotation_point = grasp + [0, 0, 0.300]
        yield from ctx.move(arm, rotation_point, final_gripper, 0.0, 11)
        yield from ctx.move(arm, rotation_point, final_gripper, 0.0, 2)
        ctx.stage = f"tube_transport_{hole_index}"
        yield from ctx.move(arm, above, final_gripper, 0.0, 15)
        ctx.stage = f"tube_held_measure_{hole_index}"
        # Keep both image and proprioception timestamps stable before triangulation.
        yield from ctx.hold(3)
        measured_tcp_pose = np.asarray(ctx.state[f"{arm}_ee_pose"])
        measured_tcp = measured_tcp_pose[:3] + _rotate(measured_tcp_pose[3:], [0.145, 0, 0])
        cap = _held_tube_cap(ctx, arm, measured_tcp + [0, 0, TUBE_LENGTH-pinch_height], hole_index)
        lateral_offset = np.zeros(3)
        if cap is not None:
            cap_position, ray_gap, head_pixel, wrist_pixel = cap
            offset = cap_position-measured_tcp
            corrected_pinch = TUBE_LENGTH-offset[2]
            if np.linalg.norm(offset[:2]) < 0.040 and 0.055 < corrected_pinch < 0.130:
                pinch_height = corrected_pinch
                lateral_offset[:2] = offset[:2]
                ctx.memory[f"tube_held_{hole_index}"] = {
                    "cap": cap_position.tolist(), "tcp": measured_tcp.tolist(),
                    "ray_gap": ray_gap, "head_pixel": head_pixel, "wrist_pixel": wrist_pixel,
                    "lateral_offset": lateral_offset.tolist(), "pinch_height": pinch_height}
        from .precision_geometry import tube_cap_disk_pose
        disk = tube_cap_disk_pose(ctx, arm, np.asarray(cap[0])) if cap is not None else None
        ctx.memory[f"tube_disk_{hole_index}"] = disk if disk is not None else {"matched": False}
        surface = hole_position - lateral_offset + [0, 0, pinch_height + 0.003]
        from .precision_geometry import tube_tip_registration
        registration = tube_tip_registration(disk, measured_tcp,
                                               lateral_offset-[0, 0, pinch_height], upright=True)
        if registration is not None:
            kinematics = getattr(ctx, "kinematics", None) or getattr(ctx, "precision_kinematics", None)
            if kinematics is None:
                from coap.kinematics import X5Kinematics
                kinematics = X5Kinematics(ctx.initial_state)
            registered_surface = hole_position-np.asarray(registration["tip_offset"])+[0, 0, .003]
            registered_gripper = _multiply(registration["rotation"], final_gripper)
            candidates = [kinematics.solve(arm, np.r_[point, registered_gripper], tcp=True,
                                              current_joints=ctx.state[f"{arm}_arm_joint_state"],
                                              max_starts=8)
                          for point in (registered_surface, registered_surface-[0, 0, .051])]
            from .precision_geometry import occupied_tube_margin
            accepted = all(result["status"] == "Success" and
                           _public_self_margin(ctx, arm, result["joint_value"]) >= .001 and
                           occupied_tube_margin(kinematics, ctx.precision_collision_geometry, arm,
                                                result["joint_value"], occupied_holes[:-1]) >= .004
                           for result in candidates)
            registration["accepted"] = accepted
            ctx.memory[f"tube_tip_registration_{hole_index}"] = registration
            if accepted:
                surface = registered_surface
                final_gripper = registered_gripper
                disk = dict(disk, center=(measured_tcp + _rotate(registration["rotation"],
                                np.asarray(disk["center"])-measured_tcp)).tolist(),
                            normal=_rotate(registration["rotation"], disk["normal"]).tolist())
        ctx.stage = f"tube_align_{hole_index}"
        yield from ctx.move(arm, surface, final_gripper, 0.0, 14)
        surface, final_gripper, disk, measured_tcp = yield from _tube_low_recheck(
            ctx, arm, hole_position, surface, final_gripper, disk, measured_tcp,
            occupied_holes[:-1], hole_index)
        seated = surface - [0, 0, 0.051]
        ctx.stage = f"tube_insert_{hole_index}"
        yield from ctx.move(arm, seated, final_gripper, 0.0, 21)
        seated = yield from _tube_retry(ctx, arm, hole_position, seated, final_gripper,
                                        disk, measured_tcp, hole_index)
        ctx.stage = f"tube_release_{hole_index}"
        yield from ctx.move(arm, seated, final_gripper, 1.0, 6)
        yield from ctx.move(arm, seated + [0, 0, 0.15], final_gripper, 1.0, 14)
        ctx.stage = f"tube_home_{hole_index}"
        final_scout_home = hole_index == 2 and ctx.memory.get("tube_visibility_scout", {}).get("accepted")
        yield from _home(ctx, arm, duration=10 if final_scout_home else 12)
    ctx.stage = "tube_settle"
    yield from ctx.hold(3)


def insert_key(ctx):
    """Locate the key in RGB and transfer its blade grasp to the flat bow faces."""
    from coap.tasks.key_handoff import run
    ctx.stage = "key_measure"
    key = _best(ctx,"key")
    slot = _best(ctx,"key_slot")
    if key is None or slot is None:
        yield from ctx.hold(10)
        key, slot = _best(ctx,"key"), _best(ctx,"key_slot")
    if key is None or slot is None:
        raise RuntimeError("Key and slot must both be visually resolved")
    yield from run(ctx,key,slot)


def _mug_handle(mug):
    if "handle_local" in mug:
        return np.asarray(mug["handle_local"], dtype=float)
    asset_index = mug.get("asset_index")
    if asset_index is not None and 0 <= int(asset_index) < len(MUG_HANDLES):
        return np.asarray(MUG_HANDLES[int(asset_index)], dtype=float)
    size = mug.get("size", [0.105, 0.080, 0.073])
    width = float(max(size[:2]))
    index = min(range(len(MUG_WIDTHS)), key=lambda number: abs(MUG_WIDTHS[number] - width))
    return np.asarray(MUG_HANDLES[index], dtype=float)


def _mug_body_frame(ctx, mug, height):
    """Estimate the circular body separately from the protruding handle."""
    center = _position(mug)
    body_offset = 0.013 if mug.get("asset_index") == 5 else 0.012
    yaw = float(mug.get("handle_yaw", _observed_yaw(mug, directed=True)))
    if "body_position" in mug:
        body = np.asarray(mug["body_position"], dtype=float)
        if 'body_origin_offset' in mug:
            offset = np.r_[np.asarray(mug['body_origin_offset'],float),0.]
            center[:2] = body[:2]+_rotate(_yaw_quaternion(yaw),offset)[:2]
        else:
            center[:2] = body[:2] + _direction(yaw)[:2] * body_offset
        return center, yaw
    contour = mug.get("contour")
    if contour is None or len(contour) < 6:
        return center, yaw
    import cv2
    camera = mug.get("camera", "cam_head")
    vision = ctx.obs["vision"][camera]
    image = np.asarray(vision["color"])
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
    cv2.drawContours(mask, [points], -1, 255, -1)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, radius, _, pixel = cv2.minMaxLoc(distance)
    if radius < 5:
        return center, yaw
    surface_z = ctx.table_z + height
    body = np.asarray(ctx.project(pixel, z=surface_z, camera=camera), dtype=float)
    world = np.array([ctx.project(point, z=surface_z, camera=camera)[:2]
                      for point in np.asarray(contour)])
    radial = world - body[:2]
    radii = np.linalg.norm(radial, axis=1)
    outer = radial[radii >= np.quantile(radii, 0.92)]
    direction = np.median(outer, axis=0)
    hsv = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2HSV)
    u, v = pixel
    if hsv[v, u, 1] < 80:
        # White mugs can have a detached red handle in the thresholded silhouette.
        x, y, width, box_height = mug["bbox"]
        expanded = np.zeros(mask.shape, dtype=np.uint8)
        expanded[max(0, y-12):y+box_height+12, max(0, x-15):x+width+15] = 255
        red = (((hsv[..., 0] < 8) | (hsv[..., 0] > 172)) &
               (hsv[..., 1] > 180) & (hsv[..., 2] > 100) & (expanded > 0) &
               (image[..., 0].astype(float) > 2.5 * image[..., 1]))
        yy, xx = np.nonzero(red)
        if len(xx) >= 8:
            handle_pixel = [float(np.median(xx)), float(np.median(yy))]
            handle_position = ctx.project(handle_pixel, z=surface_z, camera=camera)
            direction = np.asarray(handle_position)[:2] - body[:2]
    if np.linalg.norm(direction) > 0.025:
        yaw = math.atan2(direction[1], direction[0])
    # Cup models place the body slightly opposite the local positive X handle.
    center[:2] = body[:2] + body_offset * _direction(yaw)[:2]
    return center, yaw


def _red_mug_profile(ctx, mug):
    """Recognize distinctive public cup colors and matching body dimensions."""
    if mug.get("asset_index", -1) >= 10:
        return mug
    color = np.asarray(mug.get("mean_rgb", [0, 0, 0]))
    width = float(mug.get("size", [0.0])[0])
    if color[0] > 170 and color[1] > 150 and color[2] < 60 and 0.085 < width < 0.120:
        # The yellow asset's flat outer handle wall lies at local X = 46.5 mm.
        return dict(mug, asset_index=5, size=[0.09913398105, 0.07529970170, 0.069],
                    handle_local=[0.047, 0.0, 0.0], handle_grasp=True)
    if np.max(color) <= 170 or np.ptp(color) >= 80:
        return mug
    expected = _position(mug)
    expected[2] = ctx.table_z + 0.0343 + 0.005
    observed = _red_handle_position(ctx, expected)
    if observed is None:
        return mug
    # The physical opening is inward of the asset's outer-wall reward point.
    return dict(mug, asset_index=8, size=[0.1205, 0.0929, 0.0686],
                handle_local=[0.044, 0.0, 0.006], handle_grasp=True)


def _mug_frame(ctx, mug, height):
    """Locate the body and red handle at their separate public heights."""
    center, yaw = _mug_body_frame(ctx, mug, height)
    if mug.get("asset_index") == 8 and mug.get("handle_grasp") and mug.get("camera", "cam_head") == "cam_head":
        expected = center.copy()
        expected[2] = ctx.table_z + 0.0343 + 0.005
        observed = _red_handle_position(ctx, expected)
        if observed is not None:
            body = center - 0.012 * _direction(yaw)
            delta = observed[:2] - body[:2]
            if np.linalg.norm(delta) > 0.025:
                yaw = math.atan2(delta[1], delta[0])
                center[:2] = observed[:2] - 0.046891 * _direction(yaw)[:2]
    return center, yaw


def _round_body(mug):
    contour = mug.get("contour")
    if contour is None:
        return True
    import cv2
    points = np.asarray(contour, np.int32).reshape(-1, 1, 2)
    x, y, width, height = cv2.boundingRect(points)
    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    points = points - [x - 1, y - 1]
    cv2.drawContours(mask, [points], -1, 255, -1)
    _, radius, _, _ = cv2.minMaxLoc(cv2.distanceTransform(mask, cv2.DIST_L2, 5))
    area_ratio = cv2.contourArea(points) / max(math.pi * radius * radius, 1.0)
    return radius >= 6 and 0.7 < area_ratio < 4.0


def _match_mug(ctx, source, camera="cam_head"):
    anchor = _position(source)
    source_color = source.get("mean_rgb")
    candidates = []
    max_distance = 0.025 if "wrist" in camera else 0.075
    for candidate in ctx.detect("mug", camera=camera):
        distance = np.linalg.norm(_position(candidate)[:2] - anchor[:2])
        if distance > max_distance or not _round_body(candidate):
            continue
        if source_color is not None and "mean_rgb" in candidate:
            color_distance = np.linalg.norm(np.asarray(source_color) - np.asarray(candidate["mean_rgb"])) / math.sqrt(3)
            if color_distance > 65:
                continue
        candidates.append((distance, candidate))
    return min(candidates, key=lambda pair: pair[0])[1] if candidates else source



def _mug_pose_review(ctx, arm, position, quaternion):
    """Reuse identical public IK checks across alternate rotation staging heights."""
    kinematics = ctx.kinematics
    other = 'right' if arm == 'left' else 'left'
    reference = np.asarray(ctx.state[f'{arm}_arm_joint_state'],float)
    other_joints = np.asarray(ctx.state[f'{other}_arm_joint_state'],float)
    target = np.r_[position,quaternion]
    if target[3]<0:
        target[3:] *= -1
    key = (id(kinematics),id(getattr(ctx,"precision_collision_geometry",None)),float(ctx.table_z),
           arm,*reference.tolist(),*other_joints.tolist(),*target.tolist())
    cache = getattr(ctx,'_mug_pose_cache',None)
    if cache is None:
        cache = {}
        ctx._mug_pose_cache = cache
    if key not in cache:
        result = kinematics.solve(arm,target,tcp=True,current_joints=reference,max_starts=8)
        clearance = kinematics.public_clearance(arm,result['joint_value'],ctx.table_z,other,other_joints)
        if (result['status'] == 'Success' and min(clearance.values()) < .001
                and hasattr(kinematics, 'base_transforms')):
            from coap.robot_motion_geometry import check_joint_path
            both = {side: np.asarray(ctx.state[side+'_arm_joint_state']) for side in ('left', 'right')}
            both[arm] = np.asarray(result['joint_value'])
            joints = np.concatenate([both[side] for side in ('left', 'right')])
            complete = check_joint_path(kinematics, np.stack([joints, joints]), ctx.table_z)
            if complete['accepted']:
                margins = complete['minimum_margin_m']
                clearance = {'table': margins['table'], 'self': margins['self'], 'other_arm': margins['cross']}
        self_margin = (_public_self_margin(ctx,arm,result['joint_value'])
                       if result['status']=='Success' else None)
        if len(cache)>=512:
            cache.pop(next(iter(cache)))
        cache[key] = (result,clearance,self_margin)
    return cache[key]

def _mug_support(ctx, arm, center, initial_rotation, grasp_rotation, grasp_local,
                 handle_local, holder_center, holder_rotation, used_pegs, handle_grasp=False,
                 peg_frames=None, staging=None, return_options=False, handle_frames=None):
    from coap.kinematics import X5Kinematics
    kinematics = getattr(ctx, "kinematics", None) or X5Kinematics(ctx.initial_state)
    if getattr(ctx,"kinematics",None) is None:
        ctx.kinematics = kinematics
    relative = _multiply(_inverse(initial_rotation), grasp_rotation)
    if handle_frames is None:
        handle_rotations=([0.5,0.5,0.5,0.5],[0.5,-0.5,0.5,-0.5])
    else:
        handle_rotations=[]
        for frame in handle_frames:
            frame=np.asarray(frame,float)
            if (frame.shape!=(7,) or not np.isfinite(frame).all() or
                    np.linalg.norm(frame[3:])<1e-8 or
                    np.linalg.norm(frame[:3]-np.asarray(handle_local))>1e-6):
                raise RuntimeError('Native cup thread frame and point are inconsistent')
            handle_rotations.append(_unit(frame[3:]))
        if not handle_rotations:
            raise RuntimeError('No verified native cup thread frame is available')
    options = []
    rejected_poses = []
    for index, (offset, quaternion) in enumerate(peg_frames or PEG_FRAMES + LOWER_PEG_FRAMES):
        if index in used_pegs:
            continue
        point = holder_center + _rotate(holder_rotation, offset)
        peg_rotation = _multiply(holder_rotation, quaternion)
        peg_axis = _rotate(peg_rotation, [1, 0, 0])
        for handle_frame in handle_rotations:
            for roll in ((0.0, math.radians(50), -math.radians(50), math.pi/2, -math.pi/2) if handle_grasp else (0.0,)):
                final_rotation = _multiply(_axis_rotation(peg_axis, roll), _multiply(peg_rotation, _inverse(handle_frame)))
                final_gripper = _multiply(final_rotation, relative)
                seated = point + _rotate(final_rotation, grasp_local - handle_local)
                preinsert = seated + 0.075 * peg_axis + [0, 0, 0.05]
                reference = ctx.state[f"{arm}_arm_joint_state"]
                rotation_staging = staging if staging is not None else (np.array([-0.18 if arm == "left" else 0.18, -0.30, 0.94]) if handle_grasp else np.array([-0.24 if arm == "left" else 0.24, -0.18, 0.94]))
                poses = ((preinsert,seated+.045*peg_axis,seated) if staging is False else
                         (rotation_staging,preinsert,seated+.045*peg_axis,seated))
                pose_kinds = ('preinsert','aligned','seat') if staging is False else ('staging','preinsert','aligned','seat')
                checked = []
                for position in poses:
                    review = _mug_pose_review(ctx,arm,position,final_gripper)
                    checked.append(review)
                    result,clearance,self_margin = review
                    if result['status']!='Success' or min(clearance.values())<.001 or self_margin is None or self_margin<.001:
                        break
                results,clearances,self_margins = map(list,zip(*checked))
                minimum_clearance = min(min(values.values()) for values in clearances)
                feasible = len(checked)==len(poses) and all(result['status']=='Success' for result in results) and minimum_clearance>=.001 and all(value is not None and value>=.001 for value in self_margins)
                if not feasible:
                    rejected_poses.append({'peg_index':index,'roll_rad':roll,
                        'staging_height_m':None if staging is False else float(rotation_staging[2]),
                        'poses':[{'kind':kind,'position':np.asarray(position).tolist(),
                                  'status':result['status'],'position_error_m':result['position_error'],
                                  'rotation_error_rad':result.get('rotation_error'),
                                  'clearance_m':clearance,'self_margin_m':self_margin}
                                 for kind,position,result,clearance,self_margin in zip(
                                     pose_kinds,poses,results,clearances,self_margins)]})
                error = sum(result["position_error"] for result in results) + max(0.0, -minimum_clearance)
                distance = float(np.linalg.norm(point[:2] - center[:2]))
                options.append((not feasible, error if not feasible else 0.0, distance,
                                index, point, peg_axis, final_gripper, seated, preinsert))
    ctx.memory['mug_support_pose_candidates'] = {'rejected':rejected_poses,
        'feasible_count':sum(not option[0] for option in options), 'private_task_state_used':False}
    feasible_options = sorted((option for option in options if not option[0]),
                              key=lambda option: option[:3])
    if not feasible_options:
        ctx.memory['mug_support_rejections'] = {
            'candidates': len(options), 'public_geometry_margin_m': .001,
            'no_infeasible_pose_selected': True,
            'smallest_rejected_error': None if not options else min(option[1] for option in options)}
        raise RuntimeError('No mug support pose passed public IK and geometry checks')
    return feasible_options if return_options else feasible_options[0]


def _red_handle_position(ctx, expected):
    import cv2
    image = np.asarray(ctx.obs["vision"]["cam_head"]["color"])[..., :3]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    red = (((hsv[..., 0] < 8) | (hsv[..., 0] > 172)) &
           (hsv[..., 1] > 180) & (hsv[..., 2] > 100) &
           (image[..., 0].astype(float) > 2.5 * image[..., 1])).astype(np.uint8) * 255
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    choices = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 12:
            continue
        moments = cv2.moments(contour)
        pixel = [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
        point = np.asarray(ctx.project(pixel, z=float(expected[2]), camera="cam_head"))
        if np.linalg.norm(point[:2] - expected[:2]) < 0.13:
            choices.append((area, point))
    return max(choices, key=lambda choice: choice[0])[1] if choices else None


def _red_handle_rays(ctx, camera):
    """Extract handle ray candidates using the current public camera pose."""
    import cv2
    if camera not in ctx.obs["vision"]:
        return []
    image = np.asarray(ctx.obs["vision"][camera]["color"])[..., :3]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = ((((hsv[..., 0] < 8) | (hsv[..., 0] > 172)) &
             (hsv[..., 1] > 180) & (hsv[..., 2] > 100) &
             (image[..., 0].astype(float) > 2.5*image[..., 1])).astype(np.uint8)*255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    origin, rotation, fx, fy = ctx.perception.camera_geometry(camera, ctx.state, image.shape)
    rays = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 8:
            continue
        moments = cv2.moments(contour)
        pixel = np.array([moments["m10"]/moments["m00"], moments["m01"]/moments["m00"]])
        ray = _unit(rotation @ np.array([(pixel[0]-image.shape[1]/2)/fx,
                                         -(pixel[1]-image.shape[0]/2)/fy, -1.0]))
        rays.append((area, origin.copy(), ray, pixel.tolist()))
    return sorted(rays, key=lambda candidate: -candidate[0])[:4]


def _red_handle_ray_pair(first_candidates, second_candidates, expected):
    """Match the same red loop between two static RGB camera observations."""
    if not first_candidates or not second_candidates:
        return None
    choices = []
    for first_candidate in first_candidates:
        if first_candidate[0] < first_candidates[0][0]*0.6:
            continue
        for second_candidate in second_candidates:
            if second_candidate[0] < second_candidates[0][0]*0.6:
                continue
            _, first, first_ray, first_pixel = first_candidate
            _, second, second_ray, second_pixel = second_candidate
            if abs(float(np.dot(first_ray, second_ray))) > math.cos(math.radians(2)):
                continue
            distances = np.linalg.lstsq(np.column_stack([first_ray, -second_ray]), second-first, rcond=None)[0]
            if np.min(distances) <= 0:
                continue
            one, two = first+distances[0]*first_ray, second+distances[1]*second_ray
            point = (one+two)/2
            gap = float(np.linalg.norm(one-two))
            if gap < 0.007 and np.linalg.norm(point-expected) < 0.12:
                choices.append((gap, point, first_pixel, second_pixel))
    return min(choices, key=lambda choice: choice[0]) if choices else None


def _red_handle_stereo(ctx, camera, expected):
    """Match a red loop between simultaneous head and wrist observations."""
    result = _red_handle_ray_pair(_red_handle_rays(ctx, "cam_head"),
                                  _red_handle_rays(ctx, camera), expected)
    if hasattr(ctx, 'steps'):
        object_id = 'active_mug_handle_geometry'
        evidence = visual_evidence(ctx, object_id, 'geometry', accepted=result is not None,
            source='rgb_mug_multiview_handle_geometry', cameras=tuple(dict.fromkeys(('cam_head', camera))),
            metrics=dict(handle_position_m=None if result is None else np.asarray(result[1]).tolist(),
                         ray_gap_m=None if result is None else float(result[0]),
                         ray_gap_limit_m=.007, ray_gap_is_position_error_bound=False,
                         position_error_m=None, retention_verified=False),
            reason='' if result is not None else 'Current red handle stereo association was not resolved',
            limits='Matched loop geometry does not establish complete mug pose, calibrated metric error or retention.')
        verification(ctx, 'mug_handle_geometry', object_id, evidence, 'geometry')
    return result


def _mug_carry(ctx, arm, target, quaternion, index):
    """Ramp lateral carry speed without changing the measured rigid grasp pose."""
    start = np.asarray(ctx.tcp(arm), dtype=float)
    delta = np.asarray(target, dtype=float)-start
    distance = float(np.linalg.norm(delta))
    count = max(18, int(math.ceil(distance/.018))+8)
    weights = np.minimum(np.minimum(np.arange(1, count+1), np.arange(count, 0, -1)), 9).astype(float)
    total = float(weights.sum())
    fractions = np.cumsum(weights)/total
    diagnostic = {"start": start.tolist(), "target": np.asarray(target).tolist(),
                  "planned_samples": count, "maximum_displacement_per_sample": distance*weights.max()/total,
                  "maximum_displacement_change_per_sample": distance/total}
    ctx.memory[f"mug_carry_{index}"] = diagnostic
    before = ctx.steps
    for fraction in fractions:
        yield from ctx.move(arm, start+fraction*delta, quaternion, 0.0, 1)
    diagnostic["actual_actions"] = ctx.steps-before



def _mug_observer_target(ctx, arm, initial_rotation, grasp_rotation, grasp_local, handle_local):
    """Aim at the nominal rigid handle offset from the actual held tool pose."""
    initial_offset = _rotate(initial_rotation,np.asarray(handle_local)-np.asarray(grasp_local))
    actual_rotation = np.asarray(ctx.state[f'{arm}_ee_pose'],float)[3:]
    correction = _multiply(actual_rotation,_inverse(grasp_rotation))
    return np.asarray(ctx.tcp(arm))+_rotate(correction,initial_offset)


def _mug_origin_height(ctx, mug, size):
    """Preserve a public mesh root offset instead of forcing a bounding-box center."""
    return ctx.table_z+float(mug.get('object_origin_z_offset',float(size[2])/2))

def _mug_observer(ctx, arm, expected, index):
    """Use the unoccupied wrist to view a handle hidden behind its own cup."""
    from coap.kinematics import X5Kinematics
    observer = "right" if arm == "left" else "left"
    sign = 1 if observer == "right" else -1
    from coap.perception import ry, rz
    from scipy.spatial.transform import Rotation
    kinematics = getattr(ctx, "kinematics", None) or X5Kinematics(ctx.initial_state)
    mounting = ry(.3491) @ ry(math.radians(-80)) @ rz(-math.pi/2)
    for distance in (0.20, 0.24, 0.16):
        # Aim the camera at the handle while choosing an unblocked wrist roll.
        origin = np.asarray(expected) + [sign*distance, -0.02, 0.08]
        back = origin-np.asarray(expected)
        back /= np.linalg.norm(back)
        right = np.cross([0, 0, 1], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        camera_rotation = np.column_stack([right, up, back]) @ rz(sign*math.radians(-60))
        tool_rotation = camera_rotation @ mounting.T
        scalar_last = Rotation.from_matrix(tool_rotation).as_quat()
        quaternion = np.r_[scalar_last[3], scalar_last[:3]]
        position = origin + tool_rotation @ np.array([.145-.0845, 0, -.051])
        result = kinematics.solve(observer, np.r_[position, quaternion], tcp=True,
                                   current_joints=ctx.state[f"{observer}_arm_joint_state"], max_starts=8)
        clearance = kinematics.public_clearance(observer, result["joint_value"], ctx.table_z,
                                                arm, ctx.state[f"{arm}_arm_joint_state"])
        if result["status"] != "Success" or min(clearance.values()) < 0.001 or \
                _public_self_margin(ctx, observer, result["joint_value"]) < 0.001:
            continue
        ctx.stage = f"mug_observer_{index}"
        yield from ctx.move(observer, position, quaternion, 1.0, 12)
        # The renderer may lag one control step, so acquire only static views.
        yield from ctx.hold(3)
        camera = f"cam_{observer}_wrist"
        # A held-object measurement is anchored to the actual tool, not a planned peg pose.
        measurement_anchor = np.asarray(ctx.tcp(arm))
        measured = _red_handle_stereo(ctx, camera, measurement_anchor)
        baseline = 0.0
        from .mug_registration import marker_candidates, moving_red_mug_pose
        import copy
        first_snapshot = copy.copy(ctx)
        first_snapshot.obs = copy.deepcopy(ctx.obs)
        first_rays = _red_handle_rays(ctx, camera)
        first_anchor = first_rays[0][3] if first_rays else None
        markers = marker_candidates(ctx, camera, first_anchor) if first_anchor is not None else {}
        identifiable = all(len(markers.get(label, [])) >= count
                           for label, count in ((1, 1), (2, 1), (3, 2), (4, 1)))
        ctx.memory[f"mug_body_fit_{index}"] = {"accepted": False, "reason": "marker_counts",
            "first_feature_counts": {str(label): len(points) for label, points in markers.items()}}
        if measured is None or identifiable:
            second_position = position + [0, -0.025, 0]
            second_result = kinematics.solve(observer, np.r_[second_position, quaternion], tcp=True,
                                              current_joints=ctx.state[f"{observer}_arm_joint_state"], max_starts=8)
            second_clearance = kinematics.public_clearance(observer, second_result["joint_value"], ctx.table_z,
                                                            arm, ctx.state[f"{arm}_arm_joint_state"])
            if first_rays and second_result["status"] == "Success" and min(second_clearance.values()) > 0.001 and \
                    _public_self_margin(ctx, observer, second_result["joint_value"]) > 0.001:
                ctx.stage = f"mug_observer_second_{index}"
                yield from ctx.move(observer, second_position, quaternion, 1.0, 4)
                yield from ctx.hold(3)
                second_rays = _red_handle_rays(ctx, camera)
                if second_rays:
                    baseline = float(np.linalg.norm(first_rays[0][1]-second_rays[0][1]))
                measurement_anchor = np.asarray(ctx.tcp(arm))
                temporal = _red_handle_ray_pair(first_rays, second_rays, measurement_anchor)
                measured = temporal if temporal is not None else measured
                if identifiable and second_rays:
                    body_pose, fit_diagnostic = moving_red_mug_pose(first_snapshot, ctx, camera,
                                                                    [first_anchor, second_rays[0][3]])
                    ctx.memory[f"mug_body_fit_{index}"] = fit_diagnostic
                    if body_pose is not None:
                        ctx.memory[f"mug_body_pose_{index}"] = body_pose
        ctx.memory[f"mug_observer_measure_{index}"] = {
            "matched": measured is not None, "expected": np.asarray(expected).tolist(),
            "measurement_anchor": measurement_anchor.tolist(),
            "camera_baseline": baseline,
            "held_tcp_pose": np.r_[ctx.tcp(arm), np.asarray(ctx.state[f"{arm}_ee_pose"])[3:]].tolist()}
        ctx.stage = f"mug_observer_return_{index}"
        yield from _home(ctx, observer, duration=6)
        return measured
    return None


def _holder_frame(ctx, holder):
    """Fit the six public support centerlines to the initial gold RGB silhouette."""
    import cv2
    from scipy.optimize import least_squares
    center = _position(holder)
    center[2] = ctx.table_z + 0.1744
    image = np.asarray(ctx.obs["vision"]["cam_head"]["color"])[..., :3]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    base_pixel = holder.get("base_pixel", holder.get("pixel"))
    if base_pixel is None:
        return center, _observed_yaw(holder)
    u, v = base_pixel
    region = np.zeros(image.shape[:2], np.uint8)
    region[:max(0, int(v)-20), max(0, int(u)-130):min(image.shape[1], int(u)+150)] = 255
    mask = (((hsv[..., 0] > 15) & (hsv[..., 0] < 45) &
             (hsv[..., 1] > 35) & (hsv[..., 2] > 130) & (region > 0))
            .astype(np.uint8) * 255)
    if np.count_nonzero(mask) < 100:
        return center, _observed_yaw(holder)
    distances = cv2.distanceTransform(255-mask, cv2.DIST_L2, 5)
    samples = [np.c_[np.zeros(100), np.zeros(100), np.linspace(-0.1244, 0.1744, 100)]]
    for offset, quaternion in LOWER_PEG_FRAMES + PEG_FRAMES:
        point = np.asarray(offset)
        axis = _rotate(quaternion, [1, 0, 0])
        stem_intersection = point - axis * np.linalg.norm(point[:2]) / np.linalg.norm(axis[:2])
        samples.append(np.linspace(stem_intersection, point + 0.045*axis, 35))
    local = np.concatenate(samples)
    origin, camera_rotation, fx, fy = ctx.perception.camera_geometry("cam_head", image_shape=image.shape)

    def residual(parameters):
        x, y, yaw = parameters
        cosine, sine = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]])
        world = local @ rotation + [x, y, center[2]]
        camera = (world-origin) @ camera_rotation
        pixels_x = image.shape[1]/2 - fx*camera[:, 0]/camera[:, 2]
        pixels_y = image.shape[0]/2 + fy*camera[:, 1]/camera[:, 2]
        error = cv2.remap(distances, pixels_x.astype(np.float32)[:, None],
                          pixels_y.astype(np.float32)[:, None], cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=100).ravel()
        return np.r_[error, (parameters[:2]-center[:2])*30]

    def jacobian(parameters):
        epsilon = np.array([0.0002, 0.0002, 0.005])
        return np.column_stack([(residual(parameters + np.eye(3)[i]*epsilon[i]) -
                                 residual(parameters - np.eye(3)[i]*epsilon[i])) / (2*epsilon[i])
                                for i in range(3)])

    candidates = []
    for yaw in np.linspace(-0.45, 0.45, 9):
        result = least_squares(residual, [*center[:2], yaw], jac=jacobian,
                               bounds=([*(center[:2]-0.035), -0.48],
                                       [*(center[:2]+0.035), 0.48]), max_nfev=80)
        candidates.append(result)
    best = min(candidates, key=lambda result: float(np.mean(residual(result.x)**2)))
    if float(np.mean(residual(best.x)**2)) < 25:
        center[:2] = best.x[:2]
        return center, float(best.x[2])
    return center, _observed_yaw(holder)


def hang_mugs(ctx):
    """Place observed handle holes onto three visible mug-rack supports."""
    # Keep public mesh branch selection local to this task's numerical controller.
    if ctx.kinematics is not None:
        from .mesh_kinematics import MeshFilteredKinematics
        from coap.public_mesh import PublicRobotConvexGeometry
        model = getattr(ctx, "precision_collision_geometry", None) or PublicRobotConvexGeometry()
        ctx.precision_collision_geometry = model
        if not isinstance(ctx.kinematics, MeshFilteredKinematics):
            ctx.kinematics = MeshFilteredKinematics(ctx.kinematics, model)
        ctx.memory["mug_mesh_filter"] = ctx.kinematics.stats
    from .mug_scene import rack_candidates, acquire_mug_sources, refresh_mug_source
    from .mug_motion import plan_mug_pickup
    ctx.stage = "mug_measure"
    holder_candidates = []
    for attempt in range(2):
        holder_candidates = rack_candidates(ctx, ctx.detect('cup_holder'))
        if len(holder_candidates) == 1:
            break
        ctx.memory['mug_scene_failure'] = {'kind':'rack_unresolved', 'attempt':attempt,
                                          'candidate_count':len(holder_candidates)}
        yield from ctx.hold(3)
    if len(holder_candidates) != 1:
        raise RuntimeError('Exactly one complete public mug-rack model must be visually resolved')
    holder = holder_candidates[0]
    holder_center, holder_yaw = _position(holder), holder['yaw']
    holder_rotation = _yaw_quaternion(holder_yaw)
    frames = holder['support_frames']
    # Keep the established upper/middle/lower support ordering for the standard rack.
    peg_order = (4,5,2,3,0,1)
    peg_frames = [(np.asarray(frames[index][:3])*holder['geometry_scale'], frames[index][3:])
                  for index in peg_order]
    ctx.memory['mug_holder_frame'] = {'position':holder_center.tolist(), 'yaw':holder_yaw,
        'asset_index':holder['asset_index'], 'geometry_source':holder['geometry_source'],
        'geometry_scale':holder['geometry_scale'], 'model_fit':holder['model_fit']}
    used_positions = []
    used_pegs = set()
    source_mugs = yield from acquire_mug_sources(ctx,holder)
    if len(source_mugs) != 3:
        ctx.memory['mug_scene_failure'] = {'kind':'source_identity_unresolved',
            'candidate_count':len(source_mugs), 'required_count':3,
            'random_source_models_supported':True if holder['asset_index']==1 else None,
            'source_acquisition_policy':ctx.memory.get('mug_source_acquisition_policy'),
            'grasp_started':False,
            'private_task_state_used':False}
        raise RuntimeError('Three distinct source cups with supported public geometry must be resolved')
    for mug_index, source_mug in enumerate(source_mugs[:3]):
        ctx.stage = f"mug_measure_{mug_index}"
        source_mug=refresh_mug_source(ctx,source_mug,holder,mug_index)
        # Preserve initial RGB identities so moved robot shells cannot become cups.
        mug = source_mug if source_mug.get("asset_index",-1)>=10 else _red_mug_profile(ctx,_match_mug(ctx,source_mug))
        size = mug.get("size", [0.105, 0.080, 0.073])
        handle_grasp = bool(mug.get("handle_grasp"))
        center, mug_yaw = _mug_frame(ctx, mug, float(size[2]))
        center[2] = _mug_origin_height(ctx,mug,size)
        mug_initial_rotation = _yaw_quaternion(mug_yaw)
        if mug.get('asset_index',-1)>=10:
            thread_geometry=mug.get('thread_geometry',{})
            if (not thread_geometry.get('accepted') or
                    thread_geometry.get('private_task_state_used') is not False or
                    'thread_local' not in mug or not mug.get('thread_frames')):
                ctx.memory['mug_scene_failure']={'kind':'native_thread_geometry_unresolved',
                    'source_index':mug_index,'asset_index':mug['asset_index'],'grasp_started':False,
                    'private_task_state_used':False}
                raise RuntimeError('Random cup native thread geometry must be verified before pickup')
            handle_local=np.asarray(mug['thread_local'],float)
            ctx.memory[f'mug_thread_geometry_{mug_index}']={
                'functional_handle_local':np.asarray(mug['handle_local']).tolist(),
                'thread_local':handle_local.tolist(),'thread_frames':mug['thread_frames'],
                'geometry':thread_geometry,'private_task_state_used':False}
        else:
            handle_local = _mug_handle(mug)
        # Pinch the outer flat handle faces, whose normals match the jaw closing axis.
        grasp_local = np.asarray(mug.get("handle_grasp_local", [0.0465 if mug.get("asset_index") == 5 else 0.055, 0.0, 0.012])) if handle_grasp else \
            np.asarray(mug.get("grasp_local",[-0.012,0.0,0.030-float(size[2])/2]))
        grasp = center + _rotate(mug_initial_rotation, grasp_local)
        arm = _arm(ctx, grasp)
        is_red_handle = mug.get('asset_index') == 8 and handle_grasp
        route = plan_mug_pickup(ctx,mug,arm,center,mug_yaw,grasp_local,handle_local,
                              holder_center,holder_rotation,used_pegs,peg_frames,
                              handle_grasp,mug_index,is_red_handle)
        ctx.memory[f'mug_path_{mug_index}'] = route
        grasp_local = route['grasp_local']
        grasp = center+_rotate(mug_initial_rotation,grasp_local)
        grasp_quaternion = route['quaternion']
        ctx.stage = f"mug_grasp_{mug_index}"
        lift = 0.05 if handle_grasp else 0.12
        yield from ctx.move(arm, grasp + [0, 0, lift], grasp_quaternion, 1.0, 8)
        wrist_camera = f"cam_{arm}_wrist"
        # The fixed head view avoids using the moving source wrist before a refresh.
        wrist_mug = mug if (mug.get("asset_index",-1)>=10 or mug.get("asset_index")==5 or route["grasp_kind"]=="near_wall") else _match_mug(ctx,source_mug,camera=wrist_camera)
        if wrist_mug.get("camera") == wrist_camera:
            mug = dict(wrist_mug, asset_index=mug.get("asset_index", 8), size=size, handle_local=handle_local,
                       handle_grasp=True) if handle_grasp else wrist_mug
            center, mug_yaw = _mug_frame(ctx, mug, float(size[2]))
            center[2] = _mug_origin_height(ctx,mug,size)
            mug_initial_rotation = _yaw_quaternion(mug_yaw)
            grasp = center + _rotate(mug_initial_rotation, grasp_local)
            route = plan_mug_pickup(ctx,mug,arm,center,mug_yaw,grasp_local,handle_local,
                                  holder_center,holder_rotation,used_pegs,peg_frames,
                                  handle_grasp,mug_index,is_red_handle)
            ctx.memory[f'mug_path_{mug_index}'] = route
            grasp_local = route['grasp_local']
            grasp = center+_rotate(mug_initial_rotation,grasp_local)
            grasp_quaternion = route['quaternion']
        yield from ctx.move(arm, grasp, grasp_quaternion, 1.0, 8)
        yield from ctx.grip(arm, 0.0, 6)
        ctx.stage = f"mug_lift_{mug_index}"
        yield from ctx.move(arm, grasp + [0, 0, lift], grasp_quaternion, 0.0, 10)
        used_positions.append(center.copy())
        # MetaControl caches clipped commands, not reliable contact aperture.
        # A thin cup wall can remain securely held with a cached opening of zero.
        _, _, _, peg_index, peg_position, peg_axis, final_gripper, seated, preinsert = route["support"]
        # Keep the broad white cup farther from the central post while threading.
        # This shift remains inside the public 45 mm support-point tolerance.
        seating_shift = 0.025 * peg_axis if is_red_handle else np.zeros(3)
        peg_position = peg_position + seating_shift
        seated = seated + seating_shift
        preinsert = preinsert + seating_shift
        used_pegs.add(peg_index)
        ctx.memory[f"mug_target_{mug_index}"] = {
            "peg_index": peg_index, "position": peg_position.tolist(),
            "quaternion": final_gripper.tolist(), "seated": seated.tolist(),
            "seating_shift": seating_shift.tolist()}
        ctx.memory[f"mug_grasp_{mug_index}"] = {
            "asset_index": mug.get("asset_index"),
            "handle_grasp":handle_grasp and route["grasp_kind"]=="handle",
            "grasp_kind":route["grasp_kind"],"source":grasp.tolist(),
            "object_yaw": mug_yaw, "grasp_local": grasp_local.tolist(),
            "handle_local": handle_local.tolist()}
        # Retain upright staging when both orientations and the entire path are legal.
        if route.get('route_kind') != 'direct_pose_transfer':
            staging = route['staging']
            ctx.stage = f'mug_transport_upright_{mug_index}'
            yield from ctx.move(arm,staging,grasp_quaternion,0.,10)
            ctx.stage = f'mug_rotate_{mug_index}'
            yield from ctx.move(arm,staging,final_gripper,0.,10)
            yield from ctx.hold(2)
        ctx.stage = f'mug_transport_rotate_{mug_index}'
        if is_red_handle:
            preinsert = np.array([-.06 if arm=='left' else .06,-.17,1.10])
        transfer_duration = 32 if route.get('route_kind')=='direct_pose_transfer' else 12
        yield from ctx.move(arm,preinsert,final_gripper,0.,transfer_duration)
        if not is_red_handle:
            # Refresh the carried point after rotation, using actual tool coordinates.
            from .mug_held import refresh_handle
            from .mug_motion import preflight_segments
            corrected = refresh_handle(ctx, arm, mug, mug_initial_rotation, grasp_quaternion,
                grasp_local, handle_local, final_gripper, peg_position, seated, mug_index)
            live = ctx.memory[f'mug_live_handle_{mug_index}']
            if live['accepted']:
                future = [{'kind': 'move', 'stage': f'mug_{phase}_{mug_index}', 'arm': arm,
                    'position': target, 'quaternion': final_gripper, 'grip': 0., 'duration': duration}
                    for phase, target, duration in (
                        ('align', corrected+peg_axis*.045, 8),
                        ('thread', corrected+[0, 0, .008], 12), ('seat', corrected, 6))]
                live['corrected_route_preflight'] = preflight_segments(ctx, future)
                if live['corrected_route_preflight']['accepted']:
                    seated = corrected
                else:
                    live.update(accepted=False, reason='Corrected route rejected; retain original delivery')
        if is_red_handle:
            expected_handle = ctx.tcp(arm)+[0,0,-.025]
            if route['grasp_kind']=='near_wall':
                expected_handle = _mug_observer_target(ctx,arm,mug_initial_rotation,
                    grasp_quaternion,grasp_local,handle_local)
                ctx.memory[f'mug_handle_observer_prior_{mug_index}'] = {
                    'position':expected_handle.tolist(),'source':'nominal_rigid_grasp_geometry_from_actual_tool',
                    'retention_verified':False,'metric_object_pose_error_bound_m':None}
            measured = yield from _mug_observer(ctx, arm, expected_handle, mug_index)
            body_registered = False
            body_pose = ctx.memory.get(f"mug_body_pose_{mug_index}")
            if body_pose is not None:
                from .precision_geometry import red_mug_registered_target
                tool_pose = ctx.memory[f"mug_observer_measure_{mug_index}"]["held_tcp_pose"]
                candidate_peg = 4 if 4 not in used_pegs or peg_index == 4 else peg_index
                candidate_offset, candidate_localq = peg_frames[candidate_peg]
                candidate_axis = _rotate(_multiply(holder_rotation, candidate_localq), [1, 0, 0])
                candidate_position = holder_center + _rotate(holder_rotation, candidate_offset) + candidate_axis*.025
                registered = red_mug_registered_target(body_pose, tool_pose, final_gripper,
                                                        candidate_position, candidate_axis)
                diagnostic = {"accepted": False, "reason": "large_angle"}
                ctx.memory[f"mug_body_registration_{mug_index}"] = diagnostic
                if registered is not None:
                    corrected_q = np.asarray(registered["quaternion"])
                    corrected = np.asarray(registered["seated"])
                    rigid_initial = _rotate(mug_initial_rotation, handle_local-grasp_local)
                    nominal = candidate_position-_rotate(
                        _multiply(corrected_q, _inverse(grasp_quaternion)), rigid_initial)
                    correction = corrected-nominal
                    diagnostic.update(registered, correction=correction.tolist())
                    diagnostic["reason"] = "large_translation"
                    if np.linalg.norm(correction[:2]) < .035 and abs(correction[2]) < .035:
                        from coap.kinematics import X5Kinematics
                        from coap.public_mesh import PublicRobotConvexGeometry
                        kinematics = ctx.kinematics or X5Kinematics(ctx.initial_state)
                        model = getattr(ctx, "precision_collision_geometry", None) or PublicRobotConvexGeometry()
                        reference = ctx.state[f"{arm}_arm_joint_state"]
                        waypoint = ctx.tcp(arm).copy()
                        waypoint[0] = corrected[0] + candidate_axis[0]*.045
                        targets = (ctx.tcp(arm), waypoint, corrected+candidate_axis*.045,
                                   corrected+[0, 0, .008], corrected, corrected+[0, 0, .04])
                        results = [kinematics.solve(arm, np.r_[target, corrected_q], tcp=True,
                                                     current_joints=reference, max_starts=8)
                                   for target in targets]
                        other = "right" if arm == "left" else "left"
                        feasible = all(result["status"] == "Success" and
                                       min(model.check(kinematics, arm, result["joint_value"], grip=0).values()) >= .001 and
                                       min(kinematics.public_clearance(arm, result["joint_value"], ctx.table_z,
                                                                      other, ctx.state[f"{other}_arm_joint_state"]).values()) >= .001
                                       for result in results)
                        diagnostic["reason"] = "unreachable"
                        if feasible:
                            used_pegs.remove(peg_index)
                            peg_index, peg_position, peg_axis = candidate_peg, candidate_position, candidate_axis
                            used_pegs.add(peg_index)
                            final_gripper, seated = corrected_q, corrected
                            body_registered = True
                            diagnostic.update(accepted=True, reason="matched")
                            ctx.stage = f"mug_body_rotate_{mug_index}"
                            yield from ctx.move(arm, ctx.tcp(arm), final_gripper, 0.0, 8)
                            ctx.stage = f"mug_body_route_{mug_index}"
                            yield from _mug_carry(ctx, arm, waypoint, final_gripper, mug_index)
                            diagnostic["peg_index"] = peg_index
                            diagnostic["peg_position"] = peg_position.tolist()
            if not body_registered and measured is not None:
                gap, point, head_pixel, wrist_pixel = measured
                # Use the observed object-to-tool vector, independently of goal tracking error.
                observed_tool = np.asarray(ctx.memory[f"mug_observer_measure_{mug_index}"]["held_tcp_pose"])
                relative_offset = point-observed_tool[:3]
                tool_correction = _multiply(final_gripper, _inverse(observed_tool[3:]))
                target_offset = _rotate(tool_correction, relative_offset)
                corrected = peg_position-target_offset
                correction = corrected-seated
                from coap.kinematics import X5Kinematics
                kinematics = getattr(ctx, "kinematics", None) or X5Kinematics(ctx.initial_state)
                results = [kinematics.solve(arm, np.r_[position, final_gripper], tcp=True,
                                           current_joints=ctx.state[f"{arm}_arm_joint_state"], max_starts=8)
                           for position in (corrected+peg_axis*0.045, corrected)]
                if all(result["status"] == "Success" for result in results):
                    seated = corrected
                    ctx.memory[f"mug_handle_stereo_{mug_index}"] = {
                        "position": point.tolist(), "correction": correction.tolist(),
                        "ray_gap": gap, "head_pixel": head_pixel, "wrist_pixel": wrist_pixel}
        aligned = seated + peg_axis * 0.045
        ctx.stage = f"mug_align_{mug_index}"
        yield from ctx.move(arm, aligned, final_gripper, 0.0, 8)
        if is_red_handle and not body_registered and f"mug_handle_stereo_{mug_index}" not in ctx.memory:
            ctx.stage = f"mug_handle_refine_{mug_index}"
            expected_handle = peg_position + peg_axis * 0.045
            observed_handle = _red_handle_position(ctx, expected_handle)
            if observed_handle is not None:
                correction = expected_handle[:2] - observed_handle[:2]
                if np.linalg.norm(correction) < 0.095:
                    from coap.kinematics import X5Kinematics
                    kinematics = getattr(ctx, "kinematics", None) or X5Kinematics(ctx.initial_state)
                    adjusted = seated.copy()
                    adjusted[:2] += correction
                    result = kinematics.solve(arm, np.r_[adjusted, final_gripper], tcp=True,
                                               current_joints=ctx.state[f"{arm}_arm_joint_state"], max_starts=8)
                    if result["status"] == "Success":
                        seated = adjusted
        ctx.stage = f"mug_thread_{mug_index}"
        yield from ctx.move(arm, seated + [0, 0, 0.008], final_gripper, 0.0, 12)
        ctx.stage = f"mug_seat_{mug_index}"
        yield from ctx.move(arm, seated, final_gripper, 0.0, 6)
        ctx.stage = f"mug_release_{mug_index}"
        yield from ctx.grip(arm, 1.0, 6)
        withdraw = seated + [0, 0, .04] if is_red_handle and body_registered else \
            seated + peg_axis * .08 + [0, 0, .06]
        yield from ctx.move(arm, withdraw, final_gripper, 1.0, 8)
        if mug_index == 2:
            ctx.stage = f"mug_home_{mug_index}"
            # Both arms must return after mixed-arm placements and camera scouts.
            yield from _home(ctx, None, duration=6)
        else:
            ctx.stage = f"mug_refresh_{mug_index}"
            yield from ctx.hold(3)
    ctx.stage = "mug_settle"
    yield from ctx.hold(3)


TASKS = {"insert_tubes": insert_tubes, "insert_key": insert_key, "hang_mugs": hang_mugs}


def run(ctx):
    yield from TASKS[ctx.task_name](ctx)
