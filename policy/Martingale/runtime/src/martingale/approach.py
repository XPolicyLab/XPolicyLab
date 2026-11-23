# Copyright © 2026 Martingale. All Rights Reserved.

import math

TRANSPORT_STEP_M = 0.008
WAYPOINT_LEAD_M = 0.00075


def _angle(a, b):
    scale = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
    if not math.isfinite(scale) or scale <= 0:
        return math.inf
    return 2 * math.acos(min(1.0, abs(sum(x * y for x, y in zip(a, b))) / scale))


def approach_to(robot, xyz, arm, *, clearance_m, empty_hand, vertical, max_waypoints,
                transport=False):
    """Implementation of RobodojoRobot.approach_to."""
    target = tuple(float(x) for x in xyz)
    clearance = float(clearance_m)
    if len(target) != 3 or not all(math.isfinite(x) for x in target):
        raise ValueError("xyz must be three finite numbers")
    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError("clearance_m must be finite and nonnegative")
    if any(type(v) is not bool for v in (empty_hand, vertical, transport)):
        raise ValueError("empty_hand, vertical and transport must be booleans")
    if transport and (empty_hand or vertical):
        raise ValueError("transport is a loaded upward lift, not an empty-hand approach")
    if type(max_waypoints) is not int or not 1 <= max_waypoints <= 200:
        raise ValueError("max_waypoints must be an integer from 1 to 200")
    if arm not in ("left", "right"):
        raise ValueError("arm must be left or right")

    state = robot.state()
    initial = state.arm(arm)
    other = "left" if arm == "right" else "right"
    other_start = state.arm(other).tcp_pos
    start = tuple(initial.tcp_pos)
    if not all(math.isfinite(x) for x in start) or not math.isfinite(
        _angle(initial.eef_quat, initial.eef_quat)
    ):
        raise ValueError("invalid initial robot pose")
    if vertical:
        if abs(initial.pitch - 90) > 3 or math.dist(start[:2], target[:2]) > 0.001:
            raise ValueError("vertical pickup requires pitch=90 and TCP aligned above target")
        if target[2] > start[2]:
            raise ValueError("vertical pickup must descend")
        target = (start[0], start[1], target[2])
    if transport:
        if abs(initial.pitch - 90) > 3 or math.dist(start[:2], target[:2]) > 0.001:
            raise ValueError("transport requires a downward-facing, vertically aligned grasp")
        if target[2] < start[2]:
            raise ValueError("transport only supports upward lifts after a verified test lift")
        target = (start[0], start[1], target[2])

    clearance = max(0.0, clearance - 0.005)
    steps = waypoints = 0
    position = start
    scheduled_z = start[2]
    path_length = math.dist(start, target)
    path_axis = tuple((b - a) / path_length for a, b in zip(start, target)) if path_length else (0., 0., 0.)
    scheduled_distance = 0.0

    def finish(reason):
        return dict(reached=reason == "reached", reason=reason, steps_used=steps,
                    waypoints=waypoints, final_xyz=position,
                    remaining_m=math.dist(position, target))

    for _ in range(max_waypoints):
        remaining = math.dist(position, target)
        if state.truncated:
            return finish("episode_ended")
        if remaining <= 0.001:
            return finish("reached")
        distance = min(remaining, clearance)
        size = 0.002
        boundary_room = math.inf
        if empty_hand and distance > 0.0325 + 1e-9:
            size = min(0.010, distance - 0.030)
            boundary_room = distance - 0.030
        elif empty_hand and distance > 0.0125 + 1e-9:
            size = min(0.005, distance - 0.010)
            boundary_room = distance - 0.010
        size = min(size, remaining)
        if transport:
            size = 0.002
            boundary_room = math.inf
            if clearance > 0.0125 + 1e-9:
                size = min(TRANSPORT_STEP_M, clearance - 0.010)
                boundary_room = clearance - 0.010
            next_z = min(target[2], max(scheduled_z, position[2]) + size,
                         position[2] + size + WAYPOINT_LEAD_M,
                         position[2] + boundary_room)
            waypoint = (start[0], start[1], next_z)
            max_lead = min(size + WAYPOINT_LEAD_M, boundary_room)
            if math.dist(position, waypoint) > max_lead + 1e-9:
                lateral = math.dist(position[:2], start[:2])
                if lateral >= max_lead:
                    return finish("lateral_drift")
                waypoint = (start[0], start[1], min(next_z, position[2]
                             + math.sqrt(max_lead * max_lead - lateral * lateral)))
            scheduled_z = waypoint[2]
        elif empty_hand:
            progress = sum((p - s) * u for p, s, u in zip(position, start, path_axis))
            lateral_sq = sum((p - s - progress * u) ** 2
                             for p, s, u in zip(position, start, path_axis))
            max_lead = min(size + WAYPOINT_LEAD_M, boundary_room)
            if lateral_sq >= max_lead * max_lead:
                return finish("lateral_drift")
            scheduled_distance = min(path_length, max(scheduled_distance, progress) + size,
                                     progress + math.sqrt(max_lead * max_lead - lateral_sq))
            waypoint = tuple(s + scheduled_distance * u for s, u in zip(start, path_axis))
        else:
            waypoint = tuple(a + (b - a) * size / remaining for a, b in zip(position, target))
        if vertical:
            waypoint = (start[0], start[1], waypoint[2])
        prior = state.arm(arm)
        result = robot.move_to(waypoint, arm=arm, gripper=robot.KEEP, path="line",
                               pitch=initial.pitch, yaw=initial.yaw, roll=initial.roll)
        steps += result.steps_used
        waypoints += 1
        state = robot.state()
        current = state.arm(arm)
        previous, position = position, tuple(current.tcp_pos)
        if state.truncated or result.truncated:
            return finish("episode_ended")
        if not all(math.isfinite(v) for v in position):
            return finish("invalid_pose")
        if result.blocked or not result.reached or result.ik_error:
            return finish("blocked")
        if math.dist(position, waypoint) > 0.0025:
            return finish("tracking_error")
        if _angle(initial.eef_quat, current.eef_quat) > math.radians(3):
            return finish("orientation_drift")
        if math.dist(other_start, state.arm(other).tcp_pos) > 0.003:
            return finish("other_arm_moved")
        if (vertical or transport) and math.dist(position[:2], start[:2]) > 0.001:
            return finish("lateral_drift")
        if remaining - math.dist(position, target) < 0.0001:
            return finish("no_progress")
        clearance = max(0.0, clearance - math.dist(previous, position)
                        - 0.20 * _angle(prior.eef_quat, current.eef_quat))
    return finish("reached" if math.dist(position, target) <= 0.001 else "waypoint_limit")
