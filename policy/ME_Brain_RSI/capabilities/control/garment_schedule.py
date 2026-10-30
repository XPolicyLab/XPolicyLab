"""Reuse measured stationary observations without reducing cloth motion timing."""
from copy import deepcopy

import numpy as np

from b88r42wq.motion import (FINAL_JOINT_TOLERANCE, MAX_ACTUAL_TARGET_LEAD, MotionProfile,
                         _cartesian_errors, _pose_arrived, link_to_tcp,
                         quat_matrix, tcp_to_link)


def same_tcp_goal(record, arm, point, quat):
    """Match a reported TCP goal, including quaternion double-cover equivalence."""
    if record.get("goal_frame") != "tcp":
        return False
    goal = record.get("nominal_frame_goals", {}).get(arm)
    if goal is None:
        return False
    goal = np.asarray(goal, float)
    return (goal.shape == (7,) and
            np.allclose(goal[:3], point, rtol=0., atol=1e-9) and
            abs(float(np.dot(goal[3:], quat))) >= 1.-1e-10)


def pose_ready(ctx, arm, point, quat, *, precise=False):
    """Use public measured pose and joints to identify an already reached task goal."""
    actual = np.asarray(ctx.state[f"{arm}_ee_pose"], float)
    joint_error = float(np.max(np.abs(np.asarray(ctx.state[f"{arm}_arm_joint_state"])
                                      - ctx.joint_targets[arm])))
    angle = float(np.arccos(np.clip((np.trace(quat_matrix(actual[3:]).T @
                                            quat_matrix(quat))-1.)/2., -1., 1.)))
    return (joint_error <= FINAL_JOINT_TOLERANCE and
            np.linalg.norm(link_to_tcp(actual)-point) <= (.0005 if precise else .002) and
            angle <= (.003 if precise else np.pi/180))


def move_with_receipt(ctx, arm, point, quat, grip, duration):
    """Record the actual final unchanged-command holds issued by Context.move."""
    last_command = None
    stable_holds = 0
    for action in ctx.move(arm, point, quat, grip, duration):
        if f"{arm}_arm_joint_state" not in action:
            yield action
            continue
        command = np.asarray(action[f"{arm}_arm_joint_state"], float).copy()
        report = getattr(ctx, "motion_plan", {})
        unchanged = last_command is not None and np.array_equal(command, last_command)
        if (unchanged and report.get("phase") in ("pose_arrival_hold", "pose_correction_hold")
                and report.get("pose_stable_observations", 0) > 0):
            stable_holds += 1
        else:
            stable_holds = 0
        last_command = command
        yield action
    if last_command is None:
        return None
    report = deepcopy(getattr(ctx, "motion_plan", {}))
    return {"step": ctx.steps, "arm": arm, "point": np.asarray(point).copy(),
            "quat": np.asarray(quat).copy(), "command": last_command,
            "emitted_final_stable_holds": stable_holds, "report": report}


def rendered_settle(ctx, receipt, minimum_holds=3):
    """Complete three post-target render steps using a fresh verified arrival window."""
    reusable = 0
    if receipt is not None:
        report = receipt["report"]
        live = getattr(ctx, "motion_plan", {})
        arm = receipt["arm"]
        stable = int(report.get("pose_stable_observations", 0))
        if (receipt["step"] == ctx.steps and report.get("phase") == "pose_arrived"
                and live.get("phase") == "pose_arrived"
                and report.get("joint_and_pose_share_stability_window") is True
                and stable >= 3 and live.get("pose_stable_observations", 0) >= 3
                and receipt["emitted_final_stable_holds"] >= stable-1
                and same_tcp_goal(live, arm, receipt["point"], receipt["quat"])
                and same_tcp_goal(report, arm, receipt["point"], receipt["quat"])
                and receipt["command"] is not None
                and np.array_equal(receipt["command"], ctx.joint_targets[arm])
                and float(np.max(np.abs(np.asarray(ctx.state[arm+"_arm_joint_state"])
                                        -receipt["command"]))) <= FINAL_JOINT_TOLERANCE
                and _pose_arrived(_cartesian_errors(ctx.state, {arm: tcp_to_link(
                    receipt["point"], receipt["quat"])}, tcp=True),
                    MotionProfile(**report.get("motion_profile", {"name": "legacy"})))):
            reusable = min(minimum_holds, stable-1)
    additional = max(0, minimum_holds-reusable)
    ctx.memory.setdefault("garment_render_wait", []).append({
        "stage": getattr(ctx, "stage", ""), "required_post_target_holds": minimum_holds,
        "reused_issued_stable_holds": reusable, "additional_holds": additional,
        "recorded_final_stable_holds": 0 if receipt is None else receipt["emitted_final_stable_holds"],
        "reported_stable_observations": 0 if receipt is None else
            receipt["report"].get("pose_stable_observations", 0)})
    if additional:
        yield from ctx.hold(additional)


def release_same_pose(ctx, left, right, left_quat, right_quat, duration=8):
    """Keep eight opening actions and verify the final three measured observations."""
    report = getattr(ctx, "motion_plan", {})
    reusable = (ctx.numeric_ik and report.get("phase") == "pose_arrived"
                and report.get("joint_and_pose_share_stability_window") is True
                and report.get("pose_stable_observations", 0) >= 3
                and same_tcp_goal(report, "left", left, left_quat)
                and same_tcp_goal(report, "right", right, right_quat)
                and report.get("robot_geometry", {}).get("accepted") is True)
    if not reusable:
        yield from ctx.move_both(left, right, left_quat, right_quat, 1., 1., duration)
        return
    goals = {"left": tcp_to_link(left, left_quat), "right": tcp_to_link(right, right_quat)}
    targets = {arm: {"pose": pose} for arm, pose in goals.items()}
    profile = ctx.motion_profile(targets)
    commanded = {arm: ctx.joint_targets[arm].copy() for arm in goals}
    # Match numeric_move_actions' measured-to-cached segment-start geometry check.
    from b88r42wq.robot_motion_geometry import check_joint_path
    actual = np.concatenate([ctx.state[f"{arm}_arm_joint_state"] for arm in goals])
    cached = np.concatenate([commanded[arm] for arm in goals])
    geometry = check_joint_path(ctx.kinematics, np.stack([actual, cached]), ctx.table_z)
    if not geometry["accepted"]:
        raise RuntimeError(f"Public stationary garment release rejected before movement: {geometry}")
    ctx.gripper_targets.update(left=1., right=1.)
    stable = 0
    for index in range(duration):
        error = max(float(np.max(np.abs(np.asarray(ctx.state[f"{arm}_arm_joint_state"])
                                        -commanded[arm]))) for arm in goals)
        if error > MAX_ACTUAL_TARGET_LEAD:
            raise RuntimeError(f"Stationary garment release exceeded actual lead: {error:.6f}")
        yield ctx.hold_action()
        errors = _cartesian_errors(ctx.state, goals, tcp=True)
        joint_ready = all(float(np.max(np.abs(np.asarray(ctx.state[f"{arm}_arm_joint_state"])
                                               -commanded[arm]))) <= FINAL_JOINT_TOLERANCE
                          for arm in goals)
        if index >= duration-3:
            stable = stable+1 if joint_ready and _pose_arrived(errors, profile) else 0
    ctx.memory["garment_stationary_release"] = {"opening_actions": duration,
        "measured_final_stable_observations": stable, "same_checked_pose": True,
        "retention_verified": False}
    if stable >= 3:
        ctx.report_motion({"phase": "pose_arrived", "pose_stable_observations": stable,
                           "joint_and_pose_share_stability_window": True,
                           "measured_pose_errors": errors,
                           "stationary_opening_actions": duration})
        return
    # The shared arrival/correction budget remains the only authority for residuals.
    yield from ctx.move_both(left, right, left_quat, right_quat, 1., 1., 1)


def preview_body_headings(ctx, body, points):
    """Select a robot-feasible equivalent pinch heading for the entire nominal fold.

    This is public FK preflight with unchanged current RGB, not cloth retention or
    future garment prediction. The real acquisition gates remain in the caller.
    """
    from copy import copy
    from b88r42wq.tasks import garment as g
    from b88r42wq.motion import down_quat
    candidates = []
    failures = []
    yaw = np.arctan2(body["right"][1]-body["left"][1], body["right"][0]-body["left"][0])
    for left_flip, right_flip in ((0, 0), (0, 1), (1, 0), (1, 1)):
        preview = copy(ctx)
        preview.memory = deepcopy(ctx.memory)
        candidate = deepcopy(body)
        candidate["quats"] = [down_quat(yaw+left_flip*np.pi),
                              down_quat(yaw+(1+right_flip)*np.pi)]
        frames = 0
        minimum = {}
        try:
            for action in _nominal_body_actions(preview, candidate, points):
                frames += 1
                geometry = getattr(preview, "motion_plan", {}).get("robot_geometry", {})
                for kind, margin in geometry.get("minimum_margin_m", {}).items():
                    minimum[kind] = min(minimum.get(kind, float("inf")), float(margin))
                state = dict(preview.state)
                for arm in ("left", "right"):
                    state[arm+"_arm_joint_state"] = np.asarray(action[arm+"_arm_joint_state"]).copy()
                    state[arm+"_ee_pose"] = preview.kinematics.forward(arm, state[arm+"_arm_joint_state"])
                    state[arm+"_ee_joint_state"] = np.asarray(action[arm+"_ee_joint_state"]).copy()
                preview.update_obs({"state": state, "vision": ctx.vision})
            candidates.append((frames, candidate, minimum, [left_flip, right_flip]))
        except RuntimeError as error:
            failures.append({"flips": [left_flip, right_flip], "error": str(error)})
    if not candidates:
        raise RuntimeError(f"No complete public garment body path: {failures}")
    frames, chosen, minimum, flips = min(candidates, key=lambda candidate: candidate[0])
    original_frames = frames
    chosen["fold_duration"] = 10
    available = 500-int(ctx.steps)
    if frames > available:
        faster = deepcopy(chosen)
        faster["fold_duration"] = 9
        preview = copy(ctx)
        preview.memory = deepcopy(ctx.memory)
        faster_frames = 0
        faster_minimum = {}
        try:
            for action in _nominal_body_actions(preview, faster, points):
                faster_frames += 1
                geometry = getattr(preview, "motion_plan", {}).get("robot_geometry", {})
                for kind, margin in geometry.get("minimum_margin_m", {}).items():
                    faster_minimum[kind] = min(faster_minimum.get(kind, float("inf")), float(margin))
                state = dict(preview.state)
                for arm in ("left", "right"):
                    state[arm+"_arm_joint_state"] = np.asarray(action[arm+"_arm_joint_state"]).copy()
                    state[arm+"_ee_pose"] = preview.kinematics.forward(arm, state[arm+"_arm_joint_state"])
                    state[arm+"_ee_joint_state"] = np.asarray(action[arm+"_ee_joint_state"]).copy()
                preview.update_obs({"state": state, "vision": ctx.vision})
            chosen, frames, minimum = faster, faster_frames, faster_minimum
        except RuntimeError as error:
            failures.append({"flips": flips, "fold_duration": 9, "error": str(error)})
    ctx.memory["garment_body_heading_plan"] = {"flips": flips,
        "original_fold_nominal_frames": 10, "selected_fold_nominal_frames": chosen["fold_duration"],
        "original_seven_fold_nominal_seconds": 2.8,
        "selected_seven_fold_nominal_seconds": 7*chosen["fold_duration"]*.04,
        "original_nominal_robot_frames": original_frames,
        "remaining_episode_actions_at_prediction": available,
        "predicted_within_cap": frames <= available,
        "nominal_robot_frames_including_precontact_and_home": frames,
        "minimum_complete_robot_clearances_m": minimum,
        "candidate_frames": [{"flips": candidate[3], "frames": candidate[0]} for candidate in candidates],
        "rejections": failures, "current_RGB_static": True,
        "future_cloth_prediction": False, "retention_verified": False}
    return chosen


def _nominal_body_actions(ctx, body, points):
    """Check nominal no-retry controls without declaring a visual acquisition gate passed."""
    from b88r42wq.tasks import garment as g
    left, right = body["left"].copy(), body["right"].copy()
    targets = [point.copy() for point in body["targets"]]
    ctx.memory["garment_fold_targets"] = [target.tolist() for target in targets]
    quats = body["quats"]
    ctx.memory["garment_fold_duration"] = int(body.get("fold_duration", 10))
    left_point, entry, rake_opening, _, _ = g._edge_entry(ctx, "left", left, quats[0], targets[0]-left)
    right_point, _, _, _, _ = g._edge_entry(ctx, "right", right, quats[1], targets[1]-right)
    right_point[:2] += .012*(left[:2]-right_point[:2])/max(np.linalg.norm(left[:2]-right_point[:2]), 1e-8)
    ctx.stage = "garment_right_sleeve_retreat_contact"
    yield from ctx.move_both(left_point, g._lift(right_point, .075), entry, quats[1], rake_opening, 1., 10)
    lift = g.edge_lift_height(left, right)
    ctx.stage = "garment_left_edge_acquire"
    left, _ = yield from g._acquire_edge(ctx, "left", left, quats[0], lift, targets[0]-left)
    ctx.stage = "garment_create_right_edge"
    yield from ctx.hold(3)
    ctx.stage = "garment_right_edge_acquire"
    previous = right.copy()
    g._snap_sewn_hem(ctx, g._current_garment(ctx), (right,), (targets[1],))
    correction = right[:2]-previous[:2]
    right[:2] = previous[:2]+correction*min(1., .020/max(np.linalg.norm(correction), 1e-8))
    right[:2] += .012*(left[:2]-right[:2])/max(np.linalg.norm(left[:2]-right[:2]), 1e-8)
    right, _ = yield from g._acquire_edge(ctx, "right", right, quats[1], .040, targets[1]-right)
    center = (points["left_shoulder"]+points["right_shoulder"])/2
    center[1] -= .010
    targets = g.spread_fold_targets(targets, left, right, center=center)
    height = g.fold_arc_height(left, right, targets)
    tilts = [g._select_fold_path_tilt(ctx, ctx.kinematics, arm, source, target, quat, height)
             for arm, source, target, quat in zip(("left", "right"), (left, right), targets, quats)]
    for index, fraction in enumerate(np.linspace(.16, 1., 7)):
        ctx.stage = f"garment_fold_arc_{index}"
        extra = height*np.sin(np.pi*fraction)+.021*fraction
        carry = [g._fold_carry_quat(ctx.kinematics, arm, target, quat, tilt, fraction,
                    ctx.memory.get("garment_fold_closing_axis", {}).get(arm, False),
                    ctx.memory.get("garment_fold_fixed_tilt", {}).get(arm, False))
                 for arm, target, quat, tilt in zip(("left", "right"), targets, quats, tilts)]
        lp = g.clear_low_point(ctx, g._lift((1-fraction)*left+fraction*targets[0], extra), carry[0])
        rp = g.clear_low_point(ctx, g._lift((1-fraction)*right+fraction*targets[1], extra), carry[1])
        yield from ctx.move_both(lp, rp, *carry, 0., 0., ctx.memory["garment_fold_duration"])
    ctx.stage = "garment_release"
    lp = g.clear_low_point(ctx, g._lift(targets[0], .021), carry[0])
    rp = g.clear_low_point(ctx, g._lift(targets[1], .021), carry[1])
    yield from release_same_pose(ctx, lp, rp, *carry, duration=8)
    ctx.stage = "garment_home"
    yield from ctx.home(duration=18)
    yield from ctx.hold(2)
