"""Bounded raw-step stage and recovery bookkeeping from the H4/H5 closed-loop include. Preloaded; moves nothing
at load. guarded_move drives one arm with raw `step` end-effector actions (the other arm holds its measured pose);
prefer move_to, which plans the path and checks reach and budget first.
"""
import numpy as np


def guarded_move(arm, xyz, quaternion, grip, max_steps, position_tolerance=.003,
                 angle_tolerance=.08, translation_step=.02, rotation_step=.15):
    """One bounded stage of raw end-effector steps toward xyz/quaternion.

    Reaching a pose is not attachment or placement success. Stops on tolerance, lack of progress over five
    actions, episode end, or max_steps.
    """
    if arm not in ("left", "right") or max_steps < 1 or int(max_steps) != max_steps:
        raise RuntimeError("invalid arm or action budget")
    if min(position_tolerance, angle_tolerance, translation_step, rotation_step) <= 0:
        raise RuntimeError("motion bounds must be positive")
    xyz, quaternion = np.asarray(xyz, dtype=float), np.asarray(quaternion, dtype=float)
    if xyz.shape != (3,) or quaternion.shape != (4,) or not np.all(np.isfinite(xyz)) or \
            not np.all(np.isfinite(quaternion)):
        raise RuntimeError("finite XYZ and quaternion required")
    if np.linalg.norm(quaternion) < 1e-12:
        raise RuntimeError("zero quaternion")
    quaternion = quaternion / np.linalg.norm(quaternion)
    declaration = {"arm": arm, "goal_pose": np.concatenate([xyz, quaternion]).tolist(),
                   "position_tolerance": float(position_tolerance),
                   "orientation_tolerance": float(angle_tolerance), "max_calls": int(max_steps)}
    other = "right" if arm == "left" else "left"
    obs = get_observation()
    hold = obs["state"][other + "_ee_pose"].copy()
    hold_grip = [float(arm_status()[other]["gripper_command"])]
    errors = []
    last_receipt = None
    for index in range(int(max_steps)):
        current = obs["state"][arm + "_ee_pose"]
        delta = xyz - current[:3]
        command_xyz = current[:3] + delta * min(1., translation_step / max(float(np.linalg.norm(delta)), 1e-12))
        q0 = current[3:] / max(float(np.linalg.norm(current[3:])), 1e-12)
        q1 = quaternion if np.dot(q0, quaternion) >= 0 else -quaternion
        theta = float(np.arccos(np.clip(np.dot(q0, q1), -1., 1.)))
        fraction = min(1., rotation_step / max(2 * theta, 1e-12))
        if theta < 1e-6:
            q = q1
        else:
            q = (np.sin((1 - fraction) * theta) * q0 + np.sin(fraction * theta) * q1) / np.sin(theta)
        action = {arm + "_ee_pose": np.concatenate([command_xyz, q]),
                  other + "_ee_pose": hold, arm + "_ee_joint_state": [grip],
                  other + "_ee_joint_state": hold_grip}
        obs, reward, terminated, truncated, info = step(action, stage=declaration)
        last_receipt = info.get("execution_receipt")
        if terminated or truncated:
            return {"status": "episode_over", "calls": index + 1, "receipt": last_receipt}
        pose = obs["state"][arm + "_ee_pose"]
        distance = float(np.linalg.norm(pose[:3] - xyz))
        angle = 2 * float(np.arccos(np.clip(abs(np.dot(pose[3:] / max(float(np.linalg.norm(pose[3:])), 1e-12),
                                                       quaternion)), 0., 1.)))
        errors.append((distance, angle))
        if distance <= position_tolerance and angle <= angle_tolerance:
            return {"status": "tracking_reached", "calls": index + 1, "position_error_m": distance,
                    "orientation_error_rad": angle, "attachment": "unknown", "placement": "unknown",
                    "receipt": last_receipt}
        if len(errors) >= 6:
            old = errors[-6]
            recent = errors[-5:]
            position_stall = distance > position_tolerance and old[0] - min(e[0] for e in recent) < .001
            rotation_stall = angle > angle_tolerance and old[1] - min(e[1] for e in recent) < .01
            if position_stall or rotation_stall:
                return {"status": "tracking_stalled", "calls": index + 1, "position_error_m": distance,
                        "orientation_error_rad": angle, "attachment": "unknown", "receipt": last_receipt}
    return {"status": "stage_budget_exhausted", "calls": int(max_steps), "receipt": last_receipt,
            "attachment": "unknown", "placement": "unknown"}


def bounded_recovery(history, stage, reason, observation_ref, changed_parameter, max_attempts=2):
    """Account for an agent-selected recovery; no motion and no automatic semantics.

    Keep `history` (a list) in the persistent namespace. Describe fresh evidence and a real changed parameter
    before retrying a failed stage.
    """
    if not observation_ref or not changed_parameter or max_attempts < 1:
        raise RuntimeError("recovery requires current observation, a change, and finite budget")
    previous = [r for r in history if r["stage"] == stage]
    if len(previous) >= max_attempts:
        raise RuntimeError("stage recovery budget exhausted; replan or defer this subgoal")
    if any(r["changed_parameter"] == changed_parameter for r in previous):
        raise RuntimeError("same recovery already attempted; select a different hypothesis")
    row = {"stage": stage, "reason": reason, "observation": observation_ref,
           "changed_parameter": changed_parameter, "attempt": len(previous) + 1,
           "basis": "agent_selected_not_automatically_verified"}
    history.append(row)
    return row
