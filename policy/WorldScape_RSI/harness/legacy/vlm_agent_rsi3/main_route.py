"""Discrete commands on RoboDojo, each executed as one whole motion.

This is the RoboDojo counterpart of ``RoboTwinDiscreteEnv.execute`` in
``harness/robotwin/env.py``. The model answers a turn with a few command
strings (:mod:`commands`); the policy server parses them and ships the list to
the simulation client as a ``sequence`` envelope (:mod:`motion`); the client
runs them here one after another. Every command is resolved from the arm's
*live* pose at that moment, so a command that failed leaves the next one
relative to where the arm really is; every command is run as one complete
planned motion; and every command is answered with an ``ok`` / ``FAILED``
result whose note says what actually happened, which the model reads on its
next turn.

Two things differ from RoboTwin and are stated plainly:

* the control point is the fingertip centre (tool centre point), 0.145 m ahead
  of the wrist that RoboDojo reports and plans for. ``move`` translates that
  point, ``rotate`` turns the gripper about a world axis *through* that point,
  ``point`` snaps the orientation while keeping that point fixed; the wrist
  target handed to the planner is derived here;
* RoboDojo charges every 25 Hz control step to the episode budget, so a whole
  motion costs 10-20 steps rather than RoboTwin's one. The result records the
  steps each command consumed so the model can budget.

Only numpy (through :mod:`controller`) is imported: the simulation client runs
the executor, the policy server only uses the state helpers.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import types
from pathlib import Path
from typing import Any

import numpy as np


def _sibling(name: str):
    if __package__:
        return importlib.import_module(f".{name}", __package__)
    spec = importlib.util.spec_from_file_location(f"vlm_agent_{name}", Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


controller = _sibling("controller")
commands = _sibling("commands")
reward_feedback = _sibling("reward_feedback")

ENV_IDX = 0  # eval_batch is false, so RoboDojo forces num_envs to 1.

# Fingertip centre ahead of the wrist along the fingers (grounding.tool_tip uses the same value).
TCP_OFFSET_M = 0.145
# A motion command counts as reached within these tolerances (harness/robotwin/env.py).
POSITION_TOLERANCE_CM = 0.5
ORIENTATION_TOLERANCE_DEG = 4.0
# Below these the arm is reported as not having moved at all.
STILL_CM = 0.5
STILL_DEG = 2.0
# Planned motions complete their full schedules; the episode cap remains authoritative.
DEFAULT_HOME_PLAN_STEPS = 10000
DEFAULT_MAX_PLAN_STEPS = 10000
# Maximum joint displacement per control step (0.5 rad/s at 25 Hz).
JOINT_RETREAT_RAD_PER_STEP = 0.02
JOINT_RETREAT_MIN_STEPS = 8
JOINT_RETREAT_MAX_STEPS = 10000

# Fingers straight down, tilted towards +y in 15 degree steps, or straight forward (+y).
PRESET_DIRECTIONS = {
    "down5in": (0.050795547, 0.070821318, -0.996194698),
    "down30y90": (-0.5, 0.0, -0.8660254037844387),
    "down": (0.0, 0.0, -1.0),
    "forward": (0.0, 1.0, 0.0),
    "down15": (0.0, 0.2588190451025207, -0.9659258262890683),
    "down30": (0.0, 0.4999999999999999, -0.8660254037844387),
    "down45": (0.0, 0.7071067811865475, -0.7071067811865476),
    "down60": (0.0, 0.8660254037844386, -0.5000000000000001),
    "down75": (0.0, 0.9659258262890683, -0.2588190451025207),
}
WORLD_AXES = {"roll": (1.0, 0.0, 0.0), "pitch": (0.0, 1.0, 0.0), "yaw": (0.0, 0.0, 1.0)}

def preset_quat(current, preset):
    if preset == "down30y90":
        base = controller.approach_quat(current, np.array(PRESET_DIRECTIONS["down30"]), 0.0)
        yaw = controller._axis_angle_quat(np.array([0.0, 0.0, 1.0]), math.pi / 2)
        return controller._normalize_quat(controller.quat_mul(yaw, base))
    return controller.approach_quat(current, np.array(PRESET_DIRECTIONS[preset]), 0.0)


# ------------------------------------------------------------------ geometry
def forward_axis(quat: Any) -> np.ndarray:
    """Direction the fingers point (link6 local +X)."""
    return controller.forward_axis(np.asarray(quat, dtype=float))


def finger_axis(quat: Any) -> np.ndarray:
    """Direction the fingers open and close along (link6 local +Y)."""
    return controller._rotate_vector(controller._normalize_quat(np.asarray(quat, dtype=float)), np.array([0.0, 1.0, 0.0]))


def tcp_of(pos: Any, quat: Any, offset: float = TCP_OFFSET_M) -> np.ndarray:
    return np.asarray(pos, dtype=float).reshape(3) + float(offset) * forward_axis(quat)


def wrist_of(tcp: Any, quat: Any, offset: float = TCP_OFFSET_M) -> np.ndarray:
    return np.asarray(tcp, dtype=float).reshape(3) - float(offset) * forward_axis(quat)


def quat_angle_deg(q1: Any, q2: Any) -> float:
    a = controller._normalize_quat(np.asarray(q1, dtype=float))
    b = controller._normalize_quat(np.asarray(q2, dtype=float))
    return math.degrees(2.0 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


def arm_state(prefix: str, pos: Any, quat: Any, opening: float | None = None, closed_floor: float | None = None,
              offset: float = TCP_OFFSET_M, grasp_open_threshold: float = 0.05) -> dict:
    """The per-arm state block of the prompt (``main_prompts.state_text``), in centimetres."""
    pos = np.asarray(pos, dtype=float).reshape(3)
    quat = controller._normalize_quat(np.asarray(quat, dtype=float).reshape(4))
    tcp = tcp_of(pos, quat, offset)
    held = None
    if opening is not None and closed_floor is not None:
        held = bool(float(opening) - float(closed_floor) > grasp_open_threshold)
    return {
        "position_cm": [round(float(v) * 100.0, 1) for v in tcp],
        "wrist_cm": [round(float(v) * 100.0, 1) for v in pos],
        "approach": [round(float(v), 2) for v in forward_axis(quat)],
        "finger_axis": [round(float(v), 2) for v in finger_axis(quat)],
        "quat_wxyz": [round(float(v), 4) for v in quat],
        "opening": None if opening is None else round(float(opening), 3),
        "closed_floor": None if closed_floor is None else round(float(closed_floor), 3),
        "held": held,
    }


def resolve(cmd: Any, pos: Any, quat: Any, home_pose: Any = None, offset: float = TCP_OFFSET_M) -> dict:
    """Wrist target for one motion command, from the arm's current wrist pose.

    Returns ``{"pos", "quat", "tcp"}``: the wrist target the planner is handed
    and the fingertip centre it corresponds to (what the reached-pose check
    compares against).
    """
    pos = np.asarray(pos, dtype=float).reshape(3)
    quat = controller._normalize_quat(np.asarray(quat, dtype=float).reshape(4))
    tcp = tcp_of(pos, quat, offset)
    if cmd.kind == "pose3d":
        target_tcp = tcp + np.asarray(cmd.value, dtype=float).reshape(3) / 100.0
        target_quat = preset_quat(quat, cmd.axis)
    elif cmd.kind in ("move3d", "push3d"):
        target_tcp = tcp + np.asarray(cmd.value, dtype=float).reshape(3) / 100.0
        target_quat = quat
    elif cmd.kind == "move":
        delta = np.zeros(3)
        delta["xyz".index(cmd.axis)] = float(cmd.value) / 100.0
        target_tcp = tcp + delta
        target_quat = quat
    elif cmd.kind == "rotate":
        turn = controller._axis_angle_quat(np.array(WORLD_AXES[cmd.axis]), math.radians(float(cmd.value)))
        target_quat = controller._normalize_quat(controller.quat_mul(turn, quat))
        target_tcp = tcp
    elif cmd.kind == "point":
        target_quat = preset_quat(quat, cmd.axis)
        target_tcp = tcp
    elif cmd.kind in ("home", "homeplan"):
        if home_pose is None:
            raise commands.CommandError("home pose unknown")
        home = np.asarray(home_pose, dtype=float).reshape(7)
        target_quat = controller._normalize_quat(home[3:])
        return {"pos": home[:3].copy(), "quat": target_quat, "tcp": tcp_of(home[:3], target_quat, offset)}
    else:
        raise commands.CommandError(f"{cmd.kind} is not a motion command")
    return {"pos": wrist_of(target_tcp, target_quat, offset), "quat": target_quat, "tcp": target_tcp}


def motion_outcome(cmd: Any, target: dict, before: tuple, after: tuple, offset: float = TCP_OFFSET_M,
                   suffix: str = "") -> tuple[bool, str, dict]:
    """Judge a motion by where the fingertips ended up, and say what happened when they did not arrive."""
    before_tcp = tcp_of(before[0], before[1], offset)
    after_tcp = tcp_of(after[0], after[1], offset)
    pos_err_cm = float(np.linalg.norm(after_tcp - target["tcp"])) * 100.0
    ang_err = quat_angle_deg(after[1], target["quat"])
    moved = (after_tcp - before_tcp) * 100.0
    moved_cm = float(np.linalg.norm(moved))
    turned = quat_angle_deg(after[1], before[1])
    metrics = {"position_error_cm": round(pos_err_cm, 2), "orientation_error_deg": round(ang_err, 1),
               "moved_cm": round(moved_cm, 2), "turned_deg": round(turned, 1)}
    if pos_err_cm <= POSITION_TOLERANCE_CM and ang_err <= ORIENTATION_TOLERANCE_DEG:
        return True, "", metrics
    if moved_cm < STILL_CM and turned < STILL_DEG:
        what = "the arm did NOT move"
    elif cmd.kind == "rotate":
        what = f"the gripper turned only {turned:.0f} deg of the requested {abs(float(cmd.value)):.0f}"
    else:
        what = (f"the fingertips moved ({moved[0]:+.1f}, {moved[1]:+.1f}, {moved[2]:+.1f}) cm, i.e. only part "
                f"of the way")
    hint = ""
    if (not suffix and cmd.kind == "move" and getattr(cmd, "axis", None) == "z" and cmd.value is not None
            and float(cmd.value) < 0 and moved[2] > 0.5 * float(cmd.value)):
        # a descent that stops short is the fingers resting on something, usually the top of the
        # object; closing there closes on nothing
        hint = (" The descent stopped early: the fingers are resting ON something - most likely the top of the "
                "object (the fingers are not straddling it) or the table. Open, lift 3 cm, shift by the offset "
                "the wrist-camera cross shows, and descend again.")
    note = (f"the target was not reached: {what}; remaining error {pos_err_cm:.1f} cm / {ang_err:.0f} deg "
            f"(see CURRENT STATE for the actual pose){suffix}. Try a smaller step, a different direction, or move "
            f"away from the table/robot body first.{hint}")
    return False, note, metrics


def gripper_outcome(value: float, opening: float | None, floor: float | None, threshold: float,
                    status: str | None) -> tuple[bool | None, str]:
    """``(held, note)`` after a gripper command, from the measured finger opening."""
    if value >= 0.5:
        return (False if opening is not None else None), "gripper opened"
    if opening is None or floor is None:
        return None, "fingers closed (opening could not be measured)"
    if status not in (None, "settled", "closed_at_limit", "disabled"):
        return None, f"fingers still moving ({status}); opening {opening:.2f} is not a settled reading"
    if opening - max(floor, 0.0) > threshold:
        return True, f"contact detected at opening {opening:.2f}; target retention is unverified, inspect a short lift"
    # Thin straps were retained at opening 0.20-0.22 in real collection runs.
    # Width alone cannot distinguish a thin grasp from an empty closure.
    return None, (f"fingers near closed limit (opening {opening:.2f}): grasp uncertain; "
                  "a thin object may still be held. Lift and verify visually before transport.")


# ------------------------------------------------------------------ simulator side
def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("pose contains nonfinite values")
    return array


def target_arms(env) -> list[tuple[str, Any]]:
    return [(robot.arm_name.split("_")[0], robot)
            for robot in env.robot_manager.robot_list if robot.type == "target"]


def live_poses(env) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Wrist pose of every target arm, read from the robot rather than the last observation."""
    poses = {}
    for prefix, robot in target_arms(env):
        pose = _array(env.robot_manager.get_real_endpose(robot, env_idx_list=[ENV_IDX])[ENV_IDX]).reshape(7)
        poses[prefix] = (pose[:3].copy(), controller._normalize_quat(pose[3:]))
    return poses


def steps_used(env) -> int:
    return int(env.take_action_cnt[ENV_IDX])


def _hold_actions(env, hooks, grippers: dict, count: int, poses: dict) -> list[dict]:
    """Joint-space holds when the planner module is usable, end-effector holds otherwise."""
    try:
        return hooks.joint_module.hold_actions(env, grippers, count)
    except Exception:  # noqa: BLE001 - an ee hold is the same thing on a robot without a planner
        states = {prefix: controller.ArmState(prefix, pos, quat, grippers.get(prefix, 1.0))
                  for prefix, (pos, quat) in poses.items()}
        return [controller.action_dict(states) for _ in range(max(1, int(count)))]



def _read_joints(env, hooks: Any) -> dict[str, list[float]]:
    """Current joint positions per arm through the planner helper module; empty without one."""
    module = getattr(hooks, "joint_module", None)
    if module is None:
        return {}
    try:
        return {prefix: [float(v) for v in module.current_joints(env, robot)]
                for prefix, robot in module.target_arms(env)}
    except Exception:  # noqa: BLE001 - a robot without joint access simply has no joint-space retreat
        return {}


def _joint_home_actions(env, hooks: Any, arm: str, session: dict, grippers: dict) -> list[dict]:
    """A direct joint-space path back to the joints recorded on the first turn; empty when unavailable.

    The planner refuses a good share of the ``home`` commands on X5, typically
    from a stretched or folded pose, and an arm that cannot go home wastes the
    rest of the episode. Interpolating the joints ignores obstacles; this is
    the explicit home operation and the result reports it. All ordinary moves
    continue to use cuRobo.
    """
    goal = (session.get("home_joints") or {}).get(arm)
    module = getattr(hooks, "joint_module", None)
    if goal is None or module is None:
        return []
    try:
        joints = _read_joints(env, hooks)
        start = np.asarray(joints[arm], dtype=float)
        target = np.asarray(goal, dtype=float)
        if start.shape != target.shape or start.size == 0:
            return []
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(target)):
            return []
        span = float(np.max(np.abs(target - start)))
        steps = int(min(JOINT_RETREAT_MAX_STEPS,
                        max(JOINT_RETREAT_MIN_STEPS, math.ceil(math.pi / 2 * span / JOINT_RETREAT_RAD_PER_STEP))))
        actions = []
        for step in range(1, steps + 1):
            fraction = 0.5 - 0.5 * math.cos(math.pi * step / steps)  # ease in and out
            targets = dict(joints)
            targets[arm] = (start + fraction * (target - start)).tolist()
            actions.append(module.joint_action(env, targets, dict(grippers)))
        return actions
    except Exception:  # noqa: BLE001 - without joint access there is simply no retreat
        return []


def planner_failure_note(reachable: bool | None, error: str | None = None) -> str:
    """Explain a planner failure so the model changes strategy instead of retrying the same command."""
    detail = f" ({error})" if error else ""
    if reachable is False:
        return ("Planning failed and the collision-aware IK diagnostic also found no accepted solution. "
                f"This finite search does NOT prove geometric unreachability{detail}. No motion executed. "
                "Keep the actual task target; inspect pose constraints, frames, collision settings and search "
                "diagnostics before changing the approach. Never substitute an easier off-slot landing.")
    if reachable:
        return ("a joint solution exists but no collision-free path was found by cuRobo; its collision world "
                f"checks the table and self-collision, not tabletop objects. The arm did NOT move{detail}. Lift straight up a "
                "few cm first, move back toward the base, or take a smaller step.")
    return (f"motion planner could not find a path to the target: the arm did NOT move{detail}. Try a smaller "
            "step, a different direction, or move away from the table/robot body first.")


RETREAT_NOTE = ("home reached by a direct joint-space return without collision planning; objects near "
                "the arm may have been pushed - check the images")


def run_sequence(env, envelope: dict, config: dict, hooks: Any, session: dict, deadline: float,
                 control_dt: float, commanded_grippers: dict) -> tuple[dict, str | None]:
    """Execute one turn of commands and describe every one of them.

    ``hooks`` supplies the simulation-client primitives (``execute_control``,
    ``plan_keypose``, ``gripper_real``, ``gripper_closed_floor``,
    ``stop_reason``, ``ik_ok``, ``settle_module``, ``joint_module``) so this
    module stays testable without Isaac Sim. ``session`` persists across the
    turns of one episode (the home poses ``home`` returns to).
    """
    reward_feedback.begin_batch(env, envelope)
    options = dict(envelope.get("options") or {})
    cfg = controller.ControllerConfig(**dict(config.get("controller") or {}))
    offset = float(options.get("tcp_offset_m", TCP_OFFSET_M))
    hold_steps = int(options.get("hold_steps", cfg.hold_steps))
    gripper_steps = int(options.get("gripper_steps", cfg.gripper_steps))
    home_plan_steps = int(options.get("home_plan_steps", DEFAULT_HOME_PLAN_STEPS))
    max_plan_steps = int(options.get("max_plan_steps", DEFAULT_MAX_PLAN_STEPS))
    motion_settle_steps = int(options.get("motion_settle_steps", 10))
    threshold = float(options.get("grasp_open_threshold", config.get("grasp_open_threshold", 0.05)))
    settle_module = hooks.settle_module

    report: dict[str, Any] = {
        "requested": 0, "executed": 0, "ik_failed": 0, "ik_failed_key": None, "ik_failed_pose": None,
        "motion_requested": 0, "motion_executed": 0, "settle_steps": 0, "plan_status": None,
        "backend": "curobo", "commands_executed": 0, "command_results": [],
    }
    settle_reports: dict = {}
    stopped = None

    if session.get("home_poses") is None:
        session["home_poses"] = {prefix: [*pos, *quat] for prefix, (pos, quat) in live_poses(env).items()}
    if session.get("home_joints") is None:
        session["home_joints"] = _read_joints(env, hooks)

    parsed = []
    for raw in envelope.get("commands") or []:
        try:
            parsed.append(commands.Command.from_dict(raw))
        except commands.CommandError as exc:
            report["command_results"].append({"command": "(invalid)", "kind": "invalid", "arm": None,
                                              "ok": False, "note": str(exc)})

    def run_actions(actions: list[dict], settle, planned_steps=None) -> tuple[int, str | None]:
        """Run env actions until the episode stops; returns (steps, stop)."""
        before = report["executed"]
        stop = None
        if not result.get('budget_assessment'):
            full_cost=(len(actions) if planned_steps is None else planned_steps)+motion_settle_steps+int(math.ceil(settle.max_s/control_dt))
            verdict=reward_feedback.budget_primitive(env,full_cost,
                is_finishing=cmd.kind in ('home','homeplan') and (envelope.get('plan',{}).get('stage_context') or {}).get('phase')=='finish',kind=cmd.kind)
            result['budget_assessment']=verdict
            if verdict['decision']!='allow':
                result.update(ok=False,note='Full primitive plus final reset reserve does not fit native remaining steps')
                return 0,'supervisor_budget'
        report["requested"] += len(actions)
        report["motion_requested"] += len(actions)
        for action in actions:
            stop = hooks.stop_reason(env, deadline)
            if stop:
                break
            hooks.execute_control(env, action, report, settle, settle_module)
        return report["executed"] - before, stop

    for cmd in parsed:
        stopped = hooks.stop_reason(env, deadline)
        if stopped:
            break
        poses = live_poses(env)
        opening_before = hooks.gripper_real(env)
        result: dict[str, Any] = {"command": cmd.text(), "kind": cmd.kind, "arm": cmd.arm, "ok": True, "note": "",
                                  "steps_before": steps_used(env)}
        if cmd.kind == "homeplan":
            joints = _read_joints(env, hooks)
            home = (session.get("home_joints") or {}).get(cmd.arm)
            if home is not None and cmd.arm in joints:
                current_j = np.asarray(joints[cmd.arm], dtype=float)
                home_j = np.asarray(home, dtype=float)
                home_pose = session["home_poses"][cmd.arm]
                p, q = poses[cmd.arm]
                if (current_j.shape == home_j.shape and current_j.size
                        and np.max(np.abs(current_j - home_j)) <= 0.01
                        and np.linalg.norm(tcp_of(p, q, offset) - tcp_of(home_pose[:3], home_pose[3:], offset)) <= 0.003
                        and quat_angle_deg(q, home_pose[3:]) <= 2.0):
                    result.update(note="already at recorded home; no motion needed", steps_after=steps_used(env), steps_cost=0)
                    reward_feedback.note_primitive(env, result)
                    report["command_results"].append(result)
                    report["commands_executed"] += 1
                    continue
        settle = settle_module.GripperSettle(config.get("gripper_settle"), control_dt, commanded_grippers)

        if cmd.kind == "done":
            # The policy server answers "done" itself; a client that still receives one treats it as a wait.
            cmd = commands.Command("wait", raw=cmd.raw)
            result.update(command="wait", kind="wait")

        if cmd.kind == "wait":
            actions = _hold_actions(env, hooks, dict(commanded_grippers), hold_steps, poses)
            executed, stopped = run_actions(actions, settle)
            result["note"] = "waited"

        elif cmd.kind == "sync" and all(v["kind"] != "gripper" for v in cmd.value):
            sync = _sibling("sync_route")
            outcome, stopped = sync.execute(types.SimpleNamespace(**globals()), cmd, env, hooks, session, poses,
                commanded_grippers, settle, run_actions, report, offset, hold_steps, max_plan_steps, motion_settle_steps)
            result.update(outcome)

        elif cmd.kind == "sync":
            grippers = dict(commanded_grippers)
            grippers.update({v["arm"]: float(v["value"]) for v in cmd.value})
            settle.expect(grippers)
            actions = _hold_actions(env, hooks, grippers, gripper_steps, poses)
            executed, stopped = run_actions(actions, settle)
            # Keep the original joint targets: re-reading joints here lets contact drift
            # become a new hold target on every settling step.
            hold = actions[-1]
            while settle.pending and not stopped:
                _, stopped = run_actions([hold], settle)
            if stopped: settle.interrupt(stopped)
            commanded_grippers.update(grippers)
            settle_reports.update(settle.report())
            contacts = {}
            for child in cmd.value:
                arm, value = child["arm"], float(child["value"])
                opening = hooks.gripper_real(env).get(arm)
                floor = hooks.gripper_closed_floor(env).get(arm)
                status = (settle.report().get(arm) or {}).get("status")
                held, note = gripper_outcome(value, opening, floor, threshold, status)
                contacts[arm] = {"value": value, "held": held, "opening": opening,
                                 "closed_floor": floor, "note": note}
            result.update(note="both grippers commanded and observed together", settle=settle.report(),
                          gripper_contacts=contacts)

        elif cmd.kind == "gripper":
            grippers = dict(commanded_grippers)
            grippers[cmd.arm] = float(cmd.value)
            settle.expect(grippers)
            actions = _hold_actions(env, hooks, grippers, gripper_steps, poses)
            executed, stopped = run_actions(actions, settle)
            # Preserve the original arm target while fingers settle. Re-reading
            # joints here turns contact displacement into the new hold target.
            hold = actions[-1]
            while settle.pending and not stopped:
                stopped = hooks.stop_reason(env, deadline)
                if stopped:
                    break
                if hold is None:
                    hold = _hold_actions(env, hooks, grippers, 1, poses)[0]
                report["requested"] += 1
                hooks.execute_control(env, hold, report, settle, settle_module, added=True)
            if stopped:
                settle.interrupt(stopped)
            commanded_grippers[cmd.arm] = float(cmd.value)
            settle_report = settle.report()
            settle_reports.update(settle_report)
            opening = hooks.gripper_real(env).get(cmd.arm)
            floor = hooks.gripper_closed_floor(env).get(cmd.arm)
            status = (settle_report.get(cmd.arm) or {}).get("status")
            held, note = gripper_outcome(float(cmd.value), opening, floor, threshold, status)
            result.update(value=float(cmd.value), note=note, held=held, opening=opening, closed_floor=floor,
                          settle=settle_report.get(cmd.arm))

        elif cmd.kind == "push3d":
            result.update(ok=False, note="push3d currently requires sync with both arms")

        else:  # move / rotate / point / home
            pos, quat = poses[cmd.arm]
            try:
                target = resolve(cmd, pos, quat, session["home_poses"].get(cmd.arm), offset)
            except commands.CommandError as exc:
                result.update(ok=False, note=str(exc))
                result.update(steps_after=steps_used(env),steps_cost=0)
                reward_feedback.note_primitive(env, result)
                report["command_results"].append(result)
                report["commands_executed"] += 1
                report["batch_aborted"] = "invalid_target"
                break
            result.update(target_wrist=[float(v) for v in (*target["pos"], *target["quat"])],
                          target_tcp_cm=[round(float(v) * 100.0, 1) for v in target["tcp"]])
            executed = 0
            suffix = ""
            home_actions = (_joint_home_actions(env, hooks, cmd.arm, session, commanded_grippers)
                            if cmd.kind == "home" else [])
            if home_actions:
                result.update(retreat="joint_path", plan_status="joint_home", truncated=False)
                actions = home_actions + [home_actions[-1]] * motion_settle_steps
                executed, stopped = run_actions(actions, settle)
            else:
                request = {"arm": cmd.arm, "target_pose": [float(v) for v in (*target["pos"], *target["quat"])],
                           "start_pose": [float(v) for v in (*pos, *quat)], "grippers": dict(commanded_grippers),
                           "executed_motion": True, "settle_steps": hold_steps, "duration_s": None}
                if cmd.kind == "homeplan":
                    request["target_joints"] = (session.get("home_joints") or {}).get(cmd.arm)
                request["max_plan_steps"] = home_plan_steps if cmd.kind in ("home", "homeplan") else max_plan_steps
                actions, info = hooks.plan_keypose(env, request)
                result.update(plan_status=info.get("plan_status"), plan_points=info.get("plan_points"),
                              plan_seconds=info.get("plan_seconds"), truncated=bool(info.get("truncated")))
                report["plan_status"] = info.get("plan_status")
                if info.get("plan_status") != "Success" or not actions:
                    error = info.get("planner_error")
                    ik_ok = getattr(hooks, "ik_ok", None)
                    reachable = ik_ok(env, cmd.arm, request["target_pose"]) if ik_ok is not None else None
                    result.update(ok=False, blocked_reason="planner", reachable=reachable,
                                  note=planner_failure_note(reachable, error))
                else:
                    # Execute the COMPLETE planner schedule, then settle from observed robot pose.
                    # Never shorten the trajectory, increase speed, or alter native termination.
                    executed, stopped = run_actions(actions, settle)
                    extra = 0
                    stable = 0
                    previous = live_poses(env)[cmd.arm]
                    for _ in range(motion_settle_steps):
                        if stopped:
                            break
                        count, stopped = run_actions([actions[-1]], settle)
                        extra += count
                        current = live_poses(env)[cmd.arm]
                        close = (np.linalg.norm(tcp_of(*current, offset) - target["tcp"]) <= 0.003
                                 and quat_angle_deg(current[1], target["quat"]) <= 2.0)
                        still = (np.linalg.norm(tcp_of(*current, offset) - tcp_of(*previous, offset)) <= 0.0005
                                 and quat_angle_deg(current[1], previous[1]) <= 0.2)
                        stable = stable + 1 if close and still else 0
                        previous = current
                        if stable >= 2:
                            break
                    executed += extra
                    result["settle_executed"] = extra
                    result["settle_saved"] = motion_settle_steps - extra
                    if info.get("truncated"):
                        suffix = " (the motion was cut by the per-command step limit)"
            if result["ok"]:
                after = live_poses(env)[cmd.arm]
                ok, note, metrics = motion_outcome(cmd, target, (pos, quat), after, offset, suffix)
                if ok and result.get("retreat"):
                    note = RETREAT_NOTE
                result.update(ok=ok, note=note, **metrics)
            result["motion_steps"] = executed

        grip_statuses = ({cmd.arm: (result.get("settle") or {}).get("status")} if cmd.kind == "gripper" else
                         {v["arm"]: (result.get("settle", {}).get(v["arm"]) or {}).get("status")
                          for v in cmd.value} if cmd.kind == "sync" and all(v["kind"] == "gripper" for v in cmd.value) else {})
        unsettled = {arm: status for arm, status in grip_statuses.items()
                     if status not in (None, "settled", "closed_at_limit", "disabled")}
        if unsettled:
            result.update(ok=False, gripper_acceptance="needs_observation", unsettled_grippers=unsettled)
            result["note"] += " Gripper settling unverified; remaining batch paused before dependent motion."

        if cmd.clipped:
            result["note"] = (result["note"] + " " if result["note"] else "") + "(magnitude was clipped to the limit)"
        if getattr(cmd, "rounded", False):
            result["note"] = (result["note"] + " " if result["note"] else "") + f"(rounded to the 15 deg step: {cmd.value:+.0f})"
        result["steps_after"] = steps_used(env)
        result["steps_cost"] = result["steps_after"] - result["steps_before"]
        result['native_terminal'] = stopped == 'native'
        measured = hooks.gripper_real(env)
        floors = hooks.gripper_closed_floor(env)
        result["after"] = {prefix: arm_state(prefix, p, q, measured.get(prefix), floors.get(prefix), offset, threshold)
                           for prefix, (p, q) in live_poses(env).items()}
        result["opening_before"] = opening_before
        # Check both directions after each primitive; no hidden scene state.
        suspect = []
        changes = {}
        sync_grip = cmd.kind == "sync" and all(v['kind'] == 'gripper' for v in cmd.value)
        if cmd.kind not in ("gripper", "wait") and not sync_grip:
            for arm in ("left", "right"):
                change = retention_change(opening_before.get(arm), measured.get(arm), floors.get(arm),
                                          float(commanded_grippers.get(arm, 1.0)))
                if change:
                    suspect.append(arm)
                    changes[arm] = change
        if suspect:
            result["retention_warning"] = suspect
            result["retention_change"] = changes
            result["note"] += " Abrupt gripper opening change on " + ", ".join(suspect) + "; possible slip, pivot, wedging, or changed contact. Remaining batch paused for visual inspection."
            report["batch_aborted"] = "retention_uncertain"
        reward_feedback.note_primitive(env, result)
        report["command_results"].append(result)
        report["commands_executed"] += 1
        if stopped:
            break
        if result.get("ok") is False:
            report["batch_aborted"] = "command_failed"
            break
        if suspect:
            break
        if env.is_episode_end():
            stopped = "native"
            break

    report["gripper_settle"] = settle_reports
    return report, stopped


def retention_change(prior, current, floor, commanded):
    """Proprioceptive anomaly only; never asserts object identity or loss."""
    if None in (prior, current, floor) or commanded >= 0.5:
        return None
    if not all(math.isfinite(float(v)) for v in (prior, current, floor, commanded)):
        return None
    if prior - floor <= 0.08:
        return None
    if current - floor <= 0.025 or (prior - current > 0.12 and current - floor < 0.5 * (prior - floor)):
        return 'collapse'
    if current - prior > 0.12:
        return 'expansion'
    return None


def update_contact_facts(facts: dict, results: list[dict]) -> None:
    """Remember measured contacts, including sync commands; invalidate after warnings.

    This contains no object identity, support height, or assertion of a stable grasp.
    """
    for result in results:
        for arm in result.get("retention_warning") or []:
            facts.pop(arm, None)
        for arm in result.get("unsettled_grippers") or {}:
            facts.pop(arm, None)
        if not result.get("ok"):
            continue
        contacts = result.get("gripper_contacts") or {}
        if result.get("kind") == "gripper" and result.get("arm") in ("left", "right"):
            contacts = {result["arm"]: result}
        for arm, contact in contacts.items():
            value = contact.get("value")
            if value is None:
                continue
            facts.pop(arm, None)
            if float(value) < 0.5 and contact.get("held"):
                position = ((result.get("after") or {}).get(arm) or {}).get("position_cm") or []
                if len(position) == 3:
                    facts[arm] = {"grasp_z": round(float(position[2]), 1),
                                  "opening": contact.get("opening")}
