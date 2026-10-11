"""Episode logic of robo-server, independent of how the robot is driven.

An Executor provides the robot state, runs joint-target chunks and returns
observations. Two executors exist: the direct one drives the RoboDojo
environment in-process (main.py); the bridge one hands the chunks to the
official evaluation client through an XPolicyLab policy server (bridge.py).
Everything the agent sees is produced here, so both modes behave the same.
"""

import io
import json
import os
import time

import numpy as np

from roboshell.contract import v0 as C
from roboshell.server import geometry as geo
from roboshell.server import motion
from roboshell.server import tools as toolkit

CAMERA_NAMES = {"cam_head": "head", "cam_left_wrist": "wrist_l", "cam_right_wrist": "wrist_r"}
SUCCESS_THRESHOLD = 1 - 1e-3
# No waiting of our own after a motion. The only wait is physical: the arm must have arrived before the command returns,
# otherwise the reported pose and the next command would be wrong. It is measured, not assumed: hold until the joints are
# within SETTLE_TOL_RAD of the commanded target, at most SETTLE_MAX_STEPS control steps (usually none).
SETTLE_STEPS = int(os.environ.get("ROBOSHELL_SETTLE_STEPS", 0))        # extra fixed holds, none by default
SETTLE_TOL_RAD = float(os.environ.get("ROBOSHELL_SETTLE_TOL", 0.01))
SETTLE_MAX_STEPS = int(os.environ.get("ROBOSHELL_SETTLE_MAX", 10))
# Control steps the official gripper actuator needs for a full stroke (measured in the simulator).
GRIPPER_STEPS = int(os.environ.get("ROBOSHELL_GRIPPER_STEPS", 8))
WORKSPACE = {"x": (-0.75, 0.75), "y": (-0.75, 0.60), "z": (0.74, 1.45)}
APPROACH = {"down": (0.0, 0.0, -1.0), "forward": (0.0, 1.0, 0.0), "down45": (0.0, 0.7071068, -0.7071068)}
AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
TCP_OFFSET_M = 0.145  # RoboDojo's gripper_bias: end link to the point between the fingertips, along the tool x axis


def log(message):
    print(f"[robo-server] {message}", flush=True)


def depth_preview(depth):
    """8-bit grayscale PNG: near is bright, far is dark, 0 (no data) is black. 0.2 m to 2.0 m."""
    import cv2

    valid = depth > 0
    scaled = np.clip((2.0 - depth) / 1.8, 0.0, 1.0)
    image = (scaled * 255).astype(np.uint8)
    image[~valid] = 0
    ok, encoded = cv2.imencode(".png", image)
    return encoded.tobytes()


class BadRequest(Exception):
    pass


def number(payload, key, default=0.0):
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError):
        raise BadRequest(f"`{key}` must be a number")
    if not np.isfinite(value):
        raise BadRequest(f"`{key}` must be finite")
    return value


def tool_rotation(preset, open_axis, current):
    """Tool frame: x is the approach direction, y the axis along which the fingers open, z = x cross y.

    The fingers are symmetric, so the sign of y is free: the one nearer to the current orientation is used.
    """
    approach = np.asarray(APPROACH[preset])
    axis = np.asarray(AXES[open_axis])
    across = axis - np.dot(axis, approach) * approach
    if np.linalg.norm(across) < 0.2:
        raise BadRequest(f"the fingers cannot open along {open_axis} while the gripper points {preset}")
    across = across / np.linalg.norm(across)
    candidates = [np.stack([approach, sign * across, np.cross(approach, sign * across)], axis=1) for sign in (1.0, -1.0)]
    return min(candidates, key=lambda rotation: geo.angle_between_deg(current, rotation))


class Arm:
    """One controllable arm: the executor supplies the state, this object keeps the commanded targets."""

    def __init__(self, tag, executor):
        self.tag = tag
        self.executor = executor
        self.ee_to_tcp = np.eye(4)
        self.ee_to_tcp[:3, 3] = [TCP_OFFSET_M, 0.0, 0.0]
        self.tcp_to_ee = np.linalg.inv(self.ee_to_tcp)
        self.joint_target = None
        self.gripper_target = 1.0
        self.home_joints = None

    def joints(self):
        return np.asarray(self.executor.joints(self.tag), dtype=float)

    def ee(self):
        return self.executor.ee(self.tag)

    def tcp(self):
        return self.ee() @ self.ee_to_tcp

    def gripper(self):
        """The commanded opening: the official observation carries no measured finger position."""
        return float(self.gripper_target)

    def sync(self):
        self.joint_target = self.joints()
        self.gripper_target = float(self.executor.gripper_command(self.tag))

    def report(self):
        tcp = self.tcp()
        pose = geo.matrix_to_pose(tcp)
        return {
            "tcp_pos": [round(v, 4) for v in pose[:3]],
            "tcp_quat": [round(v, 5) for v in pose[3:]],
            "tcp_rpy": [round(v, 2) for v in geo.rpy_deg(tcp[:3, :3])],
            "approach_dir": [round(float(v), 3) for v in tcp[:3, 0]],
            "open_dir": [round(float(v), 3) for v in tcp[:3, 1]],
            "gripper": round(self.gripper(), 3),
        }


class Episode:
    GRIPPER_STEPS = GRIPPER_STEPS

    def __init__(self, executor, task, seed, layout, directory, budget, registry=None):
        self.executor = executor
        self.registry = registry or {}
        self.arms = {tag: Arm(tag, executor) for tag in executor.arm_tags()}
        self.task, self.seed, self.layout = task, seed, layout
        self.directory = directory
        self.budget = budget
        self.budget_left = budget if budget > 0 else None
        self.commands = 0
        self.over = False
        self.end_reason = None
        self.done_called = False
        self.success_at_done = None
        self.last_feedback = None
        self.last_cmd = None
        self.targets = []
        self.instruction = None
        self.started = time.time()
        self.obs_cache = {}
        self.obs_saved = 0
        self.result = None
        self._plan_method = None
        os.makedirs(os.path.join(directory, "obs"), exist_ok=True)
        self.log_file = open(os.path.join(directory, "commands.jsonl"), "a")
        for arm in self.arms.values():
            arm.sync()
            arm.home_joints = arm.joints()

    # ---- simulator access -------------------------------------------------
    @property
    def steps(self):
        return int(self.executor.steps())

    def sim_time(self):
        return self.steps / motion.CONTROL_HZ

    def sim_time_left(self):
        return max(0.0, (int(self.executor.step_lim()) - self.steps) / motion.CONTROL_HZ)

    def run(self, sequences):
        """Execute joint sequences (tag -> [steps, dof]) in lockstep; arms without a sequence hold their target.

        Returns False when the episode ended during the motion.
        """
        length = max([len(s) for s in sequences.values()] + [0])
        chunk = []
        for index in range(length):
            step = {}
            for tag, arm in self.arms.items():
                if tag in sequences and len(sequences[tag]):
                    arm.joint_target = np.asarray(sequences[tag][min(index, len(sequences[tag]) - 1)], dtype=float)
                step[tag] = (arm.joint_target.copy(), float(arm.gripper_target))
            chunk.append(step)
            self.targets.append(np.concatenate([np.append(step[t][0], step[t][1]) for t in self.arms]))
        if not chunk:
            return not self.over
        alive = self.executor.run_chunk(chunk)
        if not alive and not self.over:
            self.finish(self.executor.end_reason() or "sim_time")
        return alive

    def hold(self, steps):
        return self.run({tag: np.repeat(arm.joint_target[None], steps, axis=0) for tag, arm in self.arms.items()})

    def settle(self, tags):
        """Hold until the given arms have physically reached their commanded joints. Returns the steps it took."""
        used = 0
        if SETTLE_STEPS > 0:
            self.hold(SETTLE_STEPS); used += SETTLE_STEPS
        for _ in range(SETTLE_MAX_STEPS):
            if self.over or all(float(np.abs(self.arms[tag].joints() - self.arms[tag].joint_target).max()) < SETTLE_TOL_RAD for tag in tags):
                break
            self.hold(1); used += 1
        return used

    def refresh_obs(self):
        if self.over:
            return
        observation = self.executor.observe()
        cache = {}
        cameras = {}
        for source, name in CAMERA_NAMES.items():
            cache[f"{name}.png"] = observation["png"][source]
            depth = observation.get("depth", {}).get(source)
            if depth is not None:
                depth = np.asarray(depth, dtype=np.float32)
                depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
                buffer = io.BytesIO()
                np.save(buffer, depth)
                cache[f"{name}_depth.npy"] = buffer.getvalue()
                cache[f"{name}_depth.png"] = depth_preview(depth)
            camera = observation["cameras"][source]
            cameras[name] = {
                "intrinsics": np.round(np.asarray(camera["intrinsics"], dtype=float), 3).tolist(),
                "extrinsics_world": np.round(np.asarray(camera["extrinsics_world"], dtype=float), 5).tolist(),
                "size": [int(v) for v in camera["size"]],
            }
        state = {tag: arm.report() for tag, arm in self.arms.items()}
        state.update(
            cameras=cameras,
            sim_time_left_s=round(self.sim_time_left(), 2),
            budget_left=self.budget_left,
            step=self.commands,
        )
        cache["state.json"] = json.dumps(state).encode()
        self.obs_cache = cache
        prefix = f"{self.obs_saved:03d}_{self.last_cmd or 'reset'}_"
        for name, data in cache.items():
            if name.endswith("_depth.npy"):
                # the agent gets the plain float32 array; the archive keeps it compressed (about 5x smaller)
                depth = np.load(io.BytesIO(data))
                np.savez_compressed(os.path.join(self.directory, "obs", prefix + name[:-4] + ".npz"), depth=depth)
                continue
            with open(os.path.join(self.directory, "obs", prefix + name), "wb") as handle:
                handle.write(data)
        self.obs_saved += 1

    # ---- commands ---------------------------------------------------------
    def tail(self, feedback):
        feedback.update(
            sim_time_s=round(self.sim_time(), 2),
            sim_time_left_s=round(self.sim_time_left(), 2),
            budget_left=self.budget_left,
            step=self.commands,
        )
        if self.over:
            feedback["episode_over"] = True
        return feedback

    def move_tcp(self, arm, target_tcp, feedback):
        clipped = False
        for index, axis in enumerate("xyz"):
            low, high = WORKSPACE[axis]
            value = float(np.clip(target_tcp[index, 3], low, high))
            if abs(value - target_tcp[index, 3]) > 1e-9:
                clipped = True
                target_tcp[index, 3] = value
        feedback["workspace_limited"] = clipped
        if clipped:
            feedback["clipped"] = True
        try:
            sequence, method = motion.plan(
                self.executor.planner(arm.tag), self.executor.robot(arm.tag), arm.joints(), arm.ee(),
                target_tcp @ arm.tcp_to_ee, self.executor.dt(), self.executor.limits(arm.tag),
            )
        except motion.PlanFailure as failure:
            feedback.update(plan_ok=False, plan_fail_reason=failure.reason if failure.reason in C.PLAN_FAIL_REASONS else "ik_unreachable",
                            plan_detail=failure.detail)
            return C.EXIT_EXEC_FAILED
        feedback.update(plan_ok=True, plan_fail_reason=None)
        self._plan_method = method
        self.run({arm.tag: sequence})
        feedback["settle_steps"] = self.settle([arm.tag])
        reached = arm.tcp()
        error_m = float(np.linalg.norm(reached[:3, 3] - target_tcp[:3, 3]))
        feedback.update(
            reached_tcp={
                "pos": [round(float(v), 4) for v in reached[:3, 3]],
                "quat": [round(v, 5) for v in geo.matrix_to_pose(reached)[3:]],
                "rpy": [round(v, 2) for v in geo.rpy_deg(reached[:3, :3])],
            },
            error_m=round(error_m, 4),
            error_deg=round(geo.angle_between_deg(reached[:3, :3], target_tcp[:3, :3]), 2),
            settled=bool(error_m < 0.01),
        )
        return C.EXIT_OK

    def validate(self, payload):
        """Raises BadRequest before anything is charged to the budget. Returns parsed values."""
        cmd = payload["cmd"]
        if cmd in self.registry:
            return self.validate_tool(self.registry[cmd]["spec"], payload)
        if cmd in ("move", "rotate", "point", "gripper") and payload.get("arm") not in self.arms:
            raise BadRequest("`arm` must be left or right")
        if cmd == "home" and payload.get("arm") not in tuple(self.arms) + ("both",):
            raise BadRequest("`arm` must be left, right or both")
        if cmd == "move":
            return np.array([number(payload, k) for k in ("dx", "dy", "dz")])
        if cmd == "rotate":
            if payload.get("frame", "world") not in C.ROTATE_FRAMES:
                raise BadRequest("`frame` must be world or tool")
            return np.array([number(payload, k) for k in ("roll", "pitch", "yaw")])
        if cmd == "point":
            if payload.get("preset") not in C.POINT_PRESETS:
                raise BadRequest("the direction must be down, forward or down45")
            if payload.get("open") not in C.OPEN_AXES:
                raise BadRequest("`--open` is required: x, y or z")
            return tool_rotation(payload["preset"], payload["open"], self.arms[payload["arm"]].tcp()[:3, :3])
        if cmd == "gripper":
            value = number(payload, "value", None)
            if not 0.0 <= value <= 1.0:
                raise BadRequest("the gripper value must be in [0, 1]")
            return value
        if cmd == "wait":
            seconds = number(payload, "sec")
            if seconds < 0:
                raise BadRequest("`sec` must not be negative")
            return seconds
        return None

    def validate_tool(self, spec, payload):
        """Check a tool command's arguments against its declaration; returns the parsed argument dict."""
        parsed = {}
        for arg in spec.get("args", []):
            name = arg["name"]
            value = payload.get(name)
            if value is None:
                if arg.get("required") or arg.get("positional"):
                    raise BadRequest(f"`{name}` is required")
                parsed[name] = arg.get("default")
                continue
            kind = arg.get("type", "str")
            if kind == "float":
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    raise BadRequest(f"`{name}` must be a number")
                if not np.isfinite(value):
                    raise BadRequest(f"`{name}` must be finite")
            elif kind == "int":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    raise BadRequest(f"`{name}` must be an integer")
            if arg.get("choices") and value not in arg["choices"]:
                raise BadRequest(f"`{name}` must be one of {', '.join(map(str, arg['choices']))}")
            parsed[name] = value
        return parsed

    def execute(self, payload):
        cmd = payload["cmd"]
        try:
            parsed = self.validate(payload)
        except BadRequest as error:
            return {"error": str(error), "exit_code": C.EXIT_BAD_ARGS}
        feedback = {"cmd": cmd}
        if "arm" in payload:
            feedback["arm"] = payload["arm"]
        exit_code = C.EXIT_OK
        self.last_cmd = cmd
        self._plan_method = None
        first_step = self.steps
        started = time.time()

        entry = self.registry.get(cmd)
        if cmd in C.BUDGETED or (entry is not None and entry["spec"].get("budget", True)):
            if self.budget_left is not None:
                self.budget_left -= 1
            self.commands += 1
        arm = self.arms.get(payload.get("arm"))

        if entry is not None:
            try:
                tool_feedback, exit_code = entry["module"].run(toolkit.EpisodeAPI(self), cmd, parsed)
            except BadRequest as error:
                tool_feedback, exit_code = {"error": str(error)}, C.EXIT_EXEC_FAILED
            feedback.update(tool_feedback or {})
            feedback["requested"] = parsed
        elif cmd == "move":
            limited = np.clip(parsed, -C.MOVE_CLIP_M, C.MOVE_CLIP_M)
            feedback["requested"] = {"dx": parsed[0], "dy": parsed[1], "dz": parsed[2]}
            feedback["clipped"] = bool(np.any(limited != parsed))
            target = arm.tcp()
            target[:3, 3] += limited
            exit_code = self.move_tcp(arm, target, feedback)
        elif cmd == "rotate":
            limited = np.clip(parsed, -C.ROTATE_CLIP_DEG, C.ROTATE_CLIP_DEG)
            frame = payload.get("frame", "world")
            feedback["requested"] = {"roll": parsed[0], "pitch": parsed[1], "yaw": parsed[2], "frame": frame}
            feedback["clipped"] = bool(np.any(limited != parsed))
            delta = geo.rotation_from_rpy_deg(*limited)
            target = arm.tcp()
            target[:3, :3] = delta @ target[:3, :3] if frame == "world" else target[:3, :3] @ delta
            exit_code = self.move_tcp(arm, target, feedback)
        elif cmd == "point":
            feedback["requested"] = {"direction": payload["preset"], "open": payload["open"]}
            feedback["clipped"] = False
            target = arm.tcp()
            target[:3, :3] = parsed
            exit_code = self.move_tcp(arm, target, feedback)
        elif cmd == "gripper":
            feedback["requested"] = {"value": parsed}
            arm.gripper_target = float(parsed)
            self.hold(GRIPPER_STEPS)
        elif cmd == "home":
            tags = list(self.arms) if payload["arm"] == "both" else [payload["arm"]]
            sequences = {}
            for tag in tags:
                current = self.arms[tag]
                start = current.joints()
                sequences[tag] = motion.time_path(np.stack([start, current.home_joints]))
            self.run(sequences)
            feedback["settle_steps"] = self.settle(tags)
            feedback.update(plan_ok=True, plan_fail_reason=None)
        elif cmd == "wait":
            seconds = min(parsed, C.WAIT_MAX_S)
            feedback["requested"] = {"sec": parsed}
            feedback["clipped"] = seconds != parsed
            self.hold(int(round(seconds * motion.CONTROL_HZ)))
        elif cmd == "done":
            self.done_called = True
            self.success_at_done = self.executor.success_now()
            self.finish("done", success=self.success_at_done)
            feedback["success"] = bool(self.success_at_done)

        if arm is not None and "gripper" not in feedback:
            feedback["gripper"] = round(arm.gripper(), 3)
        if not self.over and self.budget_left is not None and self.budget_left <= 0 and cmd != "done":
            self.finish("budget", success=self.executor.success_now())
        if not self.over:
            self.refresh_obs()
        feedback = self.tail(feedback)
        self.last_feedback = dict(feedback)
        self.log_file.write(json.dumps({
            "t": round(started, 3), "wall_s": round(time.time() - started, 3), "request": payload,
            "feedback": feedback, "exit_code": exit_code, "steps": [first_step, self.steps],
            "plan_method": self._plan_method, "server_only": self.executor.private_note(),
        }) + "\n")
        self.log_file.flush()
        return dict(feedback, exit_code=exit_code, obs_updated=cmd != "done")

    # ---- end of episode ---------------------------------------------------
    def finish(self, reason, success=None):
        if self.over:
            return
        self.over = True
        self.end_reason = reason
        if success is None:
            success = self.executor.success_now() if reason == "auto_success" else False
        result = {
            "task": self.task, "seed": self.seed, "layout": self.layout,
            "success_official": None if success is None else bool(success),
            "success_at_done": self.success_at_done,
            "done_called": self.done_called,
            "end_reason": reason,
            "progress_score": self.executor.progress_score(success),
            "commands": self.commands,
            "budget": self.budget,
            "action_steps": self.steps,
            "step_lim": int(self.executor.step_lim()),
            "sim_time_s": round(self.sim_time(), 2),
            "wall_s": round(time.time() - self.started, 1),
            "contract": C.CONTRACT_VERSION,
            "mode": self.executor.mode,
        }
        self.result = result
        with open(os.path.join(self.directory, "result.json"), "w") as handle:
            json.dump(result, handle, indent=1)
        np.savez_compressed(os.path.join(self.directory, "final_state.npz"), **self.executor.final_state(self))
        np.savez_compressed(
            os.path.join(self.directory, "targets.npz"),
            targets=np.asarray(self.targets, dtype=np.float64),
            arms=np.asarray(list(self.arms)),
            control_hz=motion.CONTROL_HZ,
        )
        self.executor.on_finish(self, bool(success))
        self.log_file.flush()
        log(f"episode finished: {json.dumps(result)}")
