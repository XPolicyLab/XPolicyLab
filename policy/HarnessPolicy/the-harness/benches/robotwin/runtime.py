"""RoboTwin 2.0 worker for the three-layer runtime, one task per process.

Serves the same calls as the RoboDojo runtime worker, so the kernel session,
driver and shield are shared:

    profile                 static facts: joints, limits, cameras, embodiment constants
    state                   robot state
    tick                    one RoboTwin action (absolute joint targets, the native
                            ``qpos`` contract), checked by the shield
    observe                 camera images, intrinsics, camera-to-world extrinsics (OpenCV)
    fk / ik / plan          kinematics (pinocchio) and cuRobo planning, world frame
    claim_outcome / outcome evaluator result, readable only by the episode driver

Episodes follow the official evaluator (``scripts/eval_policy_xpolicylab.py``):
seeds start at 100000 * (1 + seed_group), a seed counts only if RoboTwin's expert
solves it (the expert check), the instruction is drawn from the task's unseen
descriptions, and success is ``eval_success`` within the task's step limit.
``reset(task, n)`` is the n-th valid seed; valid seeds are cached per task so an
episode index always names the same scene. Development seeds start at 10**7,
disjoint from every evaluation seed.

Poses are RoboTwin's end-effector poses ("ee", on the approach axis 0.12 m behind
the gripper centre, x along the approach): the kernel's link6 with a 0.12 m tip
offset. A tick executes one waypoint the way RoboTwin does (TOPP-timed), so
planned paths are thinned to one waypoint per 0.1 s (``control_hz`` = 10).

Runs in the harness-policy environment with RoboTwin's cuRobo v0.7.8 first on
PYTHONPATH (environment/robotwin_curobo.sh). Importing this module does not
import SAPIEN or RoboTwin.
"""

import importlib
import json
import math
import os
from pathlib import Path
import secrets
import sys
import uuid

from harness_policy.core.contracts import Action, Observation, Transition
from harness_policy.runtime.shield import Shield, ShieldProfile

from harness_policy.core.codec import EpisodeCursor

CAMERAS = {"cam_head": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"}
METHODS = ("profile", "state", "tick", "observe", "fk", "ik", "plan", "claim_outcome", "outcome")
EVAL_SEED_BASE = 100000
DEV_SEED_BASE = 10_000_000
TIP_OFFSET_M = 0.12
CONTROL_HZ = 10.0
PLAN_DT = 1 / 250  # RoboTwin's cuRobo interpolation step
# The shield rejects commands that are bugs. A tick is one TOPP-timed waypoint (10 per
# second), not a 25 Hz control period, so a jump is larger than RoboDojo's.
DEFAULT_SHIELD = {
    "max_joint_jump_rad": 0.6,
    "workspace": {"x": [-0.9, 0.9], "y": [-0.8, 0.6], "z": [0.70, 1.6]},
    "max_cartesian_jump_m": 0.3,
}


def create_environment(**options):
    allowed = {"checkout", "task", "task_config", "split", "seed_group", "instruction_type", "seed_cache_dir",
               "max_seed_attempts", "shield"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown RoboTwin runtime options: {sorted(unknown)}")
    shield = options.get("shield") or {}
    if not isinstance(shield, dict) or set(shield) - set(DEFAULT_SHIELD):
        raise ValueError(f"shield options must be a table with keys from {sorted(DEFAULT_SHIELD)}")
    return RoboTwinRuntimeEnvironment(**options)


class RoboTwinRuntimeEnvironment:
    name = "robotwin2-sapien-runtime"
    methods = METHODS

    def __init__(self, checkout=None, task="stack_blocks_two", task_config="demo_randomized", split="eval",
                 seed_group=0, instruction_type="unseen", seed_cache_dir=None, max_seed_attempts=5000, shield=None):
        root = Path(checkout or os.environ.get("HARNESS_ROBOTWIN_ROOT", "third_party/RoboTwin")).resolve()
        if split not in ("eval", "dev"):
            raise ValueError("split must be 'eval' or 'dev'")
        # RoboTwin resolves assets and configs relative to its checkout.
        os.chdir(root)
        sys.path[:0] = [str(root), str(root / "scripts")]
        self._evaluator = importlib.import_module("eval_policy_xpolicylab")
        self._unstable = importlib.import_module("envs.utils.create_actor").UnStableError
        self._task = task
        self._task_config = task_config
        self._instruction_type = instruction_type
        self._first_seed = (EVAL_SEED_BASE if split == "eval" else DEV_SEED_BASE) + EVAL_SEED_BASE * int(seed_group)
        self._max_attempts = int(max_seed_attempts)
        cache = Path(seed_cache_dir or root / "harness_seeds") / f"{split}-{int(seed_group)}" / task_config
        cache.mkdir(parents=True, exist_ok=True)
        self._seed_file = cache / f"{task}.json"
        self._seeds = json.loads(self._seed_file.read_text()) if self._seed_file.exists() else {
            "first": self._first_seed, "tried": 0, "valid": [], "info": {}}
        self._args, self._embodiment = self._evaluator.load_task_args(
            {"task_name": task, "task_config": task_config, "policy_name": "harness-policy"})
        self._args.update(render_freq=0, eval_mode=True, eval_video_log=False)
        self._env = getattr(importlib.import_module(f"envs.{task}"), task)()
        self._cursor = EpisodeCursor()
        self._outcome_token = None
        self._instruction = ""
        self._open = False
        self._shield_options = {**DEFAULT_SHIELD, **(shield or {})}
        self._shield = None
        self._pinocchio = None

    # -- Worker protocol ---------------------------------------------------------

    def tools(self):
        return []

    def execute(self, action: Action, episode_id: str, step: int) -> Transition:
        raise RuntimeError("The runtime worker accepts motion only through call('tick'), which the shield checks")

    def call(self, name, arguments):
        if name not in METHODS:
            raise ValueError(f"Unknown runtime method: {name}")
        return getattr(self, f"_call_{name}")(**arguments)

    def reset(self, task, seed):
        if task != self._task:
            raise ValueError(f"This worker serves {self._task!r}, not {task!r}")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed is the episode index, a non-negative integer")
        chosen = self._valid_seed(seed)
        self._close()
        self._env.setup_demo(now_ep_num=seed, seed=chosen, is_test=True, **self._args)
        self._open = True
        info = self._seeds["info"].get(str(chosen), {"info": {}})
        self._instruction = self._evaluator.build_instruction(self._args, info, self._instruction_type, 100, chosen)
        self._env.set_instruction(instruction=self._instruction)
        self._setup_kinematics()
        self._outcome_token = None
        episode = uuid.uuid4().hex
        self._cursor.start(episode)
        state = self._robot_state()
        state.update(task=self._task, task_config=self._task_config, embodiment=self._embodiment,
                     robotwin_seed=chosen, episode_index=seed)
        images = self._call_observe(episode)["images"]
        return Observation(episode, 0, self._instruction, state, images)

    def close(self):
        self._close()

    # -- Episode seeds -----------------------------------------------------------

    def _valid_seed(self, index):
        seeds = self._seeds
        while len(seeds["valid"]) <= index:
            if seeds["tried"] >= self._max_attempts:
                raise RuntimeError(f"Only {len(seeds['valid'])} valid seeds for {self._task} "
                                   f"in {self._max_attempts} attempts")
            seed = seeds["first"] + seeds["tried"]
            seeds["tried"] += 1
            info = self._expert_check(seed)
            if info is not None:
                seeds["valid"].append(seed)
                seeds["info"][str(seed)] = _jsonable(info)
            self._seed_file.write_text(json.dumps(seeds))
        return seeds["valid"][index]

    def _expert_check(self, seed):
        """The official evaluator's check: the expert must solve the seed."""
        self._close()
        try:
            self._env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **self._args)
            self._open = True
            info = self._env.play_once()
            solved = bool(self._env.plan_success and self._env.check_success())
        except self._unstable:
            return None
        except Exception as error:
            print(f"expert check failed on seed {seed}: {type(error).__name__}: {error}", file=sys.stderr)
            return None
        finally:
            self._close()
        return info if solved else None

    def _close(self):
        if self._open:
            try:
                self._env.close_env()
            finally:
                self._open = False
        self._cursor.invalidate()

    # -- Calls -------------------------------------------------------------------

    def _call_profile(self):
        return {
            "task": self._task,
            "benchmark": "RoboTwin 2.0",
            "task_config": self._task_config,
            "embodiment": self._embodiment,
            "step_limit": int(self._env.step_lim or 0),
            "control_hz": CONTROL_HZ,
            "groups": [{"name": arm, "joints": joints, "gripper": gripper} for arm, joints, gripper in _GROUPS],
            "action_low": self._lower,
            "action_high": self._upper,
            # Each arm's first joint (its shoulder). The entity origins are the robot's
            # root for a single-entity dual arm such as aloha-agilex, the same for both.
            "arm_bases": {"left_arm": _pose(self._env.robot.left_arm_joints[0].get_global_pose()),
                          "right_arm": _pose(self._env.robot.right_arm_joints[0].get_global_pose())},
            "cameras": list(CAMERAS),
            "tip_offset_m": TIP_OFFSET_M,
            # Measured opening of an empty closed gripper (normalized 0 closed .. 1 open).
            "gripper_empty_opening": 0.18,
            # Distance between the fingers fully open (2 x 0.04765 m travel, aloha-agilex URDF).
            "gripper_open_width_m": 0.0953,
            "table_z": 0.74,
            "table_z_note": "demo_randomized raises or lowers the table by up to 3 cm per episode",
            "shield": self._shield_options,
            "planner_world": "self collision and the table only; scene objects and the other arm are not modeled",
        }

    def _call_state(self, episode_id):
        self._cursor.check(episode_id, self._cursor.step)
        return self._robot_state()

    def _call_tick(self, episode_id, step, command, cameras=None):
        import numpy as np

        self._cursor.check(episode_id, step)
        if self._ended():
            raise ValueError("The episode has ended; the command was not executed")
        decision = self._shield.check(command, self._joints(), fk=self._fk_positions)
        if decision.command is None:
            return {"executed": False, "shield": decision.to_dict(), "state": self._robot_state()}
        try:
            self._env.take_action(np.asarray(decision.command, dtype=float), action_type="qpos")
        except self._unstable as error:
            self._cursor.invalidate()
            raise RuntimeError(f"RoboTwin marked the scene unstable: {error}") from error
        except Exception:
            self._cursor.invalidate()
            raise
        self._cursor.advance()
        result = {"executed": True, "shield": decision.to_dict(), "state": self._robot_state()}
        if cameras:
            result["observation"] = self._call_observe(episode_id, cameras=cameras)
        return result

    def _call_observe(self, episode_id, cameras=None, image_format="jpeg", quality=90):
        import cv2
        import numpy as np

        self._cursor.check(episode_id, self._cursor.step)
        cameras = list(cameras or CAMERAS)
        unknown = set(cameras) - set(CAMERAS)
        if unknown:
            raise ValueError(f"Unknown cameras: {sorted(unknown)}")
        observation = self._env.get_obs()["observation"]
        images, intrinsics, extrinsics = {}, {}, {}
        for camera in cameras:
            data = observation[CAMERAS[camera]]
            rgb = np.asarray(data["rgb"], dtype=np.uint8)
            encoded = cv2.imencode(".jpg" if image_format == "jpeg" else ".png",
                                   cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if image_format == "jpeg" else [])[1]
            images[camera] = _data_url(encoded.tobytes(), image_format)
            intrinsics[camera] = np.asarray(data["intrinsic_cv"], dtype=float).tolist()
            world_to_camera = np.eye(4)
            world_to_camera[:3, :] = np.asarray(data["extrinsic_cv"], dtype=float)
            extrinsics[camera] = np.linalg.inv(world_to_camera).tolist()
        return {"state": self._robot_state(), "images": images, "intrinsics": intrinsics,
                "camera_to_world": extrinsics}

    def _call_fk(self, command=None):
        return self._fk(command if command is not None else self._joints())

    def _call_ik(self, episode_id, arm, pose):
        plan = self._plan_pose(arm, pose)
        if plan is None:
            return {"success": False, "joints": None}
        return {"success": True, "joints": plan[-1]}

    def _call_plan(self, episode_id, arm, pose=None, joints=None):
        import numpy as np

        self._cursor.check(episode_id, self._cursor.step)
        if (pose is None) == (joints is None):
            raise ValueError("plan needs exactly one of pose or joints")
        if pose is not None:
            path = self._plan_pose(arm, pose)
            if path is None:
                return {"success": False, "waypoints": []}
            return {"success": True, "waypoints": path}
        start = np.asarray(self._arm_joints(arm), dtype=float)
        goal = np.asarray(joints, dtype=float)
        # Joint-space targets: a straight line, one waypoint per 0.05 rad on the largest joint.
        count = max(1, int(math.ceil(float(np.max(np.abs(goal - start))) / 0.05)))
        return {"success": True, "waypoints": [(start + (goal - start) * k / count).tolist()
                                               for k in range(1, count + 1)]}

    def _call_claim_outcome(self, episode_id):
        self._cursor.check(episode_id, self._cursor.step)
        if self._outcome_token is not None:
            raise PermissionError("The outcome token for this episode was already claimed")
        self._outcome_token = secrets.token_hex(16)
        return {"token": self._outcome_token}

    def _call_outcome(self, token):
        if self._outcome_token is None or not secrets.compare_digest(str(token), self._outcome_token):
            raise PermissionError("Invalid outcome token")
        success = bool(self._env.eval_success)
        steps = int(self._env.take_action_cnt)
        return {"reward": 1.0 if success else 0.0, "success": success, "terminated": success,
                "truncated": not success and steps >= int(self._env.step_lim or 0),
                "process_score": 100.0 if success else 0.0, "steps": steps}

    # -- Robot -------------------------------------------------------------------

    def _setup_kinematics(self):
        robot = self._env.robot
        self._articulations = {"left_arm": robot.left_entity, "right_arm": robot.right_entity}
        self._pinocchio = {arm: entity.create_pinocchio_model() for arm, entity in self._articulations.items()}
        self._arm_joint_index = {}
        for arm, joints in (("left_arm", robot.left_arm_joints), ("right_arm", robot.right_arm_joints)):
            names = [j.get_name() for j in self._articulations[arm].get_active_joints()]
            self._arm_joint_index[arm] = [names.index(j.get_name()) for j in joints]
        self._lower, self._upper = [], []
        for arm, joints in (("left_arm", robot.left_arm_joints), ("right_arm", robot.right_arm_joints)):
            for joint in joints:
                low, high = joint.get_limits()[0]
                self._lower.append(float(low) if math.isfinite(low) else -math.pi)
                self._upper.append(float(high) if math.isfinite(high) else math.pi)
            self._lower.append(0.0)
            self._upper.append(1.0)
        profile = {"groups": {arm: {"joints": joints, "gripper": gripper} for arm, joints, gripper in _GROUPS},
                   **self._shield_options}
        self._shield = Shield(ShieldProfile.from_dict(profile, self._lower, self._upper))

    def _joints(self):
        """Measured arm joints and gripper openings (0 closed .. 1 open). RoboTwin's own
        jointState reports drive targets and the commanded gripper value, which would
        hide an arm held back by contact and an object between the fingers."""
        robot = self._env.robot
        values = []
        for side in ("left", "right"):
            values += [float(v) for v in getattr(robot, f"get_{side}_arm_real_jointState")()[:-1]]
            values.append(self._gripper_opening(side))
        return values

    def _gripper_opening(self, side):
        robot = self._env.robot
        joint = getattr(robot, f"{side}_gripper")[0][0]
        entity = getattr(robot, f"{side}_entity")
        names = [j.get_name() for j in entity.get_active_joints()]
        position = float(entity.get_qpos()[names.index(joint.get_name())])
        low, high = getattr(robot, f"{side}_gripper_scale")
        return float(min(1.0, max(0.0, (position - low) / (high - low))))

    def _arm_joints(self, arm):
        joints = self._joints()
        return joints[0:6] if arm == "left_arm" else joints[7:13]

    def _ended(self):
        return bool(self._env.eval_success) or int(self._env.take_action_cnt) >= int(self._env.step_lim or 0)

    def _robot_state(self):
        robot = self._env.robot
        return {"step": int(self._env.take_action_cnt), "step_limit": int(self._env.step_lim or 0),
                "ended": self._ended(), "joint_position": self._joints(),
                "left_ee_pose": [float(v) for v in robot.get_left_ee_pose()],
                "right_ee_pose": [float(v) for v in robot.get_right_ee_pose()]}

    def _fk(self, joints):
        """ee poses (RoboTwin convention) for 14 joint values, without moving the robot."""
        import numpy as np
        import transforms3d as t3d

        robot = self._env.robot
        result = {}
        for arm, side, offset in (("left_arm", "left", 0), ("right_arm", "right", 7)):
            entity = self._articulations[arm]
            model = self._pinocchio[arm]
            qpos = np.asarray(entity.get_qpos(), dtype=float).copy()
            for index, value in zip(self._arm_joint_index[arm], joints[offset:offset + 6]):
                qpos[index] = float(value)
            model.compute_forward_kinematics(qpos)
            # RoboTwin's "ee" is a joint; its global pose is child link * pose_in_child.
            joint = getattr(robot, f"{side}_ee")
            link = joint.get_child_link()
            world = entity.get_root_pose() * model.get_link_pose(link.get_index()) * joint.get_pose_in_child()
            rotation = (t3d.quaternions.quat2mat(world.q) @ getattr(robot, f"{side}_global_trans_matrix")
                        @ getattr(robot, f"{side}_delta_matrix"))
            bias = getattr(robot, f"{side}_gripper_bias") - TIP_OFFSET_M
            position = np.asarray(world.p) + rotation @ np.array([bias, 0.0, 0.0])
            result[arm] = [float(v) for v in position] + [float(v) for v in t3d.quaternions.mat2quat(rotation)]
        return result

    def _fk_positions(self, command):
        return {arm: pose[:3] for arm, pose in self._fk(command).items()}

    def _plan_pose(self, arm, pose):
        import numpy as np

        robot = self._env.robot
        plan = (robot.left_plan_path if arm == "left_arm" else robot.right_plan_path)([float(v) for v in pose])
        if plan.get("status") != "Success":
            return None
        path = np.asarray(plan["position"], dtype=float)
        stride = max(1, int(round(1 / (CONTROL_HZ * PLAN_DT))))
        indices = list(range(stride - 1, len(path), stride))
        if not indices or indices[-1] != len(path) - 1:
            indices.append(len(path) - 1)
        return path[indices].tolist()


# Command layout: left arm joints 0-5, left gripper 6, right arm joints 7-12, right gripper 13.
_GROUPS = (("left_arm", [0, 1, 2, 3, 4, 5], 6), ("right_arm", [7, 8, 9, 10, 11, 12], 13))


def _pose(pose):
    return [float(v) for v in pose.p] + [float(v) for v in pose.q]


def _data_url(data, image_format):
    import base64

    return f"data:image/{image_format};base64," + base64.b64encode(data).decode()


def _jsonable(value):
    return json.loads(json.dumps(value, default=lambda item: item.tolist() if hasattr(item, "tolist") else str(item)))
