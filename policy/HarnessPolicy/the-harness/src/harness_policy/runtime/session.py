"""The robot's session: the episode core (runtime/episode.py) for robot arms.

A robot session connects to a robot worker (the Robot API: profile, state, tick,
observe, ...). A step is a control tick; its command is the vector of joint targets
(each arm's joints and gripper, embodiment groups); guards observe joints, grippers and
link6 poses; an execution's scope is a region of the workspace one arm's link6 must stay
in; a guard can also open the grippers. The robot skills' constants (ARMS, CAMERAS)
live here.

Bootstrap: harness_policy.runtime.kernel --bootstrap harness_policy.runtime.session:create_session
(the episode core's, which builds the session of options["embodiment"], default robot).
"""

import base64
import math

from harness_policy.geometry import FINGERTIP_OFFSET
from harness_policy.runtime.episode import EpisodeCore, create_session, session_class  # noqa: F401
from harness_policy.runtime.reactive import BUILTIN_ACTIONS, OPEN_GRIPPER


ARMS = ("left_arm", "right_arm")
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
# Executions that move nothing: they run before start(strategy, kind) too.
STILL_EXECUTIONS = ("hold", "watch_scene")
# An arm an execution moved keeps its last command as the target it holds afterwards when
# it ended within this of it (rad, every joint); one that stopped farther away (contact,
# an interrupted motion) holds where it stopped instead of pushing on.
HOLD_AT_COMMAND_RAD = 0.01


class Observation:
    """Camera images (RGB uint8 arrays), camera parameters, and robot state."""

    def __init__(self, data):
        import cv2
        import numpy as np

        self.state = data["state"]
        self.images = {}
        for camera, url in data["images"].items():
            raw = np.frombuffer(base64.b64decode(url.split(",", 1)[1]), dtype=np.uint8)
            self.images[camera] = cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        self.intrinsics = {camera: np.asarray(k) for camera, k in data["intrinsics"].items()}
        self.camera_to_world = {camera: np.asarray(t) for camera, t in data["camera_to_world"].items()}
        self._urls = data["images"]

    def __repr__(self):
        return f"Observation(cameras={list(self.images)}, step={self.state['step']})"


class RobotSession(EpisodeCore):
    """The episode core for robot arms: the embodiment's groups and constants, the
    gripper commands, and the joint-vector commands of every tick."""

    embodiment = "robot"
    guard_actions = BUILTIN_ACTIONS
    still_executions = STILL_EXECUTIONS
    strategy_skill = "vla"
    strategy_hint = ("before start only looking runs (locate, watch_scene, get_observation): the learned "
                     "policy starts from the untouched start pose")

    def _embodiment_setup(self, options):
        # Motions are timed as the planner times them (its joint velocity and acceleration).
        planner = self.profile.get("planner_limits") or {}
        self.joint_speed_limit = planner.get("joint_velocity")
        self.joint_acceleration_limit = planner.get("joint_acceleration")
        self.groups = {group["name"]: group for group in self.profile["groups"]}
        # Embodiment constants come from the worker; the defaults are RoboDojo's ARX X5.
        from harness_policy.geometry import TABLE_Z
        self.tip_offset = float(self.profile.get("tip_offset_m", FINGERTIP_OFFSET))
        self.table_z = float(self.profile.get("table_z", TABLE_Z))
        self._held = {}  # arm -> what pick recorded about the held object
        # Last commanded normalized gripper value per arm (policies observe commands).
        self._gripper_commands = {}
        # The joints each arm holds while no skill moves it (_base_command).
        self._arm_commands = {}
        # Where the arms started: the benchmark may count success only with them back there.
        self._start_joints = list(self.state["joint_position"])

    def _policy_observation(self, instruction=None):
        """The XPolicyLab observation RoboDojo's evaluation client hands a policy
        (TASK_ENV.get_obs()): lossless camera colors, RoboDojo's own state, its
        instruction (or `instruction`) and additional_info."""
        import cv2
        import numpy as np

        raw = self.env.call("policy_obs", episode_id=self.episode_id)
        self.state = raw["robot_state"]
        vision = {}
        for camera, url in raw["vision"].items():
            data = np.frombuffer(base64.b64decode(url.split(",", 1)[1]), dtype=np.uint8)
            vision[camera] = {"color": cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)}
        return {"vision": vision, "state": {key: np.asarray(value) for key, value in raw["state"].items()},
                "instruction": raw.get("instruction") if instruction is None else str(instruction),
                "additional_info": raw.get("additional_info") or {"frequency": self.hz},
                "data_format_version": raw.get("data_format_version"), "env_idx": 0}

    def _guard_command(self, action, state):
        """hold: the measured joints with the grippers as commanded; open_gripper: the
        same with every gripper open."""
        hold = self._base_command(state)
        if action == OPEN_GRIPPER:
            for group in self.groups.values():
                hold[group["gripper"]] = 1.0
        return hold

    def _base_command(self, state):
        """What every joint holds while a skill does not move it: each arm at the joints
        it holds (its last command, or where it stopped: _motion_stopped; measured before
        its first command) and each gripper at its last commanded value. Commanded to its
        measured joints each tick an idle arm has no stiffness: whatever pushes it moves
        it, and it stays there (the other arm, whose planned path does not avoid it,
        pushed one left above a button 26 mm down onto it). Commanding the measured
        opening would stop squeezing a held object, which then slips when the arm moves."""
        target = list(state["joint_position"])
        commands = self.__dict__.get("_arm_commands", {})
        for arm, group in self.groups.items():
            for index, value in zip(group["joints"], commands.get(arm, ())):
                target[index] = value
            if arm in self._gripper_commands:
                target[group["gripper"]] = self._gripper_commands[arm]
        return target

    def _tick(self, command, cameras=None):
        extra = {"cameras": list(cameras)} if cameras else {}
        result = self.env.call("tick", episode_id=self.episode_id, step=self.state["step"],
                               command=[float(v) for v in command], **extra)
        if result.get("observation") is not None:
            self._live = Observation(result["observation"])
        if result["executed"]:
            executed = result["shield"]["command"]
            commands = self.__dict__.setdefault("_arm_commands", {})
            for arm, group in self.groups.items():
                self._gripper_commands[arm] = float(executed[group["gripper"]])
                commands[arm] = [float(executed[i]) for i in group["joints"]]
        self.state = result["state"]
        return result

    def _motion_started(self, state):
        self._arm_motion = {"holding": self._base_command(state), "moved": set()}

    def _motion_ticked(self, command):
        motion = self.__dict__.get("_arm_motion")
        if motion is not None:
            motion["moved"].update(arm for arm, group in self.groups.items()
                                   if any(abs(command[i] - motion["holding"][i]) > 1e-9 for i in group["joints"]))

    def _motion_stopped(self, state, arrived=False):
        """The arms the execution moved hold their measured joints from now on, instead of
        their last command: an interrupted motion stops where the arm is, and an arm that
        contact held back does not push on. When it arrived, an arm that ended within
        HOLD_AT_COMMAND_RAD of its last command keeps holding that command."""
        motion = self.__dict__.get("_arm_motion")
        if motion is None:
            return
        commands = self.__dict__.setdefault("_arm_commands", {})
        joints = state["joint_position"]
        for arm in motion["moved"]:
            measured = [float(joints[i]) for i in self.groups[arm]["joints"]]
            last = commands.get(arm)
            if arrived and last is not None and max(abs(a - b) for a, b in zip(measured, last)) <= HOLD_AT_COMMAND_RAD:
                continue
            commands[arm] = measured
        motion["moved"].clear()

    def _sample(self, state):
        return {"step": state["step"], "joints": state["joint_position"]}

    def _arm(self, arm):
        if arm not in self.groups:
            raise ValueError(f"arm must be one of {list(self.groups)}")
        return self.groups[arm]

    def _reactive_view(self, state, tick):
        joints = state["joint_position"]
        view = {"step": state["step"], "t": tick / self.hz, "joints": list(joints)}
        for name, group in self.groups.items():
            side = name.split("_")[0]
            view[f"{side}_arm"] = [joints[i] for i in group["joints"]]
            view[f"{side}_gripper"] = joints[group["gripper"]]
            view[f"{side}_ee_pose"] = list(state[f"{side}_ee_pose"])
            view[f"{side}_ee"] = list(state[f"{side}_ee_pose"][:3])
        return view

    def _outside(self, region, state):
        """region: {"arm": "left_arm", "center": [x, y, z], "radius": r} or an axis box
        {"arm": ..., "x": [lo, hi], "y": [...], "z": [...]}."""
        side = region.get("arm", "left_arm").split("_")[0]
        point = state[f"{side}_ee_pose"][:3]
        if "radius" in region:
            return math.dist(point, region["center"]) > float(region["radius"])
        return any(axis in region and not region[axis][0] <= value <= region[axis][1]
                   for axis, value in zip("xyz", point))


def __getattr__(name):
    if name == "Session":
        return session_class("robot")
    raise AttributeError(name)
