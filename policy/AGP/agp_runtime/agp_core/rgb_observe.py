"""The worker's observation under official leaderboard conditions (switch OURS_RGB_ONLY, default off).

The official RoboDojo observation (env_cfg/arx_x5.yml of the clean checkout) holds three 640x480 RGB images, the
joint states, `<arm>_ee_pose` (link6, env frame) and the last gripper command; depth, approximate depth and the
camera intrinsics/extrinsics are off. With the switch on, sim_main installs RgbOnlyMixin in front of the worker
class, and its observe() is RgbObserver.observe(), which reads only

    env.get_obs()           the official observation: vision.<cam>.color, state.<arm>_arm_joint_state,
                            state.<arm>_ee_pose, state.<arm>_ee_joint_state, additional_info.frequency
    env.step_lim            the episode's official step limit (under official conditions OfficialEnv.step_lim: the
                            public task code's value, agp_core/official_env.py public_step_lim, not TASK_ENV.step_lim)
    env.is_episode_end()    the official end check (the same call the worker makes before every sample)

plus the worker's own records: how many take_action calls it made (StepCountingEnv, a forwarding proxy of the
environment that counts them) and its last gripper commands. Camera intrinsics and poses come from
agp_core/public_camera.py: the head camera is constant, a wrist camera is the official ee_pose times the fixed mount
of the public URDF. No camera prim, depth image, simulator clock, render dispatcher, measured finger joint,
benchmark action counter or end flag is read here.

Each capture directory gets
    top.png, left_wrist.png, right_wrist.png
    calibration.json   per camera: K, T_world_optical, T_base_optical_by_arm, T_base_tcp_by_arm (same layout as
                       the depth pipeline, from the public constants; no "depth" entry, no depth files)
    robot_state.json   per arm: official joints, ee_pose and gripper command, and the world pose of every URDF
                       link (forward kinematics of the public URDF anchored at the official ee_pose; finger joints
                       at the commanded end position, since the measured opening is not observed). Used by
                       agp_core/rgb_measure.py for the robot self-silhouette. Also the finger joints estimated
                       from the arm's wrist image (finger_q_estimate_m, finger_q_estimate_quality); the
                       silhouette still uses finger_q_assumed_m.
The observation state has no gripper_joints_m / gripper_geometry / transit velocity (measured fingers) and no
sim_time_s; low_level_actions and remaining_actions come from the own count and step_lim (the remaining count is
also given as steps_left, the name the compact view uses, so both views show it).

Visual finger gap: each arm's state carries gripper_gap_estimate, computed by
agp_core/visual_gap.py from that arm's wrist image of this observation and public constants only:
    estimate: true, source           marks it as an estimate from the image, not a measurement
    pad_gap_m, finger_joints_m        nominal pad gap (q7 + q8 + 0.807 mm) and the two finger joints
    quality                           "ok" | "low" (a finger hidden, a dark scene, no unique fit) | "unavailable"
    closed_empty                      after a close (command < 0.5) with quality ok: true when nothing is between
                                      the pads (gap < 1.5 mm), false when the fingers stopped apart; else null
    fingers                           per finger: joint, fit score, margin to the next fit 2 mm away, quality

The execution path (planning, sampling, settle, gripper outcome) is not changed by this module and uses the worker's
robot-manager access; the TASK_ENV whitelist (agp_core/official_env.py) restricts what that access reaches.

Grip width (OURS_GRIP_WIDTH): for an arm whose last worker command is a partial width
(partial_grips), robot_state.json places its fingers at that command's finger target and records gripper_width_command,
its state carries gripper_width_mm (the commanded pad gap) and its gap estimate has closed_empty null (not after a
close). With the switch off no command is partial and nothing changes.
"""
from collections import defaultdict
import json
from pathlib import Path
import time
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from agp_core import features
from agp_core import public_camera as pc
from agp_core import visual_gap

FLAG = "rgb_only"
CAMERAS = (("top", "cam_head"), ("left_wrist", "cam_left_wrist"), ("right_wrist", "cam_right_wrist"))
ARMS = pc.ARMS
SOURCE = "public camera model (agp_core/public_camera.py) and the official observation; no depth, no camera prims"


def atomic(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False, indent=2))
    tmp.replace(path)


# ----------------------------------------------------------------------------- kinematics (public URDF)
def link_poses_in_base(model, joints, finger_q):
    """base_link <- link for every URDF link: arm joints from `joints`, every prismatic (finger) joint at finger_q."""
    q = dict(zip(model.arm_joint_names, np.asarray(joints, float)))
    children = defaultdict(list)
    for name, joint in model._joints.items():
        children[joint["parent"]].append(name)
    poses = {model.base_link: np.eye(4)}
    stack = [model.base_link]
    while stack:
        parent = stack.pop()
        for name in children[parent]:
            joint = model._joints[name]
            T = poses[parent] @ pc._urdf_origin(joint)
            if joint["type"] in ("revolute", "continuous"):
                T = T @ pc.transform(Rotation.from_rotvec(joint["axis"] * q[name]).as_matrix())
            elif joint["type"] == "prismatic":
                T = T @ pc.transform(t=joint["axis"] * float(finger_q))
            poses[joint["child"]] = T
            stack.append(joint["child"])
    return poses


def finger_limits(model):
    """(lower, upper) of the prismatic finger joints of the public URDF."""
    limits = []
    root = ET.parse(Path(model.sources["urdf"]["path"])).getroot()
    for joint in root.findall("joint"):
        if joint.get("type") == "prismatic":
            limit = joint.find("limit")
            limits.append((float(limit.get("lower")), float(limit.get("upper"))))
    if not limits or len(set(limits)) != 1:
        raise NotImplementedError(f"expected identical prismatic finger joints, got {limits}")
    return limits[0]


# ----------------------------------------------------------------------------- the official observation
def official_arm(state, arm, grips=None):
    """Joints, ee_pose and gripper command of one arm from the official observation's state."""
    joints = np.asarray(state[f"{arm}_arm_joint_state"], float).ravel()
    ee_pose = np.asarray(state[f"{arm}_ee_pose"], float).ravel()
    if joints.shape != (6,) or ee_pose.shape != (7,) or not (np.isfinite(joints).all() and np.isfinite(ee_pose).all()):
        raise ValueError(f"official {arm} state has joints {joints.shape} and ee_pose {ee_pose.shape}")
    command = state.get(f"{arm}_ee_joint_state")
    source = "official observation"
    if command is None:                      # before the first command: the worker's own record
        command, source = (grips or {}).get(arm, 1.0), "worker's last command"
    command = float(np.asarray(command, float).ravel()[0])
    return {"joints": joints, "ee_pose": ee_pose, "gripper_command": command, "gripper_command_source": source}


def partial_grips(grips):
    """Grip width (OURS_GRIP_WIDTH): {arm: command} of the arms whose last worker command is partial (0 < c < 1, sent
    only by the width action); {} otherwise (always with the switch off: open is 1.0, close 0.0)."""
    out = {}
    for arm in ARMS:
        value = (grips or {}).get(arm)
        if (isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)
                and 0.0 < float(value) < 1.0):
            out[arm] = float(value)
    return out


def capture_geometry(model, arms, finger_range, widths=None):
    """calibration.json and robot_state.json contents for one capture, from public constants and official state.
    widths (grip width, OURS_GRIP_WIDTH): {arm: the worker's partial gripper command} -- that arm's fingers are placed
    at the command's finger target (where they rest with nothing between) and its entry records the command
    (gripper_width_command); with none (always with the switch off) everything is as before."""
    calibration = {}
    tcp = {arm: model.tcp_in_base(arm, arms[arm]["ee_pose"]) for arm in ARMS}
    for label, name in CAMERAS:
        own = model.wrist_arm.get(name)
        T_world = model.T_world_optical(name, ee_pose=arms[own]["ee_pose"] if own else None)
        calibration[label] = {
            "K": model.K(name).tolist(),
            "T_world_optical": T_world.tolist(),
            "T_base_optical_by_arm": {arm: (pc.inverse(model.T_world_base[arm]) @ T_world).tolist() for arm in ARMS},
            "T_base_tcp_by_arm": {arm: tcp[arm].tolist() for arm in ARMS},
            "pixel_center": pc.PIXEL_CENTER,
            "source": SOURCE,
        }
    lower, upper = finger_range
    robot = {"source": SOURCE, "finger_joint_range_m": [lower, upper], "arms": {}}
    for arm in ARMS:
        a = arms[arm]
        finger_q = upper if a["gripper_command"] >= 0.5 else lower
        in_base = link_poses_in_base(model, a["joints"], finger_q)
        base = pc.pose_matrix(a["ee_pose"]) @ pc.inverse(in_base[model.ee_link])     # simulated base, from ee_pose
        robot["arms"][arm] = {
            "joints_rad": a["joints"].tolist(), "ee_pose": a["ee_pose"].tolist(),
            "gripper_command": a["gripper_command"], "gripper_command_source": a["gripper_command_source"],
            "finger_q_assumed_m": finger_q,
            "finger_q_meaning": "commanded end position (open: upper limit, closed: lower limit); not measured",
            "T_world_base_from_ee_pose": base.tolist(),
            "T_world_links": {link: (base @ T).tolist() for link, T in in_base.items()},
        }
    for arm, command in (widths or {}).items():         # grip width: the fingers at the commanded width's target
        from agp_core import grip_width
        a = arms[arm]
        finger_q = grip_width.finger_of_command(command, limits=(lower, upper))
        in_base = link_poses_in_base(model, a["joints"], finger_q)
        base = pc.pose_matrix(a["ee_pose"]) @ pc.inverse(in_base[model.ee_link])
        robot["arms"][arm].update(
            finger_q_assumed_m=finger_q, gripper_width_command=float(command),
            finger_q_meaning=("commanded width's finger target (where the fingers rest with nothing between the pads); "
                              "not measured"),
            T_world_base_from_ee_pose=base.tolist(), T_world_links={link: (base @ T).tolist() for link, T in
                                                                    in_base.items()})
    return calibration, robot, tcp


class StepCountingEnv:
    """The benchmark environment as the worker sees it, plus the worker's own count of take_action calls.

    The worker calls env.is_episode_end() before every sample and stops once it is true, so every take_action
    it makes is one the official counter also counts; `steps_sent` therefore equals the official count without
    reading it. Every other attribute is the environment's own (reads and writes are forwarded)."""

    def __init__(self, env):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "steps_sent", 0)

    def take_action(self, action):
        result = object.__getattribute__(self, "_env").take_action(action)
        object.__setattr__(self, "steps_sent", object.__getattribute__(self, "steps_sent") + 1)
        return result

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_env"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_env"), name, value)


class RgbObserver:
    """observe() from the official observation only (see the module docstring for the exact reads)."""

    def __init__(self, env, session, model=None, counter=None):
        self.env = env
        self.session = Path(session)
        self.model = model if model is not None else pc.default_model()
        self.counter = counter
        self.finger_range = finger_limits(self.model)
        visual_gap.default_estimator()          # build the finger silhouette tables now, not at the first observe

    def observe(self, grips=None):
        obs = self.env.get_obs()
        captured = time.monotonic_ns()
        ended = bool(self.env.is_episode_end())
        limit = int(self.env.step_lim)
        used = int(self.counter.steps_sent) if self.counter is not None else 0
        frequency = float((obs.get("additional_info") or {}).get("frequency") or 0.0)
        state = obs["state"]
        arms = {arm: official_arm(state, arm, grips) for arm in ARMS}
        widths = partial_grips(grips)            # grip width: {} unless the worker commanded a partial width
        calibration, robot, tcp = (capture_geometry(self.model, arms, self.finger_range, widths) if widths else
                                   capture_geometry(self.model, arms, self.finger_range))

        capture = str(captured)
        directory = self.session / "captures" / capture
        directory.mkdir(parents=True)
        images, times, gaps = [], {}, {}
        for label, name in CAMERAS:
            color = np.asarray(obs["vision"][name]["color"])
            k = self.model.intrinsics[name]
            if color.shape[:2] != (k.height, k.width):
                raise ValueError(f"{name} image {color.shape[:2]} differs from the public camera "
                                 f"{(k.height, k.width)}")
            rgb = color[..., :3].astype(np.uint8)
            path = directory / f"{label}.png"
            Image.fromarray(rgb).save(path, compress_level=1)
            images.append({"label": label, "path": str(path)})
            times[label] = captured
            own = self.model.wrist_arm.get(name)
            if own is not None:                  # the finger gap of this arm, from its own wrist image only
                gaps[own] = visual_gap.observation_entry(rgb, None if own in widths else arms[own]["gripper_command"])
        for arm in ARMS:
            robot["arms"][arm]["finger_q_estimate_m"] = gaps[arm]["finger_joints_m"]
            robot["arms"][arm]["finger_q_estimate_quality"] = gaps[arm]["quality"]
        atomic(directory / "calibration.json", calibration)
        atomic(directory / "robot_state.json", robot)

        arm_states = {}
        lower, upper = self.finger_range
        for arm in ARMS:
            a = arms[arm]
            arm_states[arm] = {
                "frame": arm + "_base", "tcp_pose_base": pc.pose_vector(tcp[arm]).tolist(),
                "joints_rad": a["joints"].tolist(), "gripper_command": a["gripper_command"],
                "ee_pose": a["ee_pose"].tolist(), "T_world_base": self.model.T_world_base[arm].tolist(),
                "T_link6_tcp": self.model.T_link6_tcp.tolist(),
                "gripper_joint_limits_m": [[lower, upper], [lower, upper]],
                "gripper_gap_estimate": gaps[arm]}
            if arm in widths:                    # grip width: the gap the gripper is commanded to (mm)
                from agp_core import grip_width
                arm_states[arm]["gripper_width_mm"] = round(grip_width.target_gap_m(
                    widths[arm], limits=(lower, upper)) * 1000.0, 1)
        state_out = {
            "arms": arm_states,
            "low_level_actions": used,
            "remaining_actions": max(0, limit - used),
            "steps_left": max(0, limit - used),      # the compact view's name, also in the full view
            "step_limit": limit,
            "episode_ended": ended,
            "control_time_s": used / frequency if frequency > 0 else None,
            "observation": "rgb_only",
        }
        return {"ok": True, "capture": capture, "camera_times": times, "captured_monotonic_ns": captured,
                "state": state_out, "calibration": calibration, "images": images}


class RgbOnlyMixin:
    """Put in front of a migration Worker class: observe() is RgbObserver.observe(), steps are counted locally."""

    def __init__(self, env, session):
        super().__init__(env, session)
        self.env = StepCountingEnv(self.env)
        self.rgb_observer = RgbObserver(self.env, self.session, counter=self.env)
        atomic(self.session / "rgb_only.json", {
            "switch": features.FLAGS[FLAG], "observation": "official get_obs only (RGB, joints, ee_pose, command)",
            "camera_model": self.rgb_observer.model.constants(),
            "step_count": "own count of take_action calls; limit from env.step_lim"})

    def observe(self):
        grips = {name.split("_")[0]: value for name, value in self.grips.items()}
        return self.rgb_observer.observe(grips=grips)


def worker_class(base_cls, environ=None):
    """`base_cls` unchanged when the switch is off; else a subclass with RgbOnlyMixin in front."""
    if not features.enabled(FLAG, environ):
        return base_cls
    if issubclass(base_cls, RgbOnlyMixin):
        return base_cls
    return type(f"RgbOnly{base_cls.__name__}", (RgbOnlyMixin, base_cls),
                {"__module__": __name__, "__doc__": f"{base_cls.__qualname__} observing RGB only."})


def install(worker_module, environ=None):
    """Replace `worker_module.Worker` when OURS_RGB_ONLY=1; a no-op otherwise. Returns the class in use."""
    cls = worker_class(worker_module.Worker, environ)
    if cls is not worker_module.Worker:
        worker_module.Worker = cls
        print(f"[ours] rgb_only: {cls.__name__} (observe from get_obs + public camera model, own step count)",
              flush=True)
    return cls
