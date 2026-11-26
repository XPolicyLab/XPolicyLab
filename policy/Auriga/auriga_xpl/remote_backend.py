"""RobotBackend for Auriga inside the XPolicyLab policy server.

Auriga's own RoboDojo backend runs ``RoboDojoAdapter`` (auriga/robot/backends/robodojo/robot_worker.py)
in a worker process that owns the simulator. In the leaderboard deployment the simulator belongs to
the RoboDojo eval client, so the same adapter class is constructed in-process over
:class:`EnvFacade`, which answers the simulator attributes the adapter reads from XPolicyLab
observations and robot constants. Planning, execution (per-row dispatch, arrival tolerances, stall
and grip guards, paired/independent dual-arm progression, arm queues, cancellation) and observation
code are the adapter's own; only the environment underneath is different:

    adapter reads                         facade source
    -----------------------------------   ---------------------------------------------------------
    robot_manager.get_joint               obs["state"]["<side>_arm_joint_state"]
    robot_manager.get_real_endpose        obs["state"]["<side>_ee_pose"]  ([x, y, z, qw, qx, qy, qz])
    get_end_effector_real_val             additional_info["auriga"]["proprioception"][side]["gripper_position_m"]
    robot_key[i].data.joint_vel           additional_info["auriga"]["proprioception"][side]["arm_qdot_rad_s"]
    soft_joint_pos_limits / vel limits    ARX X5 constants (RoboDojo Assets/Robots/x5 and robot_config/x5.py)
    control_manager.prev_control          commands of the last dispatched row, interpolated like EvalEnv
    robot_manager.planner[robot_name]     auriga_xpl.planner (RoboDojo CuroboPlanner, vendored)
    robot_manager.get_link_pose           FK of the measured joints on the vendored X5A.urdf + base pose
    camera_manager (K, poses)             auriga_xpl.calibration constants (+ FK for wrist cameras)
    obs_manager.get_obs                   latest observation received through the bridge
    env.take_action(row)                  bridge.act([row]) -> observation after that row
    env.take_action_cnt / step_lim        additional_info["auriga"]["native_step" / "native_step_limit"]
    env.end_flag / success                bridge EpisodeOver (verdict from trial_end)
    env.reward_manager / scene_manager    not available: evaluate / observer_snapshot raise
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .bridge import EpisodeOver

ARMS = ("left", "right")
ADAPTER_ROOT = Path(__file__).resolve().parents[1]
# One robot tree for everything that reads the X5 description: the vendored planner, the calibration
# FK and this adapter. It mirrors RoboDojo's layout, so Auriga's own binding paths apply unchanged
# (Assets/Robots/x5/{X5A.urdf, robot_config.yml, curobo.yml, meshes/}).
DEFAULT_ROBODOJO_ROOT = ADAPTER_ROOT / "vendor" / "robodojo_planner"

# Stock RoboDojo arx_x5 cell (RoboDojo 726e9aa): env_cfg/robot/dual_x5.yml, Assets/Robots/x5/robot_config.yml,
# env/robot_manager/robot_config/x5.py (velocity_limit_sim=5.0), env_cfg/sim/sim_config.yml (dt 0.004),
# env_cfg/arx_x5.yml (collect_freq 25). Joint limits are the articulation's soft limits as reported by the
# simulator (float32 of the USD values), checked against Auriga worker run manifests.
ARX_X5_CELL = {
    "robot_name": "x5",
    "base_pose": {"left": [-0.3, -0.45, 0.765, 0.707, 0.0, 0.0, 0.707],
                  "right": [0.3, -0.45, 0.765, 0.707, 0.0, 0.0, 0.707]},
    "arm_joints_name": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    "gripper_joints_name": ["joint7", "joint8"],
    "gripper_move": {"base": "joint7", "sign": 1.0, "mimic": ["joint8", 1.0, 0.0]},
    "gripper_scale": [-0.01, 0.044],
    "ee_link": "link6",
    "base_link": "base_link",
    "soft_joint_pos_limits": [[-9.999999046325684, 9.999999046325684]] * 5 + [[-3.1399998664855957, 3.1399998664855957]]
                             + [[0.0, 0.044], [0.0, 0.044]],
    "joint_vel_limits": [5.0] * 6 + [5.0, 5.0],
    "sim_dt": 0.004,
    "collect_freq": 25,
}



class EvaluatorUnavailable(RuntimeError):
    """The leaderboard deployment has no evaluator or scene access by construction."""


class HostArray(np.ndarray):
    """numpy array with the torch-like accessors the adapter calls on simulator tensors."""

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def detach(self):
        return self

    def clone(self):
        return np.ndarray.copy(self).view(HostArray)


def host(values, dtype=np.float32):
    return np.array(values, dtype=dtype).view(HostArray)


def _list(value) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _matrix(pose):
    from transforms3d.quaternions import quat2mat
    pose = np.asarray(pose, dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = quat2mat(pose[3:] / np.linalg.norm(pose[3:]))
    transform[:3, 3] = pose[:3]
    return transform


def _pose(transform):
    from transforms3d.quaternions import mat2quat
    transform = np.asarray(transform, dtype=float)
    return list(transform[:3, 3]) + list(mat2quat(transform[:3, :3]))


class FacadeRobot:
    """One target arm, with the RobotManager attributes the adapter reads (identity equality)."""

    def __init__(self, side: str, cell: dict, curobo_yml_path: str):
        self.arm_name, self.gripper_name = f"{side}_arm", f"{side}_ee"
        self.robot_name, self.robot_type, self.type, self.ee_type = cell["robot_name"], "arm", "target", "gripper"
        self.arm_joints_name = list(cell["arm_joints_name"])
        self.gripper_joints_name = list(cell["gripper_joints_name"])
        self.arm_joint_indices = list(range(len(self.arm_joints_name)))
        self.gripper_joint_indices = [len(self.arm_joints_name) + i for i in range(len(self.gripper_joints_name))]
        self.gripper_move = deepcopy(cell["gripper_move"])
        self.gripper_scale = list(cell["gripper_scale"])
        self.entity_origin_pose = list(cell["base_pose"][side])
        self.ee_link_name, self.base_link = cell["ee_link"], cell["base_link"]
        self.curobo_yml_path = curobo_yml_path


class FacadeAsset:
    def __init__(self, cell: dict, body_names: list[str]):
        dofs = len(cell["arm_joints_name"]) + len(cell["gripper_joints_name"])
        self.body_names = list(body_names)
        self.data = SimpleNamespace(
            joint_pos=host(np.zeros((1, dofs))), joint_vel=host(np.zeros((1, dofs))),
            joint_vel_limits=host([cell["joint_vel_limits"]]),
            soft_joint_pos_limits=host([cell["soft_joint_pos_limits"]]))


class FacadeCamera:
    def __init__(self, facade: "EnvFacade", camera_id: str, item: dict):
        self.facade, self.camera_id, self.item = facade, camera_id, item

    def get_world_pose(self, camera_axes="ros"):
        if camera_axes != "ros":
            raise ValueError("facade_camera_supports_ros_axes_only")
        item = self.item
        if "T_link_camera" in item:
            link = self.facade.link_transform(item["arm"], item["link_name"])
            transform = link @ np.asarray(item["T_link_camera"], dtype=float)
        else:
            transform = np.asarray(item["T_world_camera"], dtype=float)
        pose = _pose(transform)
        return host(pose[:3]), host(pose[3:])

    def get_intrinsics_matrix(self):
        return host(self.item["K"])

    def get_resolution(self):
        return tuple(int(v) for v in self.item["image_size"])

    def get_clipping_range(self):
        raise ValueError("depth_unavailable_in_xpolicylab_deployment")


class FacadeRobotManager:
    def __init__(self, facade: "EnvFacade", planners: dict, cell: dict, curobo_yml_path: str, body_names: list[str]):
        self.facade = facade
        self.robot_list = [FacadeRobot(side, cell, curobo_yml_path) for side in ARMS]
        self.robot_key = [FacadeAsset(cell, body_names) for _ in ARMS]
        self.planner = planners
        self.target_arm_nums = 2
        self.control_manager = SimpleNamespace(prev_control=[{}])

    def process_name(self, name):
        return name if name.endswith("_state") else name + "_joint_state"

    def get_robot_by_arm_name(self, arm_name):
        for robot in self.robot_list:
            if robot.arm_name == arm_name:
                return robot
        raise ValueError(f"No robot found with arm name: {arm_name}")

    def get_robot_by_gripper_name(self, gripper_name):
        for robot in self.robot_list:
            if robot.gripper_name == gripper_name:
                return robot
        raise ValueError(f"No robot found with gripper name: {gripper_name}")

    def _side(self, robot):
        return robot.arm_name.split("_")[0]

    def get_joint(self, robot, env_idx_list=None):
        return {0: np.array(self.facade.measured(self._side(robot))["q"], dtype=np.float32)}

    def get_real_endpose(self, robot, env_idx_list=None, is_relative=True):
        return {0: np.array(self.facade.measured(self._side(robot))["ee_pose"], dtype=np.float32)}

    def get_end_effector_real_val(self, robot, env_idx_list=None):
        return {0: np.array(self.facade.measured(self._side(robot))["gripper_m"], dtype=np.float32)}

    def get_link_pose(self, robot, link_name, env_idx_list=None, is_relative=False):
        return {0: np.array(_pose(self.facade.link_transform(self._side(robot), link_name)), dtype=np.float32)}


class EnvFacade:
    """Stand-in for the RoboDojo EvalEnv attributes used by RoboDojoAdapter (one environment)."""

    def __init__(self, bridge, planners: dict, calibration: dict, *, urdf_path: Path, curobo_yml_path: str = "",
                 cell: dict | None = None):
        self.bridge = bridge
        self.cell = deepcopy(cell or ARX_X5_CELL)
        self.calibration = calibration or {}
        self.dt = float(self.cell["sim_dt"])
        self.num_envs = 1
        self._ticks = 0
        self.sim = SimpleNamespace(sim=_Clock(self))
        self._lock = threading.RLock()
        self._ended = False
        self._verdict: dict | None = None
        self.warnings: list[str] = []
        from auriga.robot.model.robot_model import urdf_manifest
        self.robot_model = urdf_manifest(urdf_path)
        cameras = self._camera_items()
        # Frozen simulator base pose of each arm when the calibration carries it (wrist cameras).
        self._bases = {item["arm"]: np.asarray(item["T_world_base"], dtype=float)
                       for item in cameras.values() if "arm" in item and "T_world_base" in item}
        links = {joint["child"] for joint in self.robot_model["joints"]} | {self.robot_model["base_link"]}
        # The simulator articulation merges fixed-joint links; keep moving links plus the declared mounts.
        fixed_children = {joint["child"] for joint in self.robot_model["joints"] if joint["type"] == "fixed"}
        body_names = sorted((links - fixed_children) | {item["link_name"] for item in cameras.values() if "link_name" in item})
        self.robot_manager = FacadeRobotManager(self, planners, self.cell, curobo_yml_path, body_names)
        self.obs_manager = SimpleNamespace(collect_freq=self.cell["collect_freq"],
                                           collect_interval=int(round(1.0 / (self.dt * self.cell["collect_freq"]))),
                                           render_for_capture=lambda: None, get_obs=self._get_obs)
        self.camera_manager = SimpleNamespace(
            camera_names=[list(cameras)],
            cameras=[[FacadeCamera(self, name, item) for name, item in cameras.items()]],
            camera_config={name: SimpleNamespace(camera={"mount_link": item["link_name"]} if "link_name" in item else {})
                           for name, item in cameras.items()})
        self.env_origins = [host(np.zeros(3))]
        self._pending_controls: list[dict] = []
        seq, obs = bridge.latest()
        if obs is None:
            raise RuntimeError("environment_facade_requires_first_observation")
        self._obs_seq, self._obs = seq, obs
        status = self._status(obs)
        self._count = int(status.get("native_step") or 0)
        limit = status.get("native_step_limit")
        if type(limit) is not int or limit <= 0:
            raise RuntimeError("native_step_limit_missing_from_observation")
        self.step_lim = limit
        self._last_command = {side: self._normalized_gripper(obs, side) for side in ARMS}
        self._set_controls(self._measured_controls())
        self._refresh_assets()

    # -- calibration -----------------------------------------------------------------------------
    def _camera_items(self) -> dict:
        calibration = self.calibration
        cameras = calibration.get("cameras")
        if cameras is None and isinstance(calibration.get("camera_calibration"), dict):
            cameras = calibration["camera_calibration"].get("cameras")
        if not isinstance(cameras, dict) or not cameras:
            raise ValueError("calibration_cameras_missing")
        for name, item in cameras.items():
            if "K" not in item or "image_size" not in item:
                raise ValueError(f"calibration_camera_incomplete:{name}")
            if "T_link_camera" not in item and "T_world_camera" not in item:
                raise ValueError(f"calibration_camera_pose_missing:{name}")
        return {name: dict(item) for name, item in cameras.items()}

    def link_transform(self, side: str, link_name: str):
        measured = self.measured(side)
        custom = self.calibration.get("link_pose")
        if callable(custom):
            return _matrix(custom(side, link_name, list(measured["q"]), list(measured["gripper_m"])))
        from auriga.robot.model.geometry import fk
        names = self.cell["arm_joints_name"] + self.cell["gripper_joints_name"]
        values = [float(v) for v in measured["q"]] + [float(v) for v in measured["gripper_m"]]
        base = self._bases.get(side)
        if base is None:
            base = _matrix(self.cell["base_pose"][side])
        links = fk({**self.robot_model, "T_world_base": base},
                   {"joint_names": names, "q_rad": values, "source_kind": "measurement"})
        if link_name not in links:
            raise ValueError(f"Link name {link_name} not found in robot {self.cell['robot_name']}")
        return np.asarray(links[link_name], dtype=float)

    # -- observation -----------------------------------------------------------------------------
    @staticmethod
    def _status(obs: dict) -> dict:
        return dict(((obs or {}).get("additional_info") or {}).get("auriga") or {})

    def _normalized_gripper(self, obs: dict, side: str) -> float:
        value = (obs.get("state") or {}).get(f"{side}_ee_joint_state")
        return float(_list(value)[0]) if value is not None else 1.0

    def measured(self, side: str) -> dict:
        with self._lock:
            obs = self._obs
        state = obs.get("state") or {}
        q = [float(v) for v in _list(state[f"{side}_arm_joint_state"])]
        pose = [float(v) for v in _list(state[f"{side}_ee_pose"])]
        proprioception = (self._status(obs).get("proprioception") or {}).get(side) or {}
        gripper = proprioception.get("gripper_position_m")
        if gripper is None:
            # Debug clients do not pass measured jaws; fall back to the command (never in RoboDojo).
            lower, upper = self.cell["gripper_scale"]
            opening = lower + float(np.clip(self._normalized_gripper(obs, side), 0, 1)) * (upper - lower)
            gripper = [opening, opening * self.cell["gripper_move"]["mimic"][1] + self.cell["gripper_move"]["mimic"][2]]
            self._warn("gripper_position_m_missing:" + side)
        qdot = proprioception.get("arm_qdot_rad_s")
        if qdot is None:
            qdot = [0.0] * len(q)
            self._warn("arm_qdot_rad_s_missing:" + side)
        return {"q": q, "ee_pose": pose, "gripper_m": [float(v) for v in _list(gripper)],
                "qdot": [float(v) for v in _list(qdot)]}

    def _warn(self, text: str) -> None:
        if text not in self.warnings:
            self.warnings.append(text)

    def _refresh_assets(self) -> None:
        for robot, asset in zip(self.robot_manager.robot_list, self.robot_manager.robot_key):
            measured = self.measured(robot.arm_name.split("_")[0])
            asset.data.joint_pos = host([measured["q"] + measured["gripper_m"]])
            asset.data.joint_vel = host([measured["qdot"] + [0.0] * len(measured["gripper_m"])])

    def _get_obs(self, env_idx_list=None):
        with self._lock:
            obs = self._obs
        vision = {}
        for name, values in (obs.get("vision") or {}).items():
            color = values.get("color") if isinstance(values, dict) else None
            if color is None:
                continue
            vision[name] = {"color": np.asarray(color, dtype=np.uint8)}
        return {0: {"vision": vision, "state": dict(obs.get("state") or {}), "instruction": obs.get("instruction"),
                    "additional_info": {"frequency": self.obs_manager.collect_freq}}}

    # -- control ---------------------------------------------------------------------------------
    def _gripper_command(self, robot: FacadeRobot, normalized: float) -> list:
        lower, upper = robot.gripper_scale
        value = float(np.clip(normalized, 0, 1))
        value = value * (upper - lower) + lower if robot.gripper_move["sign"] == 1 else (1 - value) * (upper - lower) + lower
        mimic = robot.gripper_move["mimic"]
        return [value, value * mimic[1] + mimic[2]]

    def _measured_controls(self) -> dict:
        controls = {}
        for robot in self.robot_manager.robot_list:
            side = robot.arm_name.split("_")[0]
            controls[self.robot_manager.process_name(robot.arm_name)] = {"position": list(self.measured(side)["q"])}
            controls[self.robot_manager.process_name(robot.gripper_name)] = {
                "position": self._gripper_command(robot, self._last_command[side])}
        return controls

    def _set_controls(self, controls: dict) -> None:
        self.robot_manager.control_manager.prev_control[0] = deepcopy(controls)

    def _control_sequence(self, action: dict) -> list[dict]:
        """EvalEnv.process_control_info: 80 % linear interpolation from the measured state, then hold."""
        steps = int(self.obs_manager.collect_interval)
        interp = int(np.floor(steps * 0.8))
        sequence = [dict() for _ in range(steps)]
        for robot in self.robot_manager.robot_list:
            side = robot.arm_name.split("_")[0]
            measured = self.measured(side)
            arm_key = self.robot_manager.process_name(robot.arm_name)
            target = np.asarray(action[f"{side}_arm_joint_state"], dtype=float)
            current = np.asarray(measured["q"], dtype=float)
            jaw_key = self.robot_manager.process_name(robot.gripper_name)
            goal = self._gripper_command(robot, float(action[f"{side}_ee_joint_state"][0]))[0]
            lower, upper = robot.gripper_scale
            mimic = robot.gripper_move["mimic"]
            for index in range(steps):
                if index < interp:
                    alpha = (index + 1) / (interp + 1)
                    arm = ((1 - alpha) * current + alpha * target).tolist()
                    jaw = float(np.clip((1 - alpha) * measured["gripper_m"][0] + alpha * goal, lower, upper))
                else:
                    arm, jaw = target.tolist(), float(np.clip(goal, lower, upper))
                sequence[index][arm_key] = {"position": arm}
                sequence[index][jaw_key] = {"position": [jaw, jaw * mimic[1] + mimic[2]]}
        return sequence

    @staticmethod
    def _row_action(action: dict) -> dict:
        row = {}
        for side in ARMS:
            arm = [float(v) for v in _list(action[f"{side}_arm_joint_state"])]
            jaw = [float(v) for v in _list(action[f"{side}_ee_joint_state"])]
            if len(arm) != 6 or len(jaw) != 1 or not np.isfinite(arm + jaw).all():
                raise ValueError("invalid_native_joint_action")
            row[f"{side}_arm_joint_state"], row[f"{side}_ee_joint_state"] = arm, jaw
        return row

    def step(self, env_idx_list, decimation=1):
        """One physics tick of the row the client is executing (bookkeeping only; see take_action)."""
        if self._pending_controls:
            self._set_controls(self._pending_controls.pop(0))
        self._ticks += int(decimation)

    def take_action(self, action: dict) -> None:
        if self.end_flag[0] or self._count >= self.step_lim:
            return  # EvalEnv skips actions after its terminal condition.
        row = self._row_action(action)
        controls = self._control_sequence(row)
        before = self._count
        try:
            seq, obs = self.bridge.act([row])
        except EpisodeOver as over:
            verdict = dict(over.args[0]) if over.args else {}
            self._finish(verdict)
            executed = verdict.get("native_step")
            if verdict.get("abandoned") or (type(executed) is int and executed <= before):
                return  # the client left or ended the trial before running this row
            self._count = executed if type(executed) is int else before + 1
            obs = None
        for side in ARMS:
            self._last_command[side] = float(np.clip(row[f"{side}_ee_joint_state"][0], 0, 1))
        self._pending_controls = controls
        for _ in range(len(controls)):
            self.step([0])  # the adapter wraps this attribute to count physics ticks and log commands
        if obs is not None:
            with self._lock:
                self._obs_seq, self._obs = seq, obs
            status = self._status(obs)
            count = status.get("native_step")
            self._count = count if type(count) is int else before + 1
            self._refresh_assets()
        if self._count >= self.step_lim:
            self._ended = True

    def _finish(self, verdict: dict) -> None:
        with self._lock:
            self._ended = True
            self._verdict = dict(verdict)

    # -- episode state ---------------------------------------------------------------------------
    @property
    def take_action_cnt(self):
        return [self._count]

    @property
    def end_flag(self):
        if not self._ended:
            verdict = self.bridge.verdict
            if verdict is not None or self.bridge.over:
                self._finish(verdict or {"abandoned": True})
        return [self._ended]

    @property
    def success(self):
        verdict = self._verdict or {}
        return [bool(verdict.get("success")) if self._ended else True]

    @property
    def verdict(self):
        return None if self._verdict is None else dict(self._verdict)

    @property
    def sim_time_s(self):
        return self._ticks * self.dt

    @property
    def reward_manager(self) -> Any:
        raise EvaluatorUnavailable("reward_manager is not available to the policy")

    @property
    def scene_manager(self) -> Any:
        raise EvaluatorUnavailable("scene_manager is not available to the policy")

    def close(self) -> None:
        return None


class _Clock:
    def __init__(self, facade: EnvFacade):
        self.facade = facade

    @property
    def current_time(self):
        return self.facade.sim_time_s


class _App:
    config: dict = {"fast_shutdown": None}

    def close(self):
        return None


def _planner_robodojo_root(planners: dict, urdf_path: Path) -> Path | None:
    """Root R with R/Assets/Robots/x5/X5A.urdf == the URDF of the (shared) planner."""
    for planner in (planners or {}).values():
        try:
            kinematics = planner.robot_cfg["robot_cfg"]["kinematics"]
            path = Path(kinematics["urdf_path"])
            if not path.is_absolute():
                path = Path(kinematics["asset_root_path"]) / path
            path = path.resolve()
        except (AttributeError, KeyError, TypeError):
            continue
        root = path.parents[len(Path(urdf_path).parts) - 1]
        if path.is_file() and (root / urdf_path).resolve() == path:
            return root
    return None


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


class RemoteRobotBackend:
    """Same public surface as auriga.robot.backends.robodojo.robodojo.RobotBackend, in-process.

    Every reply passes through JSON like the worker IPC does, so receipts and observations have
    exactly the types the Harness receives from the worker.
    """

    def __init__(self, bridge, *, planners: dict, calibration: dict, config: Any, run_dir, task: str,
                 seed: int = 0, robodojo_root: Path | str | None = None, versions: dict | None = None):
        from auriga.core.config import motion_thresholds
        from auriga.core.storage.images import image_settings
        from auriga.robot.backends.robodojo import robodojo
        from auriga.robot.model.embodiments import ARX_X5, get_robodojo_robot
        self.validate_action = robodojo.validate_action
        self.compile_action = robodojo.compile_action
        if config.backend.robot_model_id != ARX_X5.robot_model_id:
            raise ValueError("xpolicylab_adapter_supports_arx_x5_only")
        self.robot_binding = get_robodojo_robot(ARX_X5.robot_model_id)
        # The adapter checks that the planner's URDF is the robot model it reads (joint replanning
        # is skipped otherwise), so default to the tree the planner was built from.
        self.robodojo_root = (Path(robodojo_root).resolve() if robodojo_root else
                              _planner_robodojo_root(planners, self.robot_binding.urdf_path) or DEFAULT_ROBODOJO_ROOT)
        self.bridge, self.planners, self.calibration = bridge, planners, calibration
        self.config, self.task, self.seed = config, task, seed
        self.run_dir = Path(run_dir).resolve()
        features = config.features
        self.observation_profile = features.observation_profile
        self.tracking_guard, self.joint_replan = features.tracking_guard, features.joint_replan
        self.motion_safety_guard, self.trajectory_preview = features.motion_safety_guard, features.trajectory_preview
        self.motion_execution_mode = features.motion_execution_mode
        self.image_settings = image_settings(config.images.to_dict())
        self.motion_thresholds = motion_thresholds(features)
        self.extra_versions = dict(versions or {})
        self.operation_lock = threading.Lock()
        self.process = None
        self.failure = None
        self.capabilities = None
        self.adapter = None
        self.facade = None
        self.last_activity_at = time.monotonic()
        self.closed = False

    # -- lifecycle -------------------------------------------------------------------------------
    def _versions(self) -> dict:
        from auriga.robot.backends.robodojo.robot_worker import ADAPTER_SOURCE_SHA256
        root, binding = self.robodojo_root, self.robot_binding
        return {"adapter_source_sha256": ADAPTER_SOURCE_SHA256, "environment_host": "xpolicylab_robodojo_eval_client",
                "robot_config_sha256": _sha256(root / binding.robot_config_path),
                "curobo_config_sha256": _sha256(root / binding.curobo_config_path),
                "urdf_sha256": _sha256(root / binding.urdf_path), **self.extra_versions}

    def start(self):
        if self.adapter is not None:
            return self.capabilities
        from auriga.robot.backends.robodojo.robot_worker import RoboDojoAdapter
        self.run_dir.mkdir(parents=True, exist_ok=True)
        facade = EnvFacade(self.bridge, self.planners, self.calibration,
                           urdf_path=self.robodojo_root / self.robot_binding.urdf_path,
                           curobo_yml_path=str(self.robodojo_root / self.robot_binding.curobo_config_path))
        arguments = SimpleNamespace(
            run_dir=str(self.run_dir), robodojo_root=str(self.robodojo_root), task=self.task, seed=self.seed,
            layout_id=None, gpu=None, observation_profile=self.observation_profile,
            robot_model_id=self.robot_binding.robot.robot_model_id, tracking_guard=self.tracking_guard,
            joint_replan=self.joint_replan, motion_safety_guard=self.motion_safety_guard,
            trajectory_preview=self.trajectory_preview, motion_execution_mode=self.motion_execution_mode,
            scene_truth_log=None)
        environment = SimpleNamespace(
            env=facade, app=_App(), robot_binding=self.robot_binding, versions=self._versions(),
            layout_acceptance={"valid": None, "stable": None,
                               "authority": "environment host (RoboDojo eval client); not visible to the policy"},
            image_settings=dict(self.image_settings), motion_thresholds=dict(self.motion_thresholds))
        self.facade = facade
        self.adapter = RoboDojoAdapter(arguments, environment)
        self.capabilities = self._reply(self.adapter.capabilities())
        return self.capabilities

    def close(self):
        if self.closed:
            return dict(self._close_report)
        self.closed = True
        report = {"native_close_acknowledged": True, "environment_close_completed": True,
                  "application_close_started": True, "application_close_completed": True,
                  "forced_termination": None, "process_exit_confirmed": True, "reader_exit_confirmed": True,
                  "in_process": True, "errors": []}
        if self.adapter is None:
            report.update(native_close_acknowledged=False, not_started=True)
        else:
            try:
                self.adapter.close()
            except Exception as error:  # noqa: BLE001 - recorded in the cleanup report
                report["errors"].append(f"{type(error).__name__}: {error}")
        self._close_report = report
        return dict(report)

    # -- operations ------------------------------------------------------------------------------
    @staticmethod
    def _reply(value):
        return json.loads(json.dumps(value, allow_nan=False))

    def _call(self, operation, *args):
        from auriga.robot.backends.robodojo.robot_process import GPU_OOM_FAILURE, GPU_OOM_REPLY
        if self.failure:
            raise RuntimeError(self.failure)
        if self.adapter is None:
            raise RuntimeError("robot_backend_not_started")
        try:
            result = self._reply(getattr(self.adapter, operation)(*args))
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            if GPU_OOM_REPLY.match(message):
                self.failure = GPU_OOM_FAILURE
            raise RuntimeError(message) from error
        finally:
            self.last_activity_at = time.monotonic()
        reason = result.get("reason") if isinstance(result, dict) else None
        if isinstance(reason, str) and GPU_OOM_REPLY.match(reason):
            self.failure = GPU_OOM_FAILURE
        return result

    def observe(self):
        with self.operation_lock:
            return self._call("observe")

    def preview_motion(self, action):
        from auriga.robot.model.arx_contract import validate_preview_action
        validate_preview_action(action)
        if "preview_plan_id" in action:
            raise ValueError("motion_preview_cannot_reference_existing_plan")
        with self.operation_lock:
            return self._call("preview_motion", action)

    def preview_batch(self, action):
        from auriga.robot.motion.batch_contract import validate_batch_action
        validate_batch_action(action)
        with self.operation_lock:
            return self._call("preview_batch", action)

    def execute(self, action, progress_callback=None, cancel_flag=None):
        from auriga.robot.model.arx_contract import validate_joint_prefix
        if action.get("type") == "JointPrefixAction":
            validate_joint_prefix(action)
        else:
            self.validate_action(action)
        cancellation = cancel_flag if cancel_flag is not None else threading.Event()

        def progress(update):
            self.last_activity_at = time.monotonic()
            if progress_callback is not None:
                progress_callback(self._reply(update))

        with self.operation_lock:
            return self._call("execute", action, progress, cancellation)

    def execute_joint_prefix(self, action, progress_callback=None, cancel_flag=None):
        from auriga.robot.model.arx_contract import validate_joint_prefix
        validate_joint_prefix(action)
        return self.execute(action, progress_callback, cancel_flag)

    def record(self, enabled=True):
        with self.operation_lock:
            return self._call("record", bool(enabled))

    def save_video(self):
        raise EvaluatorUnavailable("observer video is recorded by the RoboDojo eval client")

    def evaluate(self, final_check=True):
        raise EvaluatorUnavailable("native evaluator is owned by the RoboDojo eval client")

    def observer_snapshot(self):
        raise EvaluatorUnavailable("scene state is not available to the policy")

    def validate_reset(self, seed=0, layout_id=0):
        raise ValueError("reset_is_owned_by_the_robodojo_eval_client")

    def reset(self, seed=0, layout_id=0):
        raise ValueError("reset_is_owned_by_the_robodojo_eval_client")

    def fast_forward_slice(self, checkpoint_spec, timeout=180.0):
        raise ValueError("checkpoint_replay_unavailable_in_xpolicylab_deployment")
