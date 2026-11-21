"""RoboDojo worker for one pinned task.

Importing this module does not import Isaac, torch, or RoboDojo. The simulator
starts only when ``create_environment`` runs inside the RoboDojo interpreter.
The client speaks the harness HTTP worker protocol. This process does not
connect to an XPolicyLab policy server.
"""

import math
import os
from pathlib import Path
import sys
import uuid

from harness_policy.core.contracts import Action, Observation, Transition

from .codec import (
    ACTION_NAME,
    ARM_DIMS,
    CAMERAS,
    ENV_CONFIG,
    GRIPPER_DIMS,
    LAYOUT_PACK_SEED,
    ROBOT_CONFIG,
    STACK,
    SUPPORTED_TASK,
    EpisodeCursor,
    action_dict,
    data_url,
    episode_outcome,
    finite_bounds,
    instruction_text,
    policy_observation,
    tool_schema,
    validate_position,
)
from .video import MotionClip


def create_environment(**options):
    if os.environ.get("HARNESS_ROBODOJO_RUNTIME") != "1":
        raise RuntimeError(
            "RoboDojo adapter must run in the RoboDojo Python runtime with "
            "HARNESS_ROBODOJO_RUNTIME=1. See benches/robodojo/README.md."
        )
    unknown = set(options) - {"device", "headless", "checkout"}
    if unknown:
        raise ValueError(f"Unknown RoboDojo options: {sorted(unknown)}")
    device = options.get("device", "cuda:0")
    if device != "cuda:0":
        raise ValueError("RoboDojo worker uses cuda:0 inside CUDA_VISIBLE_DEVICES")
    return RoboDojoEnvironment(
        device=device,
        headless=options.get("headless", True),
        checkout=options.get("checkout"),
    )


class RoboDojoEnvironment:
    """One sequential ``stack_bowls`` environment behind the harness worker."""

    name = "robodojo-isaac51"

    def __init__(self, device="cuda:0", headless=True, checkout=None):
        self.device = device
        self.headless = bool(headless)
        self.checkout = _checkout(checkout)
        self._app = None
        self._env = None
        self._bounds = None
        self._layout = None
        self._tools = None
        self._task = SUPPORTED_TASK
        self._seed = None
        self._cursor = EpisodeCursor()
        try:
            self._boot()
        except Exception as error:
            if self._app is not None:
                print(
                    f"RoboDojo worker failed during startup: {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            try:
                self.close()
            except Exception:
                pass
            if isinstance(error, RuntimeError):
                raise
            raise RuntimeError(
                f"RoboDojo worker failed during startup: {type(error).__name__}: {error}"
            ) from error

    def tools(self):
        if self._tools is None:
            raise RuntimeError("RoboDojo action schema is not available")
        return self._tools

    def reset(self, task: str, seed: int) -> Observation:
        if task != SUPPORTED_TASK:
            raise ValueError(f"RoboDojoEnvironment supports only task={SUPPORTED_TASK!r}")
        if type(seed) is not int or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self._reseed(seed)
        self._cursor.start(uuid.uuid4().hex)
        return self._observe()

    def execute(self, action: Action, episode_id: str, step: int) -> Transition:
        self._cursor.check(episode_id, step)
        if self._env is None or self._bounds is None:
            raise ValueError("Stale or missing observation; action was not executed")
        if bool(self._env.end_flag[0]):
            raise ValueError("Episode already ended; action was not executed")
        if action.name != ACTION_NAME:
            raise ValueError(f"Unknown action: {action.name}")
        command = validate_position(action.arguments, self._bounds)
        payload = action_dict(self._layout, command)
        self._reset_clip()
        try:
            self._env.take_action(payload)
        except Exception:
            self._cursor.invalidate()
            raise
        self._cursor.advance()
        try:
            videos = self._encode_clip()
            observation = self._observe()
            # get_reward already ran inside take_action. Calling get_score with
            # no cache treats the episode as successful, so read the cache only.
            scores = getattr(self._env.reward_manager, "_gated_score_lst", None)
            process_score = 0.0 if not scores else float(scores[0])
            reward, success, terminated, truncated = episode_outcome(
                bool(self._env.end_flag[0]),
                bool(self._env.success[0]),
                process_score,
                int(self._env.take_action_cnt[0]),
                int(self._env.step_lim),
            )
            return Transition(
                observation,
                reward,
                success,
                terminated,
                truncated,
                {"executed_position": command},
                videos,
            )
        except Exception:
            self._cursor.invalidate()
            raise

    def close(self):
        env = self._env
        self._env = None
        self._cursor.invalidate()
        self._reset_clip()
        if env is not None:
            env.close()
        if self._app is not None:
            self._app.close()
            self._app = None

    def _boot(self):
        runtime = Path(os.environ.get("TMPDIR", "/tmp")) / "harness-robodojo"
        runtime.mkdir(parents=True, exist_ok=True)
        os.chdir(runtime)
        os.environ.setdefault("ROBODOJO_RUN_ID", "harness-worker")
        checkout = str(self.checkout)
        xpolicy = str(self.checkout / "XPolicyLab")
        for entry in (xpolicy, checkout):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        self._launch_kit()
        self._open()
        self._reseed(LAYOUT_PACK_SEED)
        self._read_action_spec()
        print(
            f"RoboDojo worker ready task={SUPPORTED_TASK} config={ENV_CONFIG} "
            f"robot={ROBOT_CONFIG} device={self.device}",
            file=sys.stderr,
            flush=True,
        )

    def _launch_kit(self):
        import argparse

        try:
            from isaaclab.app import AppLauncher
        except ImportError as error:
            raise RuntimeError(
                "isaaclab is not importable. Start the worker with the RoboDojo conda interpreter "
                "and a pinned RoboDojo checkout on PYTHONPATH."
            ) from error
        try:
            # Present from RoboDojo 82e1ba3. Earlier checkouts, which Simate was
            # trained against, have no zero-delay Kit settings.
            from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
        except ImportError:
            add_zero_delay_kit_args = None
        parser = argparse.ArgumentParser(add_help=False)
        AppLauncher.add_app_launcher_args(parser)
        kit_args = "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
        # Machine-specific Kit flags. Examples: "--/rtx/verifyDriverVersion/enabled=false" for
        # drivers such as 535.309, whose minor version Vulkan wraps past 255 (to 535.53), failing
        # the RTX driver check; "--/app/hangDetector/enabled=false" on a desktop, where Kit takes
        # an idle worker for a hang and offers to kill it.
        if os.environ.get("HARNESS_KIT_ARGS"):
            kit_args += " " + os.environ["HARNESS_KIT_ARGS"]
        arguments = ["--enable_cameras", "--device", self.device, "--kit_args", kit_args]
        if self.headless:
            arguments.append("--headless")
        args, _unknown = parser.parse_known_args(arguments)
        args.headless = self.headless
        args.enable_cameras = True
        args.device = self.device
        if add_zero_delay_kit_args is not None:
            add_zero_delay_kit_args(args)
        launcher = AppLauncher(args)
        self._app = launcher.app

    def _open(self):
        import importlib

        from omegaconf import OmegaConf

        from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
        from src.eval_client import eval_env as eval_env_module
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        eval_env_module.WsModelClient = _NoPolicyClient
        eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, f"{ENV_CONFIG}.yml"))
        eval_cfg["task_name"] = SUPPORTED_TASK
        eval_cfg["num_envs"] = 1
        eval_cfg["device_id"] = 0
        eval_cfg["eval_batch"] = False
        eval_cfg["policy_name"] = "harness"
        eval_cfg["additional_info"] = "harness"
        eval_cfg["seed"] = LAYOUT_PACK_SEED
        eval_cfg["eval_num"] = 1
        eval_cfg["physx_monitor_enabled"] = False
        deploy_cfg = {
            "policy_name": "harness",
            "port": 1,
            "host": "127.0.0.1",
            "protocol": "ws",
            "policy_server_url": "ws://127.0.0.1:1",
            "evaluation_id": os.environ.get("ROBODOJO_RUN_ID", "harness-worker"),
            "trial_id": f"{SUPPORTED_TASK}-harness",
            "action_case_id": f"{SUPPORTED_TASK}_case",
            "repeat_index": None,
        }
        task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
        benchmark_config = os.path.join(ROOT_DIR, "task", BENCHMARK, "config")
        env_cfg = OmegaConf.create({
            "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", f"{eval_cfg['config']['sim']}.yml")),
            "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", f"{eval_cfg['config']['scene']}.yml")),
            "camera": load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", f"{eval_cfg['config']['camera']}.yml")),
            "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", f"{eval_cfg['config']['robot']}.yml")),
            "task_env": load_yaml(str(task_registry.task_config_path(benchmark_config, SUPPORTED_TASK))),
            "eval_cfg": eval_cfg,
            "deploy_cfg": deploy_cfg,
        })
        if env_cfg.robot is None or eval_cfg["config"]["robot"] != ROBOT_CONFIG:
            raise RuntimeError(f"RoboDojoEnvironment supports only robot config {ROBOT_CONFIG}")
        OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
        OmegaConf.update(env_cfg, "eval_cfg.num_envs", 1, force_add=True)
        env_cfg = process_randomization(env_cfg)
        env_cfg, _eval_num = process_config(env_cfg, task_name=SUPPORTED_TASK)
        OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
        OmegaConf.update(env_cfg, "eval_cfg.num_envs", 1, force_add=True)
        OmegaConf.update(env_cfg, "eval_cfg.eval_num", 1, force_add=True)
        OmegaConf.update(
            env_cfg,
            "camera.default_frequency",
            int(eval_cfg["observation"].get("collect_freq", 0)),
            force_add=True,
        )
        env_cfg.sim.seed = [0]
        self._env = eval_env_module.create_eval_env(env_cfg, self._app)
        self._env._stream_vision = _discard_eval_video
        self._install_motion_video()

    def _reseed(self, seed: int):
        # RoboDojo creates articulated objects (toaster, laptop, spring button, egg holder)
        # only when the simulation launches: a reset of a loaded scene reuses them, yet
        # renames them in its layout records, so their lookup fails ('NoneType' object has no
        # attribute '_get_object_transform'). The official eval loop closes the env before
        # every round; do the same whenever the new layout has an articulated object.
        layout = self._env.seed_manager.get_seed_scene_info(seed) or {}
        if layout.get("Articulation") and getattr(self, "_scene_loaded", False):
            self._env.close()
        # Set before reset: a reset that fails (e.g. an unstable layout) has already built the scene.
        self._scene_loaded = True
        self._env.reset(seed=[seed])
        if bool(self._env.end_flag[0]):
            raise RuntimeError("RoboDojo reset ended the episode before an action")
        # Official eval registers reward and score checks after reset. An empty
        # check list makes get_reward return 1, which would mark a fresh episode
        # successful on the first step.
        self._env.run_reward()
        if hasattr(self._env, "get_score"):
            self._env.get_score()
        # Official eval (run_eval) also loads the support arm's trajectory before the
        # policy's first step in interactive tasks (make_kong's opponent discarding a
        # tile, play_tic_tac_toe's opponent, imitate_sorting_sequence's demonstration).
        if getattr(self._env, "interact", False) and hasattr(self._env, "query_support_arm_traj"):
            self._env.query_support_arm_traj(env_idx=0)
        self._seed = seed

    def _read_action_spec(self):
        info = self._env.robot_action_dim_info
        arm_dims = tuple(int(value) for value in info["arm_dim"])
        gripper_dims = tuple(int(value) for value in info["ee_dim"])
        if arm_dims != ARM_DIMS or gripper_dims != GRIPPER_DIMS:
            raise RuntimeError(
                f"Expected dual-X5 action dims arm={ARM_DIMS} gripper={GRIPPER_DIMS}, "
                f"got arm={arm_dims} gripper={gripper_dims}"
            )
        robots = [
            robot for robot in self._env.robot_manager.robot_list
            if getattr(robot, "type", None) == "target" and getattr(robot, "robot_type", None) == "arm"
        ]
        if [robot.arm_name for robot in robots] != ["left_arm", "right_arm"]:
            raise RuntimeError("RoboDojoEnvironment expected target arms left_arm and right_arm")
        if [robot.gripper_name for robot in robots] != ["left_ee", "right_ee"]:
            raise RuntimeError("RoboDojoEnvironment expected grippers left_ee and right_ee")
        layout = []
        low = []
        high = []
        manager = self._env.robot_manager
        for robot in robots:
            if len(list(robot.arm_joints_name)) != int(robot.action_dim):
                raise RuntimeError(f"{robot.arm_name} joint names do not match its action dimension")
            articulation = manager.robot_key[manager.robot_list.index(robot)]
            arm_bounds = _joint_bounds(articulation, robot.arm_joint_indices)
            if len(arm_bounds) != len(list(robot.arm_joints_name)):
                raise RuntimeError(f"{robot.arm_name} limits do not match its joints")
            layout.append({
                "term": str(robot.arm_name),
                "key": str(manager.process_name(robot.arm_name)),
                "joints": [str(name) for name in robot.arm_joints_name],
                "size": len(list(robot.arm_joints_name)),
                "command": "absolute_joint",
            })
            for lower, upper in arm_bounds:
                low.append(lower)
                high.append(upper)
            layout.append({
                "term": str(robot.gripper_name),
                "key": str(manager.process_name(robot.gripper_name)),
                "joints": ["normalized_opening"],
                "size": 1,
                "command": "gripper_normalized",
            })
            low.append(0.0)
            high.append(1.0)
        self._bounds = finite_bounds(low, high)
        self._layout = layout
        self._tools = tool_schema(self._bounds, layout)
        self._physics_device = str(getattr(self._env, "device", "unknown"))

    def _observe(self):
        import numpy as np

        raw = policy_observation(self._env.obs_manager)[0]
        state_in = raw.get("state") if isinstance(raw, dict) else None
        if not isinstance(state_in, dict):
            raise RuntimeError("RoboDojo observation is missing robot state")
        joint_position = []
        poses = {}
        manager = self._env.robot_manager
        for robot in self._target_robots():
            arm_key = manager.process_name(robot.arm_name)
            arm = _vector(state_in.get(arm_key), len(list(robot.arm_joints_name)), arm_key)
            measured = manager.get_end_effector_real_val(robot, env_idx_list=[0])[0]
            opening = _normalized_opening(robot, _vector(measured, None, robot.gripper_name)[0])
            joint_position.extend(arm)
            joint_position.append(opening)
            pose_key = f"{str(robot.arm_name).split('_')[0]}_ee_pose"
            poses[pose_key] = _vector(state_in.get(pose_key), 7, pose_key)
        if len(joint_position) != len(self._bounds):
            raise RuntimeError("Observed joint position does not match the action layout")
        from .codec import encode_png

        images = {}
        vision = raw.get("vision") if isinstance(raw.get("vision"), dict) else {}
        for camera in CAMERAS:
            frame = vision.get(camera, {}).get("color") if isinstance(vision.get(camera), dict) else None
            if frame is None:
                raise RuntimeError(f"RoboDojo observation is missing {camera}")
            array = np.asarray(frame.detach().cpu().numpy() if hasattr(frame, "detach") else frame)
            if array.ndim != 3 or array.shape[-1] not in (3, 4):
                raise RuntimeError(f"{camera} image must be RGB")
            if array.shape[-1] == 4:
                array = array[..., :3]
            if array.dtype != np.uint8:
                peak = float(array.max()) if array.size else 0.0
                array = array * 255.0 if peak <= 1.0 else array
                array = np.clip(array, 0, 255).astype(np.uint8)
            array = np.ascontiguousarray(array)
            height, width = int(array.shape[0]), int(array.shape[1])
            images[camera] = data_url(encode_png(width, height, array.tobytes()))
        interval = self._env.obs_manager.collect_interval
        state = {
            "stack": STACK,
            "task": self._task,
            "env_config": ENV_CONFIG,
            "robot_config": ROBOT_CONFIG,
            "layout_pack_seed": LAYOUT_PACK_SEED,
            "seed": self._seed,
            "physics_device": self._physics_device,
            "step_limit": int(self._env.step_lim),
            "control_steps": int(interval),
            "joint_position": joint_position,
            "action_terms": self._layout,
            "action_low": [lower for lower, _upper in self._bounds],
            "action_high": [upper for _lower, upper in self._bounds],
        }
        state.update(poses)
        return Observation(
            self._cursor.episode_id,
            self._cursor.step,
            instruction_text(raw.get("instruction")),
            state,
            images,
        )

    def _reset_clip(self):
        """Drop the current motion clip (stopping its encoders) and start an empty one."""
        clip = getattr(self, "_clip", None)
        if clip is not None:
            clip.discard()
        self._clip = MotionClip()
        self._physics_steps = 0

    def _install_motion_video(self):
        self._reset_clip()
        env = self._env
        original = env.step

        def step(env_idx_list, decimation=1):
            original(env_idx_list, decimation=decimation)
            # One control step is collect_interval physics steps (10 at RoboDojo's 250 Hz
            # physics and 25 Hz control). Capture once per control step, so the clip's
            # frames match its collect_freq frame rate and play back in real time.
            self._physics_steps += 1
            if self._physics_steps % max(1, int(self._env.obs_manager.collect_interval)) == 0:
                self._capture_motion_frame()

        env.step = step

    def _capture_motion_frame(self):
        import numpy as np

        raw = policy_observation(self._env.obs_manager)[0]
        vision = raw.get("vision") if isinstance(raw, dict) and isinstance(raw.get("vision"), dict) else {}
        fps = float(getattr(self._env.obs_manager, "collect_freq", 25))
        for camera in CAMERAS:
            frame = vision.get(camera, {}).get("color") if isinstance(vision.get(camera), dict) else None
            if frame is None:
                raise RuntimeError(f"RoboDojo motion video is missing {camera}")
            # Streamed to the camera's encoder now; frames are not kept in memory.
            self._clip.add(camera, _rgb_array(np, frame), fps)

    def _encode_clip(self):
        """Finish the streamed clip: one MP4 data URL per camera that received frames."""
        clip, self._clip = self._clip, MotionClip()
        self._physics_steps = 0
        return {camera: data_url(payload, "video/mp4") for camera, payload in clip.finish().items()}

    def _target_robots(self):
        return [
            robot for robot in self._env.robot_manager.robot_list
            if getattr(robot, "type", None) == "target" and getattr(robot, "robot_type", None) == "arm"
        ]


class _NoPolicyClient:
    """Stand-in for XPolicyLab's websocket client. It never connects."""

    def __init__(self, *args, **kwargs):
        pass

    def call(self, *args, **kwargs):
        return None

    def close(self):
        return None


def _discard_eval_video(*_args, **_kwargs):
    """Keep RoboDojo's eval_result mp4 writer idle. Harness clips are separate."""
    return None


def _rgb_array(np, frame):
    array = np.asarray(frame.detach().cpu().numpy() if hasattr(frame, "detach") else frame)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise RuntimeError("Camera frame must be RGB")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        peak = float(array.max()) if array.size else 0.0
        array = array * 255.0 if peak <= 1.0 else array
        array = np.clip(array, 0, 255).astype(np.uint8)
    array = np.ascontiguousarray(array)
    pad_h = int(array.shape[0] % 2)
    pad_w = int(array.shape[1] % 2)
    if pad_h or pad_w:
        array = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return array


def _checkout(explicit):
    if explicit:
        path = Path(explicit)
    elif os.environ.get("HARNESS_ROBODOJO_ROOT"):
        path = Path(os.environ["HARNESS_ROBODOJO_ROOT"])
    else:
        path = Path(__file__).resolve().parents[2] / "third_party" / "RoboDojo"
    marker = path / "src" / "eval_client" / "eval_env.py"
    if not marker.is_file():
        raise RuntimeError(f"RoboDojo checkout not found at {path}")
    return path


def _joint_bounds(articulation, indices):
    data = articulation.data
    tensor = getattr(data, "soft_joint_pos_limits", None)
    if tensor is None:
        tensor = data.joint_pos_limits
    row = tensor.detach().cpu() if hasattr(tensor, "detach") else tensor
    if hasattr(row, "ndim") and row.ndim == 3:
        row = row[0]
    values = row.tolist()
    bounds = []
    for index in _index_list(indices):
        pair = values[index]
        lower, upper = float(pair[0]), float(pair[1])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise RuntimeError("Arm joint limits must be finite and ordered")
        bounds.append((lower, upper))
    return bounds


def _index_list(indices):
    if hasattr(indices, "detach"):
        indices = indices.detach().cpu().reshape(-1).tolist()
    elif hasattr(indices, "tolist"):
        indices = indices.tolist()
    return [int(index) for index in list(indices)]


def _vector(value, expected, label):
    if value is None:
        raise RuntimeError(f"RoboDojo observation is missing {label}")
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "reshape"):
        value = value.reshape(-1).tolist()
    elif isinstance(value, (bytes, str)):
        raise RuntimeError(f"{label} is not a numeric vector")
    numbers = []
    for item in list(value):
        if isinstance(item, (list, tuple)):
            raise RuntimeError(f"{label} must be a one-dimensional vector")
        number = float(item)
        if not math.isfinite(number):
            raise RuntimeError(f"{label} contains a non-finite value")
        numbers.append(number)
    if expected is not None and len(numbers) != expected:
        raise RuntimeError(f"{label} length {len(numbers)} does not match {expected}")
    return numbers


def _normalized_opening(robot, raw):
    scale = robot.gripper_scale
    lower, upper = float(scale[0]), float(scale[1])
    if not math.isfinite(lower) or not math.isfinite(upper) or upper == lower:
        raise RuntimeError(f"{robot.gripper_name} gripper scale is invalid")
    if float(robot.gripper_move["sign"]) != 1.0:
        raise RuntimeError(f"{robot.gripper_name} gripper sign is not +1")
    opening = (float(raw) - lower) / (upper - lower)
    if not math.isfinite(opening):
        raise RuntimeError(f"{robot.gripper_name} opening is not finite")
    return opening
