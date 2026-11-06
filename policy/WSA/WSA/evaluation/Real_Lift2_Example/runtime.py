#!/usr/bin/env python

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np
import yaml

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
SRC_ROOT = ROOT / "src"
ROBOT_RUNTIME_ROOT = Path(
    os.environ.get("REAL_LIFT2_RUNTIME_ROOT", str(Path.home() / "ROS2_LIFT_Play" / "act"))
).expanduser()

for candidate in [THIS_DIR, ROOT, SRC_ROOT, ROBOT_RUNTIME_ROOT]:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    import rclpy
except Exception as exc:
    rclpy = None
    RCLPY_IMPORT_ERROR = exc
else:
    RCLPY_IMPORT_ERROR = None

try:
    from utils.ros_operator import Rate, RosOperator
    from utils.setup_loader import setup_loader
    from utils.utils import set_seed
except Exception as exc:
    Rate = None
    RosOperator = None
    setup_loader = None
    set_seed = None
    REAL_ROBOT_IMPORT_ERROR = exc
else:
    REAL_ROBOT_IMPORT_ERROR = None

_shutdown_in_progress = False
_observation_warning_cache: set[str] = set()
DEFAULT_MANUAL_HOME_DURATION_S = 1.0
DEFAULT_MANUAL_HOME_RESUME_GUARD_STEPS = 12
MANUAL_HOME_INACTIVE = 0
MANUAL_HOME_ACTIVE = 1
MANUAL_HOME_RESUME_GUARD = 2


def set_manual_home_state(manual_home_command, state: int) -> None:
    with manual_home_command.get_lock():
        manual_home_command.value = int(state)


def ensure_compat_args(args) -> None:
    """Populate legacy robot-runtime flags expected by the ROS deployment helpers."""
    compat_defaults = {
        "use_depth_image": False,
        "use_qvel": False,
        "use_effort": False,
        "use_eef_states": False,
        "manual_home_publish_steps": 0,
        "manual_home_resume_guard_steps": DEFAULT_MANUAL_HOME_RESUME_GUARD_STEPS,
    }
    for key, value in compat_defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def ensure_runtime_available() -> None:
    if rclpy is None:
        raise ImportError(
            "rclpy is not available. Please run this script in the same robot runtime environment."
        ) from RCLPY_IMPORT_ERROR
    if RosOperator is None or Rate is None or setup_loader is None:
        raise ImportError(
            "Robot deployment helpers (utils.ros_operator / utils.setup_loader / utils.utils) are not available. "
            "Please run this script in the same robot runtime environment."
        ) from REAL_ROBOT_IMPORT_ERROR


def load_yaml(yaml_file: str | Path) -> dict[str, Any] | None:
    try:
        with open(yaml_file, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: file not found - {yaml_file}")
        return None
    except yaml.YAMLError as exc:
        print(f"Error: failed to parse YAML file - {exc}")
        return None


def log_once(message: str) -> None:
    if message in _observation_warning_cache:
        return
    _observation_warning_cache.add(message)
    print(message)


def make_shm_name_dict(args, shapes):
    shm_name_dict = {}
    for cam in args.camera_names:
        shm_name_dict[cam] = f"shm_img_{cam}"
    for state_key in shapes["states"]:
        shm_name_dict[state_key] = f"shm_state_{state_key}"
    shm_name_dict["action"] = "shm_action"
    return shm_name_dict


def create_shm_dict(config, shm_name_dict, shapes, dtypes):
    shm_dict = {}
    for cam, shape in shapes["images"].items():
        size = int(np.prod(shape) * np.dtype(dtypes[cam]).itemsize)
        shm = SharedMemory(name=shm_name_dict[cam], create=True, size=size)
        shm_dict[cam] = (shm, shape, dtypes[cam])

    for state_key, shape in shapes["states"].items():
        size = int(np.prod(shape) * np.dtype(np.float32).itemsize)
        shm = SharedMemory(name=shm_name_dict[state_key], create=True, size=size)
        shm_dict[state_key] = (shm, shape, np.float32)

    action_shape = (config["policy_config"]["action_dim"],)
    size = int(np.prod(action_shape) * np.dtype(np.float32).itemsize)
    shm = SharedMemory(name=shm_name_dict["action"], create=True, size=size)
    shm_dict["action"] = (shm, action_shape, np.float32)
    return shm_dict


def connect_shm_dict(shm_name_dict, shapes, dtypes, config):
    shm_dict = {}
    for cam, shape in shapes["images"].items():
        shm = SharedMemory(name=shm_name_dict[cam], create=False)
        shm_dict[cam] = (shm, shape, dtypes[cam])

    for state_key, shape in shapes["states"].items():
        shm = SharedMemory(name=shm_name_dict[state_key], create=False)
        shm_dict[state_key] = (shm, shape, np.float32)

    action_shape = (config["policy_config"]["action_dim"],)
    shm = SharedMemory(name=shm_name_dict["action"], create=False)
    shm_dict["action"] = (shm, action_shape, np.float32)
    return shm_dict


def robot_action(action, shm_dict):
    shm, shape, dtype = shm_dict["action"]
    np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    np_array[:] = action


def get_model_config(args):
    ensure_compat_args(args)
    if set_seed is not None:
        set_seed(args.seed)
    else:
        np.random.seed(args.seed)

    config = {
        "episode_len": args.max_publish_step,
        "policy_class": "wsa_base_remote",
        "policy_config": {
            "action_dim": args.action_dim,
            "states_dim": args.state_dim,
            "use_base": args.use_base,
            "use_depth_image": args.use_depth_image,
        },
        "camera_names": args.camera_names,
    }
    return config


def init_robot(ros_operator, use_base, connected_event, start_event):
    init0 = [0, 0, 0, 0, 0, 0, 4]
    init1 = [0, 0, 0, 0, 0, 0, 0]

    ros_operator.follow_arm_publish_continuous(init0, init0)

    connected_event.set()
    start_event.wait()

    ros_operator.follow_arm_publish_continuous(init1, init1)
    if use_base:
        ros_operator.start_base_control_thread()


def split_bimanual_action(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    left = np.zeros((7,), dtype=np.float32)
    right = np.zeros((7,), dtype=np.float32)
    left[: min(7, action.size)] = action[: min(7, action.size)]
    if action.size > 7:
        right[: min(7, action.size - 7)] = action[7 : 7 + min(7, action.size - 7)]
    return left, right


def read_current_arm_qpos(ros_operator) -> np.ndarray:
    obs = get_observation_or_none(ros_operator, log_prefix="[SafeStop]")

    if not obs or "qpos" not in obs:
        return np.zeros((14,), dtype=np.float32)

    qpos = np.asarray(obs["qpos"], dtype=np.float32).reshape(-1)
    out = np.zeros((14,), dtype=np.float32)
    out[: min(14, qpos.size)] = qpos[: min(14, qpos.size)]
    return out


def publish_staged_home_arms(
    ros_operator,
    current_qpos: np.ndarray,
    frame_rate: int,
    total_publish_steps: int,
    log_prefix: str,
) -> None:
    current_qpos = np.asarray(current_qpos, dtype=np.float32).reshape(-1)
    current_qpos = current_qpos[:14] if current_qpos.size >= 14 else np.pad(current_qpos, (0, 14 - current_qpos.size))
    publish_steps = max(2, int(total_publish_steps))
    sleep_s = max(1.0 / max(1, int(frame_rate)), 0.01)

    print(
        f"{log_prefix} Publishing smooth interpolated arm-home targets over {publish_steps} control steps."
    )

    for step_idx in range(1, publish_steps + 1):
        progress = step_idx / publish_steps
        # Smoothstep avoids large jumps at the beginning/end of the home motion.
        eased_progress = progress * progress * (3.0 - 2.0 * progress)
        staged = (1.0 - eased_progress) * current_qpos
        left_action, right_action = split_bimanual_action(staged)
        # Avoid the blocking "continuous" helper here. It tends to wait for each
        # tiny sub-target to finish and can spam per-step done logs, which makes
        # manual home look stop-and-go on the real robot.
        ros_operator.follow_arm_publish(left_action, right_action)
        time.sleep(sleep_s)


def resolve_manual_home_publish_steps(frame_rate: int, requested_steps: int) -> int:
    requested_steps = int(requested_steps)
    if requested_steps > 0:
        return max(2, requested_steps)

    auto_steps = int(round(max(1, frame_rate) * DEFAULT_MANUAL_HOME_DURATION_S))
    return max(8, auto_steps)


def publish_safe_stop_home_arms(
    ros_operator,
    frame_rate: int,
    publish_steps: int,
) -> None:
    """Publish a staged home pose before lowering the base."""
    if rclpy is not None and not rclpy.ok():
        print("[SafeStop] ROS context is already closed, skip arm-home publish.")
        return
    try:
        current_qpos = read_current_arm_qpos(ros_operator)
        publish_staged_home_arms(
            ros_operator=ros_operator,
            current_qpos=current_qpos,
            frame_rate=frame_rate,
            total_publish_steps=publish_steps,
            log_prefix="[SafeStop]",
        )
    except Exception as exc:
        print(f"[SafeStop] Failed to publish arm-home target: {exc}")


def publish_safe_stop_base(ros_operator, safe_stop_body_height: float | None, frame_rate: int, publish_steps: int) -> None:
    if safe_stop_body_height is None:
        return
    if rclpy is not None and not rclpy.ok():
        print("[SafeStop] ROS context is already closed, skip base safe-stop publish.")
        return

    try:
        action_base = np.zeros((10,), dtype=np.float32)
        action_base[3] = float(safe_stop_body_height)
        print(
            f"[SafeStop] Publishing base target height={safe_stop_body_height} "
            f"for {publish_steps} steps before shutdown."
        )
        for _ in range(max(1, int(publish_steps))):
            ros_operator.set_robot_base_target(action_base)
            time.sleep(max(1.0 / max(1, int(frame_rate)), 0.01))
    except Exception as exc:
        print(f"[SafeStop] Failed to publish safe-stop base target: {exc}")


def signal_handler(
    _signal,
    _frame,
    ros_operator,
    use_base: bool,
    safe_stop_home_arms: bool,
    safe_stop_home_publish_steps: int,
    safe_stop_body_height: float | None,
    safe_stop_publish_steps: int,
    frame_rate: int,
):
    global _shutdown_in_progress
    if _shutdown_in_progress:
        print("Shutdown already in progress, ignoring duplicate signal.")
        return
    _shutdown_in_progress = True
    print("Caught shutdown signal")
    if rclpy is not None and not rclpy.ok():
        print("[Shutdown] ROS context is already stopped, skip safe-stop commands.")
        sys.exit(0)

    base_thread = getattr(ros_operator, "base_control_thread", None)
    if use_base:
        # Stop the background base-hold loop first, otherwise it can keep
        # re-publishing the fixed body height (for example 16) and cause the
        # chassis to rebound after we send the safe-stop target height 0.
        ros_operator.base_enable = False
        if base_thread is not None:
            try:
                base_thread.join(timeout=2.0)
            except Exception as exc:
                print(f"[Shutdown] Failed to join base control thread cleanly: {exc}")

    if safe_stop_home_arms:
        publish_safe_stop_home_arms(
            ros_operator=ros_operator,
            frame_rate=frame_rate,
            publish_steps=safe_stop_home_publish_steps,
        )
    if use_base:
        publish_safe_stop_base(
            ros_operator=ros_operator,
            safe_stop_body_height=safe_stop_body_height,
            frame_rate=frame_rate,
            publish_steps=safe_stop_publish_steps,
        )
    try:
        ros_operator.robot_base_shutdown()
    except Exception as exc:
        print(f"[Shutdown] robot_base_shutdown failed: {exc}")
    sys.exit(0)


def request_graceful_ros_shutdown(ros_proc, args) -> None:
    if ros_proc is None or not ros_proc.is_alive():
        return

    graceful_timeout = 3.0
    graceful_timeout += max(0, float(args.safe_stop_home_publish_steps)) / max(1, float(args.frame_rate))
    graceful_timeout += max(0, float(args.safe_stop_publish_steps)) / max(1, float(args.frame_rate))

    try:
        if ros_proc.pid is not None:
            os.kill(ros_proc.pid, signal.SIGINT)
    except Exception as exc:
        print(f"[Shutdown] Failed to send SIGINT to ROS process: {exc}")

    ros_proc.join(timeout=graceful_timeout)
    if ros_proc.is_alive():
        print("[Shutdown] ROS process did not exit gracefully in time, forcing terminate().")
        ros_proc.terminate()
        ros_proc.join(timeout=2.0)


def cleanup_shm(names):
    for name in names:
        try:
            shm = SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


def get_observation_or_none(ros_operator, log_prefix: str = "[ROS]"):
    try:
        return ros_operator.get_observation()
    except AttributeError as exc:
        message = str(exc)
        if "_deque" in message:
            log_once(f"[ROS Warmup] Observation buffers are not ready yet: {message}")
            return None
        log_once(f"{log_prefix} get_observation raised AttributeError: {message}")
        return None
    except Exception as exc:
        log_once(f"{log_prefix} get_observation failed: {exc}")
        return None


def wait_for_camera_deques(
    ros_operator,
    camera_names,
    use_base: bool = False,
    timeout_s: float = 8.0,
    poll_interval_s: float = 0.05,
) -> bool:
    """Wait until ROS callbacks create and fill the expected deque buffers.

    On a fresh restart of only the final inference window, the camera nodes may
    still be alive but the newly created RosOperator instance can need a short
    warmup before `get_observation()` stops complaining about missing
    `<camera>_deque` attributes.
    """
    pending = set(camera_names)
    deadline = time.time() + max(0.1, float(timeout_s))
    has_logged_wait = False

    while pending and time.time() < deadline:
        still_pending = set()
        for cam in pending:
            deque_obj = getattr(ros_operator, f"{cam}_deque", None)
            if deque_obj is None:
                still_pending.add(cam)
                continue

            try:
                if len(deque_obj) <= 0:
                    still_pending.add(cam)
            except TypeError:
                still_pending.add(cam)

        if not still_pending and use_base:
            # Some runtime variants don't expose `base_pose_deque` immediately or
            # at all, so only treat it as pending when the attribute exists but is empty.
            base_pose_deque = getattr(ros_operator, "base_pose_deque", None)
            if base_pose_deque is not None:
                try:
                    if len(base_pose_deque) <= 0:
                        still_pending.add("base_pose")
                except TypeError:
                    still_pending.add("base_pose")

        if not still_pending:
            if has_logged_wait:
                print("[ROS Warmup] Camera deques are ready.")
            return True

        if not has_logged_wait:
            print(
                "[ROS Warmup] Waiting for camera streams to populate deque buffers: "
                + ", ".join(sorted(still_pending))
            )
            has_logged_wait = True

        pending = still_pending
        time.sleep(max(0.01, float(poll_interval_s)))

    if pending:
        print(
            "[ROS Warmup] Timed out waiting for camera deque buffers: "
            + ", ".join(sorted(pending))
            + ". Will keep trying through normal observation polling."
        )
        return False

    return True


def ros_process(args, config, meta_queue, connected_event, start_event, shm_ready_event, manual_home_command):
    ensure_runtime_available()
    ensure_compat_args(args)
    setup_loader(ROBOT_RUNTIME_ROOT)

    rclpy.init()
    data = load_yaml(args.data)
    ros_operator = RosOperator(args, data, in_collect=False)

    def _spin_loop(node):
        while rclpy.ok():
            try:
                rclpy.spin_once(node, timeout_sec=0.001)
            except Exception:
                break

    spin_thread = threading.Thread(target=_spin_loop, args=(ros_operator,), daemon=True)
    spin_thread.start()

    signal.signal(
        signal.SIGINT,
        partial(
            signal_handler,
            ros_operator=ros_operator,
            use_base=args.use_base,
            safe_stop_home_arms=args.safe_stop_home_arms,
            safe_stop_home_publish_steps=args.safe_stop_home_publish_steps,
            safe_stop_body_height=args.safe_stop_body_height,
            safe_stop_publish_steps=args.safe_stop_publish_steps,
            frame_rate=args.frame_rate,
        ),
    )
    signal.signal(
        signal.SIGTERM,
        partial(
            signal_handler,
            ros_operator=ros_operator,
            use_base=args.use_base,
            safe_stop_home_arms=args.safe_stop_home_arms,
            safe_stop_home_publish_steps=args.safe_stop_home_publish_steps,
            safe_stop_body_height=args.safe_stop_body_height,
            safe_stop_publish_steps=args.safe_stop_publish_steps,
            frame_rate=args.frame_rate,
        ),
    )

    init_robot(ros_operator, args.use_base, connected_event, start_event)
    wait_for_camera_deques(
        ros_operator=ros_operator,
        camera_names=args.camera_names,
        use_base=args.use_base,
    )

    rate = Rate(args.frame_rate)
    while rclpy.ok():
        obs = get_observation_or_none(ros_operator)
        if obs:
            shapes = {"images": {}, "states": {}, "dtypes": {}}
            for cam in args.camera_names:
                img = obs["images"][cam]
                shapes["images"][cam] = img.shape
                shapes["dtypes"][cam] = img.dtype
            shapes["states"]["qpos"] = obs["qpos"].shape
            shapes["states"]["qvel"] = obs["qvel"].shape
            shapes["states"]["effort"] = obs["effort"].shape
            shapes["states"]["robot_base"] = obs["robot_base"].shape
            shapes["states"]["base_velocity"] = obs["base_velocity"].shape
            meta_queue.put(shapes)
            break
        rate.sleep()

    shm_name_dict = meta_queue.get()
    cleanup_shm(shm_name_dict.values())
    shm_dict = create_shm_dict(config, shm_name_dict, shapes, shapes["dtypes"])
    shm_ready_event.set()

    rate = Rate(args.frame_rate)
    manual_home_active = False
    manual_home_resume_guard_active = False
    manual_home_resume_guard_steps_remaining = 0
    while rclpy.ok():
        obs = get_observation_or_none(ros_operator)
        if not obs:
            rate.sleep()
            continue

        manual_home_state = manual_home_command.value
        if manual_home_state == MANUAL_HOME_ACTIVE:
            if not manual_home_active:
                current_qpos = np.asarray(obs.get("qpos", np.zeros((14,), dtype=np.float32)), dtype=np.float32)
                publish_staged_home_arms(
                    ros_operator=ros_operator,
                    current_qpos=current_qpos,
                    frame_rate=args.frame_rate,
                    total_publish_steps=resolve_manual_home_publish_steps(
                        frame_rate=args.frame_rate,
                        requested_steps=args.manual_home_publish_steps,
                    ),
                    log_prefix="[Manual Home]",
                )
                manual_home_active = True
                manual_home_resume_guard_active = False
                manual_home_resume_guard_steps_remaining = 0
            if args.use_base and args.fixed_body_height >= 0:
                fixed_h = float(args.fixed_body_height)
                action_base = np.zeros((10,), dtype=np.float32)
                action_base[3] = fixed_h
                ros_operator.set_robot_base_target(action_base)
            rate.sleep()
            continue
        elif (
            manual_home_state == MANUAL_HOME_RESUME_GUARD
            and not manual_home_resume_guard_active
        ) or (
            manual_home_state == MANUAL_HOME_INACTIVE
            and manual_home_active
            and not manual_home_resume_guard_active
        ):
            manual_home_active = False
            manual_home_resume_guard_steps_remaining = max(0, int(args.manual_home_resume_guard_steps))
            manual_home_resume_guard_active = manual_home_resume_guard_steps_remaining > 0
            if manual_home_resume_guard_steps_remaining > 0:
                print(
                    "[Manual Home] Resume requested. Holding zero actions for "
                    f"{manual_home_resume_guard_steps_remaining} control steps to flush stale commands."
                )
                set_manual_home_state(manual_home_command, MANUAL_HOME_RESUME_GUARD)
            else:
                set_manual_home_state(manual_home_command, MANUAL_HOME_INACTIVE)

        for cam in args.camera_names:
            shm, shape, dtype = shm_dict[cam]
            np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            np_array[:] = obs["images"][cam]
        for state_key in shapes["states"]:
            shm, shape, dtype = shm_dict[state_key]
            np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            np_array[:] = obs[state_key]

        shm, shape, dtype = shm_dict["action"]
        action = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
        gripper_idx = [6, 13]

        if args.use_base and args.fixed_body_height >= 0:
            fixed_h = float(args.fixed_body_height)
            action_base = np.zeros((10,), dtype=np.float32)
            action_base[3] = fixed_h
            ros_operator.set_robot_base_target(action_base)

        if manual_home_resume_guard_active:
            zero_action = np.zeros(shape, dtype=dtype)
            robot_action(zero_action, shm_dict)
            left_action = np.zeros((7,), dtype=np.float32)
            right_action = np.zeros((7,), dtype=np.float32)
            ros_operator.follow_arm_publish(left_action, right_action)
            manual_home_resume_guard_steps_remaining -= 1
            if manual_home_resume_guard_steps_remaining <= 0:
                manual_home_resume_guard_active = False
                set_manual_home_state(manual_home_command, MANUAL_HOME_INACTIVE)
                print("[Manual Home] Resume guard finished. Fresh inference can start now.")
            rate.sleep()
            continue

        if np.any(action):
            left_action = action[: gripper_idx[0] + 1].copy()
            right_action = action[gripper_idx[0] + 1 : gripper_idx[1] + 1].copy()

            if args.gripper_gate != -1:
                left_action[gripper_idx[0]] = 0 if left_action[gripper_idx[0]] < args.gripper_gate else 5
                right_action[gripper_idx[0]] = 0 if right_action[gripper_idx[0]] < args.gripper_gate else 5

            ros_operator.follow_arm_publish(left_action, right_action)

            if args.use_base and args.fixed_body_height < 0 and action.shape[0] > gripper_idx[1] + 10:
                action_base = action[gripper_idx[1] + 1 : gripper_idx[1] + 1 + 10].copy()
                ros_operator.set_robot_base_target(action_base)

        rate.sleep()

    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass

    spin_thread.join(timeout=0.2)
    for shm, _, _ in shm_dict.values():
        shm.close()
        shm.unlink()
