#!/usr/bin/env python

"""Entry point for the Real Lift2 example robot-side inference process."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
SRC_ROOT = ROOT / "src"

for candidate in [THIS_DIR, ROOT, SRC_ROOT]:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    from .runtime import (
        DEFAULT_MANUAL_HOME_RESUME_GUARD_STEPS,
        ROBOT_RUNTIME_ROOT,
        connect_shm_dict,
        ensure_runtime_available,
        get_model_config,
        make_shm_name_dict,
        request_graceful_ros_shutdown,
        ros_process,
    )
    from .inference import inference_process
except ImportError:
    from runtime import (
        DEFAULT_MANUAL_HOME_RESUME_GUARD_STEPS,
        ROBOT_RUNTIME_ROOT,
        connect_shm_dict,
        ensure_runtime_available,
        get_model_config,
        make_shm_name_dict,
        request_graceful_ros_shutdown,
        ros_process,
    )
    from inference import inference_process


def parse_args(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_publish_step", type=int, default=10000, help="Max publish step.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROBOT_RUNTIME_ROOT / "data" / "config.yaml"),
        help="Robot config YAML.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--camera_names",
        nargs="+",
        type=str,
        choices=["head", "left_wrist", "right_wrist"],
        default=["head", "left_wrist", "right_wrist"],
        help="Camera names to use.",
    )
    parser.add_argument("--use_depth_image", action="store_true", help="Enable depth image subscriptions.")
    parser.add_argument("--use_base", action="store_true", help="Use robot base.")
    parser.add_argument(
        "--fixed_body_height",
        type=float,
        default=-1.0,
        help=">=0 to lock chassis and wheel motion, and set fixed body height.",
    )
    parser.add_argument("--record", choices=["Distance", "Speed"], default="Distance", help="Record mode.")
    parser.add_argument("--frame_rate", type=int, default=60, help="Control frame rate.")
    parser.add_argument("--gripper_gate", type=float, default=-1, help="Optional gripper threshold.")
    parser.add_argument("--prompt", type=str, default="Clear the junk and items off the desktop.")
    parser.add_argument("--ws_url", type=str, default="", help="WSABase websocket URL.")
    parser.add_argument("--image_history_interval", type=int, default=15, help="History interval in frames.")
    parser.add_argument(
        "--send_image_height",
        type=int,
        default=0,
        help="Optional robot-side send height for websocket images. Set together with --send_image_width.",
    )
    parser.add_argument(
        "--send_image_width",
        type=int,
        default=0,
        help="Optional robot-side send width for websocket images. Set together with --send_image_height.",
    )
    parser.add_argument("--state_dim", type=int, default=14, help="State dimension.")
    parser.add_argument("--action_dim", type=int, default=14, help="Action dimension.")
    parser.add_argument(
        "--inference_mode",
        choices=["sync", "async", "rtc"],
        default="sync",
        help="`sync` blocks per chunk, `async` uses background prefetch, and `rtc` uses Real-Time Chunking with an action queue.",
    )
    parser.add_argument(
        "--prefetch_lead_steps",
        type=int,
        default=10,
        help="Minimum number of remaining control steps before we start prefetching the next action chunk.",
    )
    parser.add_argument(
        "--log_timing_every",
        type=int,
        default=5,
        help="Print server/client timing every N chunks. The first 3 chunks are always logged.",
    )
    parser.add_argument(
        "--rtc_execution_horizon",
        type=int,
        default=10,
        help="RTC overlap horizon in steps. This should match the server-side runtime RTC setting.",
    )
    parser.add_argument(
        "--rtc_max_guidance_weight",
        type=float,
        default=10.0,
        help="Runtime RTC guidance strength. Used locally for queue config sanity and should match the server.",
    )
    parser.add_argument(
        "--rtc_queue_threshold",
        type=int,
        default=30,
        help="Request a fresh RTC chunk when the remaining queued actions drop to this many steps or less.",
    )
    parser.add_argument(
        "--rtc_latency_lookback",
        type=int,
        default=10,
        help="How many recent RTC request latencies to track when estimating inference delay.",
    )
    parser.add_argument(
        "--safe_stop_home_arms",
        action="store_true",
        help="Before lowering the base on shutdown, first publish an all-zero home pose to both arms.",
    )
    parser.add_argument(
        "--safe_stop_home_publish_steps",
        type=int,
        default=180,
        help="Number of cycles to publish the arm-home pose before lowering the base.",
    )
    parser.add_argument(
        "--safe_stop_body_height",
        type=float,
        default=None,
        help="Optional base height target to publish for a few cycles before shutdown. Set to 0 to lower before stop.",
    )
    parser.add_argument(
        "--safe_stop_publish_steps",
        type=int,
        default=30,
        help="Number of cycles to publish the safe-stop base target before shutdown.",
    )
    parser.add_argument(
        "--manual_home_publish_steps",
        type=int,
        default=0,
        help="Number of control cycles used for Enter-triggered arm homing. Set to 0 to use the built-in short smooth default.",
    )
    parser.add_argument(
        "--manual_home_resume_guard_steps",
        type=int,
        default=DEFAULT_MANUAL_HOME_RESUME_GUARD_STEPS,
        help="Number of zero-action control cycles to hold after the second Enter before resuming inference.",
    )

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(args):
    has_send_height = args.send_image_height > 0
    has_send_width = args.send_image_width > 0
    if has_send_height != has_send_width:
        raise ValueError(
            "--send_image_height and --send_image_width must either both be set to positive values or both stay unset."
        )
    if args.inference_mode == "rtc":
        if args.rtc_execution_horizon <= 0:
            raise ValueError("--rtc_execution_horizon must be positive.")
        if args.rtc_queue_threshold < 0:
            raise ValueError("--rtc_queue_threshold must be non-negative.")
        if args.rtc_latency_lookback <= 0:
            raise ValueError("--rtc_latency_lookback must be positive.")
        if args.rtc_max_guidance_weight <= 0:
            raise ValueError("--rtc_max_guidance_weight must be positive.")
    if args.manual_home_publish_steps < 0:
        raise ValueError("--manual_home_publish_steps must be non-negative.")
    if args.manual_home_resume_guard_steps < 0:
        raise ValueError("--manual_home_resume_guard_steps must be non-negative.")

    ensure_runtime_available()
    meta_queue = mp.Queue()

    connected_event = mp.Event()
    start_event = mp.Event()
    shm_ready_event = mp.Event()

    config = get_model_config(args)
    manual_home_command = mp.Value("i", 0)

    ros_proc = mp.Process(
        target=ros_process,
        args=(args, config, meta_queue, connected_event, start_event, shm_ready_event, manual_home_command),
    )
    ros_proc.start()

    if not connected_event.wait(timeout=15.0):
        request_graceful_ros_shutdown(ros_proc, args)
        raise RuntimeError("ROS process did not reach the connected state in time.")

    input("Enter any key to continue :")
    start_event.set()

    try:
        shapes = meta_queue.get(timeout=20.0)
    except queue.Empty as exc:
        request_graceful_ros_shutdown(ros_proc, args)
        raise RuntimeError("Timed out waiting for ROS observation metadata.") from exc

    if isinstance(shapes, dict) and "error" in shapes:
        request_graceful_ros_shutdown(ros_proc, args)
        raise RuntimeError(str(shapes["error"]))

    shm_name_dict = make_shm_name_dict(args, shapes)
    meta_queue.put(shm_name_dict)
    if not shm_ready_event.wait(timeout=10.0):
        request_graceful_ros_shutdown(ros_proc, args)
        raise RuntimeError("Timed out waiting for shared-memory setup.")

    shm_dict = connect_shm_dict(shm_name_dict, shapes, shapes["dtypes"], config)

    try:
        inference_process(args, config, shm_dict, shapes, ros_proc, manual_home_command)
    except KeyboardInterrupt:
        print("[Shutdown] KeyboardInterrupt received in parent process, requesting graceful ROS shutdown...")
    finally:
        request_graceful_ros_shutdown(ros_proc, args)
        for shm, _, _ in shm_dict.values():
            try:
                shm.close()
            except FileNotFoundError:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main(parse_args())
