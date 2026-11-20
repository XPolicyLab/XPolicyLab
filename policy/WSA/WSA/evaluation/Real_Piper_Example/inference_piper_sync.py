#!/usr/bin/env python3
from __future__ import annotations

import argparse
import select
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header

THIS_DIR = Path(__file__).resolve().parent
REAL_LIFT2_DIR = THIS_DIR.parent / "Real_Lift2_Example"
REPO_ROOT = THIS_DIR.parents[1]

for candidate in [REAL_LIFT2_DIR, REPO_ROOT, REPO_ROOT / "src"]:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from request_builder import build_wsa_base_request, prepare_history_frame
from websocket_client import WebsocketClientPolicy


PIPER_CAMERA_MAP = {
    "front": "cam_high",
    "wrist": "cam_left_wrist",
}


def add_bool_arg(parser: argparse.ArgumentParser, name: str, *, default: bool, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{name: default})


def jpeg_roundtrip_rgb(image_rgb: np.ndarray) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("Failed to JPEG-encode camera image.")
    decoded_bgr = cv2.imdecode(np.frombuffer(encoded.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


def coerce_rgb_image(msg: Image, *, color_mode: str) -> np.ndarray:
    image = ros_numpy.numpify(msg)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3 or image.shape[-1] not in {3, 4}:
        raise ValueError(f"Unsupported ROS image shape: {image.shape}")

    image = np.asarray(image)
    encoding = (msg.encoding or "").lower()
    mode = color_mode.lower()
    if mode == "auto":
        if encoding in {"rgb8", "rgba8"}:
            mode = "rgb"
        elif encoding in {"bgr8", "bgra8"}:
            mode = "bgr"
        else:
            mode = "bgr"

    if image.shape[-1] == 4:
        image = image[..., :3]

    if mode == "rgb":
        rgb = image
    elif mode == "bgr":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image_color_mode={color_mode!r}")
    return np.ascontiguousarray(rgb)


def coerce_state_7d(qpos: Any, *, log_tag: str) -> np.ndarray:
    qpos_np = np.asarray(qpos, dtype=np.float32).reshape(-1)
    state = np.zeros((7,), dtype=np.float32)
    usable = min(7, qpos_np.size)
    state[:usable] = qpos_np[:usable]
    if usable < 7:
        print(f"[{log_tag}] Warning: joint state has {usable} dims, padding to 7D.")
    return state


def postprocess_piper_action(action: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    output = np.zeros((7,), dtype=np.float32)
    usable = min(7, action.size)
    output[:usable] = action[:usable]

    if args.gripper_postprocess:
        raw_gripper = float(output[6])
        if raw_gripper <= args.gripper_close_threshold:
            output[6] = max(raw_gripper - args.gripper_close_offset, args.gripper_min)
        elif raw_gripper >= args.gripper_open_threshold:
            output[6] = min(raw_gripper + args.gripper_open_offset, args.gripper_max)
    return output


def extract_piper_actions(response: dict[str, Any], args: argparse.Namespace, *, log_tag: str) -> list[np.ndarray]:
    if not isinstance(response, dict) or "actions" not in response:
        print(f"[{log_tag}] Error: server response has no `actions` field.")
        return []

    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2:
        print(f"[{log_tag}] Error: expected [T, A] actions, got shape={actions.shape}.")
        return []
    if actions.shape[1] != 7 and not args.allow_action_dim_mismatch:
        raise RuntimeError(
            f"Expected real_piper action dim 7, got {actions.shape[1]}. "
            "Check that the client is connected to a real_piper server."
        )

    return [postprocess_piper_action(step, args) for step in actions]


def validate_server_metadata(metadata: dict[str, Any], args: argparse.Namespace) -> None:
    stats_key = metadata.get("stats_key")
    if stats_key != args.expected_stats_key and not args.allow_stats_key_mismatch:
        raise RuntimeError(
            f"Connected server stats_key={stats_key!r}, expected {args.expected_stats_key!r}. "
            "Refusing to command the Piper with a mismatched model server."
        )

    target_action_dim = metadata.get("target_action_dim")
    if target_action_dim is not None and int(target_action_dim) != 7 and not args.allow_action_dim_mismatch:
        raise RuntimeError(
            f"Connected server target_action_dim={target_action_dim}, expected 7 for real_piper."
        )

    if bool(metadata.get("rtc_enabled", False)):
        raise RuntimeError("Connected server reports rtc_enabled=true, but Piper deployment is sync-only.")


def print_manual_reset_help(args: argparse.Namespace) -> None:
    if not args.manual_reset:
        return
    if args.init_joint_position is None:
        print(f"[{args.log_tag}] Manual Enter reset is disabled because INIT_JOINT_POSITION is not set.")
        return
    if not sys.stdin or not sys.stdin.isatty():
        print(f"[{args.log_tag}] Manual Enter reset is disabled because stdin is not interactive.")
        return
    print(
        f"[{args.log_tag}] Manual reset controls:\n"
        "  - Press Enter during execution: move to INIT_POS and pause the current rollout.\n"
        "  - Press Enter again while paused: start fresh inference from timestep 0."
    )


def poll_manual_reset_command(manual_reset_active: bool) -> str | None:
    if not sys.stdin or not sys.stdin.isatty():
        return None

    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    except Exception:
        return None
    if not ready:
        return None

    try:
        line = sys.stdin.readline()
    except Exception:
        return None
    if line == "":
        return None

    command = line.strip().lower()
    if command == "":
        return "resume" if manual_reset_active else "reset"
    if command in {"r", "reset", "home"}:
        return "resume" if manual_reset_active else "reset"
    return command


def hold_init_position(args: argparse.Namespace, ros_operator: "PiperRosOperator", rate: rospy.Rate, steps: int) -> None:
    if args.init_joint_position is None:
        return
    target = np.asarray(args.init_joint_position, dtype=np.float32)
    for _ in range(max(0, steps)):
        if rospy.is_shutdown():
            break
        ros_operator.publish_joint_command(target)
        rate.sleep()


def maybe_handle_manual_reset(
    args: argparse.Namespace,
    ros_operator: "PiperRosOperator",
    rate: rospy.Rate,
) -> bool:
    if not args.manual_reset or args.init_joint_position is None:
        return False

    command = poll_manual_reset_command(manual_reset_active=False)
    if command != "reset":
        return False

    print(
        "\n"
        + "=" * 72
        + f"\n[{args.log_tag}] Enter detected. Moving Piper to INIT_POS and pausing this rollout.\n"
        f"[{args.log_tag}] Press Enter again after resetting the scene to start fresh inference from timestep 0.\n"
        + "=" * 72
        + "\n"
    )
    ros_operator.move_to_initial_position(args.init_joint_position, timeout=args.init_timeout)

    last_reminder_time = 0.0
    while not rospy.is_shutdown():
        hold_init_position(args, ros_operator, rate, steps=1)
        now = time.monotonic()
        if now - last_reminder_time >= args.manual_reset_reminder_interval:
            print(f"[{args.log_tag}] Paused at INIT_POS. Press Enter again to restart inference.")
            last_reminder_time = now

        resume_command = poll_manual_reset_command(manual_reset_active=True)
        if resume_command == "resume":
            print(f"[{args.log_tag}] Second Enter detected. Resetting rollout state before fresh inference.")
            hold_init_position(args, ros_operator, rate, steps=args.manual_reset_resume_hold_steps)
            return True
        if resume_command is not None:
            print(f"[{args.log_tag}] Still paused at INIT_POS. Press Enter again to restart inference.")

    return True


class PiperRequestBuilder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.image_histories: dict[str, list[np.ndarray]] = defaultdict(list)
        self.max_history = max(args.image_history_interval + 1, 2)
        self.send_reset = True

    def reset(self) -> None:
        self.image_histories.clear()
        self.send_reset = True

    def append_images(self, images: dict[str, np.ndarray]) -> None:
        for name, image in images.items():
            prepared = prepare_history_frame(
                image,
                send_image_height=self.args.send_image_height,
                send_image_width=self.args.send_image_width,
            )
            self.image_histories[name].append(prepared)
            if len(self.image_histories[name]) > self.max_history:
                self.image_histories[name].pop(0)

    def build(self, obs: dict[str, Any], timestep: int) -> dict[str, Any]:
        self.append_images(obs["images"])
        request = build_wsa_base_request(
            qpos=coerce_state_7d(obs["qpos"], log_tag=self.args.log_tag),
            image_histories=self.image_histories,
            prompt=self.args.task_prompt,
            timestep=timestep,
            image_history_interval=self.args.image_history_interval,
            state_dim=7,
            camera_name_map=PIPER_CAMERA_MAP,
            send_image_height=self.args.send_image_height,
            send_image_width=self.args.send_image_width,
        )
        request["reset"] = bool(self.send_reset or request["reset"])
        self.send_reset = False
        return request


class PiperRosOperator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.log_tag = args.log_tag

        self.front_img: np.ndarray | None = None
        self.wrist_img: np.ndarray | None = None
        self.front_img_time: float | None = None
        self.wrist_img_time: float | None = None
        self.joint_state: JointState | None = None
        self.joint_state_time: float | None = None

        rospy.init_node("wsa_base_piper_sync_client", anonymous=True)
        rospy.Subscriber(args.front_cam_topic, Image, self.front_cam_callback, queue_size=10, tcp_nodelay=True)
        rospy.Subscriber(args.wrist_cam_topic, Image, self.wrist_cam_callback, queue_size=10, tcp_nodelay=True)
        rospy.Subscriber(args.joint_state_topic, JointState, self.joint_callback, queue_size=10, tcp_nodelay=True)
        self.joint_pub = rospy.Publisher(args.joint_cmd_topic, JointState, queue_size=10)

        print(f"[{self.log_tag}] ROS initialized")
        print(f"[{self.log_tag}]   front camera: {args.front_cam_topic}")
        print(f"[{self.log_tag}]   wrist camera: {args.wrist_cam_topic}")
        print(f"[{self.log_tag}]   joint state: {args.joint_state_topic}")
        print(f"[{self.log_tag}]   joint command: {args.joint_cmd_topic}")

    @staticmethod
    def _stamp_to_sec(msg: Any) -> float:
        stamp = msg.header.stamp.to_sec() if msg.header is not None else 0.0
        return stamp if stamp > 0.0 else rospy.Time.now().to_sec()

    def front_cam_callback(self, msg: Image) -> None:
        self.front_img = coerce_rgb_image(msg, color_mode=self.args.image_color_mode)
        if self.args.jpeg_roundtrip:
            self.front_img = jpeg_roundtrip_rgb(self.front_img)
        self.front_img_time = self._stamp_to_sec(msg)

    def wrist_cam_callback(self, msg: Image) -> None:
        self.wrist_img = coerce_rgb_image(msg, color_mode=self.args.image_color_mode)
        if self.args.jpeg_roundtrip:
            self.wrist_img = jpeg_roundtrip_rgb(self.wrist_img)
        self.wrist_img_time = self._stamp_to_sec(msg)

    def joint_callback(self, msg: JointState) -> None:
        self.joint_state = msg
        self.joint_state_time = self._stamp_to_sec(msg)

    def get_observation(self) -> dict[str, Any] | None:
        if self.front_img is None or self.wrist_img is None or self.joint_state is None:
            return None
        if self.front_img_time is None or self.wrist_img_time is None or self.joint_state_time is None:
            return None

        times = [self.front_img_time, self.wrist_img_time, self.joint_state_time]
        time_diff = max(times) - min(times)
        if time_diff > self.args.sync_tolerance:
            print(f"[{self.log_tag}] Warning: observation is not synchronized, diff={time_diff:.3f}s")
            return None

        return {
            "images": {
                "front": self.front_img.copy(),
                "wrist": self.wrist_img.copy(),
            },
            "qpos": np.asarray(self.joint_state.position, dtype=np.float32),
        }

    def publish_joint_command(self, action: np.ndarray) -> None:
        action = coerce_state_7d(action, log_tag=self.log_tag)
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.args.joint_names
        msg.position = [float(v) for v in action]
        self.joint_pub.publish(msg)

    def move_to_initial_position(self, target_position: list[float], timeout: float) -> bool:
        target = np.asarray(target_position, dtype=np.float32)
        if target.size != 7:
            raise ValueError(f"init_joint_position must contain 7 values, got {target.size}.")

        print(f"[{self.log_tag}] Moving to initial position: {target.tolist()}")
        rate = rospy.Rate(10)
        wait_start = rospy.Time.now()
        while self.joint_state is None and not rospy.is_shutdown():
            if (rospy.Time.now() - wait_start).to_sec() > 3.0:
                print(f"[{self.log_tag}] No joint state on {self.args.joint_state_topic}; skip init move.")
                return False
            rate.sleep()

        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            self.publish_joint_command(target)
            current = coerce_state_7d(self.joint_state.position, log_tag=self.log_tag)
            max_diff = float(np.max(np.abs(target - current)))
            if max_diff <= self.args.init_position_threshold:
                print(f"[{self.log_tag}] Initial position reached, max diff={max_diff:.3f}.")
                return True
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                print(f"[{self.log_tag}] Init move timeout, max diff={max_diff:.3f}.")
                return False
            rate.sleep()
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync WSABase real_piper ROS1 deployment client.")
    parser.add_argument("--ws_host", default="10.60.43.33")
    parser.add_argument("--ws_port", type=int, default=8000)
    parser.add_argument("--task_prompt", default="Sort desktop objects and place them in designated locations.")
    parser.add_argument("--publish_rate", type=float, default=15.0)
    parser.add_argument("--action_horizon", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--image_history_interval", type=int, default=15)
    parser.add_argument("--send_image_height", type=int, default=None)
    parser.add_argument("--send_image_width", type=int, default=None)
    parser.add_argument("--sync_tolerance", type=float, default=0.1)
    parser.add_argument("--front_cam_topic", default="/ob_camera_02/color/image_raw")
    parser.add_argument("--wrist_cam_topic", default="/ob_camera_01/color/image_raw")
    parser.add_argument("--joint_state_topic", default="joint_states_single")
    parser.add_argument("--joint_cmd_topic", default="js_cmd")
    parser.add_argument("--joint_names", nargs=7, default=[f"joint{i}" for i in range(7)])
    parser.add_argument("--image_color_mode", choices=["auto", "bgr", "rgb"], default="auto")
    parser.add_argument("--first_inference_check", action="store_true")
    add_bool_arg(parser, "jpeg_roundtrip", default=True, help_text="JPEG roundtrip camera frames before sending.")
    add_bool_arg(parser, "start_prompt", default=True, help_text="Wait for Enter before starting inference.")
    add_bool_arg(parser, "gripper_postprocess", default=True, help_text="Apply Piper gripper postprocess heuristic.")
    add_bool_arg(
        parser,
        "manual_reset",
        default=True,
        help_text="Enable Enter-triggered INIT_POS reset and fresh inference restart.",
    )
    parser.add_argument("--expected_stats_key", default="real_piper")
    parser.add_argument("--allow_stats_key_mismatch", action="store_true")
    parser.add_argument("--allow_action_dim_mismatch", action="store_true")
    parser.add_argument("--gripper_close_threshold", type=float, default=62000.0)
    parser.add_argument("--gripper_open_threshold", type=float, default=65000.0)
    parser.add_argument("--gripper_close_offset", type=float, default=10000.0)
    parser.add_argument("--gripper_open_offset", type=float, default=5000.0)
    parser.add_argument("--gripper_min", type=float, default=-100000.0)
    parser.add_argument("--gripper_max", type=float, default=90000.0)
    parser.add_argument("--init_joint_position", type=float, nargs=7, default=None)
    parser.add_argument("--init_wait", action="store_true")
    parser.add_argument("--init_timeout", type=float, default=10.0)
    parser.add_argument("--init_position_threshold", type=float, default=500.0)
    parser.add_argument("--manual_reset_resume_hold_steps", type=int, default=5)
    parser.add_argument("--manual_reset_reminder_interval", type=float, default=1.5)
    parser.add_argument("--log_tag", default="WSABase-Piper")
    return parser.parse_args()


def signal_handler(_sig: int, _frame: Any) -> None:
    print("\n[WSABase-Piper] Caught Ctrl+C, shutting down.")
    rospy.signal_shutdown("User interrupt")
    sys.exit(0)


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGINT, signal_handler)

    print(f"[{args.log_tag}] Starting sync Piper client")
    print(f"[{args.log_tag}] Server: {args.ws_host}:{args.ws_port}")
    print(f"[{args.log_tag}] Prompt: {args.task_prompt}")
    print(f"[{args.log_tag}] State/action: 7D single arm")
    print(f"[{args.log_tag}] Cameras: cam_high + cam_left_wrist")
    print_manual_reset_help(args)

    ros_operator = PiperRosOperator(args)
    ws_client = WebsocketClientPolicy(host=args.ws_host, port=args.ws_port)
    request_builder = PiperRequestBuilder(args)

    try:
        metadata = ws_client.get_server_metadata()
        print(f"[{args.log_tag}] Connected. Server metadata keys: {list(metadata.keys())}")
        validate_server_metadata(metadata, args)

        if args.init_joint_position is not None:
            ros_operator.move_to_initial_position(args.init_joint_position, timeout=args.init_timeout)
            if args.init_wait:
                input(f"[{args.log_tag}] Init move done. Press Enter to continue.")

        if args.start_prompt:
            input(f"[{args.log_tag}] Press Enter to start inference.")

        rate = rospy.Rate(args.publish_rate)
        action_buffer: list[np.ndarray] = []
        timestep = 0
        inference_count = 0
        auto_confirm_next_first_chunk = False

        while not rospy.is_shutdown() and timestep < args.max_steps:
            if maybe_handle_manual_reset(args, ros_operator, rate):
                action_buffer.clear()
                request_builder.reset()
                try:
                    ws_client.reset()
                except Exception:
                    pass
                timestep = 0
                inference_count = 0
                auto_confirm_next_first_chunk = True
                print(f"[{args.log_tag}] Rollout state reset. Next request will start from timestep 0.")
                continue

            obs = ros_operator.get_observation()
            if obs is None:
                rate.sleep()
                continue

            if not action_buffer or timestep % args.action_horizon == 0:
                request = request_builder.build(obs, timestep)
                print(f"[{args.log_tag}] Inference #{inference_count}, t={timestep}")
                try:
                    response = ws_client.infer(request)
                except Exception as exc:
                    print(f"[{args.log_tag}] Inference failed: {exc}")
                    rate.sleep()
                    continue

                action_buffer = extract_piper_actions(response, args, log_tag=args.log_tag)
                if not action_buffer:
                    rate.sleep()
                    continue

                print(f"[{args.log_tag}] Received {len(action_buffer)} actions; first={action_buffer[0]}")
                if args.first_inference_check and inference_count == 0:
                    print(f"[{args.log_tag}] Current qpos: {coerce_state_7d(obs['qpos'], log_tag=args.log_tag)}")
                    if auto_confirm_next_first_chunk:
                        print(
                            f"[{args.log_tag}] Reusing the second manual-reset Enter as first-chunk confirmation."
                        )
                        auto_confirm_next_first_chunk = False
                    else:
                        user_input = input(f"[{args.log_tag}] Type 'y' to execute the first action chunk: ").strip()
                        if user_input.lower() != "y":
                            print(f"[{args.log_tag}] Stopped before execution.")
                            return

                inference_count += 1

            action = action_buffer.pop(0)
            ros_operator.publish_joint_command(action)
            timestep += 1
            rate.sleep()

        print(f"[{args.log_tag}] Finished. steps={timestep}, inferences={inference_count}")
    finally:
        ws_client.close()


if __name__ == "__main__":
    main()
