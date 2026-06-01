# -- coding: UTF-8
import argparse
import threading
import time
from collections import deque
import sys
import termios
import tty
import select
import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from openpi_client import image_tools, websocket_client_policy
from piper_msgs.msg import PosCmd
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
import os
import dm_env
import collections
from deploy.data_collection.collect_data import save_data

CAMERA_NAMES = ["cam_high", "cam_right_wrist", "cam_left_wrist"]

observation_window = None
stream_buffer = None

# inference data collection control
inference_collection_active = False
inference_collection_lock = threading.Lock()

# save-data control
save_data_requested = False
save_data_lock = threading.Lock()

lang_embeddings = "Pick and sort bricks on the conveyor."
RIGHT_OFFSET = 0.003


class InferenceDataCollector:
    """Inference-mode data collector that saves episodes classified as success or failure."""

    def __init__(self, camera_names, dataset_dir="./data", task_name=None):
        self.camera_names = camera_names
        self.dataset_dir = dataset_dir
        self.task_name = task_name
        self.is_collecting = False
        self.timesteps = []
        self.actions = []
        self.frame_count = 0
        self.episode_idx_success = 0
        self.episode_idx_fail = 0
        self.full_dataset_dir = None

    def _find_next_episode_idx(self, subdir_name):
        """Find next available episode index in the given subdirectory."""
        if self.full_dataset_dir is None:
            return 0
        subdir_path = os.path.join(self.full_dataset_dir, subdir_name)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)
            print(f"Created directory: {subdir_path}")
            return 0
        existing = [
            f for f in os.listdir(subdir_path)
            if f.startswith('episode_') and f.endswith('.hdf5')
        ]
        if existing:
            indices = [int(f.split('_')[1].split('.')[0]) for f in existing]
            return max(indices) + 1
        return 0

    def update_episode_indices(self):
        """Update episode indices for success and fail subdirectories."""
        self.episode_idx_success = self._find_next_episode_idx('aloha_mobile_success')
        self.episode_idx_fail = self._find_next_episode_idx('aloha_mobile_fail')
        print(f"Next success episode: {self.episode_idx_success}")
        print(f"Next fail episode:    {self.episode_idx_fail}")

    def start_collection(self):
        """Start collecting frames and actions."""
        self.is_collecting = True
        self.timesteps = []
        self.actions = []
        self.frame_count = 0
        print(f"\n{'='*70}")
        print("Data collection started!")
        print(f"  Save dir:    {self.full_dataset_dir}")
        print(f"  Success idx: {self.episode_idx_success}  |  Fail idx: {self.episode_idx_fail}")
        print(f"  Press 's' to stop and save")
        print(f"{'='*70}\n")

    def add_frame(self, observation, action):
        """Add one frame of observation and action data."""
        if not self.is_collecting:
            return
        self.frame_count += 1
        step_type = dm_env.StepType.FIRST if self.frame_count == 1 else dm_env.StepType.MID
        self.timesteps.append(dm_env.TimeStep(
            step_type=step_type, reward=None, discount=None, observation=observation
        ))
        self.actions.append(action)
        if self.frame_count % 50 == 0:
            print(f"Collected {self.frame_count} frames")

    def save_current_episode(self, export_video=True, video_fps=30,
                             video_codec='libx264', video_quality=23):
        """Stop collection and save episode after asking user for success/fail classification."""
        if len(self.actions) == 0:
            print("\033[31m❌ No data collected, cannot save\033[0m")
            return False

        print(f"\n{'='*70}")
        print("❓ Was this episode a success or failure?")
        print("   Press '1' → Success (save to aloha_mobile_success)")
        print("   Press '0' → Failure (save to aloha_mobile_fail)")
        print(f"{'='*70}")

        local_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            is_success = None
            subdir_name = None
            episode_idx = None
            while True:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    if char == '1':
                        is_success = True
                        subdir_name = 'aloha_mobile_success'
                        episode_idx = self.episode_idx_success
                        print("\n✅ Marked as [Success]")
                        break
                    elif char == '0':
                        is_success = False
                        subdir_name = 'aloha_mobile_fail'
                        episode_idx = self.episode_idx_fail
                        print("\n❌ Marked as [Failure]")
                        break
                    else:
                        print(f"⚠️ Invalid input '{char}', press '1' or '0'")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, local_settings)

        if is_success is None:
            print("\n❌ No valid input, cancelling save")
            return False

        save_dir = os.path.join(self.full_dataset_dir, subdir_name)
        os.makedirs(save_dir, exist_ok=True)
        dataset_path = os.path.join(save_dir, f"episode_{episode_idx}")
        print(f"\n💾 Saving Episode {episode_idx} to {subdir_name}...")
        print(f"   Frames: {len(self.actions)}")

        class _Args:
            def __init__(self, camera_names, export_video, video_fps, video_codec, video_quality):
                self.camera_names = camera_names
                self.export_video = export_video
                self.video_fps = video_fps
                self.video_codec = video_codec
                self.video_quality = video_quality
                self.use_robot_base = False

        _args = _Args(self.camera_names, export_video, video_fps, video_codec, video_quality)
        try:
            save_data(_args, self.timesteps, self.actions, dataset_path)
            print(f"\n\033[32m✅ Episode {episode_idx} saved!\033[0m")
            print(f"   Path: {dataset_path}.hdf5")
            if is_success:
                self.episode_idx_success += 1
            else:
                self.episode_idx_fail += 1
            self.timesteps = []
            self.actions = []
            self.frame_count = 0
            self.is_collecting = False
            print(f"\n{'='*70}")
            print("✅ Save complete")
            print(f"   Next success episode: {self.episode_idx_success}")
            print(f"   Next fail episode:    {self.episode_idx_fail}")
            print(f"{'='*70}\n")
            return True
        except Exception as e:
            print(f"\033[31m❌ Save failed: {e}\033[0m")
            import traceback
            traceback.print_exc()
            return False

    def get_frame_count(self):
        return self.frame_count

    def has_data(self):
        return len(self.actions) > 0


def keyboard_monitor_thread():
    """Background thread: press 's' to trigger data save and exit."""
    global save_data_requested
    original_settings = None
    try:
        original_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        while not rospy.is_shutdown():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                char = sys.stdin.read(1)
                if char.lower() == 's':
                    with save_data_lock:
                        save_data_requested = True
                    print("\n" + "💾" * 35)
                    print("💾 Save requested, stopping collection...")
                    print("💾" * 35 + "\n")
    finally:
        if original_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_settings)
            except Exception:
                pass


class StreamActionBuffer:
    """
    Maintains a chunk queue for actions and the current smoothed execution sequence.
    integrate_new_chunk supports latency trimming and linear overlap blending.
    """
    def __init__(self, max_chunks=10, decay_alpha=0.25, state_dim=14, smooth_method="temporal"):
        self.chunks = deque()
        self.max_chunks = max_chunks
        self.lock = threading.Lock()
        self.decay_alpha = float(decay_alpha)
        self.state_dim = state_dim
        self.smooth_method = smooth_method
        self.cur_chunk = deque()
        self.k = 0
        self.last_action = None

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int, min_m: int = 8):
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                return
            new_chunk = [a.copy() for a in actions_chunk[drop_n:]]
            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [np.asarray(self.last_action, dtype=float).copy() for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if len(old_list) > 0 and len(old_list) < min_m:
                    tail = np.asarray(old_list[-1], dtype=float).copy()
                    old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])
                elif len(old_list) == 0:
                    self.cur_chunk = deque(new_chunk, maxlen=None)
                    self.k = 0
                    return
            new_list = list(new_chunk)
            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                return
            if len(old_list) > len(new_list):
                old_list = old_list[:len(new_list)]
                overlap_len = len(new_list)
            w_old = np.array([1.0], dtype=float) if overlap_len == 1 else np.linspace(1.0, 0.0, overlap_len, dtype=float)
            w_new = 1.0 - w_old
            smoothed = [
                (w_old[i] * np.asarray(old_list[i], dtype=float) +
                 w_new[i] * np.asarray(new_list[i], dtype=float))
                for i in range(overlap_len)
            ]
            self.cur_chunk = deque([a.copy() for a in smoothed + new_list[overlap_len:]], maxlen=None)
            self.k = 0

    def pop_next_action(self) -> np.ndarray | None:
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return act


def start_inference_thread(args, config, policy, ros_operator):
    th = threading.Thread(target=inference_fn_non_blocking_fast, args=(args, config, policy, ros_operator))
    th.daemon = True
    th.start()


def inference_fn_non_blocking_fast(args, config, policy, ros_operator):
    global stream_buffer
    rate = rospy.Rate(getattr(args, "inference_rate", 4))
    while not rospy.is_shutdown():
        try:
            update_observation_window(args, config, ros_operator)
            latest_obs = observation_window[-1]
            imgs = [
                latest_obs["images"][config["camera_names"][0]],
                latest_obs["images"][config["camera_names"][1]],
                latest_obs["images"][config["camera_names"][2]],
            ]
            imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
            imgs = image_tools.resize_with_pad(np.array(imgs), 224, 224)
            proprio = latest_obs["qpos"]
            payload = {
                "state": proprio,
                "images": {
                    "top_head":   imgs[0].transpose(2, 0, 1),
                    "hand_right": imgs[1].transpose(2, 0, 1),
                    "hand_left":  imgs[2].transpose(2, 0, 1),
                },
                "prompt": lang_embeddings,
            }
            actions = policy.infer(payload)["actions"]
            if actions is not None and len(actions) > 0:
                max_k = int(getattr(args, "latency_k", 0))
                min_m = int(getattr(args, "min_smooth_steps", 8))
                stream_buffer.integrate_new_chunk(actions, max_k=max_k, min_m=min_m)
            try:
                rate.sleep()
            except Exception:
                pass
        except Exception:
            try:
                rate.sleep()
            except Exception:
                time.sleep(0.005)
            continue


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class SimpleKalmanFilter:
    def __init__(self, process_variance=1e-6, measurement_variance=1e-7, initial_value=None):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.error_estimate = 1.0

    def update(self, measurement):
        if self.estimate is None:
            self.estimate = measurement.copy()
            return self.estimate
        # Compute Kalman gain
        kalman_gain = self.error_estimate / (self.error_estimate + self.measurement_variance)
        # Update estimate
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        # Update error estimate
        self.error_estimate = (1 - kalman_gain) * self.error_estimate + abs(self.estimate - measurement) * self.process_variance
        return self.estimate


def interpolate_action(args, prev_action, cur_action):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def minimum_jerk_interpolation(args, prev_action, cur_action):
    num_steps = args.jerk_num_steps
    t_normalized = np.linspace(0, 1, num_steps + 1)[1:]
    trajectory = []
    for tau in t_normalized:
        factor = 10 * (tau ** 3) - 15 * (tau ** 4) + 6 * (tau ** 5)
        trajectory.append(prev_action + factor * (cur_action - prev_action))
    return np.array(trajectory)


def get_config(args):
    return {
        "episode_len": args.max_publish_step,
        "state_dim": 14,
        "chunk_size": args.chunk_size,
        "camera_names": CAMERA_NAMES,
    }


def get_ros_observation(args, ros_operator):
    rate = rospy.Rate(args.publish_rate)
    print_flag = True
    # max wait time; fall back to cached frame to avoid blocking upstream
    max_wait_s = 0.6
    start_t = time.time()

    while True and not rospy.is_shutdown():
        # non-destructive peek to avoid competing with other consumers
        result = ros_operator.get_frame_peek()
        if not result:
            # timeout exceeded; use cached frame if available
            if (time.time() - start_t) > max_wait_s and ros_operator.last_frame_cache is not None:
                if print_flag:
                    print("syn timeout, using last cached frame in get_ros_observation")
                    print_flag = False
                (img_front, img_left, img_right, _, _, _, puppet_arm_left, puppet_arm_right, _) = \
                    ros_operator.last_frame_cache
                return (img_front, img_left, img_right, puppet_arm_left, puppet_arm_right)
            if print_flag:
                print("syn fail when get_ros_observation")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_front, img_left, img_right, _, _, _, puppet_arm_left, puppet_arm_right, _) = result
        return (img_front, img_left, img_right, puppet_arm_left, puppet_arm_right)


def update_observation_window(args, config, ros_operator):
    # JPEG transformation — align with training
    def jpeg_mapping(img):
        img = cv2.imencode(".jpg", img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img

    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)
        # Append the first dummy image
        observation_window.append({
            "qpos": None,
            "images": {
                config["camera_names"][0]: None,
                config["camera_names"][1]: None,
                config["camera_names"][2]: None,
            },
        })

    img_front, img_left, img_right, puppet_arm_left, puppet_arm_right = \
        get_ros_observation(args, ros_operator)
    img_front = jpeg_mapping(img_front)
    img_left  = jpeg_mapping(img_left)
    img_right = jpeg_mapping(img_right)

    qpos = np.concatenate(
        (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0
    )
    observation_window.append({
        "qpos": qpos,
        "images": {
            config["camera_names"][0]: img_front,
            config["camera_names"][1]: img_right,
            config["camera_names"][2]: img_left,
        },
    })


def inference_fn(args, config, policy):
    global observation_window
    global lang_embeddings

    while True and not rospy.is_shutdown():
        time1 = time.time()

        # fetch images in sequence [front, right, left]
        image_arrs = [
            observation_window[-1]["images"][config["camera_names"][0]],
            observation_window[-1]["images"][config["camera_names"][1]],
            observation_window[-1]["images"][config["camera_names"][2]],
        ]
        # convert BGR to RGB
        image_arrs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_arrs]
        image_arrs = image_tools.resize_with_pad(np.array(image_arrs), 224, 224)

        # get last qpos in shape [14,]
        proprio = observation_window[-1]["qpos"]

        payload = {
            "state": proprio,
            "images": {
                "top_head":   image_arrs[0].transpose(2, 0, 1),
                "hand_right": image_arrs[1].transpose(2, 0, 1),
                "hand_left":  image_arrs[2].transpose(2, 0, 1),
            },
            "prompt": lang_embeddings,
        }

        # actions shaped as [chunk, 14] in format [left, right]
        actions = policy.infer(payload)["actions"]
        print(f"Model inference time: {(time.time() - time1)*1000:.3f} ms")
        return actions


def model_inference(args, config, ros_operator):
    global lang_embeddings
    global save_data_requested
    global inference_collection_active
    global stream_buffer

    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    kalman_filters = [SimpleKalmanFilter() for _ in range(config["state_dim"])]
    print(f"Server metadata: {policy.get_server_metadata()}")

    max_publish_step = config["episode_len"]
    chunk_size = config["chunk_size"]

    # Initialize puppet arm position
    left0  = [0, 0.32, -0.36, 0, 0.24, 0, 0.07]
    right0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.07]
    ros_operator.puppet_arm_publish_continuous(left0, right0)

    # Initialize inference data collector
    data_collector = InferenceDataCollector(
        camera_names=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        dataset_dir=args.dataset_dir,
        task_name="inference_collect",
    )
    data_collector.full_dataset_dir = os.path.join(args.dataset_dir, "aloha_mobile")
    os.makedirs(data_collector.full_dataset_dir, exist_ok=True)
    data_collector.update_episode_indices()

    # Start keyboard monitor thread
    keyboard_thread = threading.Thread(target=keyboard_monitor_thread, daemon=True)
    keyboard_thread.start()
    print("\n" + "="*70)
    print("✅ Keyboard monitor started")
    print("   Press 's' → Stop and save episode")
    print("="*70 + "\n")
    time.sleep(0.3)

    print("\n" + "="*70)
    print("🤖 Robot arms reset to initial position")
    print("="*70)
    input("Press [Enter] to start inference and data collection...")
    print("\n" + "="*70)
    print("🚀 Starting inference and data collection...")
    print("   Press 's' to stop and save")
    print("="*70 + "\n")

    # Start collection
    data_collector.start_collection()
    with inference_collection_lock:
        inference_collection_active = True

    pre_action = np.zeros(config["state_dim"])
    action = None

    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            t = 0
            rate = rospy.Rate(args.publish_rate)
            action_buffer = np.zeros([chunk_size, config["state_dim"]])
            last_stream_act = None

            while t < max_publish_step and not rospy.is_shutdown():
                # check if save was requested
                with save_data_lock:
                    if save_data_requested:
                        with inference_collection_lock:
                            if inference_collection_active and data_collector.has_data():
                                print("\n" + "💾" * 35)
                                print("💾 Stopping and saving data...")
                                print("💾" * 35 + "\n")
                                ros_operator.open_grippers()
                                time.sleep(1.0)
                                print("✅ Grippers opened\n")
                                data_collector.save_current_episode(
                                    export_video=getattr(args, 'export_video', True),
                                    video_fps=getattr(args, 'video_fps', 30),
                                    video_codec=getattr(args, 'video_codec', 'libx264'),
                                    video_quality=getattr(args, 'video_quality', 23),
                                )
                                print("\n" + "✅" * 35)
                                print("✅ Data saved, exiting...")
                                print("✅" * 35 + "\n")
                                rospy.signal_shutdown("Data saved, exiting...")
                                return
                            else:
                                print("\n⚠️ No data to save\n")
                        save_data_requested = False

                # ── temporal smoothing execution path ──────────────────────
                if args.use_temporal_smoothing:
                    if stream_buffer is None:
                        stream_buffer = StreamActionBuffer(
                            max_chunks=args.buffer_max_chunks,
                            decay_alpha=args.exp_decay_alpha,
                            state_dim=config["state_dim"],
                            smooth_method="temporal",
                        )
                        start_inference_thread(args, config, policy, ros_operator)

                    act = stream_buffer.pop_next_action()
                    if act is not None:
                        if args.ctrl_type == "joint":
                            left_action  = act[:7].copy()
                            right_action = act[7:14].copy()
                            left_action[6]  = max(0.0, left_action[6]  - RIGHT_OFFSET)
                            right_action[6] = max(0.0, right_action[6] - RIGHT_OFFSET)
                            ros_operator.puppet_arm_publish(left_action, right_action)

                            # collect inference data
                            with inference_collection_lock:
                                is_collecting = inference_collection_active and data_collector.is_collecting
                            if is_collecting:
                                try:
                                    result = ros_operator.get_frame_peek()
                                    if result:
                                        (img_front, img_left, img_right, _, _, _,
                                         puppet_arm_left, puppet_arm_right, _) = result
                                        observation = collections.OrderedDict()
                                        observation['images'] = {
                                            'cam_high':        img_front.copy(),
                                            'cam_left_wrist':  img_left.copy(),
                                            'cam_right_wrist': img_right.copy(),
                                        }
                                        observation['qpos'] = np.concatenate(
                                            (puppet_arm_left.position, puppet_arm_right.position)
                                        ).copy()
                                        observation['qvel']     = np.zeros(14)
                                        observation['effort']   = np.zeros(14)
                                        observation['base_vel'] = [0.0, 0.0]
                                        action_to_save = np.concatenate([left_action, right_action]).copy()
                                        data_collector.add_frame(observation, action_to_save)
                                except Exception as e:
                                    rospy.logwarn(f"[inference collect] frame error: {e}")

                        elif args.ctrl_type == "eef":
                            ros_operator.endpose_publish(act[:7], act[7:14])

                        last_stream_act = act.copy()
                    else:
                        if last_stream_act is not None:
                            if args.ctrl_type == "joint":
                                ros_operator.puppet_arm_publish(last_stream_act[:7], last_stream_act[7:14])
                            elif args.ctrl_type == "eef":
                                ros_operator.endpose_publish(last_stream_act[:7], last_stream_act[7:14])

                    rate.sleep()
                    t += 1
                    if t % 50 == 0:
                        print(f"Step {t} | Recording: {data_collector.get_frame_count()} frames | Press 's' to save")
                    continue

                # ── non-temporal-smoothing path (blocking chunk mode) ──────
                update_observation_window(args, config, ros_operator)

                if t % chunk_size == 0:
                    action_buffer = inference_fn(args, config, policy).copy()

                raw_action = action_buffer[t % chunk_size]

                if args.use_kalman_filter:
                    action = np.array([kf.update(raw_action[i]) for i, kf in enumerate(kalman_filters)])
                else:
                    action = raw_action

                if args.use_actions_interpolation:
                    if args.interpolate_method == "linear":
                        interp_actions = interpolate_action(args, pre_action, action)
                    elif args.interpolate_method == "minimum_jerk":
                        interp_actions = minimum_jerk_interpolation(args, pre_action, action)
                    else:
                        raise NotImplementedError
                else:
                    interp_actions = action[np.newaxis, :]

                for act in interp_actions:
                    if args.ctrl_type == "joint":
                        left_action  = act[:7]
                        right_action = act[7:14]
                        if args.gripper_threshold:
                            if left_action[-1]  < 0.03: left_action[-1]  = 0
                            if right_action[-1] < 0.03: right_action[-1] = 0
                        ros_operator.puppet_arm_publish(left_action, right_action)
                    elif args.ctrl_type == "eef":
                        ros_operator.endpose_publish(act[:7], act[7:14])
                    if args.use_robot_base:
                        ros_operator.robot_base_publish(act[14:16])
                    rate.sleep()

                t += 1
                if t % 50 == 0:
                    print(f"Published Step {t}")
                pre_action = action.copy()


class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.endpose_left_publisher = None
        self.endpose_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.last_frame_cache = None
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque   = deque()
        self.img_right_deque  = deque()
        self.img_front_deque  = deque()
        self.img_left_depth_deque   = deque()
        self.img_right_depth_deque  = deque()
        self.img_front_depth_deque  = deque()
        self.puppet_arm_left_deque  = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def open_grippers(self):
        left_arm  = list(self.puppet_arm_left_deque[-1].position)
        right_arm = list(self.puppet_arm_right_deque[-1].position)
        left_arm[6]  = 1.0
        right_arm[6] = 1.0
        self.puppet_arm_publish(left_arm, right_arm)
        print("Grippers opened.")

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def endpose_publish(self, left, right):
        endpose_msg = PosCmd()
        endpose_msg.x, endpose_msg.y, endpose_msg.z = left[:3]
        endpose_msg.roll, endpose_msg.pitch, endpose_msg.yaw = left[3:6]
        endpose_msg.gripper = left[6]
        self.endpose_left_publisher.publish(endpose_msg)
        endpose_msg.x, endpose_msg.y, endpose_msg.z = right[:3]
        endpose_msg.roll, endpose_msg.pitch, endpose_msg.yaw = right[3:6]
        endpose_msg.gripper = right[6]
        self.endpose_right_publisher.publish(endpose_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x  = vel[0]
        vel_msg.linear.y  = 0
        vel_msg.linear.z  = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque)  != 0: left_arm  = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0: right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol  = [1 if left[i]  - left_arm[i]  > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff  = [abs(left[i]  - left_arm[i])  for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)
        left_arm = right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque)  != 0: left_arm  = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0: right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        traj_left_list  = np.linspace(left_arm,  left,  num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)
        for i in range(len(traj_left_list)):
            traj_left  = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1]  = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(
            target=self.puppet_arm_publish_continuous, args=(left, right)
        )
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        if (
            len(self.img_left_deque)  == 0
            or len(self.img_right_deque) == 0
            or len(self.img_front_deque) == 0
            or (self.args.use_depth_image and (
                len(self.img_left_depth_deque)  == 0
                or len(self.img_right_depth_deque) == 0
                or len(self.img_front_depth_deque) == 0
            ))
        ):
            return False

        if self.args.use_depth_image:
            frame_time = min([
                self.img_left_deque[-1].header.stamp.to_sec(),
                self.img_right_deque[-1].header.stamp.to_sec(),
                self.img_front_deque[-1].header.stamp.to_sec(),
                self.img_left_depth_deque[-1].header.stamp.to_sec(),
                self.img_right_depth_deque[-1].header.stamp.to_sec(),
                self.img_front_depth_deque[-1].header.stamp.to_sec(),
            ])
        else:
            frame_time = min([
                self.img_left_deque[-1].header.stamp.to_sec(),
                self.img_right_deque[-1].header.stamp.to_sec(),
                self.img_front_deque[-1].header.stamp.to_sec(),
            ])

        if len(self.img_left_deque)  == 0 or self.img_left_deque[-1].header.stamp.to_sec()  < frame_time: return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time: return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time: return False
        if len(self.puppet_arm_left_deque)  == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec()  < frame_time: return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time: return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque)  == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec()  < frame_time): return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time): return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time): return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time): return False

        while self.img_left_deque[0].header.stamp.to_sec()  < frame_time: self.img_left_deque.popleft()
        img_left  = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(),  "passthrough")
        while self.img_right_deque[0].header.stamp.to_sec() < frame_time: self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), "passthrough")
        while self.img_front_deque[0].header.stamp.to_sec() < frame_time: self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), "passthrough")
        while self.puppet_arm_left_deque[0].header.stamp.to_sec()  < frame_time: self.puppet_arm_left_deque.popleft()
        puppet_arm_left  = self.puppet_arm_left_deque.popleft()
        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time: self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = img_right_depth = img_front_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec()  < frame_time: self.img_left_depth_deque.popleft()
            img_left_depth  = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(),  "passthrough")
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time: self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), "passthrough")
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time: self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), "passthrough")

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time: self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        result_tuple = (img_front, img_left, img_right,
                        img_front_depth, img_left_depth, img_right_depth,
                        puppet_arm_left, puppet_arm_right, robot_base)
        self.last_frame_cache = result_tuple
        return result_tuple

    def get_frame_peek(self):
        """Thread-safe version that snapshots all deques to avoid concurrent modification."""
        if (
            len(self.img_left_deque)  == 0
            or len(self.img_right_deque) == 0
            or len(self.img_front_deque) == 0
            or (self.args.use_depth_image and (
                len(self.img_left_depth_deque)  == 0
                or len(self.img_right_depth_deque) == 0
                or len(self.img_front_depth_deque) == 0
            ))
        ):
            return False

        # snapshot all deques to avoid mutation during iteration
        try:
            img_left_snapshot   = list(self.img_left_deque)
            img_right_snapshot  = list(self.img_right_deque)
            img_front_snapshot  = list(self.img_front_deque)
            puppet_left_snapshot  = list(self.puppet_arm_left_deque)
            puppet_right_snapshot = list(self.puppet_arm_right_deque)
            if self.args.use_depth_image:
                img_left_depth_snapshot  = list(self.img_left_depth_deque)
                img_right_depth_snapshot = list(self.img_right_depth_deque)
                img_front_depth_snapshot = list(self.img_front_depth_deque)
            if self.args.use_robot_base:
                robot_base_snapshot = list(self.robot_base_deque)
        except Exception:
            # deque modified during snapshot, abort
            return False

        # compute earliest frame timestamp
        if self.args.use_depth_image:
            frame_time = min([
                img_left_snapshot[-1].header.stamp.to_sec(),
                img_right_snapshot[-1].header.stamp.to_sec(),
                img_front_snapshot[-1].header.stamp.to_sec(),
                img_left_depth_snapshot[-1].header.stamp.to_sec(),
                img_right_depth_snapshot[-1].header.stamp.to_sec(),
                img_front_depth_snapshot[-1].header.stamp.to_sec(),
            ])
        else:
            frame_time = min([
                img_left_snapshot[-1].header.stamp.to_sec(),
                img_right_snapshot[-1].header.stamp.to_sec(),
                img_front_snapshot[-1].header.stamp.to_sec(),
            ])

        # verify all snapshots have reached this timestamp
        for snapshot in [img_left_snapshot, img_right_snapshot, img_front_snapshot,
                         puppet_left_snapshot, puppet_right_snapshot]:
            if len(snapshot) == 0 or snapshot[-1].header.stamp.to_sec() < frame_time:
                return False
        if self.args.use_depth_image:
            for snapshot in [img_left_depth_snapshot, img_right_depth_snapshot, img_front_depth_snapshot]:
                if len(snapshot) == 0 or snapshot[-1].header.stamp.to_sec() < frame_time:
                    return False
        if self.args.use_robot_base:
            if len(robot_base_snapshot) == 0 or robot_base_snapshot[-1].header.stamp.to_sec() < frame_time:
                return False

        def first_ge(snapshot_list, t):
            """Find first item in snapshot_list with timestamp >= t."""
            for item in snapshot_list:
                if item.header.stamp.to_sec() >= t:
                    return item
            return snapshot_list[-1] if snapshot_list else None

        # find timestamp-matching messages
        img_left_msg   = first_ge(img_left_snapshot,  frame_time)
        img_right_msg  = first_ge(img_right_snapshot, frame_time)
        img_front_msg  = first_ge(img_front_snapshot, frame_time)
        puppet_arm_left  = first_ge(puppet_left_snapshot,  frame_time)
        puppet_arm_right = first_ge(puppet_right_snapshot, frame_time)

        # decode image messages
        img_left  = self.bridge.imgmsg_to_cv2(img_left_msg,  "passthrough")
        img_right = self.bridge.imgmsg_to_cv2(img_right_msg, "passthrough")
        img_front = self.bridge.imgmsg_to_cv2(img_front_msg, "passthrough")

        # depth images (if enabled)
        img_left_depth = img_right_depth = img_front_depth = None
        if self.args.use_depth_image:
            img_left_depth  = self.bridge.imgmsg_to_cv2(first_ge(img_left_depth_snapshot,  frame_time), "passthrough")
            img_right_depth = self.bridge.imgmsg_to_cv2(first_ge(img_right_depth_snapshot, frame_time), "passthrough")
            img_front_depth = self.bridge.imgmsg_to_cv2(first_ge(img_front_depth_snapshot, frame_time), "passthrough")

        # robot base data (if enabled)
        robot_base = None
        if self.args.use_robot_base:
            robot_base = first_ge(robot_base_snapshot, frame_time)

        return (img_front, img_left, img_right,
                img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000: self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000: self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000: self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000: self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000: self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000: self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000: self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000: self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000: self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def init_ros(self):
        rospy.init_node("joint_state_publisher", anonymous=True)
        rospy.Subscriber(self.args.img_left_topic,   Image, self.img_left_callback,   queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic,  Image, self.img_right_callback,  queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic,  Image, self.img_front_callback,  queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic,  Image, self.img_left_depth_callback,  queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic,  JointState, self.puppet_arm_left_callback,  queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher  = rospy.Publisher(self.args.puppet_arm_left_cmd_topic,  JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.endpose_left_publisher     = rospy.Publisher(self.args.endpose_left_cmd_topic,  PosCmd, queue_size=10)
        self.endpose_right_publisher    = rospy.Publisher(self.args.endpose_right_cmd_topic, PosCmd, queue_size=10)
        self.robot_base_publisher       = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_publish_step",        type=int,   default=10000)
    parser.add_argument("--seed",                    type=int,   default=None)
    parser.add_argument("--img_front_topic",         type=str,   default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic",          type=str,   default="/camera_l/color/image_raw")
    parser.add_argument("--img_right_topic",         type=str,   default="/camera_r/color/image_raw")
    parser.add_argument("--img_front_depth_topic",   type=str,   default="/camera_f/depth/image_raw")
    parser.add_argument("--img_left_depth_topic",    type=str,   default="/camera_l/depth/image_raw")
    parser.add_argument("--img_right_depth_topic",   type=str,   default="/camera_r/depth/image_raw")
    parser.add_argument("--puppet_arm_left_cmd_topic",  type=str, default="/master/joint_left")
    parser.add_argument("--puppet_arm_right_cmd_topic", type=str, default="/master/joint_right")
    parser.add_argument("--puppet_arm_left_topic",   type=str,   default="/puppet/joint_left")
    parser.add_argument("--puppet_arm_right_topic",  type=str,   default="/puppet/joint_right")
    parser.add_argument("--endpose_left_cmd_topic",  type=str,   default="/pos_cmd_left")
    parser.add_argument("--endpose_right_cmd_topic", type=str,   default="/pos_cmd_right")
    parser.add_argument("--robot_base_topic",        type=str,   default="/odom_raw")
    parser.add_argument("--robot_base_cmd_topic",    type=str,   default="/cmd_vel")
    parser.add_argument("--use_robot_base",          action="store_true", default=False)
    parser.add_argument("--publish_rate",            type=int,   default=30)
    # smoothing
    parser.add_argument("--use_temporal_smoothing",  action="store_true", default=False)
    parser.add_argument("--latency_k",               type=int,   default=8)
    parser.add_argument("--inference_rate",          type=float, default=3.0)
    parser.add_argument("--min_smooth_steps",        type=int,   default=8)
    parser.add_argument("--buffer_max_chunks",       type=int,   default=10)
    parser.add_argument("--exp_decay_alpha",         type=float, default=0.25)
    parser.add_argument("--gripper_threshold",       action="store_true", default=True)
    parser.add_argument("--chunk_size",              type=int,   default=50)
    parser.add_argument("--arm_steps_length",        type=float, default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2])
    parser.add_argument("--use_actions_interpolation", action="store_true", default=False)
    parser.add_argument("--use_kalman_filter",       action="store_true", default=True)
    parser.add_argument("--interpolate_method",      type=str,   choices=["linear", "minimum_jerk"], default="linear")
    parser.add_argument("--jerk_num_steps",          type=int,   default=10)
    parser.add_argument("--use_depth_image",         action="store_true", default=False)
    # video export
    parser.add_argument("--export_video",            action="store_true", default=True)
    parser.add_argument("--video_fps",               type=int,   default=30)
    parser.add_argument("--video_codec",             type=str,   choices=["libx264", "libx265", "libsvtav1"], default="libx264")
    parser.add_argument("--video_quality",           type=int,   default=23, help="CRF value; lower = higher quality")
    # server
    parser.add_argument("--host",                    type=str,   default="localhost")
    parser.add_argument("--port",                    type=int,   default=8000)
    parser.add_argument("--ctrl_type",               type=str,   choices=["joint", "eef"], default="joint")
    # dataset
    parser.add_argument("--dataset_dir",             type=str,   default="/home/agilex/data")
    parser.add_argument("--task_name",               type=str,   default="aloha_inference")
    parser.add_argument("--episode_idx",             type=int,   default=0)
    parser.add_argument("--max_timesteps",           type=int,   default=100000000)
    parser.add_argument("--frame_rate",              type=int,   default=30)
    parser.add_argument("--camera_names",            nargs="+",  default=["cam_high", "cam_left_wrist", "cam_right_wrist"])
    return parser.parse_args()


def main():
    args = get_arguments()
    try:
        ros_operator = RosOperator(args)
        if args.seed is not None:
            set_seed(args.seed)
        config = get_config(args)
        model_inference(args, config, ros_operator)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted, exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program exited")


if __name__ == "__main__":
    main()
