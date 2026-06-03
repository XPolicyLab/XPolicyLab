#!/usr/bin/env python
# -- coding: UTF-8 --
import os, re
import time
import numpy as np
import h5py
import argparse
import dm_env
import collections
from collections import deque
import rospy
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import threading
import av
import logging
from pathlib import Path

def encode_video_frames(images: np.ndarray, dst: Path, fps: int, vcodec: str = "libx264",
                        pix_fmt: str = "yuv420p", g: int = 2, crf: int = 23, fast_decode: int = 0,
                        log_level: int = av.logging.ERROR, overwrite: bool = False) -> bytes:
    if vcodec not in {"h264", "hevc", "libx264", "libx265", "libsvtav1"}:
        raise ValueError(f"Unsupported codec {vcodec}")
    video_path = Path(dst)
    video_path.parent.mkdir(parents=True, exist_ok=overwrite)
    if (vcodec in {"libsvtav1", "hevc", "libx265"}) and pix_fmt == "yuv444p":
        pix_fmt = "yuv420p"
    h, w, _ = images[0].shape
    options = {}
    for k, v in {"g": g, "crf": crf}.items():
        if v is not None:
            options[k] = str(v)
    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        options[key] = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
    if log_level is not None:
        logging.getLogger("libav").setLevel(log_level)
    with av.open(str(video_path), "w") as out:
        stream = out.add_stream(vcodec, fps, options=options)
        stream.pix_fmt, stream.width, stream.height = pix_fmt, w, h
        for i, img in enumerate(images):
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for pkt in stream.encode(frame):
                out.mux(pkt)
            if (i + 1) % 100 == 0 or i == len(images) - 1:
                print(f"Encoded frame {i+1}")
        for pkt in stream.encode():
            out.mux(pkt)
    if log_level is not None:
        av.logging.restore_default_callback()
    if not video_path.exists():
        raise OSError(f"Video encoding failed: {video_path}")

def create_video_from_images(images, output_path, fps=30, codec="libx264", quality=23):
    if not images:
        raise ValueError("No image data")
    print(f"Start encoding video, codec: {codec}  CRF: {quality}")
    encode_video_frames(np.asarray(images), Path(output_path), fps=fps, vcodec=codec, crf=quality, overwrite=True)
    print(f"Video saved to: {output_path}")

# ── save_data ─────────────────────────────────────────────────────────────────
def save_data(args, timesteps, actions, dataset_path):
    data_size = len(actions)
    data_dict = {k: [] for k in [
        '/observations/qpos', '/observations/qvel', '/observations/effort',
        '/action', '/base_action']}

    video_images = {cam: [] for cam in args.camera_names}
    assert args.export_video, "Please set --export_video to enable video export"
    while actions:
        action = actions.pop(0)
        ts   = timesteps.pop(0)
        for k in ['qpos', 'qvel', 'effort']:
            data_dict[f'/observations/{k}'].append(ts.observation[k])
        data_dict['/action'].append(action)
        data_dict['/base_action'].append(ts.observation['base_vel'])
        for cam in args.camera_names:
            img = ts.observation['images'][cam]
            if args.export_video:
                video_img = img if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if video_img.dtype != np.uint8:
                    video_img = (video_img * 255).astype(np.uint8) if video_img.max() <= 1.0 else video_img.astype(np.uint8)
                video_images[cam].append(video_img)

    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'], root.attrs['compress'] = False, False
        obs = root.create_group('observations')
        for k in ['qpos', 'qvel', 'effort']:
            obs.create_dataset(k, (data_size, 14))
        root.create_dataset('action', (data_size, 14))
        root.create_dataset('base_action', (data_size, 2))
        for name, arr in data_dict.items():
            root[name][...] = arr
    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n' % dataset_path)

    if args.export_video and data_size:
        print('\033[33m\nExporting videos...\033[0m')
        video_dir = os.path.join(os.path.dirname(dataset_path), "video")
        os.makedirs(video_dir, exist_ok=True)
        for cam in args.camera_names:
            if video_images[cam]:
                try:
                    cam_dir = os.path.join(video_dir, cam)
                    os.makedirs(cam_dir, exist_ok=True)
                    episode_idx = os.path.basename(dataset_path).split('_')[-1]
                    video_path = os.path.join(cam_dir, f"episode_{episode_idx}.mp4")
                    print(f"Exporting video for camera {cam}: {video_path}")
                    create_video_from_images(video_images[cam], video_path,
                                             fps=args.video_fps, codec=args.video_codec, quality=args.video_quality)
                    print(f'\033[32m✅ Camera {cam} video exported: {video_path}\033[0m')
                except Exception as e:
                    print(f'\033[31m❌ Camera {cam} video export failed: {e}\033[0m')
            else:
                print(f'\033[33m⚠️  Camera {cam} has no image data, skipping video export\033[0m')
        print(f'\033[32m\nVideo export done: {time.time() - t0:.1f} secs\033[0m')

# ── RosOperator ───────────────────────────────────────────────────────────────
from pynput import keyboard

class RosOperator:
    def __init__(self, args):
        self.args = args
        self.stop_flag = False
        self.bridge = CvBridge()
        self.init_deques()
        self.init_ros()

    def init_deques(self):
        self.img_left_deque  = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque  = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.master_arm_left_deque  = deque()
        self.master_arm_right_deque = deque()
        self.puppet_arm_left_deque  = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()

    def keyboard_listener(self):
        def on_press(key):
            if key == keyboard.Key.space:
                self.stop_flag = True
                print("\033[35m>>> Space key pressed, stopping collection and saving...\033[0m")
                return False
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

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
    def master_arm_left_callback(self, msg):
        if len(self.master_arm_left_deque) >= 2000: self.master_arm_left_deque.popleft()
        self.master_arm_left_deque.append(msg)
    def master_arm_right_callback(self, msg):
        if len(self.master_arm_right_deque) >= 2000: self.master_arm_right_deque.popleft()
        self.master_arm_right_deque.append(msg)
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
        rospy.init_node('record_episodes', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.master_arm_left_topic, JointState, self.master_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.master_arm_right_topic, JointState, self.master_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)

    def process(self):
        timesteps, actions = [], []
        count = 0
        rate = rospy.Rate(self.args.frame_rate)
        print("\033[36m>>> Collection started. Press [Space] to stop and save early...\033[0m")

        # qpos change detection variables (left/right arm separately)
        last_qpos_left, last_qpos_right = None, None
        consecutive_unchanged_count_left, consecutive_unchanged_count_right = 0, 0
        UNCHANGED_THRESHOLD = 100  # warn after 100 consecutive unchanged frames

        # start background keyboard listener thread
        threading.Thread(target=self.keyboard_listener, daemon=True).start()

        while (count < self.args.max_timesteps + 1) and not rospy.is_shutdown() and not self.stop_flag:
            result = self.get_frame()
            if not result:
                rate.sleep()
                continue
            count += 1
            (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
             puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base) = result

            image_dict = {self.args.camera_names[0]: img_front,
                          self.args.camera_names[1]: img_left,
                          self.args.camera_names[2]: img_right}
            obs = collections.OrderedDict()
            obs['images'] = image_dict
            if self.args.use_depth_image:
                obs['images_depth'] = {self.args.camera_names[0]: img_front_depth,
                                       self.args.camera_names[1]: img_left_depth,
                                       self.args.camera_names[2]: img_right_depth}
            obs['qpos'] = np.concatenate((puppet_arm_left.position, puppet_arm_right.position))
            obs['qvel'] = np.concatenate((puppet_arm_left.velocity, puppet_arm_right.velocity))
            obs['effort'] = np.concatenate((puppet_arm_left.effort, puppet_arm_right.effort))
            obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z] if self.args.use_robot_base else [0.0, 0.0]

            # left arm qpos change detection
            current_qpos_left = puppet_arm_left.position
            if last_qpos_left is not None and np.array_equal(current_qpos_left, last_qpos_left):
                consecutive_unchanged_count_left += 1
            else:
                consecutive_unchanged_count_left = 0
            if consecutive_unchanged_count_left >= UNCHANGED_THRESHOLD:
                print(f"\033[33m⚠️ Warning: left arm 'position' unchanged for {consecutive_unchanged_count_left} consecutive frames.\033[0m")
            last_qpos_left = current_qpos_left

            # right arm qpos change detection
            current_qpos_right = puppet_arm_right.position
            if last_qpos_right is not None and np.array_equal(current_qpos_right, last_qpos_right):
                consecutive_unchanged_count_right += 1
            else:
                consecutive_unchanged_count_right = 0
            if consecutive_unchanged_count_right >= UNCHANGED_THRESHOLD:
                print(f"\033[33m⚠️ Warning: right arm 'position' unchanged for {consecutive_unchanged_count_right} consecutive frames.\033[0m")
            last_qpos_right = current_qpos_right

            if count == 1:
                timesteps.append(dm_env.TimeStep(dm_env.StepType.FIRST, None, None, obs))
                continue
            timesteps.append(dm_env.TimeStep(dm_env.StepType.MID, None, None, obs))
            left_action = np.concatenate((puppet_arm_left.position[:6], [master_arm_left.position[6]]))
            right_action = np.concatenate((puppet_arm_right.position[:6], [master_arm_right.position[6]]))
            actions.append(np.concatenate((left_action, right_action)))
            print("Frame data: ", count)
            rate.sleep()

        print(f"\n>>> Collection finished, {len(actions)} frames total. Saving...")
        return timesteps, actions

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0):
            return False
        frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(),
                          self.img_right_deque[-1].header.stamp.to_sec(),
                          self.img_front_deque[-1].header.stamp.to_sec()])
        if self.args.use_depth_image:
            frame_time = min(frame_time,
                             self.img_left_depth_deque[-1].header.stamp.to_sec(),
                             self.img_right_depth_deque[-1].header.stamp.to_sec(),
                             self.img_front_depth_deque[-1].header.stamp.to_sec())
        for dq in [self.img_left_deque, self.img_right_deque, self.img_front_deque,
                   self.master_arm_left_deque, self.master_arm_right_deque,
                   self.puppet_arm_left_deque, self.puppet_arm_right_deque]:
            if not dq or dq[-1].header.stamp.to_sec() < frame_time:
                return False
        if self.args.use_depth_image:
            for dq in [self.img_left_depth_deque, self.img_right_depth_deque, self.img_front_depth_deque]:
                if not dq or dq[-1].header.stamp.to_sec() < frame_time:
                    return False
        if self.args.use_robot_base and (not self.robot_base_deque or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        def pop(dq):
            while dq[0].header.stamp.to_sec() < frame_time:
                dq.popleft()
            return dq.popleft()

        img_left  = self.bridge.imgmsg_to_cv2(pop(self.img_left_deque), 'passthrough')
        img_right = self.bridge.imgmsg_to_cv2(pop(self.img_right_deque), 'passthrough')
        img_front = self.bridge.imgmsg_to_cv2(pop(self.img_front_deque), 'passthrough')
        master_arm_left  = pop(self.master_arm_left_deque)
        master_arm_right = pop(self.master_arm_right_deque)
        puppet_arm_left  = pop(self.puppet_arm_left_deque)
        puppet_arm_right = pop(self.puppet_arm_right_deque)
        img_left_depth = img_right_depth = img_front_depth = None
        if self.args.use_depth_image:
            img_left_depth  = cv2.copyMakeBorder(self.bridge.imgmsg_to_cv2(pop(self.img_left_depth_deque), 'passthrough'), 40, 40, 0, 0, cv2.BORDER_CONSTANT, value=0)
            img_right_depth = cv2.copyMakeBorder(self.bridge.imgmsg_to_cv2(pop(self.img_right_depth_deque), 'passthrough'), 40, 40, 0, 0, cv2.BORDER_CONSTANT, value=0)
            img_front_depth = cv2.copyMakeBorder(self.bridge.imgmsg_to_cv2(pop(self.img_front_depth_deque), 'passthrough'), 40, 40, 0, 0, cv2.BORDER_CONSTANT, value=0)
        robot_base = None
        if self.args.use_robot_base:
            robot_base = pop(self.robot_base_deque)
        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base)

# ── Argument parsing ──────────────────────────────────────────────────────────
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default="./data")
    parser.add_argument('--task_name', type=str, default="aloha_mobile_dummy")
    parser.add_argument('-e', '--episode_idx', type=int, default=None)
    parser.add_argument('--max_timesteps', type=int, default=500)
    parser.add_argument('--camera_names', nargs='+', default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'])
    parser.add_argument('--img_front_topic', default='/camera_f/color/image_raw')
    parser.add_argument('--img_left_topic', default='/camera_l/color/image_raw')
    parser.add_argument('--img_right_topic', default='/camera_r/color/image_raw')
    parser.add_argument('--img_front_depth_topic', default='/camera_f/depth/image_raw')
    parser.add_argument('--img_left_depth_topic', default='/camera_l/depth/image_raw')
    parser.add_argument('--img_right_depth_topic', default='/camera_r/depth/image_raw')
    parser.add_argument('--master_arm_left_topic', default='/master/joint_left')
    parser.add_argument('--master_arm_right_topic', default='/master/joint_right')
    parser.add_argument('--puppet_arm_left_topic', default='/puppet/joint_left')
    parser.add_argument('--puppet_arm_right_topic', default='/puppet/joint_right')
    parser.add_argument('--robot_base_topic', default='/odom')
    parser.add_argument('--use_robot_base', type=bool, default=False)
    parser.add_argument('--use_depth_image', type=bool, default=False)
    parser.add_argument('--frame_rate', type=int, default=30)
    parser.add_argument('--export_video', action='store_true', help='Enable video export')
    parser.add_argument('--video_fps', type=int, default=30)
    parser.add_argument('--video_codec', choices=['libx264', 'libx265', 'libsvtav1'], default='libx264')
    parser.add_argument('--video_quality', type=int, default=23, help='CRF value; lower = higher quality')

    args = parser.parse_args()

    # auto-compute episode_idx if not specified
    if args.episode_idx is None:
        dataset_dir = os.path.join(args.dataset_dir, args.task_name)
        episode_indices = []
        if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
            pattern = re.compile(r'^episode_(\d+)\.hdf5$')
            for filename in os.listdir(dataset_dir):
                match = pattern.match(filename)
                if match:
                    episode_indices.append(int(match.group(1)))
        if episode_indices:
            args.episode_idx = max(episode_indices) + 1
            print(f"\033[34m>>> Found {len(episode_indices)} existing episodes, next index: {args.episode_idx}\033[0m")
        else:
            args.episode_idx = 0
            print(f"\033[34m>>> No existing episodes found, starting from index 0\033[0m")

    return args

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    timesteps, actions = ros_operator.process()

    if len(actions) == 0:
        print("\033[31m\nNo data collected, skipping save.\033[0m")
        return

    dataset_dir = os.path.join(args.dataset_dir, args.task_name)
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, f"episode_{args.episode_idx}")
    save_data(args, timesteps, actions, dataset_path)
    print("\033[32m>>> Saved to:", dataset_path + ".hdf5\033[0m")


if __name__ == '__main__':
    main()
