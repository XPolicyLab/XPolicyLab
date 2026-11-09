# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
from collections import deque
import datetime
import json
import time

from aloha.msg import RGBGrayscaleImage
from cv_bridge import CvBridge
# from interbotix_xs_msgs.msg import JointGroupCommand
# from interbotix_xs_msgs.msg import JointSingleCommand
import numpy as np
import rospy
from sensor_msgs.msg import JointState

from examples.mobile_aloha_AgileX import constants
import argparse
import sys
import threading
import time
import yaml
from collections import deque

import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PImage
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
import cv2


sys.path.append("./")

CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']

observation_window = None

lang_embeddings = None

# debug
preload_images = None



class ImageRecorder:
    def __init__(self, init_node=True, is_debug=False):
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

        if init_node:
            rospy.init_node("image_recorder", anonymous=True)
        for cam_name in self.camera_names:
            setattr(self, f"{cam_name}_rgb_image", None)
            setattr(self, f"{cam_name}_depth_image", None)
            setattr(self, f"{cam_name}_timestamp", 0.0)
            if cam_name == "cam_high":
                callback_func = self.image_cb_cam_high
            elif cam_name == "cam_low":
                callback_func = self.image_cb_cam_low
            elif cam_name == "cam_left_wrist":
                callback_func = self.image_cb_cam_left_wrist
            elif cam_name == "cam_right_wrist":
                callback_func = self.image_cb_cam_right_wrist
            else:
                raise NotImplementedError
            rospy.Subscriber(f"/{cam_name}", RGBGrayscaleImage, callback_func)
            if self.is_debug:
                setattr(self, f"{cam_name}_timestamps", deque(maxlen=50))

        self.cam_last_timestamps = {cam_name: 0.0 for cam_name in self.camera_names}
        time.sleep(0.5)

    def image_cb(self, cam_name, data):
        setattr(
            self,
            f"{cam_name}_rgb_image",
            self.bridge.imgmsg_to_cv2(data.images[0], desired_encoding="bgr8"),
        )
        # setattr(
        #     self,
        #     f"{cam_name}_depth_image",
        #     self.bridge.imgmsg_to_cv2(data.images[1], desired_encoding="mono16"),
        # )
        setattr(
            self,
            f"{cam_name}_timestamp",
            data.header.stamp.secs + data.header.stamp.nsecs * 1e-9,
        )
        # setattr(self, f'{cam_name}_secs', data.images[0].header.stamp.secs)
        # setattr(self, f'{cam_name}_nsecs', data.images[0].header.stamp.nsecs)
        # cv2.imwrite('/home/lucyshi/Desktop/sample.jpg', cv_image)
        if self.is_debug:
            getattr(self, f"{cam_name}_timestamps").append(
                data.images[0].header.stamp.secs + data.images[0].header.stamp.nsecs * 1e-9
            )

    def image_cb_cam_high(self, data):
        cam_name = "cam_high"
        return self.image_cb(cam_name, data)

    def image_cb_cam_low(self, data):
        cam_name = "cam_low"
        return self.image_cb(cam_name, data)

    def image_cb_cam_left_wrist(self, data):
        cam_name = "cam_left_wrist"
        return self.image_cb(cam_name, data)

    def image_cb_cam_right_wrist(self, data):
        cam_name = "cam_right_wrist"
        return self.image_cb(cam_name, data)

    def get_images(self):
        image_dict = {}
        for cam_name in self.camera_names:
            while getattr(self, f"{cam_name}_timestamp") <= self.cam_last_timestamps[cam_name]:
                time.sleep(0.00001)
            rgb_image = getattr(self, f"{cam_name}_rgb_image")
            depth_image = getattr(self, f"{cam_name}_depth_image")
            self.cam_last_timestamps[cam_name] = getattr(self, f"{cam_name}_timestamp")
            image_dict[cam_name] = rgb_image
            image_dict[f"{cam_name}_depth"] = depth_image
        return image_dict

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f"{cam_name}_timestamps"))
            print(f"{cam_name} {image_freq=:.2f}")
        print()


# class Recorder:
#     def __init__(self, side, init_node=True, is_debug=False):
#         self.secs = None
#         self.nsecs = None
#         self.qpos = None
#         self.effort = None
#         self.arm_command = None
#         self.gripper_command = None
#         self.is_debug = is_debug

#         if init_node:
#             rospy.init_node("recorder", anonymous=True)
#         rospy.Subscriber(f"/puppet_{side}/joint_states", JointState, self.puppet_state_cb)
#         rospy.Subscriber(
#             f"/puppet_{side}/commands/joint_group",
#             JointGroupCommand,
#             self.puppet_arm_commands_cb,
#         )
#         rospy.Subscriber(
#             f"/puppet_{side}/commands/joint_single",
#             JointSingleCommand,
#             self.puppet_gripper_commands_cb,
#         )
#         if self.is_debug:
#             self.joint_timestamps = deque(maxlen=50)
#             self.arm_command_timestamps = deque(maxlen=50)
#             self.gripper_command_timestamps = deque(maxlen=50)
#         time.sleep(0.1)

#     def puppet_state_cb(self, data):
#         self.qpos = data.position
#         self.qvel = data.velocity
#         self.effort = data.effort
#         self.data = data
#         if self.is_debug:
#             self.joint_timestamps.append(time.time())

#     def puppet_arm_commands_cb(self, data):
#         self.arm_command = data.cmd
#         if self.is_debug:
#             self.arm_command_timestamps.append(time.time())

#     def puppet_gripper_commands_cb(self, data):
#         self.gripper_command = data.cmd
#         if self.is_debug:
#             self.gripper_command_timestamps.append(time.time())

#     def print_diagnostics(self):
#         def dt_helper(l):
#             l = np.array(l)
#             diff = l[1:] - l[:-1]
#             return np.mean(diff)

#         joint_freq = 1 / dt_helper(self.joint_timestamps)
#         arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)
#         gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)

#         print(f"{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n{gripper_command_freq=:.2f}\n")


# def get_arm_joint_positions(bot):
#     return bot.arm.core.joint_states.position[:6]


# def get_arm_gripper_positions(bot):
#     return bot.gripper.core.joint_states.position[6]


# def move_arms(bot_list, target_pose_list, move_time=1):
#     num_steps = int(move_time / constants.DT)
#     curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
#     traj_list = [
#         np.linspace(curr_pose, target_pose, num_steps)
#         for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
#     ]
#     for t in range(num_steps):
#         for bot_id, bot in enumerate(bot_list):
#             bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)
#         time.sleep(constants.DT)


# def move_grippers(bot_list, target_pose_list, move_time):
#     print(f"Moving grippers to {target_pose_list=}")
#     gripper_command = JointSingleCommand(name="gripper")
#     num_steps = int(move_time / constants.DT)
#     curr_pose_list = [get_arm_gripper_positions(bot) for bot in bot_list]
#     traj_list = [
#         np.linspace(curr_pose, target_pose, num_steps)
#         for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
#     ]

#     with open(f"/data/gripper_traj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl", "a") as f:
#         for t in range(num_steps):
#             d = {}
#             for bot_id, bot in enumerate(bot_list):
#                 gripper_command.cmd = traj_list[bot_id][t]
#                 bot.gripper.core.pub_single.publish(gripper_command)
#                 d[bot_id] = {"obs": get_arm_gripper_positions(bot), "act": traj_list[bot_id][t]}
#             f.write(json.dumps(d) + "\n")
#             time.sleep(constants.DT)


# def setup_puppet_bot(bot):
#     return 0

# def setup_master_bot(bot):
#     return 0


# def set_standard_pid_gains(bot):
#     return 0

# def set_low_pid_gains(bot):
#     return 0


# def torque_off(bot):
#     return 0

# def torque_on(bot):
#     return 0


# # for DAgger
# def sync_puppet_to_master(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right):
#     print("\nSyncing!")

#     # activate master arms
#     torque_on(master_bot_left)
#     torque_on(master_bot_right)

#     # get puppet arm positions
#     puppet_left_qpos = get_arm_joint_positions(puppet_bot_left)
#     puppet_right_qpos = get_arm_joint_positions(puppet_bot_right)

#     # get puppet gripper positions
#     puppet_left_gripper = get_arm_gripper_positions(puppet_bot_left)
#     puppet_right_gripper = get_arm_gripper_positions(puppet_bot_right)

#     # move master arms to puppet positions
#     move_arms(
#         [master_bot_left, master_bot_right],
#         [puppet_left_qpos, puppet_right_qpos],
#         move_time=1,
#     )

#     # move master grippers to puppet positions
#     move_grippers(
#         [master_bot_left, master_bot_right],
#         [puppet_left_gripper, puppet_right_gripper],
#         move_time=1,
#     )

# Get the observation from the ROS topic
def get_ros_observation(args,ros_operator):
    rate = rospy.Rate(args.publish_rate)
    print_flag = True

    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        print("1")
        if not result:
            if print_flag:
                print("syn fail when get_ros_observation")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = result
        # print("=== img_front 信息 ===")
        # print(f"type: {type(img_front)}")

        # if isinstance(img_front, np.ndarray):
        #     print(f"shape: {img_front.shape}")
        #     print(f"dtype: {img_front.dtype}")
        #     print(f"min: {np.min(img_front)}, max: {np.max(img_front)}, mean: {np.mean(img_front):.2f}")
        #     print(f"first pixel (0,0): {img_front[0,0]}")
        # else:
        #     print("img_front is not a numpy array.")
        print(f"sync success when get_ros_observation")
        return (img_front, img_left, img_right,
         puppet_arm_left, puppet_arm_right)
    
# ROS operator class
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
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # Set timestep
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
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
            joint_state_msg.header.stamp = rospy.Time.now()  # Set the timestep
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
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

        left_arm = None
        right_arm = None

        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()
            print("sleep is ok")

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


def get_arguments():
    from types import SimpleNamespace

    args = SimpleNamespace()

    args.max_publish_step = 10000
    args.seed = None

    args.img_front_topic = '/camera_f/color/image_raw'
    args.img_left_topic = '/camera_l/color/image_raw'
    args.img_right_topic = '/camera_r/color/image_raw'

    args.img_front_depth_topic = '/camera_f/depth/image_raw'
    args.img_left_depth_topic = '/camera_l/depth/image_raw'
    args.img_right_depth_topic = '/camera_r/depth/image_raw'

    args.puppet_arm_left_cmd_topic = '/master/joint_left'
    args.puppet_arm_right_cmd_topic = '/master/joint_right'
    args.puppet_arm_left_topic = '/puppet/joint_left'
    args.puppet_arm_right_topic = '/puppet/joint_right'

    args.robot_base_topic = '/odom_raw'
    args.robot_base_cmd_topic = '/cmd_vel'
    args.use_robot_base = False
    args.publish_rate = 30
    args.ctrl_freq = 25

    args.chunk_size = 64
    args.arm_steps_length = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2]

    args.use_actions_interpolation = True
    args.use_depth_image = False
    args.disable_puppet_arm = False

    args.config_path = "configs/base.yaml"
    args.pretrained_model_name_or_path = "/home/mobilealoha/RoboticsDiffusionTransformer/checkpoints/rdt_finetune-170m/checkpoint-140000"
    args.lang_embeddings_path = "outs/Pour_water.pt"

    return args
