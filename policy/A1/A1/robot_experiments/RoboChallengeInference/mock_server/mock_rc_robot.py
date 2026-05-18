# -*- coding: utf-8 -*-
import json
import time
from abc import ABC, abstractmethod
from enum import Enum
from threading import Thread
from typing import Any

import cv2
import numpy as np

ANGLE2RAD = np.pi / 180.0


class RobotTag(Enum):
    ALOHA = 'aloha'
    UR5 = 'ur5'
    FRANKA = 'franka'
    ARX5 = 'arx5'


class MockRCRobot(ABC):
    robot_alpha: Any
    filler_thread: Thread
    frame_interval = 1 / 30

    @property
    @abstractmethod
    def dof_num(self):
        raise NotImplemented('no joint number defined')

    @property
    @abstractmethod
    def pos_num(self):
        raise NotImplemented('no pose number defined')

    @staticmethod
    def create_robot(robot_tag: RobotTag | str, realsense_ids, record_data_dir) -> 'MockRCRobot':
        robot_tag = RobotTag(robot_tag)
        if robot_tag == RobotTag.ALOHA:
            return MockRCRobotAloha(robot_tag, realsense_ids, record_data_dir)
        elif robot_tag == RobotTag.ARX5:
            return MockRCRobotArx5(robot_tag, realsense_ids, record_data_dir)
        elif robot_tag == RobotTag.UR5:
            return MockRCRobotUr5(robot_tag, realsense_ids, record_data_dir)
        elif robot_tag == RobotTag.FRANKA:
            return MockRCRobotFranka(robot_tag, realsense_ids, record_data_dir)
        else:
            raise RuntimeError('unknown robot tag')

    @staticmethod
    def iter_jsonl(file_path):
        while True:
            with open(file_path, 'r') as f:
                for line in f:
                    yield json.loads(line)

    @staticmethod
    def get_jsonl(file_path):
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def iter_mp4(file_path):
        while True:
            cap = cv2.VideoCapture(file_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            cap.release()

    @staticmethod
    def get_mp4(file_path):
        cap = cv2.VideoCapture(file_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    @abstractmethod
    def fill(self):
        pass

    def filler(self):
        for i in range(self.get_frame_number() - 1):
            start_time = time.time()
            self.fill()
            try:
                time.sleep(self.frame_interval - (time.time() - start_time))
            except ValueError:
                pass

    def get_frame_number(self):
        with open(f'{self.record_data_dir}/meta/episode_meta.json', 'r') as f:
            meta = json.load(f)
            return meta['frames']

    def __init__(self, robot_tag: RobotTag | str, realsense_ids, record_data_dir):
        self.robot_tag = RobotTag(robot_tag)
        self.record_data_dir = record_data_dir

    def left_get_enable(self):
        return True

    def right_get_enable(self):
        return True

    def left_get_joint(self):
        pass

    def right_get_joint(self):
        pass

    def left_get_pose(self):
        pass

    def right_get_pose(self):
        pass

    def left_go_joint(self, action):
        pass

    def right_go_joint(self, action):
        pass

    def left_go_pose(self, action):
        pass

    def right_go_pose(self, action):
        pass

    def get_imgs(self):
        pass

    def _start_record(self):
        self.filler_thread.start()

    def _stop_record(self):
        pass

    def terminate(self):
        pass


class MockRCRobotAloha(MockRCRobot):
    @property
    def dof_num(self):
        return 6

    @property
    def pos_num(self):
        return 7

    def __init__(self, robot_tag: RobotTag | str, realsense_ids, record_data_dir):
        super().__init__(robot_tag, realsense_ids, record_data_dir)
        self.left_arm_states = self.get_jsonl(f'{self.record_data_dir}/states/left_states.jsonl')
        self.right_arm_states = self.get_jsonl(f'{self.record_data_dir}/states/right_states.jsonl')
        self.left_images = self.get_mp4(f'{self.record_data_dir}/videos/cam_wrist_left_rgb.mp4')
        self.right_images = self.get_mp4(f'{self.record_data_dir}/videos/cam_wrist_right_rgb.mp4')
        self.high_images = self.get_mp4(f'{self.record_data_dir}/videos/cam_high_rgb.mp4')

        self.left_joint = [0.0] * self.dof_num
        self.right_joint = [0.0] * self.dof_num
        self.left_gripper = 0.0
        self.right_gripper = 0.0
        self.left_pose = [0.0] * self.pos_num
        self.right_pose = [0.0] * self.pos_num
        self.frame_left = None
        self.frame_right = None
        self.frame_high = None
        self.filler_thread = Thread(target=self.filler, daemon=True)
        self.fill()

    def fill(self):
        left_state = self.left_arm_states[0]
        right_state = self.right_arm_states[0]
        self.left_joint = left_state['joint_positions']
        self.right_joint = right_state['joint_positions']
        self.left_gripper = left_state['gripper']
        self.right_gripper = right_state['gripper']
        try:
            self.left_pose = left_state['ee_pose_quaternion']
        except:
            pass
        try:
            self.right_pose = right_state['ee_pose_quaternion']
        except:
            pass
        self.frame_left = self.left_images[0]
        self.frame_right = self.right_images[0]
        self.frame_high = self.high_images[0]

    def left_get_joint(self,idx: int = 0):
        return self.left_arm_states[idx]['joint_positions'] + [self.left_arm_states[idx]['gripper']]

    def right_get_joint(self,idx: int = 0):
        return self.right_arm_states[idx]['joint_positions'] + [self.right_arm_states[idx]['gripper']]

    def left_get_pose(self,idx: int = 0):
        return self.left_arm_states[idx]['ee_pose_quaternion'] + [self.left_arm_states[idx]['gripper']]

    def right_get_pose(self,idx: int = 0):
        return self.right_arm_states[idx]['ee_pose_quaternion'] + [self.right_arm_states[idx]['gripper']]

    def get_imgs(self, idx: int = 0):
        return [
            (time.time(), self.left_images[idx], None, None),
            (time.time(), self.right_images[idx], None, None),
            (time.time(), self.high_images[idx], None, None),
        ]

    def terminate(self):
        pass


class MockRCRobotArx5(MockRCRobot):
    @property
    def dof_num(self):
        return 6

    @property
    def pos_num(self):
        return 6

    def __init__(self, robot_tag: RobotTag | str, realsense_ids, record_data_dir):
        super().__init__(robot_tag, realsense_ids, record_data_dir)
        self.states = self.get_jsonl(f'{self.record_data_dir}/states/states.jsonl')
        self.left_images = self.get_mp4(f'{self.record_data_dir}/videos/arm_realsense_rgb.mp4')
        self.right_images = self.get_mp4(f'{self.record_data_dir}/videos/global_realsense_rgb.mp4')
        self.high_images = self.get_mp4(f'{self.record_data_dir}/videos/right_realsense_rgb.mp4')

        self.joint = [0.0] * self.dof_num
        self.gripper = 0.0
        self.pose = [0.0] * self.pos_num
        self.frame_left = None
        self.frame_right = None
        self.frame_high = None
        self.filler_thread = Thread(target=self.filler, daemon=True)
        self.fill()

    def fill(self):
        state = self.states[0]
        self.joint = state['joint_positions']
        self.gripper = state['gripper_width']
        try:
            self.pose = state['end_effector_pose']
        except:
            pass
        self.frame_left = self.left_images[0]
        self.frame_right = self.right_images[0]
        self.frame_high = self.high_images[0]

    def left_get_joint(self,idx: int = 0):
        return self.states[idx]['joint_positions'] + [self.states[idx]['gripper_width']]

    def right_get_joint(self,idx: int = 0):
        return self.states[idx]['joint_positions'] + [self.states[idx]['gripper_width']]

    def left_get_pose(self,idx: int = 0):
        return self.states[idx]['end_effector_pose'] + [self.states[idx]['gripper_width']]

    def right_get_pose(self,idx: int = 0):
        return self.states[idx]['end_effector_pose'] + [self.states[idx]['gripper_width']]

    def get_imgs(self,idx: int = 0):
        return [
            (time.time(), self.left_images[idx], None, None),
            (time.time(), self.right_images[idx], None, None),
            (time.time(), self.high_images[idx], None, None),
        ]


class MockRCRobotUr5(MockRCRobot):
    @property
    def dof_num(self):
        return 6

    @property
    def pos_num(self):
        return 7

    def __init__(self, robot_tag: RobotTag | str, realsense_ids, record_data_dir):
        super().__init__(robot_tag, realsense_ids, record_data_dir)
        self.states = self.get_jsonl(f'{self.record_data_dir}/states/states.jsonl')
        self.left_images = self.get_mp4(f'{self.record_data_dir}/videos/handeye_realsense_rgb.mp4')
        self.right_images = self.get_mp4(f'{self.record_data_dir}/videos/global_realsense_rgb.mp4')

        self.joint = [0.0] * self.dof_num
        self.gripper = 0.0
        self.pose = [0.0] * self.pos_num
        self.frame_left = None
        self.frame_right = None
        self.frame_high = None
        self.filler_thread = Thread(target=self.filler, daemon=True)
        self.fill()


    def fill(self):
        state = self.states[0]
        self.joint = state['joint_positions']
        self.gripper = state['gripper']
        try:
            self.pose = state['ee_positions']
        except:
            pass
        self.frame_left = self.left_images[0]
        self.frame_right = self.right_images[0]

    def left_get_joint(self,idx: int = 0):
        return self.states[idx]['joint_positions'] + [self.states[idx]['gripper']]

    def right_get_joint(self,idx: int = 0):
        return self.states[idx]['joint_positions'] + [self.states[idx]['gripper'] ]

    def left_get_pose(self,idx: int = 0):
        return self.states[idx]['ee_positions'] + [self.states[idx]['gripper']]

    def right_get_pose(self,idx: int = 0):
        return self.states[idx]['ee_positions'] + [self.states[idx]['gripper']]

    def get_imgs(self,idx: int = 0):
        return [
            (time.time(), self.left_images[idx], None, None),
            (time.time(), self.right_images[idx], None, None),
        ]




class MockRCRobotFranka(MockRCRobot):
    @property
    def dof_num(self):
        return 7

    @property
    def pos_num(self):
        return 7

    def __init__(self, robot_tag: RobotTag | str, realsense_ids, record_data_dir):
        super().__init__(robot_tag, realsense_ids, record_data_dir)
        self.states = self.get_jsonl(f'{self.record_data_dir}/states/states.jsonl')
        self.left_images = self.get_mp4(f'{self.record_data_dir}/videos/handeye_realsense_rgb.mp4')
        self.right_images = self.get_mp4(f'{self.record_data_dir}/videos/main_realsense_rgb.mp4')
        self.high_images = self.get_mp4(f'{self.record_data_dir}/videos/side_realsense_rgb.mp4')

        self.joint = [0.0] * self.dof_num
        self.gripper = 0.0
        self.pose = [0.0] * self.pos_num
        self.frame_left = None
        self.frame_right = None
        self.frame_high = None
        self.filler_thread = Thread(target=self.filler, daemon=True)
        self.fill()


    def fill(self):
        state = self.states[0]
        self.joint = state['joint_positions']
        self.gripper = state['gripper_width']
        try:
            self.pose = state['ee_positions'] + [state['gripper_width'][0]]
        except:
            pass
        self.frame_left = self.left_images[0]
        self.frame_right = self.right_images[0]
        self.frame_high = self.high_images[0]

    def left_get_joint(self,idx: int = 0):
        return self.states[idx]['joint_positions'] + [self.states[idx]['gripper_width'][0]]

    def right_get_joint(self,idx: int = 0):
        return self.states[idx]['joint_positions'] + [self.states[idx]['gripper_width'][0]]

    def left_get_pose(self,idx: int = 0):
        return self.states[idx]['ee_positions'] + [self.states[idx]['gripper_width'][0]]

    def right_get_pose(self,idx: int = 0):
        return self.states[idx]['ee_positions'] + [self.states[idx]['gripper_width'][0]]

    def get_imgs(self,idx: int = 0):
        return [
            (time.time(), self.left_images[idx]  , None, None),
            (time.time(), self.right_images[idx]  , None, None),
            (time.time(), self.high_images[idx]  , None, None),
        ]


if __name__ == '__main__':
    record_dir = '20250919/pour_fries_into_plate/data/episode_001533'
    robot = MockRCRobotAloha('aloha', ('233',), record_dir)
    pass
