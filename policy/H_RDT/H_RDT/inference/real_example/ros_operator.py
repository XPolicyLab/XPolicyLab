#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import threading
import time
from collections import deque
from base64 import b64decode
import requests
import json

import numpy as np
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


class RosOperator:
    """ROS operation class for handling robot communication"""
    
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.init_deques()
        self.init_ros()
        self.init_threading()
        self.timesteps = []
        
    def init_deques(self):
        """Initialize data queues"""
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        
    def init_threading(self):
        """Initialize thread locks"""
        self.master_arm_publish_lock = threading.Lock()
        self.master_arm_publish_lock.acquire()
        self.master_arm_publish_thread = None
        
    def init_ros(self):
        """Initialize ROS node and topics"""
        rospy.init_node('hrdt_inference', anonymous=True)
        
        # Subscribe to topics
        if self.args.use_puppet_left:
            rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, 
                           self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_puppet_right:
            rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, 
                           self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_robot_base:
            rospy.Subscriber(self.args.robot_base_topic, Odometry, 
                           self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
            
        # Publish topics
        self.master_arm_left_publisher = rospy.Publisher(
            self.args.master_arm_left_cmd_topic, JointState, queue_size=10)
        self.master_arm_right_publisher = rospy.Publisher(
            self.args.master_arm_right_cmd_topic, JointState, queue_size=10)
        if self.args.use_robot_base:
            self.robot_base_publisher = rospy.Publisher(
                self.args.robot_base_cmd_topic, Twist, queue_size=10)
    
    def puppet_arm_left_callback(self, msg):
        """Left arm callback function"""
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)
        
    def puppet_arm_right_callback(self, msg):
        """Right arm callback function"""
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)
        
    def robot_base_callback(self, msg):
        """Robot base callback function"""
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)
    
    def get_cam_frame(self, camera='high'):
        """Get camera frame"""
        port_map = {
            "high": 23000,
            "left": 23001, 
            "right": 23002
        }
        
        port = port_map.get(camera, 23000)
        frame = requests.get(f'http://localhost:{port}').json()
        frame['image'] = np.frombuffer(b64decode(frame['image']), dtype=np.uint8).reshape(480, 640, 3)
        frame['depth'] = np.frombuffer(b64decode(frame['depth']), dtype=np.uint16).reshape(480, 640)
        return frame['image'], frame['depth'], frame['timestamp']
    
    def get_current_observation(self):
        """Get current observation"""
        # Get camera images
        img_high, img_left, img_right = None, None, None
        
        if self.args.use_image_high:
            img_high, _, _ = self.get_cam_frame('high')
        if self.args.use_image_left:
            img_left, _, _ = self.get_cam_frame('left')
        if self.args.use_image_right:
            img_right, _, _ = self.get_cam_frame('right')
            
        # Get robot arm states
        puppet_left = puppet_right = None
        if (self.args.use_puppet_left and len(self.puppet_arm_left_deque) > 0):
            puppet_left = self.puppet_arm_left_deque[-1]
        if (self.args.use_puppet_right and len(self.puppet_arm_right_deque) > 0):
            puppet_right = self.puppet_arm_right_deque[-1]
            
        # Get robot base state
        robot_base = None
        if self.args.use_robot_base and len(self.robot_base_deque) > 0:
            robot_base = self.robot_base_deque[-1]
            
        return {
            'images': {
                'head_cam': img_high,
                'left_cam': img_left,
                'right_cam': img_right,
            },
            'puppet_left': puppet_left,
            'puppet_right': puppet_right,
            'robot_base': robot_base
        }
    
    def publish_action(self, left_action, right_action):
        """Publish action commands"""
        if self.args.disable_puppet_arm:
            return
            
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        
        # Publish left arm action
        joint_state_msg.position = left_action
        self.master_arm_left_publisher.publish(joint_state_msg)
        
        # Publish right arm action
        joint_state_msg.position = right_action
        self.master_arm_right_publisher.publish(joint_state_msg)
    
    def reset_arms(self, left_pos=None, right_pos=None):
        """Reset robot arms to specified positions"""
        # Default reset positions
        RESET_LEFT = [0] * 7
        RESET_RIGHT = [0] * 7
        
        if left_pos is None:
            left_pos = RESET_LEFT
        if right_pos is None:
            right_pos = RESET_RIGHT
            
        self.publish_action_continuous(left_pos, right_pos)
    
    def publish_action_continuous(self, left_target, right_target):
        """Continuously publish actions until reaching target positions"""
        rate = rospy.Rate(self.args.publish_rate)
        
        # Get current positions
        while not rospy.is_shutdown():
            if (len(self.puppet_arm_left_deque) > 0 and 
                len(self.puppet_arm_right_deque) > 0):
                break
            rate.sleep()
        
        left_current = list(self.puppet_arm_left_deque[-1].position)
        right_current = list(self.puppet_arm_right_deque[-1].position)
        
        # Calculate movement directions
        left_direction = [1 if left_target[i] - left_current[i] > 0 else -1 
                         for i in range(len(left_target))]
        right_direction = [1 if right_target[i] - right_current[i] > 0 else -1 
                          for i in range(len(right_target))]
        
        # Move step by step to target positions
        moving = True
        while moving and not rospy.is_shutdown():
            moving = False
            
            # Update left arm position
            for i in range(len(left_target)):
                diff = abs(left_target[i] - left_current[i])
                if diff < self.args.arm_steps_length[i]:
                    left_current[i] = left_target[i]
                else:
                    left_current[i] += left_direction[i] * self.args.arm_steps_length[i]
                    moving = True
            
            # Update right arm position
            for i in range(len(right_target)):
                diff = abs(right_target[i] - right_current[i])
                if diff < self.args.arm_steps_length[i]:
                    right_current[i] = right_target[i]
                else:
                    right_current[i] += right_direction[i] * self.args.arm_steps_length[i]
                    moving = True
                    
            # Publish action
            self.publish_action(left_current, right_current)
            rate.sleep() 