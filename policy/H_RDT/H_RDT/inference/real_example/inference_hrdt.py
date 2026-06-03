#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
import signal
import numpy as np
import torch
import cv2
import rospy

# Import custom modules - use relative path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(project_root)
from ros_operator import RosOperator
from model_inference import HRDTInference
from video_recorder import VideoRecorder, setup_video_saving, parse_video_resolution


# Global variables
start_flag = False
exit_flag = False
num_fails = 0


def get_arguments():
    parser = argparse.ArgumentParser()
    
    # Model related parameters
    parser.add_argument('--config_path', type=str, required=True, help='Path to model config file')
    parser.add_argument('--pretrained_model_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--lang_embeddings_path', type=str, required=True, help='Path to language embeddings')
    parser.add_argument('--stat_file_path', type=str, 
                       default=os.path.join(project_root, 'utils/stats.json'),
                       help='Path to statistics file for action normalization')
    
    # Inference parameters
    parser.add_argument('--max_publish_step', type=int, default=10000, help='Maximum steps to publish')
    parser.add_argument('--chunk_size', type=int, default=16, help='Action chunk size')
    parser.add_argument('--publish_rate', type=int, default=30, help='Publishing rate')
    parser.add_argument('--frame_rate', type=int, default=30, help='Frame rate')
    
    # Robot arm parameters
    parser.add_argument('--arm_steps_length', type=list, 
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0], 
                        required=False,
                        help='Arm step lengths')
    parser.add_argument('--use_actions_interpolation', action='store_true', default=False,
                       help='Use action interpolation')
    
    # ROS topic parameters
    parser.add_argument('--master_arm_left_cmd_topic', type=str, default='/master/joint_left')
    parser.add_argument('--master_arm_right_cmd_topic', type=str, default='/master/joint_right')
    parser.add_argument('--puppet_arm_left_topic', type=str, default='/puppet/joint_left')
    parser.add_argument('--puppet_arm_right_topic', type=str, default='/puppet/joint_right')
    parser.add_argument('--robot_base_topic', type=str, default='/odom')
    parser.add_argument('--robot_base_cmd_topic', type=str, default='/cmd_vel')
    
    # Camera usage flags
    parser.add_argument('--use_image_high', action='store_true', default=True)
    parser.add_argument('--use_image_left', action='store_true', default=True)
    parser.add_argument('--use_image_right', action='store_true', default=True)
    parser.add_argument('--use_puppet_left', action='store_true', default=True)
    parser.add_argument('--use_puppet_right', action='store_true', default=True)
    parser.add_argument('--use_robot_base', action='store_true', default=False)
    
    # Video recording parameters
    parser.add_argument('--record_combined_video', action='store_true', default=True,
                       help='Record combined image video during inference')
    parser.add_argument('--video_save_dir', type=str, 
                       default=os.path.join(project_root, 'videos'),
                       help='Directory to save videos')
    parser.add_argument('--video_fps', type=int, default=30, help='Video frame rate')
    parser.add_argument('--video_resolution', type=str, default='1280x720',
                       help='Video resolution (format: WIDTHxHEIGHT, e.g., 1280x720, 1920x1080)')
    
    # Other parameters
    parser.add_argument('--reset_only', action='store_true', default=False)
    parser.add_argument('--use_keyboard_end', action='store_true', default=False)
    parser.add_argument('--disable_puppet_arm', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--action_log_file', type=str, default='action_log.txt', 
                       help='File to save action chunks')
    parser.add_argument('--image_save_dir', type=str, 
                       default=os.path.join(project_root, 'images'), 
                       help='Directory to save combined images during inference')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def write_action_to_file(action_buffer, step, file_path):
    """Write action chunk to file"""
    try:
        with open(file_path, 'a') as f:
            f.write(f"Step {step}: {action_buffer.tolist()}\n")
    except Exception as e:
        print(f"Error writing action to file: {e}")


def save_combined_image(combined_image, step, save_dir):
    """Save combined image to specified directory"""
    try:
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename
        filename = f"combined_image_step_{step:06d}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, combined_image)
        return filepath
    except Exception as e:
        print(f"Error saving image: {e}")
        return None


def interpolate_action(args, prev_action, cur_action):
    """Action interpolation"""
    steps = np.array(args.arm_steps_length + args.arm_steps_length)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    
    if step <= 1:
        return cur_action[np.newaxis, :]
    
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def signal_exit(*args):
    """Signal handler function"""
    global exit_flag
    exit_flag = True


def main():
    """Main function"""
    args = get_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Signal handling
    if args.use_keyboard_end:
        signal.signal(signal.SIGINT, signal_exit)
        signal.signal(signal.SIGTERM, signal_exit)
    
    # Initialize ROS operator
    ros_operator = RosOperator(args)
    
    # If only resetting arms
    if args.reset_only:
        print("Resetting arms...")
        ros_operator.reset_arms()
        print("Arms reset completed")
        return
    
    # Initialize video recorder (only combined view)
    combined_video_recorder = None
    
    if args.record_combined_video:
        combined_video_path = setup_video_saving(args)
        # Combined video uses fixed resolution 640x720 (actual combined image size)
        combined_video_recorder = VideoRecorder(combined_video_path, fps=args.video_fps, frame_size=(640, 720))
        combined_video_recorder.start_recording()

    # Initialize H-RDT inference
    hrdt_inference = HRDTInference(args)
    
    # Initialize action log file
    action_log_path = args.action_log_file
    # Clear or create log file
    with open(action_log_path, 'w') as f:
        f.write(f"Action log started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Format: Step <step_number>: [action_chunk_list]\n\n")
    
    # Initialize image save directory
    image_save_dir = args.image_save_dir
    os.makedirs(image_save_dir, exist_ok=True)
    print(f"Images will be saved to: {image_save_dir}")
    
    # Reset arms to initial position
    print("Resetting arms to initial position...")
    # ros_operator.reset_arms()
    
    if args.use_keyboard_end:
        input("Press Enter to start inference...")
    
    # Start inference loop
    print("Starting H-RDT inference...")
    
    rate = rospy.Rate(args.publish_rate)
    t = 0
    action_buffer = None
    prev_action = np.zeros(14)
    
    try:
        while not rospy.is_shutdown() and not exit_flag and t < args.max_publish_step:
            # Get current observation
            observation = ros_operator.get_current_observation()
            
            # Check if observation is valid
            if (observation['puppet_left'] is None or 
                observation['puppet_right'] is None or
                observation['images']['head_cam'] is None):
                print("Invalid observation, skipping...")
                rate.sleep()
                continue
            
            # Update observation cache
            hrdt_inference.update_obs(observation)
            
            # Create combined image for video recording (if all cameras available)
            combined_image = None
            if (observation['images']['head_cam'] is not None and
                observation['images']['left_cam'] is not None and
                observation['images']['right_cam'] is not None):
                combined_image = hrdt_inference.create_three_view_image(
                    observation['images']['head_cam'],
                    observation['images']['left_cam'], 
                    observation['images']['right_cam']
                )
            
            # Record combined video
            if combined_video_recorder and combined_video_recorder.recording and combined_image is not None:
                combined_video_recorder.add_frame(combined_image)

            # Perform new inference when reaching end of action chunk
            if t % args.chunk_size == 0:
                print(f"Performing inference at step {t}")
                action_buffer = hrdt_inference.predict_action(observation)
                
                # Write action chunk to file
                if action_buffer is not None:
                    write_action_to_file(action_buffer, t, action_log_path)
                    print(f"Action chunk saved to {action_log_path}")
                
                # Save combined image
                if combined_image is not None:
                    saved_path = save_combined_image(combined_image, t, image_save_dir)
                    if saved_path:
                        print(f"Combined image saved to {saved_path}")

            # Get current action from action buffer
            if action_buffer is not None:
                current_action = action_buffer[t % args.chunk_size]
                
                # Action interpolation
                if args.use_actions_interpolation:
                    interpolated_actions = interpolate_action(args, prev_action, current_action)
                else:
                    interpolated_actions = current_action[np.newaxis, :]
                
                # Execute interpolated actions
                for action in interpolated_actions:
                    left_action = action[:7]
                    right_action = action[7:14]
                    
                    ros_operator.publish_action(left_action, right_action)
                    rate.sleep()
                
                prev_action = current_action.copy()
                print(f"Executed action at step {t}")
            
            t += 1
            rate.sleep()
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        # Stop video recording
        if combined_video_recorder and combined_video_recorder.recording:
            print("Stopping combined video recording...")
            combined_video_recorder.stop_recording()
        
        # Reset arms after inference completion
        print("Inference completed, resetting arms...")
        ros_operator.reset_arms()
        print("H-RDT inference finished")


if __name__ == '__main__':
    main()