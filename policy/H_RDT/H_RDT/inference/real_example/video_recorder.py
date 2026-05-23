#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import time


class VideoRecorder:
    """Video recorder class for saving inference videos"""
    
    def __init__(self, video_path, fps=30, frame_size=(1280, 720)):
        self.video_path = video_path
        self.fps = fps
        self.frame_size = frame_size
        self.video_writer = None
        self.frames = []
        self.recording = False
        
    def start_recording(self):
        """Start recording"""
        self.recording = True
        self.frames = []
        print(f"Started recording video to: {self.video_path}")
        print(f"Video resolution: {self.frame_size[0]}x{self.frame_size[1]}")
        
    def add_frame(self, frame):
        """Add frame to recording buffer"""
        if self.recording and frame is not None:
            # Resize frame
            resized_frame = cv2.resize(frame, self.frame_size)
            self.frames.append(resized_frame)
    
    def stop_recording(self):
        """Stop recording and save video"""
        if not self.recording:
            return
            
        self.recording = False
        
        if len(self.frames) == 0:
            print("No frames to save")
            return
        
        try:
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_path, fourcc, self.fps, self.frame_size)
            
            # Write all frames
            for frame in self.frames:
                self.video_writer.write(frame)
            
            # Release resources
            self.video_writer.release()
            
            print(f"Video saved successfully: {self.video_path}")
            print(f"Total frames: {len(self.frames)}")
            print(f"Duration: {len(self.frames) / self.fps:.2f} seconds")
            print(f"Resolution: {self.frame_size[0]}x{self.frame_size[1]}")
            
        except Exception as e:
            print(f"Error saving video: {e}")
        finally:
            if self.video_writer:
                self.video_writer.release()
            self.frames = []


def setup_video_saving(args):
    """Setup video saving paths and parameters"""
    # Parse model type from pretrained model path
    def parse_model_type(pretrained_model_path):
        if 'aloha_human' in pretrained_model_path:
            return 'human'
        elif 'aloha_scratch' in pretrained_model_path:
            return 'scratch'
        else:
            return 'unknown'

    # Parse task name from language embeddings path
    def parse_task_name(lang_embeddings_path):
        try:
            # Extract task name from path
            parent_dir = os.path.dirname(lang_embeddings_path)
            task_name = os.path.basename(parent_dir)
            
            # If task name is empty or generic directory name, try extracting from filename
            if not task_name or task_name in ['data', 'embeddings', 'lang_embeddings']:
                filename = os.path.basename(lang_embeddings_path)
                if filename.endswith('_lang.pt'):
                    task_name = filename[:-8]  # Remove '_lang.pt'
                else:
                    task_name = os.path.splitext(filename)[0]
            
            return task_name
        except Exception as e:
            print(f"Error parsing task name: {e}")
            return 'unknown_task'
    
    # Parse model type and task name
    model_type = parse_model_type(args.pretrained_model_path)
    task_name = parse_task_name(args.lang_embeddings_path)
    
    # Create video save directory
    video_dir = os.path.join(args.video_save_dir, model_type, task_name)
    os.makedirs(video_dir, exist_ok=True)
    
    # Generate video filename with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    combined_video_filename = f"combined_view_{timestamp}.mp4"
    combined_video_path = os.path.join(video_dir, combined_video_filename)
    
    print(f"Model type: {model_type}")
    print(f"Task name: {task_name}")
    print(f"Combined video will be saved to: {combined_video_path}")
    
    return combined_video_path


def parse_video_resolution(resolution_str):
    """Parse video resolution string"""
    try:
        width, height = map(int, resolution_str.split('x'))
        return (width, height)
    except:
        print(f"Invalid resolution format: {resolution_str}, using default 1280x720")
        return (1280, 720) 