import os
import h5py
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from lerobot.datasets.lerobot_dataset import LeRobotDataset

workspace = os.path.dirname(os.path.abspath(__file__))
RMBench_workspace = os.path.join(workspace, "..", "..", "..", "..")
Mem0_workspace = os.path.join(workspace, "..", "..")

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not installed, use simple progress display
    def tqdm(iterable, desc="", total=None, leave=True):
        if desc:
            print(f"{desc}: Processing...")
        return iterable

TASK_NAMES = [
    "battery_try",
    "blocks_ranking_try",
    "cover_blocks",
    "press_button",
    "place_block_mat",
]

# Define number of episodes to process
episode_num = 50

TASK_INSTRUCTIONS = {
    "battery_try": "There are two batteries and a battery slot on the table. Combining the two batteries in different orientations causes the dashboard needle to rotate.",
    "blocks_ranking_try": "There is a button and three colored cubes arranged in a random row on the table. Each time the cubes are rearranged, the arm presses the button until the arrangement is successful.",
    "cover_blocks": "On the table, red, green, and blue blocks are arranged randomly along with three lids. From the current viewpoint, cover the blocks from right to left using the lids, and then uncover them again in the sequence red, green, and blue.",
    "press_button": "Observe the two numbers on the table. Press the left button the number of times corresponding to the number on the left, and press the middle button the number of times corresponding to the number on the right. Then press the right button once to confirm.",
    "place_block_mat": "Pick up the blocks from the blue mat and place them on the green mat, then put them back on the original mat, starting from left to right.",
}

lerobot_dataset_name = "rmbench_data_cover_blocks"

features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (16,),
        "names": ["left_joint_0","left_joint_1","left_joint_2","left_joint_3","left_joint_4","left_joint_5","left_joint_6","right_joint_0","right_joint_1","right_joint_2","right_joint_3","right_joint_4","right_joint_5","right_joint_6","left_gripper","right_gripper"],
    },
    "action": {
        "dtype": "float32",
        "shape": (16,),
        "names": ["left_joint_0","left_joint_1","left_joint_2","left_joint_3","left_joint_4","left_joint_5","left_joint_6","right_joint_0","right_joint_1","right_joint_2","right_joint_3","right_joint_4","right_joint_5","right_joint_6","left_gripper","right_gripper"],
    },
    "observation.image.head_camera": {
        "dtype": "video",
        "shape": (240, 320, 3),
        "names": ["height", "width", "channels"],
    },
    "subtask": {
        "dtype": "string",
        "shape": (1,),
        "names": ["subtask_annotation"],
    },
    "global_task": {
        "dtype": "string",
        "shape": (1,),
        "names": ["global_task_annotation"],
    },
    "subtask_end": {
        "dtype": "bool",
        "shape": (1,),
        "names": ["subtask_end_flag"],
    },
    "episode_id": {
        "dtype": "int32",
        "shape": (1,),
        "names": ["episode_id"],
    },
}

# Statistics
total_episodes = 0
total_frames = 0

# Iterate through all tasks
task_pbar = tqdm(TASK_NAMES, desc="Overall progress", leave=True)
for dataset_name in task_pbar:
    # Create Lerobot dataset
    lerobot_dataset_name = f"{dataset_name}"
    dataset = LeRobotDataset.create(
        repo_id=lerobot_dataset_name,
        fps=30,
        features=features,
        root=Path(f"{Mem0_workspace}/lerobot_datasets/{lerobot_dataset_name}"),
        use_videos=True,
    )
    
    # Set task progress description
    task_pbar.set_description(f"Processing task: {dataset_name}")
    
    # Get global task instruction for current task
    global_task_text = TASK_INSTRUCTIONS.get(dataset_name, "")
    
    # Read language annotation file for current task
    annotation_path = Path(f"{RMBench_workspace}/data/{dataset_name}/demo_clean/language_annotation.json")
    language_annotations = {}
    
    if annotation_path.exists():
        try:
            with open(annotation_path, "r", encoding="utf-8") as f:
                language_annotations = json.load(f)
        except Exception as e:
            print(f"  ⚠️  Failed to load annotation file for {dataset_name}: {e}")
    else:
        print(f"  ⚠️  Annotation file for {dataset_name} does not exist: {annotation_path}")
    
    # Process each episode
    episode_iter = tqdm(range(episode_num), desc=f"  {dataset_name}", leave=False, unit="episode")
    for episode_idx in episode_iter:
        episode_key = f"episode_{episode_idx}"
        
        # Read hdf5 file
        hdf5_path = f"{RMBench_workspace}/data/{dataset_name}/demo_clean/data/episode{episode_idx}.hdf5"
        
        try:
            with h5py.File(hdf5_path, "r") as f:
                # Get episode length
                episode_length = len(f["joint_action"]["left_arm"])
                
                # Process language annotations (if they exist)
                subtask_boundaries = []
                if episode_key in language_annotations:
                    # Get task annotations for current episode
                    episode_annotations = language_annotations[episode_key]
                    
                    # Calculate boundary range for each subtask [start_idx, end_idx, subtask_text]
                    # Do not merge adjacent identical subtasks, maintain independence of original annotations
                    current_idx = 0
                    for subtask_info in episode_annotations:
                        subtask_text = subtask_info[0]
                        duration = subtask_info[1]
                        
                        start_idx = current_idx
                        end_idx = current_idx + duration - 1
                        subtask_boundaries.append([start_idx, end_idx, subtask_text])
                        current_idx = end_idx + 1
                
                # Pre-read all frame states to calculate actions (current frame's state is used as previous frame's action)
                states = []
                images = []
                
                frame_load_iter = tqdm(range(episode_length), desc=f"    Loading data", leave=False, 
                                      disable=(episode_length < 200), unit="frame")
                for frame_idx in frame_load_iter:
                    # Get image
                    image_bits = f["observation"]["head_camera"]["rgb"][frame_idx]
                    image_rgb = cv2.imdecode(np.frombuffer(image_bits, np.uint8), cv2.IMREAD_COLOR)
                    images.append(image_rgb)
                    
                    # Get joint states
                    left_arm = f["joint_action"]["left_arm"][frame_idx]  # shape: (6,)
                    right_arm = f["joint_action"]["right_arm"][frame_idx]  # shape: (6,)
                    left_gripper = f["joint_action"]["left_gripper"][frame_idx]  # shape: ()
                    right_gripper = f["joint_action"]["right_gripper"][frame_idx]  # shape: ()
                    
                    # Concatenate into observation.state (16,)
                    observation_state = np.concatenate([
                        left_arm,
                        [0.0],
                        right_arm,
                        [0.0],
                        np.array([left_gripper]),
                        np.array([right_gripper])
                    ]).astype(np.float32)
                    
                    states.append(observation_state)
                
                # Iterate through each frame and add using add_frame
                frame_add_iter = tqdm(range(episode_length), desc=f"    Adding frames", leave=False, 
                                      disable=(episode_length < 200), unit="frame")
                for frame_idx in frame_add_iter:
                    # Current frame's state
                    current_state = states[frame_idx]
                    
                    # Current frame's action = next frame's state (if last frame, repeat itself)
                    if frame_idx < episode_length - 1:
                        current_action = states[frame_idx + 1]
                    else:
                        current_action = states[frame_idx]  # Last frame's action repeats its own state
                    
                    # Get subtask corresponding to current frame
                    current_subtask = ""
                    subtask_end = False
                    
                    # Find which subtask the current frame belongs to (if annotations exist)
                    if subtask_boundaries:
                        for start_idx, end_idx, subtask_text in subtask_boundaries:
                            if start_idx <= frame_idx <= end_idx:
                                current_subtask = subtask_text
                                # Check if close to subtask boundary (distance to boundary <= 8)
                                distance_to_end = end_idx - frame_idx
                                if distance_to_end < 8:
                                    subtask_end = True
                                break
                    
                    # Build dictionary for current frame
                    # Note: According to features definition, episode_id and subtask_end need to be numpy arrays with shape (1,)
                    # Although subtask and global_task have shape (1,), the validation function expects string types directly
                    frame_data = {
                        "observation.state": current_state,
                        "action": current_action,
                        "observation.image.head_camera": images[frame_idx],
                        "subtask": current_subtask,  # Use string directly, validation function will handle shape (1,)
                        "global_task": global_task_text,  # Fill in global task instruction
                        "subtask_end": np.array([subtask_end], dtype=bool),  # shape: (1,)
                        "episode_id": np.array([episode_idx], dtype=np.int32),  # shape: (1,)
                    }
                    
                    # Add current frame using add_frame, with task as keyword argument
                    dataset.add_frame(frame_data, task=dataset_name)
            
            # Save when episode ends
            dataset.save_episode()
            episode_iter.set_postfix({"frames": episode_length, "total": f"{total_episodes+1}/{len(TASK_NAMES)*episode_num}"})
            total_episodes += 1
            total_frames += episode_length
            
        except FileNotFoundError:
            episode_iter.set_postfix_str(f"⚠️  File not found")
        except Exception as e:
            episode_iter.set_postfix_str(f"✗ Error: {str(e)[:30]}")
            import traceback
            traceback.print_exc()

print(f"\n{'='*60}")
print(f"All tasks processed!")
print(f"Total episodes: {total_episodes}")
print(f"Total frames: {total_frames}")
print(f"{'='*60}")
