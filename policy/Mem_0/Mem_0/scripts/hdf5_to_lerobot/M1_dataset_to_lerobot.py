import os
import h5py
import cv2
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

workspace = os.path.dirname(os.path.abspath(__file__))
RMBench_workspace = os.path.join(workspace, "..", "..", "..", "..")
Mem0_workspace = os.path.join(workspace, "..", "..")

# Define task names to process
TASK_NAMES = [
    "observe_and_pickup",
    "put_back_block",
    "rearrange_blocks",
    "swap_blocks",
    "swap_T"
]

# Define number of episodes to process
episode_num = 50

TASK_INSTRUCTIONS = {
    "observe_and_pickup": "Initially, there is one target object on the shelf and five random objects on the table. Then, a screen obscures the target object. Pick up the corresponding target object from the table and lift it up.",
    "put_back_block": "There are four mats, one block, and a button on the table. One block is on one of the mats. First, put the block to the center, then press the button. Then, put the block back in its original position.",
    "rearrange_blocks": "Move the block between the two mats onto the empty mat, press the button, then move the other block (the one that started on a mat) to the space between the two mats.",
    "swap_blocks": "There are three traies on the table, and two blocks are placed in two different traies. You may move only one block at a time, and each tray can hold at most one block. Swap the positions of the two blocks. Finally press the button.",
    "swap_T": "Swap the poses of the two T-blocks, including both position and orientation.",
}

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
    "subtask_end": {
        "dtype": "int32",
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

for dataset_name in TASK_NAMES:
    # Create Lerobot dataset
    lerobot_dataset_name = f"{dataset_name}"
    dataset = LeRobotDataset.create(
        repo_id=lerobot_dataset_name,
        fps=30,
        features=features,
        root=Path(f"{Mem0_workspace}/lerobot_datasets/{lerobot_dataset_name}"),
        use_videos=True,
    )
    
    print(f"\n{'='*60}")
    print(f"Processing task: {dataset_name}")
    print(f"{'='*60}")
    
    # User-defined subtask (same subtask used for the entire episode)
    subtask_text = TASK_INSTRUCTIONS[dataset_name]

    # Process each episode
    for episode_idx in range(episode_num):
        episode_key = f"episode_{episode_idx}"
        
        # Read hdf5 file
        hdf5_path = f"{RMBench_workspace}/data/{dataset_name}/demo_clean/data/episode{episode_idx}.hdf5"
        
        try:
            with h5py.File(hdf5_path, "r") as f:
                # Get episode length
                episode_length = len(f["joint_action"]["left_arm"])
                
                # Pre-read all frame states to calculate actions (current frame's state is used as previous frame's action)
                states = []
                images = []
                
                for frame_idx in range(episode_length):
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
                for frame_idx in range(episode_length):
                    # Current frame's state
                    current_state = states[frame_idx]
                    
                    # Current frame's action = next frame's state (if last frame, repeat itself)
                    if frame_idx < episode_length - 1:
                        current_action = states[frame_idx + 1]
                    else:
                        current_action = states[frame_idx]  # Last frame's action repeats its own state
                    
                    # Same subtask used for the entire episode
                    # Set subtask_end signal only in the last 8 frames before episode end
                    if episode_length - frame_idx <= 8:
                        subtask_end = 1
                    else:
                        subtask_end = 0
                    
                    # Build dictionary for current frame
                    # Note: According to features definition, episode_id and subtask_end need to be numpy arrays with shape (1,)
                    # Although subtask has shape (1,), the validation function expects a string type directly
                    frame_data = {
                        "observation.state": current_state,
                        "action": current_action,
                        "observation.image.head_camera": images[frame_idx],
                        "subtask": subtask_text,  # Same subtask used for the entire episode
                        "subtask_end": np.array([subtask_end], dtype=np.int32),  # shape: (1,)
                        "episode_id": np.array([episode_idx], dtype=np.int32),  # shape: (1,)
                    }
                    
                    # Add current frame using add_frame, with task as keyword argument
                    dataset.add_frame(frame_data, task=dataset_name)
            
            # Save when episode ends
            dataset.save_episode()
            print(f"  ✓ {dataset_name}/{episode_key} saved, {episode_length} frames")
            
            total_episodes += 1
            total_frames += episode_length
            
        except FileNotFoundError:
            print(f"  ⚠️  Skipping {dataset_name}/{episode_key}: file not found")
        except Exception as e:
            print(f"  ✗ Error: {dataset_name}/{episode_key}: {e}")
            import traceback
            traceback.print_exc()

print(f"\n{'='*60}")
print(f"All tasks processed!")
print(f"Total episodes: {total_episodes}")
print(f"Total frames: {total_frames}")
print(f"{'='*60}")
