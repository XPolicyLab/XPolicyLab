import argparse
import cv2
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

workspace = os.path.dirname(os.path.abspath(__file__))
Mem0_workspace = os.path.join(workspace, "..", "..")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare LLaMA-Factory format data from LeRobot dataset.")
    parser.add_argument(
        "--lerobot_dataset_path",
        type=str,
        default=os.path.join(Mem0_workspace, "lerobot_datasets", "battery_try"),
        help="Path to the LeRobot dataset.",
    )
    parser.add_argument("--episode_start_id", type=int, default=0, help="Start episode id (inclusive).")
    parser.add_argument("--episode_end_id", type=int, default=50, help="End episode id (exclusive).")
    return parser.parse_args()


# parameter need to be set (overridden by CLI args when run from automation script)
_args = parse_args()
lerobot_dataset_path = _args.lerobot_dataset_path
episode_start_id = _args.episode_start_id
episode_end_id = _args.episode_end_id

def save_image(img, path):
    # save image
    img = img.numpy()
    # Convert image format (C, H, W) -> (H, W, C)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    # Convert to uint8 and create PIL Image
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
    image = Image.fromarray(img, mode='RGB')
    
    image.save(path)

system_prompt = "You are a robotic assistant specialized in subtask planning. I will provide you with: 1. global_task: A global task instruction. 2. initial_observation: An image of the initial observation of the task. 3. finished_subtasks: A list of subtask instructions completed by the robot. Each instruction is paired with an image showing the visual observation at the end of that subtask. The indices (0, 1, 2, ...) represent the temporal order of completion, where 0 is the first completed subtask, 1 is the second, and so on.\nFormat: <global_task>: {global task instruction}. <initial_observation>: {initial observation image}. <finished_subtasks>: 0: {operation arm: finished subtask instruction}, the corresponding image is {image}. 1: {operation arm: finished subtask instruction}, the corresponding image is {image}. ...\nIMPORTANT: The numbers (0, 1, 2, ...) indicate the temporal sequence of completion. The highest index represents the most recently completed subtask. At the beginning of the task, the finished_subtasks list is null.\nBased on all the provided information, output the next subtask to execute in the format: 'next_subtask: {subtask name}.'.\n"

llamafactory_dataset_name = lerobot_dataset_path.split("/")[-1]

# set initial dataset list
high_level_finetune_data = []

# create folder
finetune_dataset_path = f"{Mem0_workspace}/llamafactory_data/{llamafactory_dataset_name}"
os.makedirs(finetune_dataset_path, exist_ok=True)

total_episodes = episode_end_id - episode_start_id
print(f"\n[LlamaFactory Data Preparation] Total episodes to process: {total_episodes} (episode_id {episode_start_id} ~ {episode_end_id - 1})\n", flush=True)

for idx, episode_id in enumerate(range(episode_start_id, episode_end_id), start=1):
    print(f"[ {idx}/{total_episodes} ] Processing episode {episode_id} ...", flush=True)
    # create episode folder
    episode_folder_path = f"{finetune_dataset_path}/{llamafactory_dataset_name}_images/episode_{episode_id}"
    os.makedirs(episode_folder_path, exist_ok=True)
    # images list
    images_list = []
    # finished subtasks list
    finished_subtasks_list = []

    dataset = LeRobotDataset(lerobot_dataset_path, video_backend="pyav", episodes=[episode_id])
    
    episode_length = len(dataset)
    
    initial_observation = dataset[0]['observation.image.head_camera']
    initial_observation_path = f"{episode_folder_path}/initial_observation.png"
    save_image(initial_observation, initial_observation_path)
    images_list.append(f"{llamafactory_dataset_name}_images/episode_{episode_id}/initial_observation.png")
    
    key_frame_id = 1
    
    for frame_id in tqdm(range(episode_length), desc=f"Episode {idx}/{total_episodes} (id={episode_id}) frames"):
        if frame_id == 0:
            frame_information = {"messages": [], "images": []}
            # system prompt message
            system_prompt_msg = {"role": "system", "content": system_prompt}
            frame_information["messages"].append(system_prompt_msg)
            # user prompt message
            global_task_txt = "<global_task>: " + dataset[frame_id]['global_task'] + "\n"
            initial_observation = "<initial_observation>: <image>.\n"
            finished_subtasks = "<finished_subtasks>: null.\n"
            combined_message = global_task_txt + initial_observation + finished_subtasks
            user_prompt_msg = {"role": "user", "content": combined_message}
            frame_information["messages"].append(user_prompt_msg)
            # assistant prompt message
            subtask_name = dataset[frame_id]['subtask']
            finished_subtasks_list.append(f"{subtask_name}")
            assistant_prompt_msg = {"role": "assistant", "content": f"next_subtask: {subtask_name}"}
            frame_information["messages"].append(assistant_prompt_msg)
            
            # add image
            frame_information["images"] = images_list.copy()
            
            high_level_finetune_data.append(frame_information.copy())
        else:
            if frame_id == episode_length - 1:
                continue
            elif (dataset[frame_id]['subtask_end'] == 1 and dataset[frame_id+1]['subtask_end'] == 0):
                # save image first
                image = dataset[frame_id]['observation.image.head_camera']
                image_path = f"{episode_folder_path}/frame_{key_frame_id}.png"
                save_image(image, image_path)
                images_list.append(f"{llamafactory_dataset_name}_images/episode_{episode_id}/frame_{key_frame_id}.png")
                key_frame_id += 1
                
                # prepare frame information
                frame_information = {"messages": [], "images": []}
                # system prompt message
                system_prompt_msg = {"role": "system", "content": system_prompt}
                frame_information["messages"].append(system_prompt_msg)
                # user prompt message
                global_task_txt = "<global_task>: " + dataset[frame_id]['global_task'] + "\n"
                initial_observation = "<initial_observation>: <image>.\n"
                finished_subtasks = "<finished_subtasks>: "
                for i in range(len(finished_subtasks_list)):
                    if i == len(finished_subtasks_list) - 1:
                        finished_subtasks += f"{i}: {finished_subtasks_list[i]} The corresponding image is: <image>.\n"
                    else:
                        finished_subtasks += f"{i}: {finished_subtasks_list[i]} The corresponding image is: <image>. "
                combined_message = global_task_txt + initial_observation + finished_subtasks
                user_prompt_msg = {"role": "user", "content": combined_message}
                frame_information["messages"].append(user_prompt_msg)
                # assistant prompt message
                subtask_name = dataset[frame_id+1]['subtask']
                finished_subtasks_list.append(f"{subtask_name}")
                assistant_prompt_msg = {"role": "assistant", "content": f"next_subtask: {subtask_name}"}
                frame_information["messages"].append(assistant_prompt_msg)
                # add image
                frame_information["images"] = images_list.copy()
                high_level_finetune_data.append(frame_information.copy())

        
# save high_level_finetune_data to json file
with open(f"{finetune_dataset_path}/{llamafactory_dataset_name}_high_level_finetune_data.json", "w") as f:
    json.dump(high_level_finetune_data, f, indent=2, ensure_ascii=False)

print(f"\n[LlamaFactory Data Preparation] Done. Processed {total_episodes} episodes (episode_id {episode_start_id} ~ {episode_end_id - 1}).", flush=True)