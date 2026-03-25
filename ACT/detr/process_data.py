import os
import h5py
import numpy as np
import cv2
import argparse
import json

from XPolicyLab.utils.load_file import load_hdf5, load_yaml
from XPolicyLab.utils.process_data import pack_robot_state, get_robot_action_dim_info, decode_image_bit

def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def data_transform(path, episode_num, load_data_dir, save_dir, robot_action_dim_info):
    begin = 0
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for current_episode in range(episode_num):
        load_path = os.path.join(load_data_dir, f"data/episode_{current_episode:07d}.hdf5")
        data = load_hdf5(load_path)
        state_all = pack_robot_state(data, action_type, robot_action_dim_info)
        
        qpos = []
        actions = []
        cam_head = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        last_state = None
        for j in range(0, state_all.shape[0]):
            
            state = state_all[j]

            if j != state.shape[0] - 1:

                state = state.astype(np.float32)
                qpos.append(state)

                cam_head_bit = data['vision']["cam_head"]['color'][j]
                cam_head = decode_image_bit(cam_head_bit)
                cam_head_resized = cv2.resize(cam_head, (640, 480))
                cam_head.append(cam_head_resized)

                camera_right_wrist_bit = data['vision']["cam_head"]["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bit, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bit = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bit, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0:
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        hdf5path = os.path.join(save_dir, f"episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            # cam_head_enc, len_high = images_encoding(cam_head)
            # cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            # cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_head", data=np.stack(cam_head), dtype=np.uint8)
            image.create_dataset("cam_right_wrist", data=np.stack(cam_right_wrist), dtype=np.uint8)
            image.create_dataset("cam_left_wrist", data=np.stack(cam_left_wrist), dtype=np.uint8)

        begin += 1
        print(f"ACT: proccess episode {i} success!", end='\r')

    return begin

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument("task_name", type=str, help="The name of the task (e.g., beat_block_hammer)",)
    parser.add_argument("env_cfg", type=str, help="The name of the environment config",)
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to process (e.g., 50)",)
    parser.add_argument("action_type", type=str, help="The type of action to process (e.g., joint)",)
    args = parser.parse_args()

    task_name = args.task_name
    env_cfg_name = args.env_cfg
    expert_data_num = args.expert_data_num
    action_type = args.action_type

    save_dir = f"processed_data/{task_name}/{env_cfg_name}-{expert_data_num}-{action_type}"

    load_data_dir = os.path.join("../../data", str(task_name), str(env_cfg_name))
    env_cfg_file = os.path.join("../../env_cfg", f"{env_cfg_name}.yml")
    env_cfg = load_yaml(env_cfg_file)

    robot_action_dim_info = get_robot_action_dim_info(env_cfg)

    begin = data_transform(os.path.join("../../data/", task_name, env_cfg_name, 'data'), expert_data_num, load_data_dir, save_dir, robot_action_dim_info)

    SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"

    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    SIM_TASK_CONFIGS[f"{task_name}-{env_cfg_name}-{expert_data_num}-{action_type}"] = {
        "dataset_dir": save_dir,
        "num_episodes": expert_data_num,
        "episode_len": 5000,
        "camera_names": ["cam_head", "cam_right_wrist", "cam_left_wrist"],
    }

    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(SIM_TASK_CONFIGS, f, indent=4)
