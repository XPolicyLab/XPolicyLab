import os
import numpy as np
import zarr
import shutil
import argparse
import cv2
from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import pack_robot_state, get_robot_action_dim_info, decode_image_bit

def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument("bench_name", type=str, help="The name of the benchmark (e.g., RoboDojo)",)
    parser.add_argument("task_name", type=str, help="The name of the task (e.g., beat_block_hammer)",)
    parser.add_argument("env_cfg_type", type=str, help="The name of the environment config",)
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to process (e.g., 50)",)
    parser.add_argument("action_type", type=str, help="The type of action to process (e.g., joint)",)
    args = parser.parse_args()

    bench_name = args.bench_name
    task_name = args.task_name
    env_cfg_type = args.env_cfg_type
    expert_data_num = args.expert_data_num
    action_type = args.action_type
    load_data_dir = os.path.join("../../../data", str(bench_name), str(task_name), str(env_cfg_type))

    robot_action_dim_info = get_robot_action_dim_info(env_cfg_type)

    frame_count = 0

    save_dir = f"./data/{bench_name}-{task_name}-{env_cfg_type}-{expert_data_num}-{action_type}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_episode = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    head_camera_arrays, left_camera_arrays, right_camera_arrays = [], [], []
    episode_ends_arrays, action_arrays, state_arrays = ([], [], [],)

    while current_episode < expert_data_num:
        print(f"DP: processing episode: {current_episode + 1} / {expert_data_num}", end="\r")

        load_path = os.path.join(load_data_dir, f"data/episode_{current_episode:07d}.hdf5")
        data = load_hdf5(load_path)
        
        state_all = pack_robot_state(data, action_type, robot_action_dim_info, source_type="dataset", state_type="state")
        action_all = pack_robot_state(data, action_type, robot_action_dim_info, source_type="dataset", state_type="action")

        for j in range(0, state_all.shape[0]):
            head_img_bit = data['vision']['cam_head']['colors'][j]
            left_img_bit = data['vision']['cam_left_wrist']['colors'][j]
            right_img_bit = data['vision']['cam_right_wrist']['colors'][j]

            state, action = state_all[j], action_all[j]

            head_img = decode_image_bit(head_img_bit)
            assert head_img.ndim == 3 and head_img.shape[-1] == 3, f"Expected HxWx3, got {head_img.shape}"
            head_img = cv2.resize(head_img, (320, 240), interpolation=cv2.INTER_AREA)  # (W, H)
            assert head_img.shape == (240, 320, 3)

            left_img = decode_image_bit(left_img_bit)
            assert left_img.ndim == 3 and left_img.shape[-1] == 3, f"Expected HxWx3, got {left_img.shape}"
            left_img = cv2.resize(left_img, (320, 240), interpolation=cv2.INTER_AREA)  # (W, H)
            assert left_img.shape == (240, 320, 3)

            right_img = decode_image_bit(right_img_bit)
            assert right_img.ndim == 3 and right_img.shape[-1] == 3, f"Expected HxWx3, got {right_img.shape}"
            right_img = cv2.resize(right_img, (320, 240), interpolation=cv2.INTER_AREA)  # (W, H)
            assert right_img.shape == (240, 320, 3)

            head_camera_arrays.append(head_img)
            left_camera_arrays.append(left_img)
            right_camera_arrays.append(right_img)
            state_arrays.append(state)
            action_arrays.append(action)

        current_episode += 1
        frame_count += state_all.shape[0]
        episode_ends_arrays.append(frame_count)

    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    action_arrays = np.array(action_arrays)
    state_arrays = np.array(state_arrays)

    head_camera_arrays = np.array(head_camera_arrays)
    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW

    left_camera_arrays = np.array(left_camera_arrays)
    left_camera_arrays = np.moveaxis(left_camera_arrays, -1, 1)  # NHWC -> NCHW

    right_camera_arrays = np.array(right_camera_arrays)
    right_camera_arrays = np.moveaxis(right_camera_arrays, -1, 1)  # NHWC -> NCHW

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    action_chunk_size = (100, action_arrays.shape[1])
    state_chunk_size = (100, state_arrays.shape[1])
    head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])
    left_camera_chunk_size = (100, *left_camera_arrays.shape[1:])
    right_camera_chunk_size = (100, *right_camera_arrays.shape[1:])

    zarr_data.create_dataset("head_camera", data=head_camera_arrays, chunks=head_camera_chunk_size, overwrite=True, compressor=compressor,)
    zarr_data.create_dataset("left_camera", data=left_camera_arrays, chunks=left_camera_chunk_size, overwrite=True, compressor=compressor,)
    zarr_data.create_dataset("right_camera", data=right_camera_arrays, chunks=right_camera_chunk_size, overwrite=True, compressor=compressor,)
    zarr_data.create_dataset("state", data=state_arrays, chunks=state_chunk_size, dtype="float32", overwrite=True, compressor=compressor,)
    zarr_data.create_dataset("action", data=action_arrays, chunks=action_chunk_size,dtype="float32", overwrite=True, compressor=compressor,)
    zarr_meta.create_dataset("episode_ends", data=episode_ends_arrays, dtype="int64", overwrite=True, compressor=compressor,)

if __name__ == "__main__":
    main()
