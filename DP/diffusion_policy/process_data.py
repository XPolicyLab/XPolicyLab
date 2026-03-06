import os
import h5py
import numpy as np
import zarr
import shutil
import argparse
import cv2
import h5py

def load_hdf5(path: str) -> dict:

    def _read(obj):
        # Dataset -> numpy / scalar / bytes->str
        if isinstance(obj, h5py.Dataset):
            v = obj[()]
            if isinstance(v, (bytes, bytearray)):
                return v.decode("utf-8", errors="replace")
            # numpy scalar -> python scalar
            try:
                return v.item()
            except Exception:
                return v

        # Group -> dict
        out = {}
        for k, v in obj.items():
            out[k] = _read(v)
        return out

    with h5py.File(path, "r") as f:
        data = _read(f)

        if len(f.attrs) > 0:
            data["_attrs"] = {k: f.attrs[k] for k in f.attrs.keys()}

        return data

def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument("task_name", type=str, help="The name of the task (e.g., beat_block_hammer)",)
    parser.add_argument("env_cfg", type=str, help="The name of the environment config",)
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to process (e.g., 50)",)
    parser.add_argument("action_type", type=str, help="The type of action to process (e.g., joint)",)
    args = parser.parse_args()

    task_name = args.task_name
    env_cfg = args.env_cfg
    expert_data_num = args.expert_data_num
    action_type = args.action_type

    load_dir = os.path.join("../../data", str(task_name), str(env_cfg))

    frame_count = 0

    save_dir = f"./data/{task_name}-{env_cfg}-{expert_data_num}-{action_type}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_episode = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    head_camera_arrays, left_camera_arrays, right_camera_arrays = ([], [], [],)
    episode_ends_arrays, action_arrays, state_arrays = ([], [], [],)

    while current_episode < expert_data_num:
        print(f"Processing episode: {current_episode + 1} / {expert_data_num}", end="\r")

        load_path = os.path.join(load_dir, f"data/episode_{current_episode:07d}.hdf5")
        data = load_hdf5(load_path)
        
        if action_type == 'joint':
            if "joint_states" in data['state'].keys(): # single arm
                joint_states = data['state']["joint_states"]
                ee_joint_states = data['state']["ee_joint_states"]
                state = np.concatenate([joint_states, ee_joint_states], axis=-1)
            else:
                assert "left_arm_joint_states" in data['state'].keys() and "right_arm_joint_states" in data['state'].keys(), "Expected joint states for both arms in the dataset."
                left_arm_joint_states = data['state']["left_arm_joint_states"]
                right_arm_joint_states = data['state']["right_arm_joint_states"]
                left_ee_joint_states = data['state']["left_ee_joint_states"]
                right_ee_joint_states = data['state']["right_ee_joint_states"]
                state = np.concatenate([left_arm_joint_states, left_ee_joint_states, right_arm_joint_states, right_ee_joint_states], axis=-1)

        elif action_type == 'ee':
            if "ee_poses" in data['state'].keys(): # single arm
                ee_poses = data['state']["ee_poses"]
                ee_joint_states = data['state']["ee_joint_states"]
                state = np.concatenate([ee_poses, ee_joint_states], axis=-1)
            else:
                assert "left_ee_poses" in data['state'].keys() and "right_ee_poses" in data['state'].keys(), "Expected ee poses for both arms in the dataset."
                left_ee_poses = data['state']["left_ee_poses"]
                right_ee_poses = data['state']["right_ee_poses"]
                left_ee_joint_states = data['state']["left_ee_joint_states"]
                right_ee_joint_states = data['state']["right_ee_joint_states"]
                state = np.concatenate([left_ee_poses, left_ee_joint_states, right_ee_poses, right_ee_joint_states], axis=-1)
        else:
            raise ValueError(f"Unsupported action type: {action_type}. Supported types are 'joint' and 'ee'.")

        for j in range(0, state.shape[0]):
            head_img_bit = data['vision']['cam_head']['colors'][j]

            if j != state.shape[0] - 1:

                head_img = cv2.imdecode(np.frombuffer(head_img_bit, np.uint8), cv2.IMREAD_COLOR)
                assert head_img.ndim == 3 and head_img.shape[-1] == 3, f"Expected HxWx3, got {head_img.shape}"
                head_img = cv2.resize(head_img, (320, 240), interpolation=cv2.INTER_AREA)  # (W, H)
                assert head_img.shape == (240, 320, 3)

                head_camera_arrays.append(head_img)
                state_arrays.append(state[j])
            if j != 0:
                action_arrays.append(state[j])

        current_episode += 1
        frame_count += state.shape[0] - 1
        episode_ends_arrays.append(frame_count)

    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    action_arrays = np.array(action_arrays)
    state_arrays = np.array(state_arrays)

    head_camera_arrays = np.array(head_camera_arrays)
    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    action_chunk_size = (100, action_arrays.shape[1])
    state_chunk_size = (100, state_arrays.shape[1])
    head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])

    zarr_data.create_dataset("head_camera", data=head_camera_arrays, chunks=head_camera_chunk_size, overwrite=True, compressor=compressor,)
    zarr_data.create_dataset("state", data=state_arrays, chunks=state_chunk_size, dtype="float32", overwrite=True, compressor=compressor,)
    zarr_data.create_dataset("action", data=action_arrays, chunks=action_chunk_size,dtype="float32", overwrite=True, compressor=compressor,)
    zarr_meta.create_dataset("episode_ends", data=episode_ends_arrays, dtype="int64", overwrite=True, compressor=compressor,)

if __name__ == "__main__":
    main()
