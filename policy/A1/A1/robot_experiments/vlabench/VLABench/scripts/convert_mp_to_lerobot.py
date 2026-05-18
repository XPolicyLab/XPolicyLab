from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import h5py
import os
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

def process_ee_state(ee_state):
    ee_pos, ee_quat, gripper = ee_state[:3], ee_state[3:7], ee_state[-1]
    ee_euler = quat2euler(ee_quat)
    ee_pos -= np.array([0, -0.4, 0.78])
    return np.concatenate([ee_pos, ee_euler, np.array([gripper]).reshape(-1)])

def quat2euler(quat, is_degree=False):
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler_angles = r.as_euler('xyz', degrees=is_degree)  
    return euler_angles

def get_all_hdf5_files(directory):
    """
    Get all HDF5 files in a directory and its subdirectories.
    """
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))
    return hdf5_files

def get_mask_frame(rgb,observation,idx):
    frame_id = 2
    rgb = rgb[frame_id]
    depth = observation['depth'][idx][-1]
    segmentation = observation['segmentation'][idx][...,0][frame_id]
    obj_geom_id = observation['obj_geom_ids'][idx]
    target_geom_id = observation['target_geom_ids'][idx]
    robot_mask = observation['robot_mask'][idx][frame_id]
    robot_mask = robot_mask.astype(np.bool_)
    mask = np.zeros_like(rgb)
    # print('mask',mask.shape)
    # print('robot_mask',robot_mask.shape)
    mask[~robot_mask,:] = (255,0,0)
    if len(obj_geom_id)>0:
        obj_mask = np.where((segmentation <= max(obj_geom_id))&(segmentation >= min(obj_geom_id)), 0, 1).astype(np.bool_)
        mask[~obj_mask,:] = (0,255,0)
    else:
        obj_mask = np.ones_like(segmentation)
    if len(target_geom_id)>0:
        target_mask = np.where((segmentation <= max(target_geom_id))&(segmentation >= min(target_geom_id)), 0, 1).astype(np.bool_)
        mask[~target_mask,:] = (0,0,255)
    else:
        target_mask = np.ones_like(segmentation)
    depth = np.clip(depth,0,1)
    depth = (depth*255).astype(np.uint8)
    depth = np.concatenate([depth,depth,depth],axis=-1)
    return mask, depth

def create_lerobot_dataset_from_hdf5(args):
    dataset = LeRobotDataset.create(
        repo_id=args.dataset_name,
        robot_type="franka",
        fps=10,
        features={
            "image":{
                "dtype": "image",
                "shape": (480, 480, 3),
                "names": ["height", "width", "channels"]
            },
            "wrist_image":{
                "dtype": "image",
                "shape": (480, 480, 3),
                "names": ["height", "width", "channels"]
            },
            "state":{
                "dtype": "float",
                "shape": (7,),
                "names": ["state"]
            },
            "actions":{
                "dtype": "float",
                "shape": (7,),
                "names": ["actions"]
            },
        },
        image_writer_processes=5,
        image_writer_threads=10
    )
    
    if args.task_list is None:
        tasks = os.listdir(args.dataset_path)
    else:
        tasks = args.task_list
    print("Task to process:", tasks)
    h5py_files = list()
    for task in tasks:
        h5py_files.extend(get_all_hdf5_files(os.path.join(args.dataset_path, task))[:args.max_files])
    print("File numbers:", len(h5py_files))
    for file in h5py_files:
        with h5py.File(file, "r") as f:
            for timestamp in f["data"].keys():
                images = f["data"][timestamp]["observation"]["rgb"]
                ee_state = f["data"][timestamp]["observation"]["ee_state"]
                q_state = f["data"][timestamp]["observation"]["q_state"]
                actions = f["data"][timestamp]["trajectory"]
                ee_pos, ee_quat, gripper = ee_state[:, :3], ee_state[:, 3:7], ee_state[:, 7]
                ee_euler = np.array([quat2euler(q) for q in ee_quat])
                ee_pos -= np.array([0, -0.4, 0.78])
                ee_state = np.concatenate([ee_pos, ee_euler, gripper.reshape(-1, 1)], axis=1)
                assert images.shape[0] == ee_state.shape[0] == q_state.shape[0] == actions.shape[0]
                for i in range(images.shape[0]):
                    action = actions[i]
                    if actions[i][-1] > 0.03:
                        action = np.concatenate([action[:6], np.array([1])])
                    else:
                        action = np.concatenate([action[:6], np.array([0])])
                    front_image = images[i][2]
                    wrist_image = images[i][3]
                    front_image, wrist_image = get_mask_frame(images[i],f["data"][timestamp]["observation"],i)

                    dataset.add_frame(
                        {
                            "image": front_image, # front camera
                            "wrist_image": wrist_image, # wrist camera
                            "state": ee_state[i],
                            "actions": action
                        }
                    )
                dataset.save_episode(task=np.array(f["data"][timestamp]["instruction"])[0].decode("utf-8"))
    dataset.consolidate(run_compute_stats=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a LeRobot dataset")
    parser.add_argument("--dataset-name", type=str, default="test", help="Name of the dataset")
    parser.add_argument("--dataset-path", type=str, default="/media/shiduo/LENOVO_USB_HDD/dataset/VLABench/select_billiards", help="Path to the dataset")
    parser.add_argument("--max-files", type=int, default=500, help="Maximum number of files to process")
    parser.add_argument("--task-list", type=str, nargs="+", default=None, help="List of tasks to process")
    args = parser.parse_args()

    create_lerobot_dataset_from_hdf5(args)