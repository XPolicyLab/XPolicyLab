import re
import os
import json
from typing import Dict
import argparse
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset,LeRobotDatasetMetadata,CODEBASE_VERSION
from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig


from a1.data.vla.utils import NormalizationType
# from a1.vla.util import FIXED_ACTION_DIM

from a1.data.dataset import Dataset  
from a1.data.vla.utils import quaternion_to_euler_numpy, quat_to_rotate6d, euler_to_rotate6d

image_transforms_cfg = ImageTransformsConfig(
    enable=True,
    max_num_transforms=2,
    random_order=True,          # Random order each time to break sample bias
    tfs={
        # "sharp": ImageTransformConfig(
        #     weight=2.0,
        #     type="SharpnessJitter",
        #     kwargs={"sharpness": (0.3, 2.0)}
        # ),
        "bright": ImageTransformConfig(
            weight=1.0,
            type="ColorJitter",
            kwargs={"brightness": (0.7, 1.3)}
        ),
    }
)

def normalize_action_and_proprio(traj: Dict, metadata: Dict,keys_to_normalize:Dict, normalization_type: NormalizationType):  
    """Normalizes the action and proprio fields of a trajectory using the given metadata."""  
    # keys_to_normalize = {"action": "actions", "state": "state"}  
      
    normalized_traj = traj.copy()  
  
    if normalization_type == NormalizationType.NORMAL:  
        for key, traj_key in keys_to_normalize.items():  
            # if traj_key in traj and key in metadata:  
            assert traj_key in traj and key in metadata, f"traj_key {traj_key} not in traj or key {key} not in metadata {metadata.keys()}"  
            
            # mask = metadata[key].get("mask", np.ones_like(metadata[key]["mean"], dtype=bool))  
            mask = np.ones_like(metadata[key]["mean"], dtype=bool)
            mask = np.array(mask) if isinstance(mask, list) else mask  
                
            mean = np.array(metadata[key]["mean"])  
            std = np.array(metadata[key]["std"])  
                
            data = traj[traj_key]  
            normalized_data = np.where(mask, (data - mean) / (std + 1e-8), data)  
            normalized_traj[traj_key] = normalized_data  
  
        return normalized_traj  
  
    elif normalization_type in [NormalizationType.BOUNDS, NormalizationType.BOUNDS_Q99]:  
        for key, traj_key in keys_to_normalize.items():  
            # if traj_key in traj and key in metadata:  
            assert traj_key in traj and key in metadata, f"traj_key {traj_key} not in traj or key {key} not in metadata {metadata.keys()}"  
            if normalization_type == NormalizationType.BOUNDS:  
                low = np.array(metadata[key]["min"])  
                high = np.array(metadata[key]["max"])  
            elif normalization_type == NormalizationType.BOUNDS_Q99:  
                low = np.array(metadata[key]["q01"])  
                high = np.array(metadata[key]["q99"])  
                
            # mask = metadata[key].get("mask", np.ones_like(metadata[key]["mean"], dtype=bool))  
            mask = np.ones_like(metadata[key]["mean"], dtype=bool)  
            mask = np.array(mask) if isinstance(mask, list) else mask  
            
            data = traj[traj_key]  

            normalized_data = np.where(  
                mask,  
                np.clip(2 * (data - low) / (high - low + 1e-8) - 1, -1, 1),  
                data,  
            )  
                
            # Set unused action dimensions (min == max) to 0  
            if "min" in metadata[key] and "max" in metadata[key]:
                zeros_mask = np.array(metadata[key]["min"]) == np.array(metadata[key]["max"])  
                normalized_data = np.where(zeros_mask, 0.0, normalized_data)  
                
            normalized_traj[traj_key] = normalized_data  
  
        return normalized_traj  
  
    raise ValueError(f"Unknown Normalization Type {normalization_type}")

def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)

class ManiparenaDatasetWrapper(Dataset):
    def __init__(self, dataset_path,
                chunk_size=8,
                fixed_action_dim=7,
                normalization_type=None,  # None means no normalization
                pad_action_and_proprio=True,
                use_proprio=True,
                use_num_images=None,
                use_wrist_image=True,
                video_backend="pyav", #decord, pyav
                num_episodes=None,
                image_aug=False,
                norm_stats_path=None,
                delta=False,
                delta_mask=None,
                action_type="ee",
                ):
        self.use_proprio = use_proprio
        self.use_wrist_image = use_wrist_image
        self.normalization_type = normalization_type
        self.norm_stats_path = norm_stats_path
        self.action_type = action_type
        if self.normalization_type is not None:
            assert self.norm_stats_path is not None, f"norm_stats_path is required when normalization_type is not None"
            self.norm_stats = json.load(open(self.norm_stats_path))
        self.delta = delta
        self.delta_mask = delta_mask
        if self.delta:
            assert self.delta_mask is not None, f"delta_mask is required when delta is True"
            self.delta_mask = make_bool_mask(*self.delta_mask)
        self.pad_action_and_proprio = pad_action_and_proprio
        self.use_num_images = use_num_images
        self.video_backend = video_backend
        self.fixed_action_dim = fixed_action_dim
        dataset_meta = LeRobotDatasetMetadata(
            os.path.basename(dataset_path),dataset_path, CODEBASE_VERSION, force_cache_sync=False
        )
        fps = dataset_meta.fps
        self.camera_keys = dataset_meta.camera_keys    

        print(f'camera_keys: {self.camera_keys}')
        
        # Compatible with three field names
        if 'observation.state' in dataset_meta.features:
            self.state_key = 'observation.state'
        elif 'state' in dataset_meta.features:
            self.state_key = 'state'
        elif 'qpos' in dataset_meta.features:
            self.state_key = 'qpos'
        else:
            raise ValueError(f"State key not found in dataset meta: {dataset_meta.features}")

        if 'action' in dataset_meta.features:
            self.action_key = 'action'
        elif 'actions' in dataset_meta.features:
            self.action_key = 'actions'
        else:
            raise ValueError(f"Action key not found in dataset meta: {dataset_meta.features}")

        print(f'camera_keys: {self.camera_keys}, state_key: {self.state_key}, action_key: {self.action_key}')
        print(f'dataset_meta.features: {dataset_meta.features}')
        # print(f'dataset_meta.stats: {dataset_demo.meta.stats}')
        

        # assert 'image' in self.camera_keys, f"Primary camera 'image' not found in dataset cameras: {self.camera_keys}"
        # assert 'wrist_image' in self.camera_keys, f"Wrist camera 'wrist_image' not found in dataset cameras: {self.camera_keys}"

        # Dynamically build delta_timestamps for camera keys (current frame)
        delta_timestamps = {key: [0,] for key in self.camera_keys}
        # Other modalities
        delta_timestamps.update({
            self.state_key: [0,],
            self.action_key: [t / fps for t in range(chunk_size)],
        })
        # Note that in any case, these delta_timestamps values need to be multiples of (1/fps) so that added to any
        # timestamp, you still get a valid timestamp.
        if num_episodes is not None:
            np.random.seed(42)
            if isinstance(num_episodes, int):
                num_episodes = min(num_episodes, dataset_meta.total_episodes)
                # Use np.random.choice to ensure no duplicates
                episodes = list(np.random.choice(
                    dataset_meta.total_episodes, 
                    size=num_episodes, 
                    replace=False
                ))
                episodes = sorted(episodes)
            elif isinstance(num_episodes, float):
                num_episodes = min(int(num_episodes*dataset_meta.total_episodes), dataset_meta.total_episodes)
                episodes = list(np.random.choice(
                    dataset_meta.total_episodes, 
                    size=num_episodes, 
                    replace=False
                ))
                episodes = sorted(episodes)
            elif isinstance(num_episodes, str) and '(' in num_episodes and ')' in num_episodes:
                from_to = num_episodes.split('(')[1].split(')')[0].split(',')
                assert len(from_to) == 2, f"Invalid num_episodes: {num_episodes}"
                from_episode = int(from_to[0])
                to_episode = min(int(from_to[1]), dataset_meta.total_episodes)
                episodes = list(range(from_episode, to_episode))
            else:
                raise ValueError(f"Invalid num_episodes: {num_episodes}")
        else:
            episodes = list(range(dataset_meta.total_episodes))
        image_transforms = ImageTransforms(image_transforms_cfg) if image_aug else None
        self.dataset = LeRobotDataset(
            repo_id=os.path.basename(dataset_path),
            root=dataset_path, 
            episodes=episodes, 
            delta_timestamps=delta_timestamps, 
            video_backend=self.video_backend,
            image_transforms=image_transforms,
            check_timestamps=False
        )
        del dataset_meta

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # 遵循上层索引；若直接通过 __getitem__ 调用，则使用全局 np.random
        return self.get(item, np.random)
    
    def get(self, item, rng):
        # 严格使用传入的 item；只有在 item 非法时才回退到 rng 随机采样
        if isinstance(item, (int, np.integer)) and 0 <= int(item) < len(self.dataset):
            idx = int(item)
        else:
            # idx = int(rng.randint(0, len(self.dataset)))
            raise ValueError(f"Invalid item: {item}")
        try:
            data_item = self.dataset[idx]
        except Exception as e:
            item = self.dataset.hf_dataset[idx]
            ep_idx = item["episode_index"].item()
            print(f"Error getting item {idx}, episode_index: {ep_idx}")
            data_item = self.dataset[0]
        images = []
        for camera_key in self.camera_keys:
            if camera_key in data_item:
                cam_chw = data_item[camera_key].cpu().numpy().squeeze()
                cam_hwc = (np.transpose(cam_chw, (1, 2, 0))*255).astype(np.uint8)
                images.append(cam_hwc)
        actions = data_item[self.action_key].cpu().numpy()
        state = data_item[self.state_key].cpu().numpy()
        if self.action_type == 'ee':
            actions = actions[..., :14]
            state = state[..., :14]
        else:
            actions = np.concatenate([actions[...,14:21],actions[...,35:42]], axis=-1)
            state = np.concatenate([state[...,14:21],state[...,35:42]], axis=-1)
        input_dict = {'actions':actions,'state':state}
        keys_to_normalize = {'actions': "actions", 'state': "state"}
        if self.delta:
            if self.delta_mask is not None:
                mask = np.asarray(self.delta_mask)
                dims = mask.shape[-1]
                state_form = np.where(mask, input_dict['state'][..., :dims], 0)
                if len(state_form.shape) == 1:
                    state_form = np.expand_dims(state_form, axis=-2)
                input_dict['actions'][..., :dims] -= state_form
            else:
                input_dict['actions'] = input_dict['actions'] - input_dict['state']
        if self.normalization_type is not None:
            normalized_data  = normalize_action_and_proprio(input_dict,self.norm_stats,keys_to_normalize,self.normalization_type)
        else:
            normalized_data = input_dict
        # print(f"action and state after normalization: {normalized_data}")

        action = normalized_data['actions']
        proprio = normalized_data['state']
        ###
        pad_len_action = self.fixed_action_dim - action.shape[-1]
        if self.pad_action_and_proprio:
            if action.shape[-1] < self.fixed_action_dim:
                action = np.pad(action, ((0, 0), (0, pad_len_action)), mode='constant')
            if proprio.shape[-1] < self.fixed_action_dim:
                pad_len_proprio = self.fixed_action_dim - proprio.shape[-1]
                proprio = np.pad(proprio, ((0, 0), (0, pad_len_proprio)), mode='constant')
        ###

        instruction = data_item['task']

        # 生成与动作同形状的padding掩码，pad位置标记为True
        action_pad_mask = np.zeros_like(action, dtype=bool)
        # if pad_len_action > 0:
        #     action_pad_mask[:, -pad_len_action:] = True

        return_dict = {  
            # "image": image_primary,  
            # "images":[image_primary,image_wrist],
            # "question": f"What action should the robot take to {instruction}?", # is the question necessary?
            "question": instruction, 
            # "message_list": conversation,  
            "timestep": data_item['timestamp'],
            "answer": "Action",
            "style": "action",
            "action": action, 
            "action_pad_mask": action_pad_mask,
            "proprio": proprio if self.use_proprio else None,

            "metadata": {  
                "timestamp": data_item['timestamp'],  
                "frame_index": data_item['frame_index'],
                "episode_index": data_item['episode_index'],
                "index": data_item['index'],
                "task_index": data_item['task_index'],
                "task": data_item['task'],  

            }
        } 
        if self.use_wrist_image:
            if self.use_num_images is not None:
                return_dict["images"] = images[:self.use_num_images]
            else:
                return_dict["images"] = images
        else:
            return_dict["image"] = [images[0],]

        return return_dict
    
    @classmethod
    def download(cls, n_procs=1):
        raise NotImplementedError()
