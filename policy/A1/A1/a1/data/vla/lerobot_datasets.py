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

class LeRobotDatasetWrapper(Dataset):
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
                ):
        self.use_proprio = use_proprio
        self.use_wrist_image = use_wrist_image
        self.normalization_type = normalization_type
        self.norm_stats_path = norm_stats_path
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
        print(f"action: {action.shape}, proprio: {proprio.shape}")
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

class LeRobotDatasetWrapperDroid(LeRobotDatasetWrapper):
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
                mode='ee',
                ):
        self.mode = mode
        super().__init__(
            dataset_path,
            chunk_size,
            fixed_action_dim,
            normalization_type,
            pad_action_and_proprio,
            use_proprio,
            use_num_images,
            use_wrist_image,
            video_backend,
            num_episodes,
            image_aug,
        )


    def get(self, item, rng):
        # Always get unnormalized data from parent
        return_dict = super().get(item, rng)

        # Apply normalization if configured (normalization_type is not None)
        if self.normalization_type is not None:
            input_dict = {
                'actions': return_dict['action'],
                'state': return_dict['proprio']
            }
            keys_to_normalize = {
                'actions': "actions",
                'state': "state"
            }
            normalized_data  = normalize_action_and_proprio(input_dict,self.dataset.meta.stats,keys_to_normalize,self.normalization_type)
            return_dict['action'] = normalized_data['actions']
            return_dict['proprio'] = normalized_data['state']

        return return_dict
    


class LeRobotDatasetWrapperAgiBotWorld(Dataset):
    def __init__(self, dataset_path,
                chunk_size=8,
                fixed_action_dim=14,
                normalization_type=NormalizationType.BOUNDS,
                pad_action_and_proprio=True,
                use_proprio=True,
                use_num_images=None,
                use_wrist_image=True,
                video_backend="pyav" #decord, pyav
                ):
        self.use_proprio = use_proprio
        self.use_wrist_image = use_wrist_image
        self.normalization_type = normalization_type
        self.pad_action_and_proprio = pad_action_and_proprio
        self.use_num_images = use_num_images
        self.video_backend = video_backend
        self.fixed_action_dim = fixed_action_dim

        dataset_demo = LeRobotDataset(repo_id=os.path.basename(dataset_path),root=dataset_path, video_backend=self.video_backend)
        fps = dataset_demo.fps
        self.camera_keys = ['observation.images.head', 'observation.images.hand_left','observation.images.hand_right']
        

        self.state_keys = ['observation.states.effector.position','observation.states.end.orientation','observation.states.end.position']

        self.action_keys = ['actions.end.position','actions.end.orientation','actions.effector.position']

        del dataset_demo
        print(f'camera_keys: {self.camera_keys}, state_keys: {self.state_keys}, action_key: {self.action_keys}')
        # Dynamically build delta_timestamps for camera keys (current frame)
        delta_timestamps = {key: [0,] for key in self.camera_keys}
        # Other modalities
        for state_key in self.state_keys:
            delta_timestamps[state_key] = [0,]
        action_deltas = [t / fps for t in range(chunk_size)]
        for action_key in self.action_keys:
            delta_timestamps[action_key] = action_deltas
        self.dataset = LeRobotDataset(repo_id=os.path.basename(dataset_path),root=dataset_path, delta_timestamps=delta_timestamps, video_backend=self.video_backend)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.get(item, np.random)
    
    def get(self, item, rng):
        if isinstance(item, (int, np.integer)) and 0 <= int(item) < len(self.dataset):
            idx = int(item)
        else:
            idx = int(rng.randint(0, len(self.dataset)))

        data_item = self.dataset[idx]
        images = []
        for camera_key in self.camera_keys:
            if camera_key in data_item:
                cam_chw = data_item[camera_key].cpu().numpy().squeeze()
                cam_hwc = (np.transpose(cam_chw, (1, 2, 0))*255).astype(np.uint8)
                images.append(cam_hwc)

        actions_ori = data_item['actions.end.orientation'].cpu().numpy() 
        actions_ori = quaternion_to_euler_numpy(actions_ori)
        actions_pos = data_item['actions.end.position'].cpu().numpy() 
        actions_eff = data_item['actions.effector.position'].cpu().numpy() 
        # 拼接成右手(pos, ori, eff)+左手(pos, ori, eff)的格式
        # 拼接 per hand: pos(3)+euler(3)+grip(1) => 7 dims per hand
        left_vec = np.concatenate([actions_pos[:, 0], actions_ori[:, 0], actions_eff[:, 0:1]], axis=-1)   # (L,7)
        right_vec = np.concatenate([actions_pos[:, 1], actions_ori[:, 1], actions_eff[:, 1:1+1]], axis=-1)  # (L,7)
        actions = np.concatenate([right_vec, left_vec], axis=-1)  # (L,14)

        state_eff = data_item['observation.states.effector.position'].cpu().numpy() 
        state_end_pos = data_item['observation.states.end.position'].cpu().numpy() 
        state_end_ori = data_item['observation.states.end.orientation'].cpu().numpy() 
        state_end_ori = quaternion_to_euler_numpy(state_end_ori)
        left_state = np.concatenate([state_end_pos[:, 0], state_end_ori[:, 0], state_eff[:, 0:1]], axis=-1)   # (L,7)
        right_state = np.concatenate([state_end_pos[:, 1], state_end_ori[:, 1], state_eff[:, 1:1+1]], axis=-1)  # (L,7)
        state = np.concatenate([right_state, left_state], axis=-1)  # (L,14)

        input_dict = {'actions':actions,'state':state}
        normalized_data = input_dict
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
        if pad_len_action > 0:
            action_pad_mask[:, -pad_len_action:] = True

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


def test_lerobot_dataset(dataset_path):
    dataset = LeRobotDatasetWrapperDroid(
        dataset_path, 
        num_episodes=10000, 
        chunk_size=50, 
        fixed_action_dim=32,
         use_proprio=True, 
         use_wrist_image=True,
         image_aug=True
    )
    print(f"Length of the dataset: {len(dataset)}")
    
    from tqdm import tqdm
    for i in tqdm(range(0,len(dataset),10)):
        item = dataset[i]
        print(f"  Action: {item['action'].shape}")
        print(f"  Proprio: {item['proprio'].shape}")
        continue
        print(f"Item {i}:")
        print(f"  Question: {item['question']}")
        print('***************image type',type(item['images'][0]),item['images'][0].shape,item['images'][1].shape)
        # print(item['images'][0].dtype)  # Print the numpy array
        print('number of images: ',len(item['images']))
        

        
 
def get_stats(dataset_path,output_path):
    dataset = LeRobotDatasetWrapper(dataset_path, chunk_size=8, use_proprio=True, use_wrist_image=True)
    stats = dataset.dataset.meta.stats
    if output_path:
        norm_stats = {'actions': stats['actions'], 'state': stats['state']}
        def _numpy_to_list(obj):
            if isinstance(obj, dict):
                return {k: _numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        norm_stats = _numpy_to_list(norm_stats)
        with open(output_path, 'w') as f:
            json.dump(norm_stats, f)
    return norm_stats

def debug_dataset_repeat_read():
     """
     固定同一索引多次读取，并比较签名；同时采样一组索引进行对比。
     用于检查：是否每条数据都一样，或者重复读到同一条数据。
     """
     import hashlib
 
     dataset_path = "/vast/users/xiaodan/zhangjian/HuggingFace/dataset/Dobot-Xtrainer/dobot_pour_water_full"
     dataset = LeRobotDatasetWrapper(dataset_path, chunk_size=8,  use_proprio=True, use_wrist_image=True)
 
     def make_signature(item):
         md5 = hashlib.md5()
         md5.update(item["action"].tobytes())
         if item.get("proprio") is not None:
             md5.update(item["proprio"].tobytes())
         imgs = item.get("images") or item.get("image")
         if imgs and len(imgs) > 0:
             md5.update(imgs[0].tobytes())
         sig = md5.hexdigest()
         meta = item["metadata"]
         ids = (meta.get("episode_index"), meta.get("frame_index"), meta.get("index"), meta.get("task_index"))
         return sig, ids
 
     print("==== 固定同一索引重复读取检查（Wrapper 层）====")
     fixed_index = 0
     repeat_times = 5
     sigs = []
     for t in range(repeat_times):
         it = dataset[fixed_index]
         sig, ids = make_signature(it)
         sigs.append(sig)
         print(f"[try {t}] idx={fixed_index}, ids={ids}, sig={sig}")
     all_same = all(s == sigs[0] for s in sigs)
     print(f"=> 同一索引重复读取，签名是否完全一致: {all_same}")
 
     print("\n==== 多个索引采样检查（Wrapper 层）====")
     indices = list(range(5))
     sigs_multi = []
     for i in indices:
         it = dataset[i]
         sig, ids = make_signature(it)
         sigs_multi.append(sig)
         print(f"[idx {i}] ids={ids}, sig={sig}")
     unique_count = len(set(sigs_multi))
     print(f"=> 采样 {len(indices)} 个索引，唯一签名数: {unique_count}")
     print("提示：如果上面两个检查都显示签名高度重复，说明读取结果可能总是同一条/被随机覆盖索引。")
 
 
def debug_image_channel_order(save_dir: str = "./lerobot_debug"):
    """
    导出原图与交换 R/B 后的图，并打印每通道均值，帮助判断数据是否为 RGB 或 BGR。
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    dataset_path = "/vast/users/xiaodan/zhangjian/HuggingFace/dataset/Dobot-Xtrainer/dobot_pour_water_full"
    dataset = LeRobotDatasetWrapper(dataset_path, chunk_size=8,  use_proprio=True, use_wrist_image=True)

    sample = dataset[0]
    imgs = sample.get("images") or sample.get("image")
    assert imgs and len(imgs) > 0, "未读取到任何图像"
    img = imgs[0]  # HWC, uint8

    def save_image(path: str, arr: np.ndarray):
        try:
            import imageio.v3 as iio
            iio.imwrite(path, arr)
            print(f"saved: {path}")
        except Exception:
            try:
                from PIL import Image
                Image.fromarray(arr).save(path)
                print(f"saved: {path}")
            except Exception:
                np.save(path.replace(".png", ".npy"), arr)
                print(f"image libs unavailable, saved numpy array: {path.replace('.png', '.npy')}")

    # 原样保存（假定为 RGB）
    as_is_path = os.path.join(save_dir, "as_is.png")
    save_image(as_is_path, img)

    # 交换 R/B 后保存（相当于把 RGB 当 BGR 或反之）
    swap_rb = img[:, :, ::-1].copy()
    swap_path = os.path.join(save_dir, "swap_rb.png")
    save_image(swap_path, swap_rb)

    # 打印每通道均值，帮助直观判断
    means = img.astype(np.float32).mean(axis=(0, 1))
    print(f"as_is channel means (C0,C1,C2): {means}")
    swap_means = swap_rb.astype(np.float32).mean(axis=(0, 1))
    print(f"swap_rb channel means (C0,C1,C2): {swap_means}")

    print("查看方式：")
    print(f"1) 用图片查看器打开 {as_is_path} 与 {swap_path}，看哪个颜色更自然。")
    print("2) 如果 as_is 正常、swap_rb 偏色明显，则 as_is 的通道顺序就是正确顺序（常见为 RGB）。")
    print("3) 若你用 OpenCV 显示（cv2.imshow），注意它期望 BGR；若颜色正常，数组多为 BGR。")
 
def export_lerobot_trajectory_video(ds,
                                    start_index: int,
                                    end_index: int,
                                    out_path: str = 'trajectory.mp4',
                                    fps: int = 30,
                                    max_action_dims: int = 6):
    """从已加载的 LeRobot 数据集对象导出某段轨迹到 mp4。

    参数：
    - ds: 已加载的 LeRobot 数据集对象（支持 __getitem__ 返回包含图像与动作的数据字典）
    - start_index, end_index: 帧索引区间 [start_index, end_index)（左闭右开）
    - out_path: 输出 mp4 路径
    - fps: 视频帧率
    - max_action_dims: 动作维度可视化的最多维数
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from math import ceil

    assert end_index > start_index, "end_index 需大于 start_index"
    num_frames = end_index - start_index

    # 先渲染一帧确定尺寸
    first_item = ds[start_index]
    # 收集所有可视图像通道
    def extract_images(item):
        imgs = []
        # 优先使用显式列表
        if 'images' in item and isinstance(item['images'], (list, tuple)) and len(item['images']) > 0:
            for im in item['images']:
                if im is None:
                    continue
                arr = im.cpu().numpy().squeeze() if hasattr(im, 'cpu') else np.asarray(im)
                if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
                    arr = (np.transpose(arr, (1, 2, 0)) * 255).astype(np.uint8)
                if arr.ndim == 3 and arr.shape[2] in (1, 3):
                    imgs.append(arr if arr.shape[2] == 3 else np.repeat(arr, 3, axis=2))
        else:
            # 从字典自动发现图像键
            for k, v in item.items():
                if k in ('action', 'actions', 'state', 'observation', 'metadata', 'timestep', 'timestamp', 'frame_index', 'episode_index', 'index', 'task', 'task_index'):
                    continue
                if v is None:
                    continue
                arr = v.cpu().numpy().squeeze() if hasattr(v, 'cpu') else np.asarray(v)
                if arr is None:
                    continue
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    arr = (np.transpose(arr, (1, 2, 0)) * 255).astype(np.uint8)
                if arr.ndim == 3 and arr.shape[2] in (1, 3):
                    imgs.append(arr if arr.shape[2] == 3 else np.repeat(arr, 3, axis=2))
        return imgs

    imgs0 = extract_images(first_item)
    if len(imgs0) == 0:
        raise ValueError("未从数据集中解析到图像通道。请确认数据项包含 'images' 或图像键。")
    print('load images success')
    # 确定动作维度数量（最多 max_action_dims）
    def extract_action_vec(item):
        act = item.get('action', None)
        if act is None:
            act = item.get('actions', None)
        if act is None:
            return None
        act_np = act.cpu().numpy() if hasattr(act, 'cpu') else np.asarray(act)
        if act_np.ndim == 2:
            return act_np[0]
        if act_np.ndim == 1:
            return act_np
        return None

    first_act_vec = extract_action_vec(first_item)
    action_dims = 0 if first_act_vec is None else min(max_action_dims, int(first_act_vec.shape[0]))
    print('load action success')
    # 预读取整段轨迹的动作序列（用于先画完整曲线）
    actions_all = None
    if action_dims > 0:
        series = []
        for offset in range(num_frames):
            item_t = ds[start_index + offset]
            vec = extract_action_vec(item_t)
            if vec is None:
                vec = np.full((action_dims,), np.nan, dtype=float)
            else:
                vec = np.asarray(vec).astype(float)
                if vec.shape[0] < action_dims:
                    pad = np.full((action_dims - vec.shape[0],), np.nan, dtype=float)
                    vec = np.concatenate([vec, pad], axis=0)
                elif vec.shape[0] > action_dims:
                    vec = vec[:action_dims]
            series.append(vec)
        actions_all = np.stack(series, axis=0)  # [T, D]
    print('load actions_all success')
    # 创建画布：顶部一行图像横跨，下面为 action 网格（每行四个）
    if action_dims > 0:
        action_cols = 4
        action_rows = int(ceil(action_dims / action_cols))
    else:
        action_cols = 4
        action_rows = 0
    print('create fig success')
    total_rows = 1 + action_rows
    height_ratios = [2] + [1] * action_rows
    fig = plt.figure(figsize=(12, 6 + 2.0 * action_rows))
    gs = gridspec.GridSpec(total_rows, action_cols, height_ratios=height_ratios)
    ax_img = fig.add_subplot(gs[0, :])
    ax_action_list = []
    if action_rows > 0:
        for d in range(action_dims):
            r = 1 + d // action_cols
            c = d % action_cols
            ax_action_list.append(fig.add_subplot(gs[r, c]))
    print('create ax_action_list success')
    # 初始绘制
    concat0 = imgs0[0]
    if len(imgs0) > 1:
        # 横向拼接同高的多张图（假设尺寸一致；若不一致，可按最小高等比缩放）
        min_h = min(img.shape[0] for img in imgs0)
        resized = []
        for img in imgs0:
            if img.shape[0] != min_h:
                scale = min_h / img.shape[0]
                resized.append(cv2.resize(img, (int(img.shape[1]*scale), min_h)))
            else:
                resized.append(img)
        concat0 = np.concatenate(resized, axis=1)
    ax_img.imshow(concat0)
    ax_img.axis('off')
    point_artists = []
    if action_dims > 0:
        # 先画完整曲线，后续逐帧叠加当前帧的粗点（并移除上一帧的点）
        for d in range(action_dims):
            ax = ax_action_list[d]
            if actions_all is not None:
                ax.plot(actions_all[:, d], color='C0', linewidth=1.0)
                # 固定坐标轴范围，避免闪烁
                ax.set_xlim(0, max(1, num_frames - 1))
                finite_mask = np.isfinite(actions_all[:, d])
                if finite_mask.any():
                    ymin = float(np.nanmin(actions_all[:, d]))
                    ymax = float(np.nanmax(actions_all[:, d]))
                else:
                    ymin, ymax = -1.0, 1.0
                if ymin == ymax:
                    ymin -= 1.0
                    ymax += 1.0
                ax.set_ylim(ymin, ymax)
            ax.set_title(f'action dim {d}')
            point_artists.append(None)
    fig.canvas.draw()
    print('draw fig success')
    # 基于画布尺寸创建视频写入器（使用 RGBA 缓冲安全抓帧）
    w, h = fig.canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    print('create writer success')
    # 不再累积历史，直接使用预先计算的 actions_all

    for offset in range(num_frames):
        item = ds[start_index + offset]

        # 图像
        imgs = extract_images(item)
        if len(imgs) == 0:
            continue
        min_h = min(img.shape[0] for img in imgs)
        row_imgs = []
        for img in imgs:
            if img.shape[0] != min_h:
                scale = min_h / img.shape[0]
                row_imgs.append(cv2.resize(img, (int(img.shape[1]*scale), min_h)))
            else:
                row_imgs.append(img)
        concat_img = np.concatenate(row_imgs, axis=1)

        # 绘制到画布
        ax_img.clear()
        # 重绘图像轴
        ax_img.imshow(concat_img)
        ax_img.axis('off')
        # 在每个 action 轴上叠加当前帧的粗点（移除上一帧的点）
        if action_dims > 0 and actions_all is not None:
            cur_idx = offset
            dims = actions_all.shape[1]
            for d in range(dims):
                ax = ax_action_list[d]
                # 移除上一帧的点
                if point_artists[d] is not None:
                    try:
                        point_artists[d].remove()
                    except Exception:
                        pass
                    point_artists[d] = None
                # 绘制当前帧的点
                if 0 <= cur_idx < actions_all.shape[0] and np.isfinite(actions_all[cur_idx, d]):
                    artist = ax.plot(cur_idx, actions_all[cur_idx, d], marker='o', markersize=7, color='C3', linewidth=0)[0]
                    point_artists[d] = artist

        # 抓帧（RGBA 更稳妥）
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = buf.reshape((h, w, 4))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        writer.write(frame_bgr)
    print('write frame success')
    writer.release()
    plt.close(fig)

def visualize_lerobot_dataset(dataset_path):
    """示例：使用本文件内的包装或你已有的 ds 导出轨迹视频。"""
    ds = LeRobotDataset(repo_id=os.path.basename(dataset_path), root=dataset_path, video_backend="pyav")
    start_index = ds.episode_data_index["from"][0].item()
    end_index = ds.episode_data_index["to"][0].item()
    print(f"start_index: {start_index}, end_index: {end_index}")
    export_lerobot_trajectory_video(ds, start_index=start_index, end_index=end_index, out_path='outputs/lerobot_trajectory.mp4', fps=5, max_action_dims=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default='data/rc_arx5_open_the_drawer', type=str, required=True)
    parser.add_argument("--mode", default="test", type=str)
    parser.add_argument("--output_path", default=None, type=str)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    if args.mode == "test":
        test_lerobot_dataset(dataset_path)
    elif args.mode == "get_stats":
        assert args.output_path is not None, "output_path is required"
        get_stats(dataset_path,args.output_path)
    elif args.mode == "visualize":
        visualize_lerobot_dataset(dataset_path)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
