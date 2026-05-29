"""
RoboMIND Dataset Reader
支持三种模式:
1. 读取数据模式 - 输出格式与 lerobot_datasets 的 return_dict 一致
2. 统计模式 - 计算 max、min、mean、std、q01、q99，按 embodiment 分别统计
3. 可视化模式 - 生成数据集的可视化视频（MP4格式）
"""

import os
import cv2
import h5py
import numpy as np
import argparse
import json
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from a1.data.vla.utils import NormalizationType
from a1.data.dataset import Dataset 
if 'NORM_STATS_PATH' in os.environ:
    NORM_STATS_PATH = os.environ['NORM_STATS_PATH']
else:
    print(f'Warning: NORM_STATS_PATH is not found')

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
                
            # 将未使用的动作维度（min == max）设为 0  
            if "min" in metadata[key] and "max" in metadata[key]:
                zeros_mask = np.array(metadata[key]["min"]) == np.array(metadata[key]["max"])  
                normalized_data = np.where(zeros_mask, 0.0, normalized_data)  
                
            normalized_traj[traj_key] = normalized_data  
  
        return normalized_traj  
  
    raise ValueError(f"Unknown Normalization Type {normalization_type}")

class RoboMINDDatasetReader(Dataset):
    """RoboMIND 数据集读取器"""
    
    # 机器人配置信息
    ROBOT_CONFIGS = {
        "h5_ur_1rgb": { 
            "camera_names": ['camera_top'], 
            "camera_sensors": ['rgb_images', 'depth_images'],
            "arms": ['puppet'],
            "controls": ['joint_position', 'end_effector'],
            "action": "ee",
        },
        "h5_franka_3rgb": {
            "camera_names": ['camera_top', 'camera_left', 'camera_right'],
            "camera_sensors": ['rgb_images', 'depth_images'],
            "arms": ['puppet'],
            "controls": ['joint_position', 'end_effector'],
            "action": "ee",
        },
        "h5_franka_1rgb": {
            "camera_names": ['camera_top'],
            "camera_sensors": ['rgb_images', 'depth_images'],
            "arms": ['puppet'],
            "controls": ['joint_position', 'end_effector'],
            "action": "ee",
        },
        "h5_tienkung_gello_1rgb": {
            "camera_names": ['camera_top'],
            "camera_sensors": ['rgb_images', 'depth_images'],
            "arms": ['puppet', 'master'],
            "controls": ['joint_position'],
            "action": "joint",
        },
        "h5_tienkung_xsens_1rgb": {
            "camera_names": ['camera_top'],
            "camera_sensors": ['rgb_images', 'depth_images'],
            "arms": ['puppet', 'master'],
            "controls": ['joint_position', 'end_effector'],
            "action": "joint",
        },
        "h5_agilex_3rgb": {
            "camera_names": ['camera_front', 'camera_left_wrist', 'camera_right_wrist'],
            "camera_sensors": ['rgb_images', 'depth_images'],
            "arms": ['puppet', 'master'],
            "controls": ['end_effector_left', 'end_effector_right', 'joint_effort_left', 
                        'joint_effort_right', 'joint_position_left', 'joint_position_right', 
                        'joint_velocity_left', 'joint_velocity_right'],
            "action": "joint",
        },
        "h5_simulation": {
            "camera_names": ['camera_front_external', 'camera_handeye', 'camera_left_external', 'camera_right_external'],
            "camera_sensors": ['rgb_images', 'depth_images'],
            "arms": ['franka'],
            "controls": ['end_effector', 'joint_effort', 'joint_position', 'joint_velocity'],
            "action": "ee",
        },
        "h5_franka_fr3_dual": {
            "camera_names": ['camera_top', 'camera_left', 'camera_right', 'camera_front'],
            "camera_sensors": ['rgb_images', 'depth_images'],
            "arms": ['puppet'],
            "controls": ['joint_position', 'end_effector'],
            "action": "ee",
        },
        "h5_sim_franka_3rgb": {
            "camera_names": ['camera_front_external', 'camera_handeye', 'camera_left_external', 'camera_right_external'],
            "camera_sensors": ['rgb_images', 'depth_images'],
            "arms": ['franka'],
            "controls": ['end_effector', 'joint_effort', 'joint_position', 'joint_velocity'],
            "action": "ee",
        },
        "h5_sim_tienkung_1rgb": {
            "camera_names": ['camera_chest', 'camera_head'],
            "camera_sensors": ['rgb_images', 'depth_images'],
            "arms": ['tiangong'],
            "controls": ['left_arm_joint_effort_seqs', 'left_arm_joint_pos_seq', 'left_arm_joint_vel_seq', 
                        'left_end_effector_waist', 'left_hand_joint_effort_seq', 'left_hand_joint_pos_seq', 
                        'left_hand_joint_vel_seq', 'right_arm_joint_effort_seq', 'right_arm_joint_pos_seq', 
                        'right_arm_joint_vel_seq', 'right_end_effector_waist', 'right_hand_joint_pos_seq', 
                        'right_hand_joint_vel_seq'],
            "action": "joint",
        },
        "h5_tienkung_prod1_gello_1rgb": {
            "camera_names": ['camera_top'],
            "camera_sensors": ['rgb_images', 'depth_images'],
            "arms": ['puppet', 'master'],
            "controls": ['joint_position'],
            "action": "joint",
        },
    }
    
    # BGR 颜色空间的 embodiments
    BGR_EMBODIMENTS = ['h5_franka_3rgb', 'h5_franka_1rgb', 'h5_ur_1rgb', 'h5_franka_fr3_dual']

    def __init__(self, dataset_path: str, 
                 embodiment: str,
                 resolution: Tuple[int, int] = None, 
                 normalization_type=None,
                 chunk_size: int = 50, 
                 fixed_action_dim=32, 
                 pad_action_and_proprio=True,
                 env_names: Optional[List[str]] = None, 
                 init_index=False):
        """
        初始化数据集读取器
        
        Args:
            embodiment: 机器人类型
            dataset_path: 数据集根路径
            resolution: 图像分辨率
            chunk_size: 动作序列长度（返回未来 chunk_size 帧的动作）
            env_names: 要加载的环境列表，如果为 None 则加载所有环境
        """
        if embodiment not in self.ROBOT_CONFIGS:
            raise ValueError(f"Unknown embodiment: {embodiment}. Available: {list(self.ROBOT_CONFIGS.keys())}")
        
        self.embodiment = embodiment
        self.dataset_path = dataset_path
        self.resolution = resolution
        self.chunk_size = chunk_size
        self.fixed_action_dim = fixed_action_dim
        self.normalization_type = normalization_type
        self.pad_action_and_proprio = pad_action_and_proprio
        self.robot_config = self.ROBOT_CONFIGS[embodiment]
        self.mode = self.robot_config['action']
        self.camera_names = self.robot_config['camera_names']
        self.camera_sensors = self.robot_config['camera_sensors']
        self.arms = self.robot_config['arms']
        self.controls = self.robot_config['controls']
        
        # 文件缓存
        self._file_cache = {}
        self._max_cache_size = 0
        
        # 构建数据集索引：[(env_name, trajectory_id, file_path, frame_idx, num_frames, episode_index), ...]
        self._index: List[Tuple[str, str, str, int, int, int]] = []
        self._build_index(env_names, init_index)
        if normalization_type is not None:
            self.norm_stats = json.load(open(NORM_STATS_PATH+'/robomind.json'))
            self.norm_stats = self.norm_stats['per_embodiment_stats'][self.embodiment]['stats']
        
    def _build_index(self, env_names: Optional[List[str]] = None, init_index=False):
        """
        构建数据集索引，遍历所有 episode 和帧
        
        Args:
            env_names: 要索引的环境列表，如果为 None 则索引所有环境
        """
        print(f"Building dataset index for embodiment: {self.embodiment}")
        
        if not init_index:
            with open(self.dataset_path+f'/{self.embodiment}_index.json', 'r') as f:
                self._index = json.load(f)['indexs']
            print(f"Index built: {len(self._index)} frames")
            return

        # 获取环境列表
        embodiment_path = os.path.join(self.dataset_path, self.embodiment)
        if not os.path.exists(embodiment_path):
            raise ValueError(f"Embodiment path does not exist: {embodiment_path}")
        
        if env_names is None:
            env_names = [d for d in os.listdir(embodiment_path) 
                        if os.path.isdir(os.path.join(embodiment_path, d))]
        
        # 遍历每个环境
        episode_counter = 0  # 全局 episode 计数器
        for env_name in tqdm(env_names, desc="Indexing environments"):
            try:
                episode_list = self.get_episode_list(env_name)
            except ValueError as e:
                print(f"Warning: {e}")
                continue
            
            # 遍历每个 episode
            for trajectory_id, file_path in episode_list:
                try:
                    # 获取该 episode 的帧数
                    f, control_dict, metadata = self._get_cached_file(file_path)
                    
                    # 获取控制数据的长度
                    num_frames = None
                    for arm_name in self.arms:
                        for control_key in self.controls:
                            if control_key in control_dict[arm_name]:
                                control_data = control_dict[arm_name][control_key]
                                if num_frames is None:
                                    num_frames = control_data.shape[0] - 1  # -1 因为 state/action 会少一帧
                                else:
                                    num_frames = min(num_frames, control_data.shape[0] - 1)
                    
                    if num_frames is None or num_frames <= 0:
                        continue
                    
                    # 为每一帧创建索引条目
                    for frame_idx in range(num_frames):
                        self._index.append((env_name, trajectory_id, file_path, frame_idx, num_frames, episode_counter))
                    
                    episode_counter += 1  # 每个 episode 递增
                
                except Exception as e:
                    print(f"Warning: Failed to index {file_path}: {e}")
                    continue
        
        print(f"Index built: {len(self._index)} frames from {len(env_names)} environments")
        with open(self.dataset_path+f'/{self.embodiment}_index.json', 'w') as f:
            json.dump({'indexs':self._index}, f)
            
    def __len__(self) -> int:
        """
        返回数据集的总帧数
        """
        return len(self._index)
    
    def get(self, item, rng=None):
        return self[item]

    def __getitem__(self, idx: int) -> Dict:
        """
        获取指定索引的数据样本，用于 PyTorch DataLoader
        
        Args:
            idx: 样本索引
            
        Returns:
            与 lerobot_datasets 格式兼容的数据字典
        """
        if idx < 0 or idx >= len(self._index):
            raise IndexError(f"Index {idx} out of range [0, {len(self._index)})")
        
        env_name, trajectory_id, file_path, frame_idx, num_frames, episode_index = self._index[idx]
        
        # 调用原有的 read_episode 方法
        return self.read_episode(file_path, frame_idx=frame_idx, task_description=env_name, episode_index=episode_index, dataset_idx=idx)
    
    def decode_image(self, camera_rgb_images, camera_depth_images=None):
        """
        解码图像数据
        
        Args:
            camera_rgb_images: RGB 图像数据
            camera_depth_images: 深度图像数据（可选）
            
        Returns:
            解码后的 RGB 和深度图像
        """
        if type(camera_rgb_images[0]) is np.uint8:
            rgb = cv2.imdecode(camera_rgb_images, cv2.IMREAD_COLOR)
            if camera_depth_images is not None:
                depth_array = np.frombuffer(camera_depth_images, dtype=np.uint8)
                depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
            else:
                depth = None
            return rgb, depth
        else:
            rgb_images = []
            depth_images = []
            for idx, camera_rgb_image in enumerate(camera_rgb_images):
                camera_rgb_image = np.array(camera_rgb_image)
                rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
                if rgb is None:
                    rgb = np.frombuffer(camera_rgb_image, dtype=np.uint8)
                    if rgb.size == 2764800:
                        rgb = rgb.reshape(720, 1280, 3)
                    elif rgb.size == 921600:
                        rgb = rgb.reshape(480, 640, 3)
                
                if camera_depth_images is not None:
                    depth_array = np.frombuffer(camera_depth_images[idx], dtype=np.uint8)
                    depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
                    if depth is None:
                        depth = np.frombuffer(camera_depth_images[idx], dtype=np.uint8)
                        if depth.size == 921600:
                            depth = depth.reshape(720, 1280)
                        elif depth.size == 307200:
                            depth = depth.reshape(480, 640)
                else:
                    depth = None
                
                rgb_images.append(rgb)
                if depth is not None:
                    depth_images.append(depth)
            
            rgb_images = np.asarray(rgb_images)
            depth_images = np.asarray(depth_images) if depth_images else None
            return rgb_images, depth_images
    
    def _get_cached_file(self, file_path: str) -> Tuple[h5py.File, Dict, Dict]:
        """
        获取缓存的 HDF5 文件句柄和控制数据
        
        Args:
            file_path: HDF5 文件路径
            
        Returns:
            f: HDF5 文件句柄（保持打开）
            control_dict: 控制数据字典（已读取）
            metadata: 元数据字典
        """
        # 打开文件并读取控制数据（不读取图像）
        f = h5py.File(file_path, 'r')
        
        control_dict = defaultdict(dict)
        for arm_name in self.arms:
            for control in self.controls:
                if control in f[arm_name]:
                    control_dict[arm_name][control] = f[arm_name][control][:]
        
        metadata = {
            'is_sim': f.attrs.get('sim', False),
            'is_compress': f.attrs.get('compress', True)
        }
        return (f, control_dict, metadata)

        # if file_path not in self._file_cache:
        #     # 如果缓存已满，清除最旧的条目
        #     if len(self._file_cache) >= self._max_cache_size:
        #         oldest_key = next(iter(self._file_cache))
        #         self._file_cache[oldest_key][0].close()
        #         del self._file_cache[oldest_key]
            
        #     # 打开文件并读取控制数据（不读取图像）
        #     f = h5py.File(file_path, 'r')
            
        #     control_dict = defaultdict(dict)
        #     for arm_name in self.arms:
        #         for control in self.controls:
        #             if control in f[arm_name]:
        #                 control_dict[arm_name][control] = f[arm_name][control][:]
            
        #     metadata = {
        #         'is_sim': f.attrs.get('sim', False),
        #         'is_compress': f.attrs.get('compress', True)
        #     }
        #     self._file_cache[file_path] = (f, control_dict, metadata)
        
        # return self._file_cache[file_path]
    
    def read_control_data_only(self, file_path: str) -> Tuple[Dict, Dict]:
        """
        仅读取控制数据（不读取图像），用于统计模式
        
        Args:
            file_path: HDF5 文件路径
            
        Returns:
            control_dict: 控制数据字典
            metadata: 元数据字典
        """
        f, control_dict, metadata = self._get_cached_file(file_path)
        return control_dict, metadata
    
    def read_h5_file(self, file_path: str) -> Tuple[Dict, Dict, Dict]:
        """
        读取单个 HDF5 文件（完整版本，用于可视化）
        
        Args:
            file_path: HDF5 文件路径
            
        Returns:
            image_dict: 图像字典
            control_dict: 控制数据字典
            base_dict: 基础数据字典
        """
        f, control_dict, metadata = self._get_cached_file(file_path)
        
        # 读取并解码所有图像
        image_dict = defaultdict(dict)
        for cam_name in self.camera_names:
            if len(self.camera_sensors) >= 2:
                rgb_images = f['observations'][self.camera_sensors[0]][cam_name][:]
                depth_images = f['observations'][self.camera_sensors[1]][cam_name][:]
            else:
                rgb_images = f['observations'][self.camera_sensors[0]][cam_name][:]
                depth_images = None
            
            decode_rgb, decode_depth = self.decode_image(rgb_images, depth_images)
            image_dict[self.camera_sensors[0]][cam_name] = decode_rgb
            if decode_depth is not None:
                image_dict[self.camera_sensors[1]][cam_name] = decode_depth
        
        base_dict = metadata
        
        return image_dict, control_dict, base_dict
    
    def process_control_data(self, control_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理控制数据，将其转换为 action 和 state
        提取 end_effector (xyz + euler angles) + gripper，与 lerobot_datasets.py 格式一致
        
        Args:
            control_dict: 控制数据字典
            
        Returns:
            action: 动作数据 (T, action_dim) - 格式: [right_xyz(3) + right_euler(3) + right_grip(1) + left_xyz(3) + left_euler(3) + left_grip(1)] = 14维
            state: 状态数据 (T, state_dim) - 格式同 action
        """
        # 根据不同的 embodiment 提取不同的控制数据
        if self.embodiment in ['h5_franka_3rgb', 'h5_franka_1rgb', 'h5_ur_1rgb']:
            # 单臂机器人：end_effector (xyz + euler) + gripper
            if self.mode == 'joint':
                left = control_dict['puppet']['joint_position'] 
            elif self.mode == 'ee':
                ee = control_dict['puppet']['end_effector']              # [T, 6]
                grip = control_dict['puppet']['joint_position'][:, -1:]  # [T, 1]
                left = np.concatenate([
                    ee,
                    grip            # gripper
                ], axis=-1)
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
            
            right = np.zeros_like(left)  # 右臂为0
            
        elif self.embodiment == 'h5_agilex_3rgb':
            # 双臂机器人：end_effector_left/right (xyz + euler + grip_raw)
            if self.mode == 'joint':
                le = control_dict['puppet']['joint_position_left']
                re = control_dict['puppet']['joint_position_right']
            elif self.mode == 'ee':
                le = control_dict['puppet']['end_effector_left']
                re = control_dict['puppet']['end_effector_right']
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
            # 转换为：xyz(3) + euler(3) + grip(1) = 7维 (每只手)
            left = np.concatenate([
                le[:, :3],      # xyz
                le[:, 3:6],     # euler angles
                (le[:, -1:] > 2.5).astype(np.float32)  # 阈值处理 gripper
            ], axis=-1)
            right = np.concatenate([
                re[:, :3],      # xyz
                re[:, 3:6],     # euler angles
                (re[:, -1:] > 2.5).astype(np.float32)
            ], axis=-1)
            
        elif self.embodiment == 'h5_franka_fr3_dual':
            # 双臂 franka：end_effector (L_xyz L_euler R_xyz R_euler) + grippers from joint_position
            ee = control_dict['puppet']['end_effector']         # [T, 12]
            jp = control_dict['puppet']['joint_position']       # [T, ...]
            if self.mode == 'joint':
                left = jp[:, :8]
                right = jp[:, 8:]
            elif self.mode == 'ee':
                left = np.concatenate([
                    ee[:, 0:3],     # xyz
                    ee[:, 3:6],     # euler angles
                    jp[:, 7:8]      # 左臂 gripper
                ], axis=-1)
                right = np.concatenate([
                    ee[:, 6:9],     # xyz
                    ee[:, 9:12],    # euler angles
                    jp[:, -1:]      # 右臂 gripper
                ], axis=-1)
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        else:
            # 其他机器人：回退到原来的逻辑（读取所有控制数据）
            action_list = []
            for arm_name in self.arms:
                control_list = []
                for control_key in self.controls:
                    if control_key in control_dict[arm_name]:
                        control = control_dict[arm_name][control_key]
                        min_len = min([control_dict[arm_name][k].shape[0] 
                                      for k in self.controls if k in control_dict[arm_name]])
                        if control.shape[0] > min_len:
                            control = control[:min_len]
                        control_list.append(control)
                
                if control_list:
                    arm_control = np.concatenate(control_list, axis=1)
                    action_list.append(arm_control)
            
            action = np.concatenate(action_list, axis=1) if action_list else np.array([])
            state = action[:-1].copy()
            action = action[1:].copy()
            return action, state
        
        # 拼接右手+左手：[right(7) + left(7)] = 14维 (与 lerobot_datasets.py 格式一致)
        action = np.concatenate([right, left], axis=1)
        
        # state 是当前时刻的 action，action 是下一时刻的
        state = action[:-1].copy()
        action = action[1:].copy()
        
        return action, state
    
    def process_images(self, image_dict: Dict, num_frames: int) -> List[np.ndarray]:
        """
        处理图像数据
        
        Args:
            image_dict: 图像字典
            num_frames: 帧数
            
        Returns:
            处理后的图像列表
        """
        images = []
        for cam_name in self.camera_names:
            if cam_name in image_dict['rgb_images']:
                rgb_imgs = image_dict['rgb_images'][cam_name][:num_frames]
                
                processed_imgs = []
                for img in rgb_imgs:
                    # 颜色空间转换
                    if self.embodiment in self.BGR_EMBODIMENTS:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 调整大小
                    if self.resolution is not None:
                        img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_AREA)
                    processed_imgs.append(img)
                
                images.extend(processed_imgs)
        
        return images
    
    def get_episode_list(self, env_name: str) -> List[str]:
        """
        获取某个任务的所有 episode 路径
        
        Args:
            env_name: 任务名称
            
        Returns:
            episode 的 HDF5 文件路径列表
        """
        dataset_root = os.path.join(self.dataset_path, self.embodiment, env_name, 'success_episodes/train')
        if not os.path.exists(dataset_root):
            raise ValueError(f"Dataset path does not exist: {dataset_root}")
        
        episode_paths = []
        for trajectory_id in sorted(os.listdir(dataset_root)):
            traj_path = os.path.join(dataset_root, trajectory_id, 'data')
            if not os.path.exists(traj_path):
                continue
            
            for file in os.listdir(traj_path):
                if file.endswith('.hdf5'):
                    file_path = os.path.join(traj_path, file)
                    episode_paths.append((trajectory_id, file_path))
                    break
        
        return episode_paths
    
    def read_episode(self, file_path: str, frame_idx: int = 0, 
                    task_description: Optional[str] = None, episode_index: int = 0, dataset_idx: int = 0) -> Dict:
        """
        读取单个 episode 的数据，返回与 lerobot_datasets 兼容的格式
        - 图像：当前帧（单帧）
        - 状态：当前帧（单时刻）
        - 动作：从当前帧开始的未来 chunk_size 帧动作序列
        
        Args:
            file_path: HDF5 文件路径
            frame_idx: 帧索引
            task_description: 任务描述
            episode_index: episode 索引
            
        Returns:
            return_dict: 与 lerobot_datasets 格式一致的数据字典
        """
        # 使用缓存的文件句柄和控制数据
        f, control_dict, metadata = self._get_cached_file(file_path)
        action, state = self.process_control_data(control_dict)
        # 确保 frame_idx 有效
        max_frames = min(action.shape[0], state.shape[0])
        if frame_idx >= max_frames:
            frame_idx = max_frames - 1
        
        # 只读取并解码需要的单帧图像（关键优化点）
        images = []
        for cam_name in self.camera_names:
            try:
                # 只读取单帧的压缩数据
                if len(self.camera_sensors) >= 2:
                    rgb_data = f['observations'][self.camera_sensors[0]][cam_name][frame_idx]
                else:
                    rgb_data = f['observations'][self.camera_sensors[0]][cam_name][frame_idx]
                
                # 解码单帧图像
                if isinstance(rgb_data, np.ndarray):
                    img = cv2.imdecode(rgb_data, cv2.IMREAD_COLOR)
                    if img is None:
                        # 尝试直接reshape
                        if rgb_data.size == 2764800:
                            img = rgb_data.reshape(720, 1280, 3)
                        elif rgb_data.size == 921600:
                            img = rgb_data.reshape(480, 640, 3)
                        else:
                            continue
                else:
                    img = np.array(rgb_data)
                
                # 颜色空间转换
                if self.embodiment in self.BGR_EMBODIMENTS:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 调整大小
                if self.resolution is not None:
                    img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_AREA)
                images.append(img)
            except Exception as e:
                img = np.zeros((480,640,3),dtype=np.uint8)
                images.append(img)
                # print(f"Warning: Failed to read image from {cam_name} at frame {frame_idx} in file {file_path}: {e}")
                continue
        
        # 获取当前时刻的 state（单帧）
        frame_state = state[frame_idx:frame_idx+1]    # (1, state_dim)
        
        # 获取从当前帧开始的未来 chunk_size 帧的动作
        end_idx = min(frame_idx + self.chunk_size, action.shape[0])
        frame_action = action[frame_idx:end_idx]  # (chunk_size, action_dim)
        
        # 如果不足 chunk_size 帧，用最后一帧补齐
        if frame_action.shape[0] < self.chunk_size:
            last_action = frame_action[-1:]
            padding = np.repeat(last_action, self.chunk_size - frame_action.shape[0], axis=0)
            frame_action = np.concatenate([frame_action, padding], axis=0)
        
        # # norm
        # input_dict = {'action':frame_action,'state':frame_state}
        # keys_to_normalize = {"action": "action", "state": "state"}  
        # normalized_data  = normalize_action_and_proprio(input_dict,self.norm_stats,keys_to_normalize,self.normalization_type)
        
        # pad
        pad_len_action = self.fixed_action_dim - frame_action.shape[-1]
        if self.pad_action_and_proprio:
            if frame_action.shape[-1] < self.fixed_action_dim:
                frame_action = np.pad(frame_action, ((0, 0), (0, pad_len_action)), mode='constant')
            if frame_state.shape[-1] < self.fixed_action_dim:
                pad_len_proprio = self.fixed_action_dim - frame_state.shape[-1]
                frame_state = np.pad(frame_state, ((0, 0), (0, pad_len_proprio)), mode='constant')

        # 任务描述
        if task_description is None:
            # 从文件路径提取任务名称
            task_description = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
        try:
            question = f['language_raw'][0].decode('utf-8')
        except Exception as e:
            question = ""
        assert len(images)<=3, f"len(images)={len(images)}"
        return_dict = {
            "question": question,
            "timestep": frame_idx / 30.0,
            "answer": "Action",
            "style": "action",
            "action": frame_action,
            "action_pad_mask": np.zeros_like(frame_action, dtype=bool),
            "proprio": frame_state,
            "images": images,
            "metadata": {
                "timestamp": frame_idx / 30.0,
                "frame_index": frame_idx,
                "episode_index": episode_index,
                "index": dataset_idx,
                "task_index": 0,
                "task": task_description,
                "file_path": file_path,
            }
        }
        
        return return_dict
    
    def compute_stats(self, env_names: List[str], return_raw_data: bool = False) -> Dict:
        """
        计算数据集的统计信息
        
        Args:
            env_names: 任务名称列表
            return_raw_data: 是否返回原始数据（用于计算总体统计）
            
        Returns:
            stats: 统计信息字典，包含 max, min, mean, std, q01, q99
            如果 return_raw_data=True，还会返回 raw_actions 和 raw_states
        """
        all_actions = []
        all_states = []
        
        print(f"Computing stats for embodiment: {self.embodiment}")
        for env_name in env_names:
            print(f"Processing environment: {env_name}")
            try:
                episode_list = self.get_episode_list(env_name)
            except ValueError as e:
                print(f"Warning: {e}")
                continue
            
            for trajectory_id, file_path in tqdm(episode_list, desc=f"Reading {env_name}"):
                try:
                    # 关键优化：只读取控制数据，不读取和解码图像
                    control_dict, metadata = self.read_control_data_only(file_path)
                    action, state = self.process_control_data(control_dict)
                    
                    if action.shape[0] > 0:
                        all_actions.append(action)
                        all_states.append(state)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
        
        if not all_actions:
            raise ValueError(f"No valid data found for embodiment {self.embodiment}")
        
        # 拼接所有数据
        all_actions = np.concatenate(all_actions, axis=0)  # (N, action_dim)
        all_states = np.concatenate(all_states, axis=0)    # (N, state_dim)
        
        print(f"Total frames: {all_actions.shape[0]}")
        print(f"Action dimension: {all_actions.shape[1]}")
        print(f"State dimension: {all_states.shape[1]}")
        
        # 计算统计量
        stats = {
            "action": {
                "max": np.max(all_actions, axis=0).tolist(),
                "min": np.min(all_actions, axis=0).tolist(),
                "mean": np.mean(all_actions, axis=0).tolist(),
                "std": np.std(all_actions, axis=0).tolist(),
                "q01": np.percentile(all_actions, 1, axis=0).tolist(),
                "q99": np.percentile(all_actions, 99, axis=0).tolist(),
            },
            "state": {
                "max": np.max(all_states, axis=0).tolist(),
                "min": np.min(all_states, axis=0).tolist(),
                "mean": np.mean(all_states, axis=0).tolist(),
                "std": np.std(all_states, axis=0).tolist(),
                "q01": np.percentile(all_states, 1, axis=0).tolist(),
                "q99": np.percentile(all_states, 99, axis=0).tolist(),
            }
        }
        
        if return_raw_data:
            return stats, all_actions, all_states
        return stats
    
    def visualize_episode(self, file_path: str, output_dir: str, 
                         trajectory_id: str, fps: int = 10,
                         show_depth: bool = False) -> List[str]:
        """
        可视化单个 episode，生成 MP4 视频
        
        Args:
            file_path: HDF5 文件路径
            output_dir: 输出目录
            trajectory_id: 轨迹 ID
            fps: 视频帧率
            show_depth: 是否同时生成深度图视频
            
        Returns:
            生成的视频文件路径列表
        """
        image_dict, control_dict, base_dict = self.read_h5_file(file_path)
        action, state = self.process_control_data(control_dict)
        
        # 确保输出目录存在
        traj_output_dir = os.path.join(output_dir, trajectory_id)
        os.makedirs(traj_output_dir, exist_ok=True)
        
        video_paths = []
        
        # 对齐帧数
        num_frames = min(action.shape[0], state.shape[0])
        
        # 为每个相机生成视频
        for sensor_type in image_dict.keys():
            if sensor_type == 'depth_images' and not show_depth:
                continue
                
            for cam_name in self.camera_names:
                if cam_name not in image_dict[sensor_type]:
                    continue
                
                images = image_dict[sensor_type][cam_name][:num_frames]
                
                if images.shape[0] == 0:
                    continue
                
                # 处理图像
                processed_frames = []
                for idx, img in enumerate(images):
                    img = np.array(img)
                    
                    if sensor_type == 'rgb_images':
                        # 颜色空间转换
                        if self.embodiment in self.BGR_EMBODIMENTS:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    elif sensor_type == 'depth_images':
                        if img is not None and img.size > 0:
                            # 归一化深度图
                            depth_min = np.min(img)
                            depth_max = np.max(img)
                            if depth_max > depth_min:
                                img = ((img - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                            else:
                                img = np.zeros_like(img, dtype=np.uint8)
                            # 应用伪彩色
                            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        else:
                            continue
                    
                    # 调整大小
                    if self.resolution is not None:
                        img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_AREA)
                    
                    # 添加文本信息（帧号、动作、状态）
                    img_with_text = img.copy()
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.3
                    thickness = 1
                    color = (255, 255, 255)
                    
                    # 添加帧号
                    cv2.putText(img_with_text, f"Frame: {idx}", (5, 15), 
                               font, font_scale, color, thickness)
                    
                    # 添加相机名称
                    cv2.putText(img_with_text, f"Cam: {cam_name}", (5, 30), 
                               font, font_scale, color, thickness)
                    
                    processed_frames.append(img_with_text)
                
                if not processed_frames:
                    continue
                
                # 生成视频文件
                video_filename = f"{cam_name}_{sensor_type}.mp4"
                video_path = os.path.join(traj_output_dir, video_filename)
                
                # 使用 OpenCV 写入视频
                height, width = processed_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                
                for frame in processed_frames:
                    # OpenCV 需要 BGR 格式
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                out.release()
                video_paths.append(video_path)
                print(f"Saved video: {video_path}")
        
        return video_paths
    
    def visualize_episodes(self, env_name: str, output_dir: str, 
                          num_episodes: Optional[int] = None,
                          fps: int = 10, show_depth: bool = False) -> Dict[str, List[str]]:
        """
        可视化多个 episodes
        
        Args:
            env_name: 任务名称
            output_dir: 输出目录
            num_episodes: 要可视化的 episode 数量（None 表示全部）
            fps: 视频帧率
            show_depth: 是否生成深度图视频
            
        Returns:
            每个 episode 生成的视频路径字典
        """
        episode_list = self.get_episode_list(env_name)
        
        if num_episodes is not None:
            episode_list = episode_list[:num_episodes]
        
        # 创建输出目录
        env_output_dir = os.path.join(output_dir, self.embodiment, env_name)
        os.makedirs(env_output_dir, exist_ok=True)
        
        all_videos = {}
        
        print(f"Visualizing {len(episode_list)} episodes for {env_name}...")
        for trajectory_id, file_path in tqdm(episode_list, desc=f"Visualizing {env_name}"):
            try:
                video_paths = self.visualize_episode(
                    file_path=file_path,
                    output_dir=env_output_dir,
                    trajectory_id=trajectory_id,
                    fps=fps,
                    show_depth=show_depth
                )
                all_videos[trajectory_id] = video_paths
            except Exception as e:
                print(f"Error visualizing {trajectory_id}: {e}")
                continue
        
        return all_videos


def main():
    """
    example:
    python a1/data/vla/robomind_datasets.py --mode read --embodiment h5_agilex_3rgb --dataset_path data/robomind/benchmark1_1_compressed
    """
    parser = argparse.ArgumentParser(description="RoboMIND Dataset Reader")
    parser.add_argument("--mode", type=str, choices=["read", "stats", "visualize", "build_index"], required=True,
                       help="Mode: 'read' for reading data, 'stats' for computing statistics, 'visualize' for generating videos")
    parser.add_argument("--embodiment", type=str, nargs="+", required=True,
                       choices=list(RoboMINDDatasetReader.ROBOT_CONFIGS.keys()),
                       help="Robot embodiment type(s). Can specify multiple embodiments.")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the RoboMIND dataset root directory")
    parser.add_argument("--env_name", type=str, nargs="+", 
                       help="Environment/task name(s). For 'read' mode: specify one (or omit to read all). For 'stats'/'visualize' mode: can specify multiple (or omit to process all).")
    parser.add_argument("--output_path", type=str, default="./robomind_stats.json",
                       help="Output path for statistics (only for 'stats' mode)")
    parser.add_argument("--output_dir", type=str, default="./visualization",
                       help="Output directory for videos (only for 'visualize' mode)")
    parser.add_argument("--frame_idx", type=int, default=0,
                       help="Frame index to read (only for 'read' mode)")
    parser.add_argument("--episode_idx", type=int, default=0,
                       help="Episode index to read (only for 'read' mode)")
    parser.add_argument("--num_episodes", type=int, default=None,
                       help="Number of episodes to visualize per task (only for 'visualize' mode, None means all)")
    parser.add_argument("--fps", type=int, default=10,
                       help="Video frame rate (only for 'visualize' mode)")
    parser.add_argument("--show_depth", action="store_true",
                       help="Also generate depth videos (only for 'visualize' mode)")
    parser.add_argument("--resize",default=False, action="store_true")
    parser.add_argument("--resolution", type=int, nargs=2, default=[128, 128],
                       help="Image resolution (width height)")
    
    args = parser.parse_args()
    if not args.resize:
        args.resolution = None
    # 将 embodiment 转换为列表
    embodiments = args.embodiment if isinstance(args.embodiment, list) else [args.embodiment]
    if args.mode == "build_index":
        print('embodiments',embodiments)
        for embodiment in embodiments:
            try:
                reader = RoboMINDDatasetReader(
                    embodiment=embodiment,
                    dataset_path=args.dataset_path,
                    resolution=args.resolution,
                    env_names=args.env_name,
                    init_index=True
                )
                print(f"Index built: {len(reader._index)} frames from {embodiment}")
            except Exception as e:
                print(f'building {embodiment} error in {args.dataset_path}')
    if args.mode == "read":
        # 读取模式只支持单个 embodiment
        if len(embodiments) != 1:
            raise ValueError("For 'read' mode, please specify exactly one embodiment")
        
        # 初始化读取器
        reader = RoboMINDDatasetReader(
            embodiment=embodiments[0],
            dataset_path=args.dataset_path,
            resolution=args.resolution,
            env_names=args.env_name  # 传入 env_names
        )
    
        # 读取数据模式
        print(f"\n=== Dataset Info ===")
        print(f"Total frames: {len(reader)}")
        print(f"Embodiment: {embodiments[0]}")
        if args.env_name:
            print(f"Environments: {', '.join(args.env_name)}")
        else:
            print(f"Environments: All available")
        
        # 示例：读取指定索引的数据
        if len(reader) == 0:
            print("Error: No data found in the dataset")
        else:
            # 随机读取几个样本作为示例
            import random
            sample_indices = random.sample(range(len(reader)), min(100, len(reader)))
            
            print(f"\n=== Sample Data ===")
            for idx in tqdm(sample_indices):
                if idx >= len(reader):
                    continue
                
                data = reader[idx]
                print(f"\nSample {idx}:")
                print(f"  Task: {data['question']}")
                print(f"  Frame index: {data['metadata']['frame_index']}")
                print(f"  Episode: {data['metadata']['file_path']}")
                print(f"  Number of images: {len(data['images'])}")
                print(f"  Image shape: {data['images'][0].shape}")
                print(f"  Action shape: {data['action'].shape}")
                print(f"  State shape: {data['proprio'].shape}")
        
    elif args.mode == "stats":
        # 统计模式 - 支持多个 embodiment
        all_stats = {}
        overall_actions = []
        overall_states = []
        
        for embodiment in embodiments:
            print(f"\n{'='*60}")
            print(f"Processing embodiment: {embodiment}")
            print(f"{'='*60}")
            
            # 初始化读取器
            reader = RoboMINDDatasetReader(
                embodiment=embodiment,
                dataset_path=args.dataset_path,
                resolution=args.resolution
            )
            
            # 获取环境列表
            if not args.env_name:
                # 如果没有指定环境，获取该 embodiment 下的所有环境
                embodiment_path = os.path.join(args.dataset_path, embodiment)
                if not os.path.exists(embodiment_path):
                    print(f"Warning: Embodiment path does not exist: {embodiment_path}")
                    continue
                
                env_names = [d for d in os.listdir(embodiment_path) 
                            if os.path.isdir(os.path.join(embodiment_path, d))]
                print(f"Found {len(env_names)} environments: {env_names}")
            else:
                env_names = args.env_name
            
            # 初始化读取器（注意：这里不会构建索引，因为 stats 模式直接调用 compute_stats）
            reader_for_stats = RoboMINDDatasetReader(
                embodiment=embodiment,
                dataset_path=args.dataset_path,
                resolution=args.resolution,
                env_names=[]  # 传入空列表，避免构建索引
            )
            
            try:
                # 如果有多个 embodiment，收集原始数据用于计算总体统计
                if len(embodiments) > 1:
                    stats, raw_actions, raw_states = reader_for_stats.compute_stats(env_names, return_raw_data=True)
                    overall_actions.append(raw_actions)
                    overall_states.append(raw_states)
                else:
                    stats = reader_for_stats.compute_stats(env_names, return_raw_data=False)
                
                all_stats[embodiment] = {
                    "environments": env_names,
                    "stats": stats
                }
            except Exception as e:
                print(f"Error computing stats for {embodiment}: {e}")
                continue
        
        if not all_stats:
            raise ValueError("No valid statistics computed for any embodiment")
        
        # 计算总体统计（如果有多个 embodiment）
        overall_stats = None
        if len(embodiments) > 1 and overall_actions:
            print(f"\n{'='*60}")
            print("Computing overall statistics across all embodiments...")
            print(f"{'='*60}")
            
            # 找到最大维度
            max_action_dim = max([a.shape[1] for a in overall_actions])
            max_state_dim = max([s.shape[1] for s in overall_states])
            
            # Pad 到相同维度
            padded_actions = []
            padded_states = []
            
            for actions, states in zip(overall_actions, overall_states):
                if actions.shape[1] < max_action_dim:
                    pad_width = max_action_dim - actions.shape[1]
                    actions = np.pad(actions, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                padded_actions.append(actions)
                
                if states.shape[1] < max_state_dim:
                    pad_width = max_state_dim - states.shape[1]
                    states = np.pad(states, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                padded_states.append(states)
            
            # 合并所有数据
            combined_actions = np.concatenate(padded_actions, axis=0)
            combined_states = np.concatenate(padded_states, axis=0)
            
            print(f"Total frames across all embodiments: {combined_actions.shape[0]}")
            print(f"Max action dimension: {max_action_dim}")
            print(f"Max state dimension: {max_state_dim}")
            
            overall_stats = {
                "action": {
                    "max": np.max(combined_actions, axis=0).tolist(),
                    "min": np.min(combined_actions, axis=0).tolist(),
                    "mean": np.mean(combined_actions, axis=0).tolist(),
                    "std": np.std(combined_actions, axis=0).tolist(),
                    "q01": np.percentile(combined_actions, 1, axis=0).tolist(),
                    "q99": np.percentile(combined_actions, 99, axis=0).tolist(),
                },
                "state": {
                    "max": np.max(combined_states, axis=0).tolist(),
                    "min": np.min(combined_states, axis=0).tolist(),
                    "mean": np.mean(combined_states, axis=0).tolist(),
                    "std": np.std(combined_states, axis=0).tolist(),
                    "q01": np.percentile(combined_states, 1, axis=0).tolist(),
                    "q99": np.percentile(combined_states, 99, axis=0).tolist(),
                }
            }
        
        # 保存统计信息
        output_data = {
            "embodiments": list(all_stats.keys()),
            "per_embodiment_stats": all_stats
        }
        
        if overall_stats:
            output_data["overall_stats"] = overall_stats
        
        with open(args.output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Statistics saved to: {args.output_path}")
        print(f"{'='*60}")
        
        # 打印每个 embodiment 的统计摘要
        for embodiment, data in all_stats.items():
            print(f"\n=== {embodiment} Statistics Summary ===")
            print(f"Environments: {', '.join(data['environments'])}")
            stats = data['stats']
            print(f"Action dimension: {len(stats['action']['mean'])}")
            print(f"State dimension: {len(stats['state']['mean'])}")
            print(f"\nAction stats:")
            print(f"  Mean range: [{np.min(stats['action']['mean']):.4f}, {np.max(stats['action']['mean']):.4f}]")
            print(f"  Std range: [{np.min(stats['action']['std']):.4f}, {np.max(stats['action']['std']):.4f}]")
            print(f"  Min: [{np.min(stats['action']['min']):.4f}, {np.max(stats['action']['min']):.4f}]")
            print(f"  Max: [{np.min(stats['action']['max']):.4f}, {np.max(stats['action']['max']):.4f}]")
            print(f"\nState stats:")
            print(f"  Mean range: [{np.min(stats['state']['mean']):.4f}, {np.max(stats['state']['mean']):.4f}]")
            print(f"  Std range: [{np.min(stats['state']['std']):.4f}, {np.max(stats['state']['std']):.4f}]")
            print(f"  Min: [{np.min(stats['state']['min']):.4f}, {np.max(stats['state']['min']):.4f}]")
            print(f"  Max: [{np.min(stats['state']['max']):.4f}, {np.max(stats['state']['max']):.4f}]")
        
        # 打印总体统计摘要
        if overall_stats:
            print(f"\n{'='*60}")
            print(f"=== Overall Statistics (All Embodiments Combined) ===")
            print(f"{'='*60}")
            print(f"Action dimension (padded): {len(overall_stats['action']['mean'])}")
            print(f"State dimension (padded): {len(overall_stats['state']['mean'])}")
            print(f"\nOverall Action stats:")
            print(f"  Mean range: [{np.min(overall_stats['action']['mean']):.4f}, {np.max(overall_stats['action']['mean']):.4f}]")
            print(f"  Std range: [{np.min(overall_stats['action']['std']):.4f}, {np.max(overall_stats['action']['std']):.4f}]")
            print(f"  Min: [{np.min(overall_stats['action']['min']):.4f}, {np.max(overall_stats['action']['min']):.4f}]")
            print(f"  Max: [{np.min(overall_stats['action']['max']):.4f}, {np.max(overall_stats['action']['max']):.4f}]")
            print(f"\nOverall State stats:")
            print(f"  Mean range: [{np.min(overall_stats['state']['mean']):.4f}, {np.max(overall_stats['state']['mean']):.4f}]")
            print(f"  Std range: [{np.min(overall_stats['state']['std']):.4f}, {np.max(overall_stats['state']['std']):.4f}]")
            print(f"  Min: [{np.min(overall_stats['state']['min']):.4f}, {np.max(overall_stats['state']['min']):.4f}]")
            print(f"  Max: [{np.min(overall_stats['state']['max']):.4f}, {np.max(overall_stats['state']['max']):.4f}]")
    
    elif args.mode == "visualize":
        # 可视化模式
        if not args.env_name:
            raise ValueError("For 'visualize' mode, please specify at least one environment name")
        
        for embodiment in embodiments:
            print(f"\n{'='*60}")
            print(f"Visualizing embodiment: {embodiment}")
            print(f"{'='*60}")
            
            # 初始化读取器（注意：visualize 模式不需要构建完整索引）
            reader = RoboMINDDatasetReader(
                embodiment=embodiment,
                dataset_path=args.dataset_path,
                resolution=args.resolution,
                env_names=[]  # 传入空列表，避免构建索引
            )
            
            # 为每个环境生成视频
            for env_name in args.env_name:
                print(f"\nProcessing environment: {env_name}")
                try:
                    all_videos = reader.visualize_episodes(
                        env_name=env_name,
                        output_dir=args.output_dir,
                        num_episodes=args.num_episodes,
                        fps=args.fps,
                        show_depth=args.show_depth
                    )
                    
                    print(f"\n{'='*60}")
                    print(f"Visualization complete for {env_name}")
                    print(f"Total episodes: {len(all_videos)}")
                    print(f"Output directory: {os.path.join(args.output_dir, embodiment, env_name)}")
                    print(f"{'='*60}")
                    
                except Exception as e:
                    print(f"Error visualizing {env_name}: {e}")
                    continue



if __name__ == "__main__":
    main()
