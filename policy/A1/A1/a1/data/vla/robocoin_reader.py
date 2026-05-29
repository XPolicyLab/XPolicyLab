"""
RoboCoin 数据集读取器，继承 LeRobot 格式的 Wrapper。

RoboCOIN 目录下每个子目录（如 AIRBOT_MMK2_bowl_storage_pepper）均为标准 LeRobot 数据集，
使用 meta/info.json + data/videos 等结构，故直接复用 LeRobotDatasetWrapper 的逻辑。
本类在此基础上预留扩展点，便于后续按 RoboCoin 需求修改（如多数据集混合、默认归一化等）。
"""

import argparse
import os
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset,LeRobotDatasetMetadata,CODEBASE_VERSION
from a1.data.vla.lerobot_datasets import (
    LeRobotDatasetWrapper,
    NormalizationType,
    normalize_action_and_proprio,
    test_lerobot_dataset,
    get_stats,
    debug_dataset_repeat_read,
    debug_image_channel_order,
    export_lerobot_trajectory_video,
    visualize_lerobot_dataset,
)

__all__ = [
    "RoboCoinDatasetWrapper",
    "LeRobotDatasetWrapper",
    "NormalizationType",
    "normalize_action_and_proprio",
    "test_lerobot_dataset",
    "get_stats",
    "debug_dataset_repeat_read",
    "debug_image_channel_order",
    "export_lerobot_trajectory_video",
    "visualize_lerobot_dataset",
]


class RoboCoinDatasetWrapper(LeRobotDatasetWrapper):
    """RoboCoin 数据集封装，继承 LeRobotDatasetWrapper。"""

    def __init__(
        self,
        dataset_path,
        chunk_size=50,
        fixed_action_dim=32,
        normalization_type=None,
        pad_action_and_proprio=True,
        use_proprio=True,
        use_num_images=3,
        use_wrist_image=True,
        video_backend="pyav",
        num_episodes=None,
    ):

        self.use_proprio = use_proprio
        self.use_wrist_image = use_wrist_image
        self.normalization_type = normalization_type
        self.pad_action_and_proprio = pad_action_and_proprio
        self.use_num_images = use_num_images
        self.video_backend = video_backend 
        self.fixed_action_dim = fixed_action_dim
        dataset_meta = LeRobotDatasetMetadata(
            os.path.basename(dataset_path),dataset_path, CODEBASE_VERSION, force_cache_sync=False
        )
        fps = dataset_meta.fps
        self.camera_keys = dataset_meta.camera_keys    

        self.state_key = 'observation.state'
        self.action_key = 'action'
        if dataset_meta.features[self.state_key]['shape'][0] >self.fixed_action_dim:
            self.state_key = 'eef_sim_pose_state'
            self.action_key = 'eef_sim_pose_action'
        delta_timestamps = {key: [0,] for key in self.camera_keys}
        delta_timestamps.update({
            self.state_key: [0,],
            self.action_key: [t / fps for t in range(chunk_size)],
        })
        if num_episodes is not None:
            np.random.seed(42)
            if isinstance(num_episodes, int):
                num_episodes = min(num_episodes, dataset_meta.total_episodes)
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
            episodes = None
        self.dataset = LeRobotDataset(
            repo_id=os.path.basename(dataset_path),
            root=dataset_path, 
            episodes=episodes, 
            delta_timestamps=delta_timestamps, 
            video_backend=self.video_backend,
            check_timestamps=False
        )
        del dataset_meta

def _make_robocoin_wrapper(dataset_path, for_test=False):
    """根据 dataset_path 创建 RoboCoinDatasetWrapper，供 main 使用。"""
    return RoboCoinDatasetWrapper(
        dataset_path=dataset_path,
        chunk_size=50 if for_test else 50,
        fixed_action_dim=32,
        use_proprio=True,
        use_wrist_image=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default="data/RoboCOIN/AIRBOT_MMK2_bowl_storage_pepper",
        type=str,
    )
    parser.add_argument("--mode", default="test", type=str)
    parser.add_argument("--output_path", default=None, type=str)
    parser.add_argument("--save_dir", default="./lerobot_debug", type=str, help="debug_image_channel 时使用")
    args = parser.parse_args()
    dataset_path = args.dataset_path
    if args.mode == "test":
        ds = _make_robocoin_wrapper(dataset_path, for_test=True)
        test_lerobot_dataset(ds)
    elif args.mode == "get_stats":
        assert args.output_path is not None, "output_path is required"
        ds = _make_robocoin_wrapper(dataset_path)
        get_stats(ds, args.output_path)
    elif args.mode == "visualize":
        ds = _make_robocoin_wrapper(dataset_path)
        visualize_lerobot_dataset(ds)
    elif args.mode == "debug_repeat":
        ds = _make_robocoin_wrapper(dataset_path)
        debug_dataset_repeat_read(ds)
    elif args.mode == "debug_image_channel":
        ds = _make_robocoin_wrapper(dataset_path)
        debug_image_channel_order(ds, args.save_dir)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
