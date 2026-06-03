#!/usr/bin/env python3
"""
示例脚本：展示如何索引lerobot v3格式数据集中每个video对应的episode和时间戳对应的帧

使用方法:
    python index_video_episode.py --dataset-dir /mnt/xlab-nas-2/vla_dataset/lerobot/agibot_convert/agibotworld/task_327
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 添加项目路径到sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agibot2lerobot.agibot_h5 import AgiBotDataset
from agibot2lerobot.agibot_utils.config import AgiBotWorld_TASK_TYPE
from agibot2lerobot.agibot_utils.lerobot_utils import generate_features_from_config


def load_dataset(dataset_dir: Path, eef_type: str = "gripper", save_depth: bool = False) -> AgiBotDataset:
    """
    加载lerobot数据集
    
    Args:
        dataset_dir: 数据集根目录
        eef_type: 末端执行器类型
        save_depth: 是否保存depth数据
        
    Returns:
        AgiBotDataset实例
    """
    agibot_world_config = AgiBotWorld_TASK_TYPE[eef_type]["task_config"]
    features = generate_features_from_config(agibot_world_config)
    
    if not save_depth:
        features.pop("observation.images.head_depth", None)
    
    dataset = AgiBotDataset(
        repo_id=dataset_dir.name,
        root=dataset_dir,
    )
    
    return dataset


def get_episode_metadata_dict(dataset: AgiBotDataset) -> Dict[int, Dict]:
    """
    从dataset.meta.episodes获取所有episode的元数据
    
    Args:
        dataset: AgiBotDataset实例
        
    Returns:
        字典，键为episode_index，值为episode元数据字典
    """
    if not hasattr(dataset.meta, 'episodes') or not dataset.meta.episodes:
        raise ValueError("无法访问dataset.meta.episodes")
    
    return dataset.meta.episodes


def get_video_path(dataset_dir: Path, video_key: str, chunk_index: int, file_index: int) -> Path:
    """
    根据video_key, chunk_index, file_index获取video文件路径
    
    Args:
        dataset_dir: 数据集根目录
        video_key: video的key，例如 "observation.images.head"
        chunk_index: chunk索引
        file_index: file索引
        
    Returns:
        video文件路径
    """
    video_path_template = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    video_path = dataset_dir / video_path_template.format(
        video_key=video_key,
        chunk_index=chunk_index,
        file_index=file_index
    )
    return video_path


def get_episode_video_info(episodes_dict: Dict[int, Dict], episode_index: int, video_key: str) -> Optional[Dict]:
    """
    获取指定episode的video信息
    
    Args:
        episodes_dict: episode元数据字典（从dataset.meta.episodes获取）
        episode_index: episode索引
        video_key: video的key
        
    Returns:
        包含video信息的字典，如果不存在则返回None
    """
    if episode_index not in episodes_dict:
        return None
    
    episode_meta = episodes_dict[episode_index]
    
    # 获取video相关的列（使用斜杠格式）
    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"
    from_ts_col = f"videos/{video_key}/from_timestamp"
    to_ts_col = f"videos/{video_key}/to_timestamp"
    
    if chunk_col not in episode_meta or episode_meta[chunk_col] is None:
        return None
    
    return {
        "episode_index": int(episode_index),
        "chunk_index": int(episode_meta[chunk_col]),
        "file_index": int(episode_meta[file_col]),
        "from_timestamp": float(episode_meta[from_ts_col]),
        "to_timestamp": float(episode_meta[to_ts_col]),
        "length": int(episode_meta["length"]),
    }


def find_episode_by_video_file(
    episodes_dict: Dict[int, Dict], 
    video_key: str, 
    chunk_index: int, 
    file_index: int
) -> List[Dict]:
    """
    根据video文件找到所有相关的episode
    
    Args:
        episodes_dict: episode元数据字典（从dataset.meta.episodes获取）
        video_key: video的key
        chunk_index: chunk索引
        file_index: file索引
        
    Returns:
        包含该video文件的所有episode信息列表
    """
    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"
    from_ts_col = f"videos/{video_key}/from_timestamp"
    to_ts_col = f"videos/{video_key}/to_timestamp"
    
    results = []
    for episode_index, episode_meta in episodes_dict.items():
        if (episode_meta.get(chunk_col) == chunk_index and 
            episode_meta.get(file_col) == file_index):
            results.append({
                "episode_index": int(episode_index),
                "from_timestamp": float(episode_meta[from_ts_col]),
                "to_timestamp": float(episode_meta[to_ts_col]),
                "length": int(episode_meta["length"]),
            })
    
    return sorted(results, key=lambda x: x["episode_index"])


def find_frame_by_timestamp(
    episodes_dict: Dict[int, Dict],
    video_key: str,
    timestamp: float,
    fps: float = 30.0
) -> Optional[Dict]:
    """
    根据时间戳找到对应的episode和frame索引
    
    Args:
        episodes_dict: episode元数据字典（从dataset.meta.episodes获取）
        video_key: video的key
        timestamp: 时间戳（秒）
        fps: 视频帧率
        
    Returns:
        包含episode和frame信息的字典，如果不存在则返回None
    """
    from_ts_col = f"videos/{video_key}/from_timestamp"
    to_ts_col = f"videos/{video_key}/to_timestamp"
    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"
    
    # 找到包含该时间戳的episode
    matching_episode = None
    for episode_index, episode_meta in episodes_dict.items():
        from_ts = float(episode_meta[from_ts_col])
        to_ts = float(episode_meta[to_ts_col])
        if from_ts <= timestamp < to_ts:
            matching_episode = (episode_index, episode_meta)
            break
    
    if matching_episode is None:
        return None
    
    episode_index, episode_meta = matching_episode
    
    # 计算在该episode内的相对时间戳
    relative_timestamp = timestamp - float(episode_meta[from_ts_col])
    
    # 计算frame索引（在该episode内的frame索引）
    frame_index_in_episode = int(relative_timestamp * fps)
    
    # 确保frame_index不超过episode长度
    episode_length = int(episode_meta["length"])
    frame_index_in_episode = min(frame_index_in_episode, episode_length - 1)
    
    return {
        "episode_index": int(episode_index),
        "chunk_index": int(episode_meta[chunk_col]),
        "file_index": int(episode_meta[file_col]),
        "timestamp": timestamp,
        "relative_timestamp": relative_timestamp,
        "frame_index_in_episode": frame_index_in_episode,
        "frame_index_in_video": int((timestamp - float(episode_meta[from_ts_col])) * fps),
    }


def main():
    parser = argparse.ArgumentParser(description="索引lerobot v3格式数据集中video和episode的对应关系")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="数据集目录路径")
    parser.add_argument("--eef-type", type=str, choices=["gripper", "dexhand", "tactile"], 
                       default="gripper", help="末端执行器类型")
    parser.add_argument("--save-depth", action="store_true", help="是否保存depth数据")
    parser.add_argument("--episode-index", type=int, help="查询指定episode的video信息")
    parser.add_argument("--video-key", type=str, help="video key（例如: observation.images.head）")
    parser.add_argument("--chunk-index", type=int, help="查询指定chunk和file的episode")
    parser.add_argument("--file-index", type=int, help="查询指定chunk和file的episode")
    parser.add_argument("--timestamp", type=float, help="查询指定时间戳对应的episode和frame")
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        print(f"错误: 数据集目录不存在: {dataset_dir}")
        return
    
    # 加载数据集
    print(f"正在加载数据集: {dataset_dir}")
    try:
        dataset = load_dataset(dataset_dir, args.eef_type, args.save_depth)
        episodes_dict = get_episode_metadata_dict(dataset)
        info = dataset.meta.info
    except Exception as e:
        print(f"错误: 无法加载数据集 - {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n数据集信息:")
    print(f"  - 总episode数: {dataset.num_episodes}")
    print(f"  - 总frame数: {dataset.meta.total_frames}")
    print(f"  - FPS: {info.get('fps', 30)}")
    
    # 获取所有video keys
    video_keys = dataset.meta.video_keys
    print(f"  - Video keys: {video_keys}\n")
    
    # 示例1: 查询指定episode的video信息
    if args.episode_index is not None:
        episode_idx = args.episode_index
        video_key = args.video_key or video_keys[0] if video_keys else None
        
        if video_key is None:
            print("错误: 需要指定--video-key")
            return
        
        print(f"查询episode {episode_idx} 的video信息 (video_key: {video_key}):")
        video_info = get_episode_video_info(episodes_dict, episode_idx, video_key)
        
        if video_info:
            print(f"  Episode索引: {video_info['episode_index']}")
            print(f"  Video文件: chunk-{video_info['chunk_index']:03d}/file-{video_info['file_index']:03d}.mp4")
            video_path = get_video_path(dataset_dir, video_key, video_info['chunk_index'], video_info['file_index'])
            print(f"  Video路径: {video_path}")
            print(f"  时间戳范围: {video_info['from_timestamp']:.3f}s - {video_info['to_timestamp']:.3f}s")
            print(f"  Episode长度: {video_info['length']} frames")
            print(f"  Video文件是否存在: {video_path.exists()}")
        else:
            print(f"  未找到episode {episode_idx} 的video信息")
    
    # 示例2: 根据video文件查找episode
    elif args.chunk_index is not None and args.file_index is not None:
        chunk_idx = args.chunk_index
        file_idx = args.file_index
        video_key = args.video_key or video_keys[0] if video_keys else None
        
        if video_key is None:
            print("错误: 需要指定--video-key")
            return
        
        print(f"查询video文件 chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4 对应的episode:")
        video_path = get_video_path(dataset_dir, video_key, chunk_idx, file_idx)
        print(f"  Video路径: {video_path}")
        print(f"  Video文件是否存在: {video_path.exists()}\n")
        
        episodes = find_episode_by_video_file(episodes_dict, video_key, chunk_idx, file_idx)
        
        if episodes:
            print(f"  找到 {len(episodes)} 个episode:")
            for ep in episodes:
                print(f"    - Episode {ep['episode_index']}: "
                      f"时间戳 {ep['from_timestamp']:.3f}s - {ep['to_timestamp']:.3f}s, "
                      f"长度 {ep['length']} frames")
        else:
            print(f"  未找到对应的episode")
    
    # 示例3: 根据时间戳查找frame
    elif args.timestamp is not None:
        timestamp = args.timestamp
        video_key = args.video_key or video_keys[0] if video_keys else None
        
        if video_key is None:
            print("错误: 需要指定--video-key")
            return
        
        fps = info.get("fps", 30.0)
        print(f"查询时间戳 {timestamp:.3f}s 对应的episode和frame (video_key: {video_key}):")
        
        frame_info = find_frame_by_timestamp(episodes_dict, video_key, timestamp, fps)
        
        if frame_info:
            print(f"  Episode索引: {frame_info['episode_index']}")
            print(f"  Video文件: chunk-{frame_info['chunk_index']:03d}/file-{frame_info['file_index']:03d}.mp4")
            video_path = get_video_path(
                dataset_dir, video_key, 
                frame_info['chunk_index'], frame_info['file_index']
            )
            print(f"  Video路径: {video_path}")
            print(f"  绝对时间戳: {frame_info['timestamp']:.3f}s")
            print(f"  相对时间戳（在episode内）: {frame_info['relative_timestamp']:.3f}s")
            print(f"  Frame索引（在episode内）: {frame_info['frame_index_in_episode']}")
            print(f"  Frame索引（在video文件内）: {frame_info['frame_index_in_video']}")
        else:
            print(f"  未找到时间戳 {timestamp:.3f}s 对应的episode")
    
    # 默认: 展示所有episode的video信息（前10个）
    else:
        print("\n展示前10个episode的video信息:")
        video_key = video_keys[0] if video_keys else None
        
        if video_key:
            print(f"使用video_key: {video_key}\n")
            sorted_episode_indices = sorted(episodes_dict.keys())[:10]
            for episode_idx in sorted_episode_indices:
                video_info = get_episode_video_info(episodes_dict, episode_idx, video_key)
                
                if video_info:
                    print(f"Episode {episode_idx}: "
                          f"chunk-{video_info['chunk_index']:03d}/file-{video_info['file_index']:03d}.mp4, "
                          f"时间戳 {video_info['from_timestamp']:.3f}s - {video_info['to_timestamp']:.3f}s, "
                          f"长度 {video_info['length']} frames")
        else:
            print("未找到video keys")


if __name__ == "__main__":
    main()






