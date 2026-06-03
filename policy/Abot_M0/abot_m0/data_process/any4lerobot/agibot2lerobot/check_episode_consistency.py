#!/usr/bin/env python3
"""
检查转换后的task中，data和video存储的episode id是否一一对应

使用示例:
    python check_episode_consistency.py --task-dir /path/to/task_dataset --eef-type gripper
    python check_episode_consistency.py --task-dir /path/to/task_dataset --eef-type gripper --save-depth
"""
import argparse
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict

from agibot_utils.config import AgiBotWorld_TASK_TYPE
from agibot_utils.lerobot_utils import generate_features_from_config
from agibot_h5 import AgiBotDataset


def get_episode_ids_from_data_files(local_dir: Path) -> set[int]:
    """
    从data目录下的parquet文件中实际读取episode_index
    返回: set[episode_index]
    """
    episode_ids = set()
    data_dir = local_dir / "data"
    
    if not data_dir.exists():
        print(f"警告: data目录不存在: {data_dir}")
        return episode_ids
    
    # 查找所有parquet文件
    parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    
    if not parquet_files:
        print(f"警告: 在 {data_dir} 中未找到parquet文件")
        return episode_ids
    
    print(f"  找到 {len(parquet_files)} 个parquet文件")
    
    # 读取每个parquet文件
    for parquet_file in parquet_files:
        try:
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            
            # 检查是否有episode_index列
            if 'episode_index' in df.columns:
                unique_episodes = df['episode_index'].unique()
                episode_ids.update([int(ep_idx) for ep_idx in unique_episodes])
            else:
                print(f"  警告: {parquet_file.name} 中没有episode_index列")
        except Exception as e:
            print(f"  错误: 读取 {parquet_file} 失败: {e}")
    
    return episode_ids


def get_episode_ids_from_video_files(dataset: AgiBotDataset, local_dir: Path) -> dict[str, set[int]]:
    """
    从episode metadata中获取每个episode对应的video文件信息，并检查文件是否存在
    返回: {video_key: set(episode_ids)} - 只包含video文件确实存在的episode
    """
    video_episode_ids = defaultdict(set)
    
    if not hasattr(dataset.meta, 'episodes') or not dataset.meta.episodes:
        print("警告: 无法访问episode metadata")
        return dict(video_episode_ids)
    
    # 获取video路径模板
    video_path_template = dataset.meta.info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    
    # 遍历所有episode
    for item in dataset.meta.episodes:
        episode_index = item.get('episode_index')
        if episode_index is None:
            continue
        
        try:
            episode_index = int(episode_index)
        except (ValueError, TypeError):
            continue
        
        # 遍历所有video keys
        for video_key in dataset.meta.video_keys:
            # 从episode metadata中获取video文件的chunk_index和file_index
            chunk_key = f"videos/{video_key}/chunk_index"
            file_key = f"videos/{video_key}/file_index"
            
            # 获取chunk_index和file_index
            chunk_index = item.get(chunk_key)
            file_index = item.get(file_key)
            
            # 检查值是否存在且不为None
            if chunk_index is not None and file_index is not None:
                # 转换为整数
                try:
                    chunk_index = int(chunk_index)
                    file_index = int(file_index)
                except (ValueError, TypeError) as e:
                    continue
                
                # 构建video文件路径
                try:
                    video_path = local_dir / video_path_template.format(
                        video_key=video_key,
                        chunk_index=chunk_index,
                        file_index=file_index
                    )
                except KeyError:
                    # 如果format失败，尝试手动构建路径
                    video_path = local_dir / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
                
                # 检查文件是否存在
                if video_path.exists():
                    video_episode_ids[video_key].add(episode_index)
    
    return dict(video_episode_ids)


def get_episode_ids_from_metadata(dataset: AgiBotDataset) -> set[int]:
    """从metadata中获取所有episode_index（用于对比）"""
    if hasattr(dataset.meta, 'episodes') and dataset.meta.episodes:
        episodes_dict = dataset.meta.episodes.to_dict()
        if 'episode_index' in episodes_dict:
            return set([int(ep_idx) for ep_idx in episodes_dict['episode_index']])
    return set()


def check_episode_consistency(local_dir: Path, eef_type: str, save_depth: bool = False):
    """
    检查指定task的数据和video的episode id是否一致
    
    Args:
        local_dir: 数据集本地目录路径
        eef_type: 末端执行器类型 (gripper, dexhand, tactile)
        save_depth: 是否保存depth数据
    """
    print(f"\n{'='*60}")
    print(f"检查任务: {local_dir.name}")
    print(f"目录路径: {local_dir}")
    print(f"{'='*60}\n")
    
    # 加载数据集配置
    agibot_world_config = AgiBotWorld_TASK_TYPE[eef_type]["task_config"]
    features = generate_features_from_config(agibot_world_config)
    
    if not save_depth:
        features.pop("observation.images.head_depth", None)
    
    # 加载数据集
    try:
        dataset = AgiBotDataset(
            repo_id=local_dir.name,
            root=local_dir,
        )
        
        print(f"成功加载数据集")
        print(f"总episode数: {dataset.num_episodes}")
        print(f"总frame数: {dataset.meta.total_frames}")
        print(f"Video keys: {dataset.meta.video_keys}\n")
    except Exception as e:
        print(f"错误: 无法加载数据集 - {e}")
        return False
    
    # 从数据中获取episode ids
    print("正在从数据中提取episode ids...")
    data_episode_ids = get_episode_ids_from_data(dataset)
    print(f"数据中的episode ids: {sorted(data_episode_ids)}")
    print(f"数据中的episode数量: {len(data_episode_ids)}\n")
    
    # 从video文件中获取episode ids（通过检查metadata中指定的video文件是否存在）
    print("正在从video metadata中提取episode ids并检查文件是否存在...")
    video_episode_ids_dict = get_episode_ids_from_videos(dataset, local_dir)
    
    if not video_episode_ids_dict:
        print("警告: 未找到任何video文件")
        return False
    
    # 检查每个video key的一致性
    all_consistent = True
    for video_key, video_episode_ids in video_episode_ids_dict.items():
        print(f"\n检查 video key: {video_key}")
        print(f"  数据中的episode数量: {len(data_episode_ids)}")
        print(f"  找到对应video文件的episode数量: {len(video_episode_ids)}")
        
        # 检查是否一致
        data_only = data_episode_ids - video_episode_ids
        video_only = video_episode_ids - data_episode_ids
        
        if data_only:
            print(f"  ❌ 数据中有但video文件不存在的episode ids: {sorted(data_only)}")
            all_consistent = False
        
        if video_only:
            print(f"  ⚠️  Video文件存在但数据中没有的episode ids: {sorted(video_only)}")
            # 这种情况可能是正常的，因为一个video文件可能包含多个episode
        
        if not data_only and not video_only and len(data_episode_ids) == len(video_episode_ids):
            print(f"  ✅ 数据中的所有episode都有对应的video文件")
        
        # 检查数量是否一致
        if len(data_episode_ids) != len(video_episode_ids):
            print(f"  ⚠️  数量不一致: 数据中有 {len(data_episode_ids)} 个episode, 但只有 {len(video_episode_ids)} 个episode有对应的video文件")
            if len(video_episode_ids) < len(data_episode_ids):
                all_consistent = False
    
    # 总结
    print(f"\n{'='*60}")
    if all_consistent:
        print("✅ 检查通过: 所有video key的episode ids都与数据中的episode ids一致")
    else:
        print("❌ 检查失败: 发现不一致的episode ids")
    print(f"{'='*60}\n")
    
    return all_consistent


def main():
    parser = argparse.ArgumentParser(description="检查转换后的task中，data和video存储的episode id是否一一对应")
    parser.add_argument("--task-dir", type=Path, default="/mnt/xlab-nas-2/vla_dataset/lerobot/agibot_convert/agibotworld/task_365", help="任务数据集目录路径")
    parser.add_argument("--eef-type", type=str, choices=["gripper", "dexhand", "tactile"], 
                       default="gripper", help="末端执行器类型")
    parser.add_argument("--save-depth", action="store_true", help="是否保存depth数据")
    args = parser.parse_args()

    with open("/mnt/xlab-nas-2/vla_dataset/lerobot/agibot_convert/.completed_tasks.txt","r") as f:
        completed_tasks = {line.strip() for line in f if line.strip()}
    for completed_task in completed_tasks:
        task_dir = Path(f"/mnt/xlab-nas-2/vla_dataset/lerobot/agibot_convert/agibotworld/{completed_task}")
        if not task_dir.exists():
            print(f"错误: 目录不存在: {task_dir}")
            continue
        check_episode_consistency(task_dir, args.eef_type, args.save_depth)



if __name__ == "__main__":
    main()
