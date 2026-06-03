#!/usr/bin/env python3
"""
简单的检查脚本：验证 data 和 video 文件中的 episode 数量是否与 info.json 一致
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def count_episodes_from_data_files(dataset_path: Path) -> int:
    """
    从 data 目录的 parquet 文件中统计 episode 数量
    """
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        return 0
    
    episode_indices = set()
    
    # 遍历所有 chunk 目录
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        if not chunk_dir.is_dir():
            continue
        
        # 遍历每个 chunk 中的所有 parquet 文件
        for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
            try:
                table = pq.read_table(parquet_file)
                if "episode_index" in table.column_names:
                    # 获取所有唯一的 episode_index
                    episode_col = table["episode_index"]
                    unique_episodes = set(episode_col.to_pylist())
                    episode_indices.update(unique_episodes)
            except Exception as e:
                print(f"Error reading {parquet_file}: {e}")
    
    return len(episode_indices)


def count_episodes_from_video_files(dataset_path: Path, video_key: str) -> dict:
    """
    从 video 目录的 mp4 文件中统计 episode 数量
    通过读取 meta/episodes 来获取每个 episode 对应的视频文件位置
    
    Returns:
        dict: {"total_episodes": count, "video_files": count, "missing_episodes": list}
    """
    video_dir = dataset_path / "videos" / video_key
    if not video_dir.exists():
        return {"total_episodes": 0, "video_files": 0, "missing_episodes": [], "expected_episodes": 0}
    
    # 读取 episode metadata 来获取每个 episode 对应的视频文件
    episodes_meta_path = dataset_path / "meta" / "episodes"
    episode_to_video = {}  # {episode_index: (chunk_idx, file_idx)}
    
    if episodes_meta_path.exists():
        # 遍历所有 chunk 的 parquet 文件
        for chunk_meta_dir in sorted(episodes_meta_path.glob("chunk-*")):
            if not chunk_meta_dir.is_dir():
                continue
            
            for meta_file in sorted(chunk_meta_dir.glob("file-*.parquet")):
                try:
                    table = pq.read_table(meta_file)
                    if "episode_index" in table.column_names:
                        chunk_col = f"videos/{video_key}/chunk_index"
                        file_col = f"videos/{video_key}/file_index"
                        
                        if chunk_col in table.column_names and file_col in table.column_names:
                            records = table.to_pylist()
                            for record in records:
                                ep_idx = record.get("episode_index")
                                chunk_idx = record.get(chunk_col)
                                file_idx = record.get(file_col)
                                if ep_idx is not None and chunk_idx is not None and file_idx is not None:
                                    episode_to_video[int(ep_idx)] = (int(chunk_idx), int(file_idx))
                except Exception as e:
                    print(f"Error reading episode metadata {meta_file}: {e}")
    
    # 统计实际存在的视频文件（所有 chunk 中的所有文件）
    video_files = set()
    for chunk_dir in sorted(video_dir.glob("chunk-*")):
        if not chunk_dir.is_dir():
            continue
        
        chunk_idx = int(chunk_dir.name.split("-")[-1])
        
        for video_file in sorted(chunk_dir.glob("file-*.mp4")):
            file_idx = int(video_file.stem.split("-")[-1])
            video_files.add((chunk_idx, file_idx))
    
    # 检查每个 episode 的视频文件是否存在
    existing_episodes = set()
    missing_episodes = []
    
    for ep_idx, (chunk_idx, file_idx) in episode_to_video.items():
        if (chunk_idx, file_idx) in video_files:
            existing_episodes.add(ep_idx)
        else:
            missing_episodes.append((ep_idx, chunk_idx, file_idx))
    
    return {
        "total_episodes": len(existing_episodes),
        "video_files": len(video_files),
        "missing_episodes": missing_episodes,
        "expected_episodes": len(episode_to_video),
    }


def check_dataset_episodes(dataset_path: Path) -> dict:
    """
    检查数据集的 episode 数量
    
    Returns:
        dict: 包含检查结果的字典
    """
    result = {
        "dataset_path": str(dataset_path),
        "info_episodes": 0,
        "data_episodes": 0,
        "video_episodes": {},
        "status": "unknown",
        "issues": [],
    }
    
    # 1. 读取 info.json
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        result["status"] = "no_info"
        result["issues"].append("info.json not found")
        return result
    
    try:
        with open(info_path, 'r') as f:
            info = json.load(f)
        result["info_episodes"] = info.get("total_episodes", 0)
        
        # 获取视频 keys
        features = info.get("features", {})
        video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    except Exception as e:
        result["status"] = "error_reading_info"
        result["issues"].append(f"Error reading info.json: {e}")
        return result
    
    # 2. 统计 data 目录中的 episode 数
    result["data_episodes"] = count_episodes_from_data_files(dataset_path)
    
    # 3. 统计每个 video key 的 episode 数
    for video_key in video_keys:
        video_info = count_episodes_from_video_files(dataset_path, video_key)
        result["video_episodes"][video_key] = video_info
    
    # 4. 检查是否匹配
    issues = []
    
    if result["info_episodes"] != result["data_episodes"]:
        issues.append(
            f"Data episodes mismatch: info.json={result['info_episodes']}, "
            f"actual data files={result['data_episodes']}"
        )
    
    for video_key, video_info in result["video_episodes"].items():
        expected = result["info_episodes"]
        actual = video_info["total_episodes"]
        if expected != actual:
            issues.append(
                f"Video {video_key} episodes mismatch: expected={expected}, actual={actual}"
            )
            if video_info["missing_episodes"]:
                missing_samples = video_info["missing_episodes"][:5]
                issues.append(
                    f"  Missing video files: {missing_samples}"
                    + (f" ... and {len(video_info['missing_episodes']) - 5} more" 
                       if len(video_info['missing_episodes']) > 5 else "")
                )
    
    if issues:
        result["status"] = "mismatch"
        result["issues"] = issues
    else:
        result["status"] = "ok"
    
    return result


def main():
    parser = argparse.ArgumentParser(description="简单检查数据集的 episode 数量")
    parser.add_argument("--dataset-path", type=Path, required=True, help="数据集路径（包含 meta/ 和 data/ 目录）")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["simple", "detailed", "json"],
        default="simple",
        help="输出格式",
    )
    
    args = parser.parse_args()
    
    if not args.dataset_path.exists():
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        return
    
    result = check_dataset_episodes(args.dataset_path)
    
    if args.output_format == "simple":
        print(f"Dataset: {args.dataset_path.name}")
        print(f"  info.json episodes: {result['info_episodes']}")
        print(f"  data episodes: {result['data_episodes']}")
        for video_key, video_info in result["video_episodes"].items():
            print(f"  video {video_key} episodes: {video_info['total_episodes']}")
        
        if result["status"] == "ok":
            print("  Status: ✓ All episodes match")
        else:
            print(f"  Status: ✗ {result['status']}")
            for issue in result["issues"]:
                print(f"    - {issue}")
    
    elif args.output_format == "detailed":
        print("=" * 80)
        print(f"Dataset: {result['dataset_path']}")
        print("=" * 80)
        print(f"\ninfo.json:")
        print(f"  total_episodes: {result['info_episodes']}")
        
        print(f"\ndata/ directory:")
        print(f"  Unique episodes found: {result['data_episodes']}")
        if result['info_episodes'] != result['data_episodes']:
            print(f"  ⚠️  Mismatch!")
        
        print(f"\nvideos/ directory:")
        for video_key, video_info in result["video_episodes"].items():
            print(f"  {video_key}:")
            print(f"    Episodes found: {video_info['total_episodes']}")
            print(f"    Video files: {video_info['video_files']}")
            if video_info.get("expected_episodes", 0) != video_info['total_episodes']:
                print(f"    ⚠️  Expected: {video_info.get('expected_episodes', 'unknown')}")
            if video_info["missing_episodes"]:
                print(f"    Missing files: {len(video_info['missing_episodes'])}")
                for ep_idx, chunk_idx, file_idx in video_info["missing_episodes"][:10]:
                    print(f"      Episode {ep_idx}: chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4")
        
        print(f"\nStatus: {result['status']}")
        if result["issues"]:
            print("\nIssues:")
            for issue in result["issues"]:
                print(f"  - {issue}")
    
    elif args.output_format == "json":
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
