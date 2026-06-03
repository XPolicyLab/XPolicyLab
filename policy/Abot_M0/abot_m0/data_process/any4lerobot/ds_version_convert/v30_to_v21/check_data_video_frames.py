#!/usr/bin/env python3
"""
检查数据集中每个子集的 data 和 video 帧数是否一致
随机抽查每个子集的一个 episode
"""
import argparse
import json
import random
import subprocess
from pathlib import Path

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    try:
        import pandas as pd
        HAS_PYARROW = False
    except ImportError:
        raise ImportError("Neither pyarrow nor pandas is available. Please install one of them.")

try:
    from lerobot.datasets.utils import load_info
except ImportError:
    # Fallback: read info.json directly
    def load_info(root: Path) -> dict:
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found at {info_path}")
        with open(info_path, 'r') as f:
            return json.load(f)


def count_data_frames(dataset_path: Path, episode_index: int) -> int:
    """统计 data 文件中指定 episode 的帧数"""
    info = load_info(dataset_path)
    data_path_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    
    # 计算 episode 所在的 chunk
    chunk_size = info.get("chunks_size", 1000)
    episode_chunk = episode_index // chunk_size
    
    data_path = dataset_path / data_path_template.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index
    )
    
    if not data_path.exists():
        return -1  # 文件不存在
    
    try:
        if HAS_PYARROW:
            table = pq.read_table(data_path)
            return len(table)
        else:
            df = pd.read_parquet(data_path)
            return len(df)
    except Exception as e:
        print(f"Error reading data file {data_path}: {e}")
        return -1


def count_video_frames(video_path: Path) -> int:
    """统计视频文件中的帧数"""
    if not video_path.exists():
        return -1
    
    try:
        import subprocess
        
        # 方法1: 使用 ffprobe 获取精确帧数（推荐）
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            try:
                frame_count = int(result.stdout.strip())
                if frame_count > 0:
                    return frame_count
            except ValueError:
                pass
        
        # 方法2: 使用 ffprobe 获取 duration 和 fps，然后计算
        from lerobot.datasets.video_utils import get_video_info
        video_info = get_video_info(video_path)
        fps = video_info.get("video.fps", 30.0)
        
        cmd_duration = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        result_duration = subprocess.run(cmd_duration, capture_output=True, text=True, timeout=30)
        if result_duration.returncode == 0 and result_duration.stdout.strip():
            try:
                duration = float(result_duration.stdout.strip())
                frame_count = int(round(duration * fps))
                if frame_count > 0:
                    return frame_count
            except ValueError:
                pass
        
        # 方法3: 使用 format duration（最后备用）
        cmd_format = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        result_format = subprocess.run(cmd_format, capture_output=True, text=True, timeout=30)
        if result_format.returncode == 0 and result_format.stdout.strip():
            try:
                duration = float(result_format.stdout.strip())
                frame_count = int(round(duration * fps))
                if frame_count > 0:
                    return frame_count
            except ValueError:
                pass
        
        return -1
    except Exception as e:
        print(f"Error counting video frames {video_path}: {e}")
        return -1


def get_video_keys(dataset_path: Path) -> list[str]:
    """获取数据集中的所有 video keys"""
    try:
        info = load_info(dataset_path)
        features = info.get("features", {})
        video_keys = [key for key, ft in features.items() if ft.get("dtype") == "video"]
        return video_keys
    except Exception as e:
        print(f"Error getting video keys: {e}")
        return []


def get_all_episodes(dataset_path: Path) -> list[int]:
    """获取数据集中所有 episode 索引"""
    try:
        info = load_info(dataset_path)
        total_episodes = info.get("total_episodes", 0)
        return list(range(total_episodes))
    except Exception as e:
        print(f"Error getting episodes: {e}")
        return []


def check_subset(dataset_path: Path, subset_name: str) -> dict:
    """检查单个子集的 data 和 video 帧数"""
    result = {
        "subset": subset_name,
        "episode_index": None,
        "data_frames": -1,
        "video_frames": {},
        "match": False,
        "error": None
    }
    
    try:
        # 获取所有 episode
        episodes = get_all_episodes(dataset_path)
        if len(episodes) == 0:
            result["error"] = "No episodes found"
            return result
        
        # 随机选择一个 episode
        episode_index = random.choice(episodes)
        result["episode_index"] = episode_index
        
        # 统计 data 帧数
        data_frames = count_data_frames(dataset_path, episode_index)
        result["data_frames"] = data_frames
        
        if data_frames <= 0:
            result["error"] = f"Failed to read data file for episode {episode_index}"
            return result
        
        # 获取 video keys
        video_keys = get_video_keys(dataset_path)
        if len(video_keys) == 0:
            result["error"] = "No video keys found"
            return result
        
        # 统计每个 camera 的视频帧数
        info = load_info(dataset_path)
        video_path_template = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
        chunk_size = info.get("chunks_size", 1000)
        episode_chunk = episode_index // chunk_size
        
        all_match = True
        for video_key in video_keys:
            video_path = dataset_path / video_path_template.format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=episode_index
            )
            print(video_path)
            import pdb
            pdb.set_trace()
            video_frames = count_video_frames(video_path)
            result["video_frames"][video_key] = video_frames
            
            if video_frames != data_frames:
                all_match = False
        
        result["match"] = all_match
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="检查数据集中 data 和 video 帧数是否一致")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="数据集根路径（包含多个子集）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（用于可重复性）"
    )
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return
    
    # 获取所有子集（子目录）
    subsets = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if len(subsets) == 0:
        print(f"No subsets found in {dataset_path}")
        return
    
    print(f"Found {len(subsets)} subsets")
    print("=" * 100)
    
    results = []
    for subset_path in sorted(subsets):
        subset_name = subset_path.name
        print(f"\nChecking subset: {subset_name}")
        result = check_subset(subset_path, subset_name)
        results.append(result)
        
        if result["error"]:
            print(f"  ❌ Error: {result['error']}")
        else:
            print(f"  Episode: {result['episode_index']}")
            print(f"  Data frames: {result['data_frames']}")
            print(f"  Video frames:")
            for video_key, video_frames in result["video_frames"].items():
                match_str = "✓" if video_frames == result["data_frames"] else "✗"
                print(f"    {video_key}: {video_frames} {match_str}")
            
            if result["match"]:
                print(f"  ✅ All video frames match data frames")
            else:
                print(f"  ❌ Frame count mismatch detected")
    
    # 汇总统计
    print("\n" + "=" * 100)
    print("Summary:")
    total_subsets = len(results)
    successful_checks = sum(1 for r in results if r["error"] is None)
    matched_subsets = sum(1 for r in results if r.get("match", False))
    
    print(f"  Total subsets: {total_subsets}")
    print(f"  Successful checks: {successful_checks}")
    print(f"  Matched subsets: {matched_subsets}")
    print(f"  Mismatched subsets: {successful_checks - matched_subsets}")
    
    if successful_checks < total_subsets:
        print(f"  Failed checks: {total_subsets - successful_checks}")
        print("\nFailed subsets:")
        for r in results:
            if r["error"]:
                print(f"    {r['subset']}: {r['error']}")


if __name__ == "__main__":
    main()
