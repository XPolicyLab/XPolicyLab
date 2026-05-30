#!/usr/bin/env python3
"""
检查数据转换进度，对比原始数据量和转换后的数据量
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_source_stats(json_path: Path) -> Dict:
    """加载原始数据统计信息"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def count_data_files(task_dir: Path) -> int:
    """统计 data 目录下的唯一 episode 数量"""
    data_dir = task_dir / "data"
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
                import pyarrow.parquet as pq
                table = pq.read_table(parquet_file)
                if "episode_index" in table.column_names:
                    # 获取所有唯一的 episode_index
                    episode_col = table["episode_index"]
                    unique_episodes = set(episode_col.to_pylist())
                    episode_indices.update(unique_episodes)
            except Exception as e:
                # 如果读取失败，跳过该文件
                pass
    
    return len(episode_indices)


def count_video_files(task_dir: Path) -> Dict[str, int]:
    """统计 videos 目录下各个相机的唯一 episode 数量
    
    目录结构：videos/camera_key/chunk-*/file-*.mp4
    通过读取 meta/episodes 来统计每个 camera 的 episode 数量
    """
    videos_dir = task_dir / "videos"
    if not videos_dir.exists():
        return {}
    
    # 读取 episode metadata 来获取视频信息
    episodes_meta_path = task_dir / "meta" / "episodes"
    if not episodes_meta_path.exists():
        return {}
    
    # 统计每个 camera 的 episode 数量
    camera_episodes = defaultdict(set)
    
    # 遍历所有 camera 目录
    camera_dirs = [d for d in videos_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for camera_dir in camera_dirs:
        camera_key = camera_dir.name
        
        # 从 meta/episodes 中读取该 camera 的 episode 信息
        for chunk_meta_dir in sorted(episodes_meta_path.glob("chunk-*")):
            if not chunk_meta_dir.is_dir():
                continue
            
            for meta_file in sorted(chunk_meta_dir.glob("file-*.parquet")):
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(meta_file)
                    
                    if "episode_index" not in table.column_names:
                        continue
                    
                    # 检查该 camera 的列是否存在
                    chunk_col = f"videos/{camera_key}/chunk_index"
                    file_col = f"videos/{camera_key}/file_index"
                    
                    if chunk_col in table.column_names and file_col in table.column_names:
                        records = table.to_pylist()
                        for record in records:
                            ep_idx = record.get("episode_index")
                            chunk_idx = record.get(chunk_col)
                            file_idx = record.get(file_col)
                            
                            if ep_idx is not None and chunk_idx is not None and file_idx is not None:
                                # 检查视频文件是否存在
                                video_file = camera_dir / f"chunk-{int(chunk_idx):03d}" / f"file-{int(file_idx):03d}.mp4"
                                if video_file.exists():
                                    camera_episodes[camera_key].add(int(ep_idx))
                except Exception as e:
                    # 读取失败，跳过
                    pass
    
    # 转换为数量
    result = {camera: len(episodes) for camera, episodes in camera_episodes.items()}
    return result


def check_info_file(task_dir: Path) -> bool:
    """检查 info.json 是否存在"""
    info_file = task_dir / "meta" / "info.json"
    return info_file.exists()


def check_tasks_file(task_dir: Path) -> bool:
    """检查 meta/tasks.jsonl 是否存在"""
    tasks_file = task_dir / "meta" / "tasks.jsonl"
    tasks_file_parquet = task_dir / "meta" / "tasks.parquet"
    return tasks_file.exists() or tasks_file_parquet.exists()


def check_task_conversion(
    benchmark: str,
    embodiment: str,
    task_name: str,
    source_stats: Dict,
    output_base: Path
) -> Dict:
    """检查单个任务的转换情况"""
    
    # 从源数据统计中获取原始数量
    original_train = 0
    original_val = 0
    original_total = 0
    
    benchmark_data = source_stats.get("benchmarks", {}).get(benchmark, {})
    for task in benchmark_data.get("tasks", []):
        if task["embodiment"] == embodiment and task["task_name"] == task_name:
            original_train = task["train"]
            original_val = task["val"]
            original_total = task["total"]
            break
    
    # 检查转换后的目录
    task_dir = output_base / benchmark / embodiment / task_name
    
    result = {
        "benchmark": benchmark,
        "embodiment": embodiment,
        "task_name": task_name,
        "original_train": original_train,
        "original_val": original_val,
        "original_total": original_total,
        "exists": task_dir.exists(),
        "data_count": 0,
        "video_counts": {},
        "has_info": False,
        "has_tasks": False,
        "status": "NOT_STARTED"
    }
    
    if not task_dir.exists():
        return result
    
    # 统计转换后的文件
    print(task_dir)
    result["data_count"] = count_data_files(task_dir)
    result["video_counts"] = count_video_files(task_dir)
    result["has_info"] = check_info_file(task_dir)
    result["has_tasks"] = check_tasks_file(task_dir)
    
    # 判断状态
    if result["data_count"] == 0:
        result["status"] = "NOT_STARTED"
    elif result["data_count"] < original_total:
        result["status"] = "IN_PROGRESS"
    elif result["data_count"] == original_total and result["has_info"] and result["has_tasks"]:
        # 检查视频数量
        video_complete = True
        if result["video_counts"]:
            for cam, count in result["video_counts"].items():
                if count < original_total:
                    video_complete = False
                    break
        
        if video_complete:
            result["status"] = "COMPLETED"
        else:
            result["status"] = "VIDEO_INCOMPLETE"
    else:
        result["status"] = "DATA_INCOMPLETE"
    
    return result


def format_table_row(result: Dict, max_cameras: int = 3) -> str:
    """格式化表格行"""
    # 基本信息
    row = f"{result['benchmark']:<25} {result['embodiment']:<25} {result['task_name']:<50} "
    
    # 原始数据量
    row += f"{result['original_train']:>6} {result['original_val']:>6} {result['original_total']:>6} "
    
    # 转换后数据量
    row += f"{result['data_count']:>6} "
    
    # Info 和 Tasks 状态
    info_status = "✓" if result['has_info'] else "✗"
    tasks_status = "✓" if result['has_tasks'] else "✗"
    row += f"{info_status:>6} {tasks_status:>6} "
    
    # 视频数量（最多显示3个相机）
    video_counts_list = list(result['video_counts'].items())
    for i in range(max_cameras):
        if i < len(video_counts_list):
            cam_name, count = video_counts_list[i]
            row += f"{count:>6} "
        else:
            row += f"{'N/A':>6} "
    
    # 状态
    row += f"{result['status']:<20}"
    
    return row


def main():
    parser = argparse.ArgumentParser(description="检查数据转换进度")
    parser.add_argument(
        "--source-json",
        type=Path,
        default=Path(__file__).parent / "source_episodes_reports" / "all_benchmarks_summary.json",
        help="原始数据统计 JSON 文件路径"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/mnt/xlab-nas-2/vla_dataset/lerobot/robomind_10_new_110"),
        help="转换后数据的输出路径"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="benchmark1_0_compressed",
        help="要检查的 benchmark"
    )
    parser.add_argument(
        "--embodiments",
        type=str,
        nargs="+",
        default=["agilex_3rgb","franka_1rgb","franka_3rgb","simulation","tienkung_gello_1rgb","tienkung_xsens_1rgb","ur_1rgb"],
        help="要检查的 embodiment 列表"
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        help="输出报告文件路径（默认：conversion_progress_report.txt）"
    )
    
    args = parser.parse_args()
    
    # 设置默认输出报告路径
    if args.output_report is None:
        args.output_report = Path(__file__).parent / "source_episodes_reports" / f"{args.benchmark}_conversion_progress.txt"
    
    # 加载原始数据统计
    print(f"加载原始数据统计: {args.source_json}")
    source_stats = load_source_stats(args.source_json)
    
    # 收集所有任务
    benchmark_data = source_stats.get("benchmarks", {}).get(args.benchmark, {})
    
    # 按 embodiment 分组任务
    tasks_by_embodiment = defaultdict(list)
    for task in benchmark_data.get("tasks", []):
        if task["embodiment"] in args.embodiments:
            tasks_by_embodiment[task["embodiment"]].append(task["task_name"])
    
    # 检查每个任务
    results = []
    total_tasks = 0
    for embodiment in args.embodiments:
        task_names = tasks_by_embodiment.get(embodiment, [])
        total_tasks += len(task_names)
        
        print(f"\n检查 {embodiment} ({len(task_names)} 个任务)...")
        # task_name = "57_potatolittleoven"
        for task_name in task_names:
            if task_name=="57_potatolittleoven":
                a = 1
            result = check_task_conversion(
                args.benchmark,
                embodiment,
                task_name,
                source_stats,
                args.output_path
            )
            results.append(result)
            
            # 打印进度
            if result["status"] == "COMPLETED":
                status_symbol = "✓"
            elif result["status"] in ["IN_PROGRESS", "VIDEO_INCOMPLETE", "DATA_INCOMPLETE"]:
                status_symbol = "⚠"
            else:
                status_symbol = "✗"
            
            print(f"  {status_symbol} {task_name}: {result['status']} "
                  f"(Data: {result['data_count']}/{result['original_total']})")
    
    # 生成报告
    print(f"\n生成报告: {args.output_report}")
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output_report, 'w', encoding='utf-8') as f:
        # 标题
        f.write("=" * 200 + "\n")
        f.write(f"RoboMIND 数据转换进度报告 - {args.benchmark}\n")
        f.write(f"原始数据统计: {args.source_json}\n")
        f.write(f"输出路径: {args.output_path}\n")
        f.write("=" * 200 + "\n\n")
        
        # 表头
        header = (
            f"{'Benchmark':<25} {'Embodiment':<25} {'Task Name':<50} "
            f"{'Train':>6} {'Val':>6} {'Total':>6} "
            f"{'Data':>6} {'Info':>6} {'Tasks':>6} "
            f"{'Video1':>6} {'Video2':>6} {'Video3':>6} "
            f"{'Status':<20}"
        )
        f.write(header + "\n")
        f.write("=" * 200 + "\n")
        
        # 数据行
        for result in results:
            if result['task_name']=="57_potatolittleoven":
                a = 1
            row = format_table_row(result)
            f.write(row + "\n")
        
        # 统计信息
        f.write("=" * 200 + "\n\n")
        
        # 按状态统计
        status_counts = defaultdict(int)
        for result in results:
            status_counts[result["status"]] += 1
        
        f.write("统计信息:\n")
        f.write(f"  总任务数: {total_tasks}\n")
        f.write(f"  已完成: {status_counts['COMPLETED']}\n")
        f.write(f"  进行中: {status_counts['IN_PROGRESS']}\n")
        f.write(f"  数据不完整: {status_counts['DATA_INCOMPLETE']}\n")
        f.write(f"  视频不完整: {status_counts['VIDEO_INCOMPLETE']}\n")
        f.write(f"  未开始: {status_counts['NOT_STARTED']}\n")
        
        # 按 embodiment 统计
        f.write("\n按 Embodiment 统计:\n")
        for embodiment in args.embodiments:
            embodiment_results = [r for r in results if r["embodiment"] == embodiment]
            completed = sum(1 for r in embodiment_results if r["status"] == "COMPLETED")
            total = len(embodiment_results)
            
            # 统计原始和转换后的 episode 总数
            original_total = sum(r["original_total"] for r in embodiment_results)
            converted_total = sum(r["data_count"] for r in embodiment_results)
            
            f.write(f"  {embodiment}:\n")
            f.write(f"    任务数: {total} (已完成: {completed}, {completed/total*100:.1f}%)\n")
            f.write(f"    Episodes: {converted_total}/{original_total} ({converted_total/original_total*100:.1f}%)\n")
        
        # 需要关注的任务
        f.write("\n需要关注的任务:\n")
        incomplete_tasks = [r for r in results if r["status"] not in ["COMPLETED", "NOT_STARTED"]]
        if incomplete_tasks:
            for result in incomplete_tasks:
                f.write(f"  [{result['status']}] {result['embodiment']}/{result['task_name']}: "
                       f"Data {result['data_count']}/{result['original_total']}\n")
        else:
            f.write("  无\n")
        
        # 完全缺失的任务
        missing_tasks = [r for r in results if r["status"] == "NOT_STARTED"]
        if missing_tasks:
            f.write(f"\n未开始的任务 ({len(missing_tasks)}):\n")
            for result in missing_tasks[:20]:  # 只显示前20个
                f.write(f"  {result['embodiment']}/{result['task_name']}\n")
            if len(missing_tasks) > 20:
                f.write(f"  ... 还有 {len(missing_tasks) - 20} 个任务\n")
    
    print(f"\n报告已生成: {args.output_report}")
    print(f"\n总结:")
    print(f"  总任务数: {total_tasks}")
    print(f"  已完成: {status_counts['COMPLETED']} ({status_counts['COMPLETED']/total_tasks*100:.1f}%)")
    print(f"  进行中: {status_counts['IN_PROGRESS']}")
    print(f"  未开始: {status_counts['NOT_STARTED']}")


if __name__ == "__main__":
    main()
