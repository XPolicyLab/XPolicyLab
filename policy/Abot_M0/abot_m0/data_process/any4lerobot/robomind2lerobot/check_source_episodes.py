#!/usr/bin/env python3
"""
检查 RoboMIND 原始格式（转换前）中每个 task 的 episode 数量
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def count_source_episodes(src_path: Path, task_type: str, embodiment: str, benchmark: str) -> dict[str, int]:
    """
    统计源数据中的轨迹数（按 split）
    
    Returns:
        dict: {"train": count, "val": count, "total": count}
    """
    counts = {"train": 0, "val": 0, "total": 0}
    
    # RoboMIND 原始格式路径结构：src_path / benchmark / h5_{embodiment} / task_type / task_type / success_episodes / {split}
    task_path = src_path / benchmark / f"h5_{embodiment}" / task_type / task_type
    
    if not task_path.exists():
        return counts
    
    for split in ["train", "val"]:
        split_path = task_path / "success_episodes" / split
        if split_path.exists():
            # 统计所有 trajectory.hdf5 文件
            hdf5_files = list(split_path.glob("**/trajectory.hdf5"))
            counts[split] = len(hdf5_files)
    
    counts["total"] = counts["train"] + counts["val"]
    return counts


def get_all_tasks(src_path: Path, embodiment: str, benchmark: str) -> list[str]:
    """获取所有任务列表"""
    src_task_path = src_path / benchmark / f"h5_{embodiment}"
    
    if not src_task_path.exists():
        return []
    
    tasks = []
    for task_type in src_task_path.iterdir():

        # 将路径拆分为部件列表
        task_type = list[str](task_type.parts)
        task_type.insert(-1, task_type[-1])
        # 重组为新路径
        task_type = Path(*task_type)

        if task_type.name.endswith(".tar.gz"):
            continue
        if task_type.is_dir():
            # 检查是否有 success_episodes 目录
            # 路径结构：task_type / task_type / success_episodes
            success_path = task_type / task_type / "success_episodes"
            if success_path.exists():
                tasks.append(task_type.name)
    
    return sorted(tasks)


def check_all_tasks(
    src_path: Path,
    benchmarks: list[str],
    embodiments: list[str],
    output_format: str = "table",
    output_file: Path | None = None
) -> None:
    """检查所有任务的 episode 数量"""
    
    all_results = []
    
    for benchmark in benchmarks:
        for embodiment in embodiments:
            tasks = get_all_tasks(src_path, embodiment, benchmark)
            
            if len(tasks) == 0:
                print(f"No tasks found for {benchmark}/{embodiment}")
                continue
            
            for task_type in tasks:
                counts = count_source_episodes(src_path, task_type, embodiment, benchmark)
                
                result = {
                    "benchmark": benchmark,
                    "embodiment": embodiment,
                    "task_type": task_type,
                    "train": counts["train"],
                    "val": counts["val"],
                    "total": counts["total"],
                }
                print(result)
                all_results.append(result)
    
    # 准备最终结果输出内容
    result_lines = []
    
    if output_format == "table":
        result_lines.append("=" * 120)
        result_lines.append(f"{'Benchmark':<25} {'Embodiment':<30} {'Task Type':<40} {'Train':<10} {'Val':<10} {'Total':<10}")
        result_lines.append("=" * 120)
        
        for result in sorted(all_results, key=lambda x: (x["benchmark"], x["embodiment"], x["task_type"])):
            result_lines.append(
                f"{result['benchmark']:<25} "
                f"{result['embodiment']:<30} "
                f"{result['task_type']:<40} "
                f"{result['train']:<10} "
                f"{result['val']:<10} "
                f"{result['total']:<10}"
            )
        
        result_lines.append("=" * 120)
        
        # 统计信息
        total_tasks = len(all_results)
        total_train = sum(r["train"] for r in all_results)
        total_val = sum(r["val"] for r in all_results)
        total_episodes = sum(r["total"] for r in all_results)
        
        result_lines.append(f"\n统计信息:")
        result_lines.append(f"  总任务数: {total_tasks}")
        result_lines.append(f"  总 train episodes: {total_train}")
        result_lines.append(f"  总 val episodes: {total_val}")
        result_lines.append(f"  总 episodes: {total_episodes}")
        
        # 按 benchmark 统计
        result_lines.append(f"\n按 Benchmark 统计:")
        benchmark_stats = defaultdict(lambda: {"tasks": 0, "train": 0, "val": 0, "total": 0})
        for result in all_results:
            bm = result["benchmark"]
            benchmark_stats[bm]["tasks"] += 1
            benchmark_stats[bm]["train"] += result["train"]
            benchmark_stats[bm]["val"] += result["val"]
            benchmark_stats[bm]["total"] += result["total"]
        
        for bm, stats in sorted(benchmark_stats.items()):
            result_lines.append(f"  {bm}:")
            result_lines.append(f"    任务数: {stats['tasks']}")
            result_lines.append(f"    Train: {stats['train']}, Val: {stats['val']}, Total: {stats['total']}")
        
        # 按 embodiment 统计
        result_lines.append(f"\n按 Embodiment 统计:")
        embodiment_stats = defaultdict(lambda: {"tasks": 0, "train": 0, "val": 0, "total": 0})
        for result in all_results:
            emb = result["embodiment"]
            embodiment_stats[emb]["tasks"] += 1
            embodiment_stats[emb]["train"] += result["train"]
            embodiment_stats[emb]["val"] += result["val"]
            embodiment_stats[emb]["total"] += result["total"]
        
        for emb, stats in sorted(embodiment_stats.items()):
            result_lines.append(f"  {emb}:")
            result_lines.append(f"    任务数: {stats['tasks']}")
            result_lines.append(f"    Train: {stats['train']}, Val: {stats['val']}, Total: {stats['total']}")
        
        result_content = "\n".join(result_lines)
    
    elif output_format == "json":
        result_content = json.dumps(all_results, indent=2, ensure_ascii=False)
    
    elif output_format == "summary":
        # 只显示有数据的任务
        tasks_with_data = [r for r in all_results if r["total"] > 0]
        
        result_lines.append(f"找到 {len(tasks_with_data)} 个有数据的任务:")
        for result in sorted(tasks_with_data, key=lambda x: (x["benchmark"], x["embodiment"], x["task_type"])):
            result_lines.append(
                f"  {result['benchmark']}/{result['embodiment']}/{result['task_type']}: "
                f"train={result['train']}, val={result['val']}, total={result['total']}"
            )
        
        # 显示没有数据的任务
        tasks_without_data = [r for r in all_results if r["total"] == 0]
        if tasks_without_data:
            result_lines.append(f"\n找到 {len(tasks_without_data)} 个没有数据的任务:")
            for result in sorted(tasks_without_data, key=lambda x: (x["benchmark"], x["embodiment"], x["task_type"])):
                result_lines.append(f"  {result['benchmark']}/{result['embodiment']}/{result['task_type']}")
        
        result_content = "\n".join(result_lines)
    
    # 打印到标准输出（中间输出和最终结果都打印）
    print(result_content)
    
    # 如果指定了输出文件，同时写入文件
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_content)
        print(f"\n结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="检查 RoboMIND 原始格式中每个 task 的 episode 数量")
    parser.add_argument(
        "--src-path",
        type=Path,
        default=Path("/mnt/nas-data-4/gaowo.cyz/RoboMIND"),
        help="源数据路径",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["benchmark1_1_compressed"],
        help="Benchmark 列表",
    )
    parser.add_argument(
        "--embodiments",
        type=str,
        nargs="+",
        default=[
            "agilex_3rgb",
            "franka_fr3_dual",
            "sim_tienkung_1rgb",
            "tienkung_prod1_gello_1rgb",
            "ur_1rgb",
            "franka_3rgb",
            "sim_franka_3rgb",
            "tienkung_gello_1rgb",
            "tienkung_xsens_1rgb",
        ],
        help="Embodiment 列表",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["table", "json", "summary"],
        default="table",
        help="输出格式",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出文件路径（如果指定，最终结果会同时保存到文件）",
    )
    
    args = parser.parse_args()
    
    if not args.src_path.exists():
        print(f"Error: Source path does not exist: {args.src_path}")
        return
    
    check_all_tasks(
        args.src_path,
        args.benchmarks,
        args.embodiments,
        args.output_format,
        args.output
    )


if __name__ == "__main__":
    main()
