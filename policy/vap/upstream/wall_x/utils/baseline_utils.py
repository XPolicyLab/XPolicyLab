#!/usr/bin/env python3
import yaml
import os
from pathlib import Path
import csv
import tempfile


def check_baseline_dump(
    baseline_path: str,
    id_val: str,
    name_val: str,
    total_time: float,
    total_episodes_run: int,
    total_successes: int,
):
    """
    对比当前评测结果与 baseline，并在检测到回归时将详细信息 dump 到 yml 文件。

    环境变量 TARSFER_DUMP_PATH 指定输出目录。
    仅当 total_successes 减少 或 final_success_rate 下降时，才生成 dump 文件。
    """

    if total_episodes_run <= 0:
        print("✅ total_episodes_run=0，未运行任何 episode，跳过回归检测。")
        return

    baseline_path = Path(baseline_path)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline 文件不存在: {baseline_path}")

    fieldnames = [
        "id",
        "name",
        "total_time",
        "total_episodes_run",
        "total_successes",
        "final_success_rate",
    ]

    baseline_row = None
    with open(baseline_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames) != set(fieldnames):
            raise ValueError(
                f"Baseline 表头不匹配。期望: {fieldnames}, 实际: {reader.fieldnames}"
            )
        for row in reader:
            if row["id"] == str(id_val):
                baseline_row = row
                break

    if baseline_row is None:
        raise ValueError(f"Baseline 中未找到 id='{id_val}' 的记录")

    try:
        base_total_time = float(baseline_row["total_time"])
        base_episodes = int(baseline_row["total_episodes_run"])
        base_successes = int(baseline_row["total_successes"])
        base_rate_str = baseline_row["final_success_rate"]
        if base_rate_str.endswith("%"):
            base_success_rate = float(base_rate_str.rstrip("%")) / 100.0
        else:
            base_success_rate = (
                float(base_rate_str) / 100.0
                if "." in base_rate_str
                else float(base_rate_str)
            )
    except (ValueError, KeyError) as e:
        raise ValueError(f"解析 baseline 行失败: {e}, 行内容: {baseline_row}")

    current_success_rate = (
        total_successes / total_episodes_run if total_episodes_run > 0 else 0.0
    )

    should_dump = False
    messages = []

    if current_success_rate < base_success_rate - 1e-1:
        should_dump = True
        messages.append(
            f"成功率下降: baseline={base_success_rate:.4f} ({base_successes}/{base_episodes}) "
            f"→ current={current_success_rate:.4f} ({total_successes}/{total_episodes_run})"
            f"成功次数减少: baseline={base_successes} → current={total_successes}"
        )

    if total_time > base_total_time:
        print(
            f"⚠️  警告: 运行时间增加: baseline={base_total_time:.2f}s → current={total_time:.2f}s"
        )

    if not should_dump:
        print("✅ 未检测到成功率下降，跳过 dump。")
        return

    dump_dir = os.environ.get("TARSFER_DUMP_PATH")
    if not dump_dir:
        raise EnvironmentError("环境变量 TARSFER_DUMP_PATH 未设置，无法保存 dump 文件")

    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_file = dump_dir / f"{id_val}.yml"

    dump_data = {
        "id": id_val,
        "name": name_val,
        "comparison_reason": messages,
        "baseline": {
            "total_time": base_total_time,
            "total_episodes_run": base_episodes,
            "total_successes": base_successes,
            "final_success_rate": base_success_rate,
            "final_success_rate_percent_str": baseline_row["final_success_rate"],
        },
        "current": {
            "total_time": total_time,
            "total_episodes_run": total_episodes_run,
            "total_successes": total_successes,
            "final_success_rate": current_success_rate,
        },
        "regression_detected": True,
    }

    with open(dump_file, "w", encoding="utf-8") as f:
        yaml.dump(dump_data, f, indent=2, allow_unicode=True)

    print(f"🚨 检测到回归！已将对比详情 dump 至: {dump_file}")


def update_baseline(
    baseline_path: str,
    id_val: str,
    name_val: str,
    total_time: float,
    total_episodes_run: int,
    total_successes: int,
):
    """
    更新 baseline 表格（CSV 格式）：
    - 若文件不存在，创建并写入表头 ['id', 'name', 'final_success_rate']
    - 若存在相同 id 的行，更新其 final_success_rate
    - 否则追加新行
    """
    final_success_rate = (
        total_successes / total_episodes_run if total_episodes_run > 0 else 0
    )
    baseline_path = Path(baseline_path)
    fieldnames = [
        "id",
        "name",
        "total_time",
        "total_episodes_run",
        "total_successes",
        "final_success_rate",
    ]

    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    found = False

    if baseline_path.exists():
        with open(baseline_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if set(reader.fieldnames) != set(fieldnames):
                raise ValueError(
                    f"Baseline 文件 {baseline_path} 表头不匹配，期望字段: {fieldnames}, 实际: {reader.fieldnames}"
                )
            for row in reader:
                if row["id"] == str(id_val):
                    row["name"] = str(name_val)
                    row["total_time"] = f"{total_time:.6f}"
                    row["total_episodes_run"] = str(total_episodes_run)
                    row["total_successes"] = str(total_successes)
                    row["final_success_rate"] = f"{final_success_rate * 100:.2f}%"
                    found = True
                rows.append(row)

    if not found:
        rows.append(
            {
                "id": str(id_val),
                "name": str(name_val),
                "total_time": f"{total_time:.6f}",
                "total_episodes_run": str(total_episodes_run),
                "total_successes": str(total_successes),
                "final_success_rate": f"{final_success_rate * 100:.2f}%",
            }
        )

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv", dir=baseline_path.parent
    ) as tmp_f:
        writer = csv.DictWriter(tmp_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = tmp_f.name

    os.replace(tmp_path, baseline_path)
    print(f"✅ Baseline updated at: {baseline_path}")
