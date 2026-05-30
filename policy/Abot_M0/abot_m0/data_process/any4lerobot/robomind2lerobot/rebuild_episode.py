#!/usr/bin/env python3
from pathlib import Path
import shutil
import pyarrow.parquet as pq

from robomind_h5_v3_new import get_all_tasks, save_as_lerobot_dataset

# ===== 配置 =====
ROOT = Path("/mnt/nas-data-4/gaowo.cyz/RoboMIND")
BENCHMARK = "benchmark1_0_compressed"
EMBODIMENT = "agilex_3rgb"
TASK_NAME = "10_packplate_2"

# 临时输出目录（不会动原始数据）
TMP_OUTPUT = Path("/mnt/workspace/yangyandan/tmp_rebuild_10_packplate_2")

# 目标输出文件（新文件，不覆盖旧文件）
TARGET_FILE = Path(
    "/mnt/xlab-nas-2/vla_dataset/lerobot/robomind_10_new_110/"
    "benchmark1_0_compressed/agilex_3rgb/10_packplate_2/"
    "meta/episodes_rebuilt/file-000.parquet"
)

# ===== 开始 =====
def main():
    TMP_OUTPUT.mkdir(parents=True, exist_ok=True)
    TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 找到目标任务
    selected = None
    for task in get_all_tasks(ROOT / BENCHMARK, TMP_OUTPUT, EMBODIMENT):
        task_type, splits, local_dir, task_instruction = task
        if task_type == TASK_NAME:
            selected = task
            break

    if selected is None:
        raise SystemExit(f"Task not found: {TASK_NAME}")

    # 只重建该 task（输出到临时目录）
    save_as_lerobot_dataset(
        selected, ROOT, BENCHMARK, EMBODIMENT,
        save_depth=False, save_images=False
    )

    rebuilt = TMP_OUTPUT / BENCHMARK / EMBODIMENT / TASK_NAME / "meta/episodes/chunk-000/file-000.parquet"
    if not rebuilt.exists():
        raise SystemExit(f"Rebuilt parquet not found: {rebuilt}")

    # 检查行数
    rows = pq.read_table(rebuilt).num_rows
    print(f"Rebuilt rows: {rows}")

    # 保存为新文件（不覆盖旧文件）
    if TARGET_FILE.exists():
        target = TARGET_FILE.with_name("file-000.rebuilt.parquet")
    else:
        target = TARGET_FILE

    shutil.copy2(rebuilt, target)
    print(f"Copied to: {target}")

if __name__ == "__main__":
    main()