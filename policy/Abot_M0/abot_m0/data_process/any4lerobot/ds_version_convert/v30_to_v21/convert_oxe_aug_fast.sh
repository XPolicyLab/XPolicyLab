#!/bin/bash
TASK_LIST="/mnt/workspace/yangyandan/workspace/any4lerobot/ds_version_convert/v30_to_v21/task_list_oxeaug.txt"
SCRIPT="convert_v30_to_v21_simple.py"
MAX_JOBS=80

# 读取所有非空、非注释行到数组
mapfile -t tasks < <(grep -v '^[[:space:]]*$' "$TASK_LIST" | grep -v '^[[:space:]]*#')

# 并发控制循环
for ((i=0; i<${#tasks[@]}; i++)); do
    task_id="${tasks[i]}"
    echo "[$(date)] Starting task: $task_id"
    # 启动新任务前，确保池中任务数 < MAX_JOBS
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        wait -n  # 等待任意1个任务完成（Bash 4.3+）
    done
    
    # 启动后台任务
    python "$SCRIPT" --task-id  "$task_id"  --input-path /mnt/xlab-nas-2/vla_dataset/oxeauge/data/oxe-auge  --output-path /mnt/xlab-nas-2/vla_dataset/oxeauge/data/oxe-auge-v21-debug/ &

done
wait  # 等待剩余任务

echo "All tasks completed."


