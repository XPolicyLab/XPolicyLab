#!/bin/bash
TASK_LIST="/mnt/workspace/yangyandan/workspace/any4lerobot/ds_version_convert/v30_to_v21/task_list.txt"
SCRIPT="convert_v30_to_v21_simple.py"
MAX_JOBS=20

# 读取所有非空、非注释行到数组
mapfile -t tasks < <(grep -v '^[[:space:]]*$' "$TASK_LIST" | grep -v '^[[:space:]]*#')

# 并发控制循环
for ((i=0; i<${#tasks[@]}; i++)); do
    task_id="${tasks[i]}"
    echo "[$(date)] Starting task: $task_id"

    # 启动后台任务
    python "$SCRIPT" --task-id  "$task_id"  --input-path /mnt/xlab-nas-2/vla_dataset/lerobot/agibot_convert_2/complete/  --output-path /mnt/xlab-nas-2/vla_dataset/lerobot/agibot_convert_21/agibotworld_complete_1_362/ &

    # 如果已启动 MAX_JOBS 个任务，就等待其中一个完成
    if (( (i + 1) % MAX_JOBS == 0 )); then
        wait  # 等待当前批次全部完成
    fi
done

# 等待剩余任务（最后一组不足 MAX_JOBS 的）
wait
echo "All tasks completed."


