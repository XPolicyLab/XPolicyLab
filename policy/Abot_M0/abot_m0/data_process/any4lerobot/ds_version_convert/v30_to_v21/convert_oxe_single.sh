#!/bin/bash

SCRIPT="convert_v30_to_v21_simple.py"
task_id=bridge_train_15000_20000_augmented

# 启动后台任务
python "$SCRIPT" --task-id  "$task_id"  --input-path /mnt/xlab-nas-2/vla_dataset/oxeauge/data/oxe-auge  --output-path /mnt/xlab-nas-2/vla_dataset/oxeauge/data/oxe-auge-v21-new1 

    
# 等待剩余任务（最后一组不足 MAX_JOBS 的）
wait
echo "All tasks completed."



