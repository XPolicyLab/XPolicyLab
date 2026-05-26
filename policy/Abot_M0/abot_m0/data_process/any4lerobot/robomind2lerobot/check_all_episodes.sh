#!/bin/bash
# 批量检查所有数据集的 episode 数量

SRC_PATH="/mnt/nas-data-4/gaowo.cyz/RoboMIND"
OUTPUT_PATH="/mnt/xlab-nas-2/vla_dataset/lerobot/robomind_11_new"
BENCHMARK="benchmark1_1_compressed"
EMBODIMENTS=("ur_1rgb" "franka_3rgb" "sim_franka_3rgb" "tienkung_gello_1rgb" "tienkung_xsens_1rgb" ) #"agilex_3rgb"  "franka_fr3_dual" "sim_tienkung_1rgb" "tienkung_prod1_gello_1rgb" "ur_1rgb" "franka_3rgb" "sim_franka_3rgb" "tienkung_gello_1rgb" "tienkung_xsens_1rgb")

# SRC_PATH="/mnt/nas-data-4/gaowo.cyz/RoboMIND"
# OUTPUT_PATH="/mnt/xlab-nas-2/vla_dataset/lerobot/robomind_10"
# BENCHMARK="benchmark1_0_compressed"
# EMBODIMENTS=("agilex_3rgb" "franka_1rgb" "franka_3rgb" "simulation" "tienkung_gello_1rgb" "tienkung_xsens_1rgb" "ur_1rgb")



for embodiment in "${EMBODIMENTS[@]}"; do
    echo "=========================================="
    echo "Checking ${embodiment}"
    echo "=========================================="
    
    dataset_base="${OUTPUT_PATH}/${BENCHMARK}/${embodiment}"
    echo $dataset_base
    if [ ! -d "$dataset_base" ]; then
        echo "Directory not found: $dataset_base"
        continue
    fi
    
    # 遍历所有任务目录
    for task_dir in "$dataset_base"/*; do
        if [ -d "$task_dir" ]; then
            task_name=$(basename "$task_dir")
            echo ""
            echo "--- ${embodiment}/${task_name} ---"
            python check_episodes_simple.py \
                --dataset-path "$task_dir" \
                --output-format simple
        fi
    done
done
