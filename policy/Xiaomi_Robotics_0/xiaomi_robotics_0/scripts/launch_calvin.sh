# Copyright (C) 2026 Xiaomi Corporation.
#!/usr/bin/env bash

# ======================== Configuration Section (Adjust as needed) ========================
BASE_PORT=10086                          # Base port number for task assignment

# ======================== Main Execution Logic ========================
# Parse input parameters
NUM_PORTS="$1"
LOG_PATH="$2"

# Create video/log directory (prevent "no such file" errors for logs/videos)
mkdir -p "${LOG_PATH}"

# Launch tasks in batch with concurrency control
echo "Starting batch task execution, concurrency limit: ${NUM_PORTS}"
for rank in $(seq 0 $((NUM_PORTS - 1))); do
    port=$((BASE_PORT + rank))
    LOG_FILE="${LOG_PATH}/rank_${rank}_eval.log"

    # -------------------------- Core task execution logic (no function wrap) --------------------------
    # Set environment variables for current task
    export MUJOCO_GL='glx'
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

    # Activate conda environment
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate calvin
    
    echo "Starting task execution: rank=${rank}, port=${port}, log=${LOG_FILE}"
    python eval_calvin/main.py --rank $rank --world_size $NUM_PORTS --CACHE_ROOT ${LOG_PATH} 2>&1 | tee "$LOG_FILE" &
    
    echo "Rank $rank started with PID $!"
done

# Wait for all remaining jobs
echo "Waiting for remaining jobs to complete..."
wait

echo "All tasks completed"

# Merge task results
echo "All tasks completed successfully! Merging results to ${LOG_PATH}"
python eval_calvin/merge_results.py --eval_log_dir "${LOG_PATH}"

# Final status message
echo "Whole process completed! All results and logs are stored in: ${LOG_PATH}"
