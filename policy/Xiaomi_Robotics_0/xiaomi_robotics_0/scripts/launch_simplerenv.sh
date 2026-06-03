# Copyright (C) 2026 Xiaomi Corporation.
#!/usr/bin/env bash

# ======================== Configuration Section (Adjust as needed) ========================
BASE_PORT=10086                          # Base port number for task assignment

# ======================== Main Execution Logic ========================
# Parse input parameters
NUM_PORTS="$1"
TASK="$2"
LOG_PATH="$3"

# Create video/log directory (prevent "no such file" errors for logs/videos)
mkdir -p "${LOG_PATH}/${TASK}"

# Launch tasks in batch with concurrency control
echo "Starting batch task execution, concurrency limit: ${NUM_PORTS}"
declare -a jobs=()

for i in $(seq 0 $((NUM_PORTS - 1))); do
    # Calculate task-specific variables
    PORT=$((BASE_PORT + i))
    LOG_FILE="${LOG_PATH}/${TASK}/${i}_eval.log"

    # -------------------------- Core task execution logic (no function wrap) --------------------------
    # Activate conda environment
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate simplerenv

    # Execute the tasks. xvfb-run -a -s
    echo "Starting task execution: task_id=${i}, port=${PORT}, log=${LOG_FILE}"
    DISPLAY="" VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json CUDA_VISIBLE_DEVICES=${i} python -u eval_simplerenv/main.py \
        --args.dataset-name "${TASK}" \
        --args.experiment-root "${LOG_PATH}/${TASK}" \
        --args.worker-id "${i}" \
        --args.num-workers "${NUM_PORTS}" \
        --args.port "${PORT}" 2>&1 | tee "${LOG_FILE}" &
        
    # --------------------------------------------------------------------------------------------------

    # Track background job PID and print status
    JOB_PID=$!
    jobs+=("${JOB_PID}")
    echo "Started task ${i} on port ${PORT} with job ID ${JOB_PID}"
done

# Wait for all remaining background tasks to finish
echo "Waiting for all remaining tasks to complete..."
wait

# Final status message
echo "Whole process completed! All results and logs are stored in: ${LOG_PATH}/${TASK}"
