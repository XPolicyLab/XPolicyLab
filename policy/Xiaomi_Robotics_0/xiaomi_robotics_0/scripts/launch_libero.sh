# Copyright (C) 2026 Xiaomi Corporation.
#!/usr/bin/env bash

# ======================== Configuration Section (Adjust as needed) ========================
TASK_IDS=(0 1 8 3 4 5 6 7 2 9)          # Fixed task ID list
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

for i in "${!TASK_IDS[@]}"; do
    # Calculate task-specific variables
    TASK_ID="${TASK_IDS[$i]}"
    PORT_OFFSET=$((i % NUM_PORTS))
    PORT=$((BASE_PORT + PORT_OFFSET))
    LOG_FILE="${LOG_PATH}/${TASK}/${TASK_ID}_eval.log"

    # -------------------------- Core task execution logic (no function wrap) --------------------------
    # Set environment variables for current task
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    export MUJOCO_GL='glx'

    # Activate conda environment
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate libero

    # Execute the task with xvfb-run (virtual display for headless servers)
    echo "Starting task execution: task_id=${TASK_ID}, port=${PORT}, log=${LOG_FILE}"
    xvfb-run -a -s "-screen 0 1400x900x24" python eval_libero/main.py \
        --args.task-suite-name "${TASK}" \
        --args.video-out-path "${LOG_PATH}/${TASK}" \
        --args.task-id "${TASK_ID}" \
        --args.port "${PORT}" 2>&1 | tee "${LOG_FILE}" &
    # --------------------------------------------------------------------------------------------------

    # Track background job PID and print status
    JOB_PID=$!
    jobs+=("${JOB_PID}")
    echo "Started task ${TASK_ID} on port ${PORT} with job ID ${JOB_PID}"

    # Enforce concurrency limit: wait for any job to finish if max ports are used
    if (( ${#jobs[@]} >= NUM_PORTS )); then
        echo "Concurrent task limit (${NUM_PORTS}) reached, waiting for any task to complete..."
        wait -n 2>/dev/null || true  # Wait for any background job, ignore "no jobs" error
        # Refresh job list (remove completed PIDs)
        jobs=()
        for job in $(jobs -p); do
            jobs+=("${job}")
        done
    fi
done

# Wait for all remaining background tasks to finish
echo "Waiting for all remaining tasks to complete..."
wait

# Merge task results
echo "All tasks completed successfully! Merging results to ${LOG_PATH}/${TASK}"
python eval_libero/merge_results.py "${LOG_PATH}/${TASK}"

# Final status message
echo "Whole process completed! All results and logs are stored in: ${LOG_PATH}/${TASK}"
