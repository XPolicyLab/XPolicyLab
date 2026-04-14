#!/bin/bash

RUN_NUM=0
export CUDA_VISIBLE_DEVICES=${RUN_NUM}
export PYTHONPATH=.

# ============================================================
# Configuration
# ============================================================

# Model checkpoint directory
MODEL_ROOT="<path-to-model-root>" # root directory for model checkpoints
MODEL_NAME="Being-H05-2B_robocasa"
MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"

# Server configuration
SERVER_PORT=1888${RUN_NUM}
SERVER_LOG_FILE="results/eval/logs/${MODEL_NAME}/server_${SERVER_PORT}.log"
mkdir -p "results/eval/logs/${MODEL_NAME}"

# Set eval conda environment name, which will be activated after launching policy server
EVAL_CONDA_ENV="robocasa"

# Initialize PID variables
SERVER_PID=""
EVAL_PID=""

echo "Model directory: ${MODEL_PATH}"

# --- Server Configuration ---
MODEL_PATH="${OUTPUT_DIR}/${MODEL_NAME}"
DATA_CONFIG_NAME="robocasa_human"
EMBODIMENT_TAG="robocasa"
DATASET_NAME="robocasa_human_posttrain"

SERVER_SEED=42
SERVER_PROMPT_TEMP=long
SERVER_MAX_VIEW_NUM=-1

# --- Evaluation Configuration ---
EVAL_SEED=41
EVAL_CHUNK_SIZE=8
EVAL_ACTION_TYPE="world_delta"
EVAL_DATA_CONFIG_NAME="robocasa_human"
NUM_TRIALS=50

# --- Helper Functions ---
kill_tree() {
    local _pid=$1
    local _sig=${2:-9}
    if [ -z "$_pid" ]; then return; fi
    local _children=$(pgrep -P "$_pid")
    for _child in $_children; do
        kill_tree "$_child" "$_sig"
    done
    if kill -0 "$_pid" 2>/dev/null; then
        echo "    -> Cleaning up process PID: $_pid"
        kill -$_sig "$_pid" 2>/dev/null
    fi
}

cleanup() {
    echo ""
    echo "=============== Cleanup ==============="
    if [ -n "$EVAL_PID" ]; then
        echo "Cleaning up evaluation task (PID: $EVAL_PID) and all child processes..."
        kill_tree "$EVAL_PID"
    fi
    if [ -n "$SERVER_PID" ]; then
        echo "Cleaning up server process (PID: $SERVER_PID)..."
        kill -9 ${SERVER_PID} 2>/dev/null
    fi
    echo "Checking port ${SERVER_PORT} occupation..."
    PIDS_ON_PORT=$(lsof -t -i:${SERVER_PORT})
    if [ -n "$PIDS_ON_PORT" ]; then
        echo "${PIDS_ON_PORT}" | xargs kill -9 2>/dev/null
    else
        echo "Port ${SERVER_PORT} released."
    fi
}

trap cleanup EXIT INT TERM

echo ""
echo "=============== Step 1: Starting Server ==============="

echo "Starting inference server in background..."
nohup python -u -m BeingH.inference.run_server_vla \
    --model-path "${MODEL_PATH}" \
    --port ${SERVER_PORT} \
    --data-config-name "${DATA_CONFIG_NAME}" \
    --dataset-name "${DATASET_NAME}" \
    --embodiment-tag "${EMBODIMENT_TAG}" \
    --seed "${SERVER_SEED}" \
    --prompt-template "${SERVER_PROMPT_TEMP}" \
    --max-view-num $SERVER_MAX_VIEW_NUM \
    --no-use-fixed-view \
    --no-enable-rtc \
    ${METADATA_VARIANT_ARGS} > "${SERVER_LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "Inference server PID: ${SERVER_PID}"

echo "Waiting for server to be ready..."
MAX_RETRIES=300
COUNTER=0
SERVER_READY=false

while [ $COUNTER -lt $MAX_RETRIES ]; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Error: Server process exited unexpectedly!"
        tail -n 10 "${SERVER_LOG_FILE}"
        exit 1
    fi
    if grep -q "Server is ready" "${SERVER_LOG_FILE}"; then
        echo "Server started successfully!"
        SERVER_READY=true
        break
    fi
    sleep 3
    ((COUNTER++))
done

if [ "$SERVER_READY" = false ]; then
    echo "Error: Server startup timeout."
    exit 1
fi

echo ""
echo "=============== Step 2: Starting Evaluation Loop ==============="

CONDA_PATH=$(conda info --base)
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate ${EVAL_CONDA_ENV}
echo "Conda environment activated: ${EVAL_CONDA_ENV}"

VIDEO_DIR="results/rollouts/${MODEL_NAME}/robocasa_"
EVAL_LOG_FILE="results/eval/logs/${MODEL_NAME}/robocasa.log"
mkdir -p "$(dirname ${EVAL_LOG_FILE})"

python -m BeingH.benchmark.robocasa.run_robocasa_eval_fast \
    --port $SERVER_PORT \
    --seed $EVAL_SEED \
    --local_log_dir "results/robocasa/${CKPT_NAME}_${MODEL_NAME}" \
    --num_open_loop_steps $EVAL_CHUNK_SIZE \
    --num_trials_per_task $NUM_TRIALS \
    --action_type $EVAL_ACTION_TYPE \
    --data_config_name $EVAL_DATA_CONFIG_NAME 2>&1 | tee "${EVAL_LOG_FILE}"

echo ""
echo "=============== Evaluation Complete ==============="
echo "Results saved to: results/eval/logs/${MODEL_NAME}"
