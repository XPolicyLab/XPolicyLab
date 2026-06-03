#!/bin/bash
# Single task evaluation script for Motus policy on RoboTwin platform

# ============================================================================
# Single Task Configuration - MODIFY THESE
# ============================================================================
TASK_NAME="click_alarmclock"  # Change this to the task you want to test
GPU_ID=0                       # GPU to use

# ============================================================================
# Script starts here
# ============================================================================
echo "Starting single task evaluation at $(date)"

# Get script directory (policy/Motus/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$SCRIPT_DIR"

# ============================================================================
# Load Configuration from paths_config.yml
# ============================================================================
CONFIG_FILE="${POLICY_DIR}/paths_config.yml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Please create paths_config.yml with required paths."
    exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"

# Parse YAML (improved - remove comments and extra whitespace)
ROBOTWIN_ROOT=$(grep "^robotwin_root:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CONDA_ENV=$(grep "^conda_env:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CHECKPOINT_PATH=$(grep "^checkpoint_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
WAN_PATH=$(grep "^wan_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
VLM_PATH=$(grep "^vlm_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Optional configurations
TASK_CONFIG=$(grep "^task_config:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
SEED=$(grep "^seed:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Default values
TASK_CONFIG=${TASK_CONFIG:-"demo_randomized"}
SEED=${SEED:-"42"}
POLICY_NAME="Motus"

# ============================================================================
# Validation
# ============================================================================
if [ -z "$ROBOTWIN_ROOT" ]; then
    echo "Error: robotwin_root is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$CONDA_ENV" ]; then
    echo "Error: conda_env is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: checkpoint_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$WAN_PATH" ]; then
    echo "Error: wan_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$VLM_PATH" ]; then
    echo "Error: vlm_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$ROBOTWIN_ROOT" ]; then
    echo "Error: RoboTwin root not found: $ROBOTWIN_ROOT"
    exit 1
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$WAN_PATH" ]; then
    echo "Error: WAN path not found: $WAN_PATH"
    exit 1
fi

if [ ! -d "$VLM_PATH" ]; then
    echo "Error: VLM path not found: $VLM_PATH"
    exit 1
fi

cd "$ROBOTWIN_ROOT" || exit 1

# Activate conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found."
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi

# Set environment
export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Create logs directory
LOG_DIR="${POLICY_DIR}/logs_single_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

ckpt_setting="${CHECKPOINT_PATH}"
log_file="${LOG_DIR}/${TASK_NAME}.log"

echo ""
echo "================================================================"
echo "Single Task Evaluation Configuration"
echo "================================================================"
echo "Task Name:         $TASK_NAME"
echo "GPU:               $GPU_ID"
echo "----------------------------------------------------------------"
echo "RoboTwin Root:     $ROBOTWIN_ROOT"
echo "Policy Dir:        $POLICY_DIR"
echo "Checkpoint:        $CHECKPOINT_PATH"
echo "WAN Path:          $WAN_PATH"
echo "VLM Path:          $VLM_PATH"
echo "Task Config:       $TASK_CONFIG"
echo "Seed:              $SEED"
echo "Log File:          $log_file"
echo "================================================================"
echo ""

# Run evaluation with WAN_PATH passed as argument
echo "Starting evaluation..."

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py \
    --config "policy/${POLICY_NAME}/deploy_policy.yml" \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${TASK_CONFIG}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${SEED}" \
    --policy_name "${POLICY_NAME}" \
    --log_dir "${LOG_DIR}" \
    --wan_path "${WAN_PATH}" \
    --vlm_path "${VLM_PATH}" \
    2>&1 | tee "$log_file"

exit_code=${PIPESTATUS[0]}

echo ""
echo "================================================================"
if [ $exit_code -eq 0 ]; then
    echo "✅ Task $TASK_NAME completed successfully"
    echo "================================================================"
    exit 0
else
    echo "❌ Task $TASK_NAME failed with exit code $exit_code"
    echo "================================================================"
    echo "Log file: $log_file"
    exit 1
fi