#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$POLICY_DIR/paths_config.yml}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: configuration not found: $CONFIG_FILE" >&2
    echo "Copy paths_config.example.yml to paths_config.yml and edit it." >&2
    exit 1
fi
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"

yaml_value() {
    local key="$1"
    awk -v key="$key" '
        $1 == key ":" {
            value = substr($0, index($0, ":") + 1)
            sub(/[[:space:]]+#.*/, "", value)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            if (substr(value, 1, 1) == "\"" && substr(value, length(value), 1) == "\"") {
                value = substr(value, 2, length(value) - 2)
            }
            print value
            exit
        }
    ' "$CONFIG_FILE"
}

TASK_NAME="${1:-${TASK_NAME:-}}"
if [[ -z "$TASK_NAME" ]]; then
    echo "Usage: $0 <task_name>" >&2
    exit 2
fi

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$(yaml_value robotwin_root)}"
CONDA_ENV="${CONDA_ENV:-$(yaml_value conda_env)}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$(yaml_value checkpoint_path)}"
WAN_PATH="${WAN_PATH:-$(yaml_value wan_path)}"
VLM_PATH="${VLM_PATH:-$(yaml_value vlm_path)}"
TASK_CONFIG="${TASK_CONFIG:-$(yaml_value task_config)}"
SEED="${SEED:-$(yaml_value seed)}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-$(yaml_value instruction_type)}"
TEST_NUM="${TEST_NUM:-$(yaml_value test_num)}"
INFERENCE_MODE="${INFERENCE_MODE:-$(yaml_value inference_mode)}"
NUM_INFERENCE_TIMESTEPS="${NUM_INFERENCE_TIMESTEPS:-$(yaml_value num_inference_timesteps)}"
HISTORY_ACTION_NOISE_STD="${HISTORY_ACTION_NOISE_STD:-$(yaml_value history_action_noise_std)}"
FUTURE_VIDEO_DENOISE_FRACTION="${FUTURE_VIDEO_DENOISE_FRACTION:-$(yaml_value future_video_denoise_fraction)}"
GPU_ID="${GPU_ID:-0}"

TASK_CONFIG="${TASK_CONFIG:-demo_randomized}"
SEED="${SEED:-42}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
TEST_NUM="${TEST_NUM:-100}"
INFERENCE_MODE="${INFERENCE_MODE:-legacy}"
NUM_INFERENCE_TIMESTEPS="${NUM_INFERENCE_TIMESTEPS:-10}"
HISTORY_ACTION_NOISE_STD="${HISTORY_ACTION_NOISE_STD:-0.02}"
FUTURE_VIDEO_DENOISE_FRACTION="${FUTURE_VIDEO_DENOISE_FRACTION:-1.0}"

for required_name in ROBOTWIN_ROOT CHECKPOINT_PATH WAN_PATH VLM_PATH; do
    required_value="${!required_name}"
    if [[ -z "$required_value" || ! -d "$required_value" ]]; then
        echo "Error: $required_name is not a directory: $required_value" >&2
        exit 1
    fi
done
DEPLOYED_POLICY_DIR="$ROBOTWIN_ROOT/policy/Motus"
if [[ ! -f "$DEPLOYED_POLICY_DIR/deploy_policy.yml" ]]; then
    echo "Error: deploy this directory to $DEPLOYED_POLICY_DIR first." >&2
    exit 1
fi

if [[ -n "$CONDA_ENV" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "Error: conda is unavailable; activate $CONDA_ENV manually or clear conda_env." >&2
        exit 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

LOG_DIR="${LOG_DIR:-$POLICY_DIR/logs_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${TASK_NAME}.log}"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONNOUSERSITE=1
unset PYTHONUSERBASE || true
export PYTHONPATH="$ROBOTWIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

cd "$ROBOTWIN_ROOT"
echo "Evaluating $TASK_NAME on GPU $GPU_ID (mode=$INFERENCE_MODE)"

set +e
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py \
    --config "policy/Motus/deploy_policy.yml" \
    --overrides \
    --task_name "$TASK_NAME" \
    --task_config "$TASK_CONFIG" \
    --ckpt_setting "$CHECKPOINT_PATH" \
    --seed "$SEED" \
    --policy_name "Motus" \
    --instruction_type "$INSTRUCTION_TYPE" \
    --log_dir "$LOG_DIR" \
    --wan_path "$WAN_PATH" \
    --vlm_path "$VLM_PATH" \
    --inference_mode "$INFERENCE_MODE" \
    --test_num "$TEST_NUM" \
    --num_inference_timesteps "$NUM_INFERENCE_TIMESTEPS" \
    --history_action_noise_std "$HISTORY_ACTION_NOISE_STD" \
    --future_video_denoise_fraction "$FUTURE_VIDEO_DENOISE_FRACTION" \
    2>&1 | tee "$LOG_FILE"
exit_code=${PIPESTATUS[0]}
set -e
exit "$exit_code"
