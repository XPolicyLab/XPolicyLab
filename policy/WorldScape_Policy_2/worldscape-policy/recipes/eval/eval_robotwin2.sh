#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Native (non-XPolicyLab) RoboTwin evaluation. Requires a RoboTwin checkout
# at ROBOTWIN_ROOT and an activated environment with both RoboTwin and the
# WorldScape runtime installed. For XPolicyLab-driven evaluation use
# policy/WorldScape_Policy_2/eval.sh instead.
if [[ -n "${CONDA_ENV:-}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to the RoboTwin checkout}"
export ROBOTWIN2_EVAL_MODEL_PATH="${ROBOTWIN2_EVAL_MODEL_PATH:-${WORLDSCAPE_CHECKPOINT:-}}"
if [[ -z "$ROBOTWIN2_EVAL_MODEL_PATH" || ! -d "$ROBOTWIN2_EVAL_MODEL_PATH" ]]; then
    echo "ERROR: Set ROBOTWIN2_EVAL_MODEL_PATH (or WORLDSCAPE_CHECKPOINT) to a checkpoint directory" >&2
    exit 2
fi
export WORLDSCAPE_CHECKPOINT="$ROBOTWIN2_EVAL_MODEL_PATH"
export WSP_MODE="${WSP_MODE:-auto}"
export ROBOTWIN_EPISODES_PER_TASK="${ROBOTWIN_EPISODES_PER_TASK:-100}"
export ROBOTWIN_MEMORY_RESET_CHUNKS="${ROBOTWIN_MEMORY_RESET_CHUNKS:-0}"
# Match training: one VLM anchor per executed action block.
export ROBOTWIN_VLM_HISTORY_NUM_FRAMES="${ROBOTWIN_VLM_HISTORY_NUM_FRAMES:-4}"
export ROBOTWIN_ACTION_HORIZON="${ROBOTWIN_ACTION_HORIZON:-48}"
case "$ROBOTWIN_ACTION_HORIZON" in
    24) default_observation_interval=3 ;;
    48) default_observation_interval=6 ;;
    *)
        echo "ERROR: ROBOTWIN_ACTION_HORIZON must be 24 or 48" >&2
        exit 2
        ;;
esac
export ROBOTWIN_REPLAN_STEPS="${ROBOTWIN_REPLAN_STEPS:-$ROBOTWIN_ACTION_HORIZON}"
export ROBOTWIN_OBSERVATION_INTERVAL="${ROBOTWIN_OBSERVATION_INTERVAL:-$default_observation_interval}"
export ROBOTWIN_VLM_HISTORY_STRIDE="${ROBOTWIN_VLM_HISTORY_STRIDE:-$ROBOTWIN_ACTION_HORIZON}"

case "$WSP_MODE" in
    interactive|auto) ;;
    *) echo "ERROR: WSP_MODE must be interactive or auto" >&2; exit 2 ;;
esac

if [[ -z "${ROBOTWIN_GPU_IDS:-}" ]]; then
    mapfile -t gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    if (( ${#gpu_ids[@]} == 0 )); then
        echo "ERROR: No available NVIDIA GPUs were detected" >&2
        exit 1
    fi
    gpu_csv="$(IFS=,; echo "${gpu_ids[*]}")"
    export ROBOTWIN_GPU_IDS="[$gpu_csv]"
    export CUDA_VISIBLE_DEVICES="$gpu_csv"
fi

export WORLDSCAPE_EVAL_OUTPUT="${WORLDSCAPE_EVAL_OUTPUT:-./evaluate_results/robotwin/$(date +%Y%m%d_%H%M%S)}"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "${WSP_PYTHON:-python}" "$ROOT/experiments/robotwin/run_robotwin_manager.py" \
    MULTIRUN.eval_phases="${ROBOTWIN_EVAL_PHASE:-clean}" \
    MULTIRUN.gpu_ids="$ROBOTWIN_GPU_IDS" \
    MULTIRUN.resume="${ROBOTWIN_RESUME:-false}" \
    "$@"
