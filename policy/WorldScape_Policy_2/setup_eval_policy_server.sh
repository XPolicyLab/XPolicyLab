#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?}
task_name=${2:?}
ckpt_name=${3:?}
env_cfg_type=${4:?}
action_type=${5:?}
seed=${6:?}
policy_gpu_id=${7:?}
policy_conda_env=${8:?}
policy_server_port=${9:?}
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="${BENCH_ROOT:-$(cd "${XPL_ROOT}/.." && pwd)}"
UTILS_DIR="${XPL_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

# WorldScape Policy source lives in worldscape-policy/ next to this script;
# WORLDSCAPE_POLICY_ROOT overrides it with another checkout.
worldscape_root="${WORLDSCAPE_POLICY_ROOT:-${SCRIPT_DIR}/worldscape-policy}"
allow_dummy_policy="${WORLDSCAPE_ALLOW_DUMMY_POLICY:-false}"
if [[ ! -d "${worldscape_root}/src/worldscape_policy" ]]; then
    echo "[SERVER][ERROR] Invalid WorldScape checkout: ${worldscape_root}" >&2
    exit 2
fi
worldscape_root="$(cd "${worldscape_root}" && pwd)"

# Checkpoint resolution happens in model.py (XPolicyLab checkpoint_resolver
# precedence: explicit path > ckpt_name as path > checkpoints/<run-dir> >
# checkpoints/<ckpt_name> > <worldscape_root>/<ckpt_name>). WORLDSCAPE_CHECKPOINT
# forces an explicit checkpoint directory.
checkpoint_path="${WORLDSCAPE_CHECKPOINT:-}"
if [[ -n "${checkpoint_path}" ]]; then
    if [[ ! -d "${checkpoint_path}" ]]; then
        echo "[SERVER][ERROR] Checkpoint directory not found: ${checkpoint_path}" >&2
        exit 2
    fi
    checkpoint_path="$(cd "${checkpoint_path}" && pwd)"
fi

if [[ "${action_type}" != "joint" ]]; then
    echo "[SERVER][ERROR] ${policy_name} requires action_type=joint." >&2
    exit 2
fi

action_horizon="${ROBOTWIN_ACTION_HORIZON:-24}"
case "${action_horizon}" in
    24) default_observation_interval=3 ;;
    48) default_observation_interval=6 ;;
    *)
        echo "[SERVER][ERROR] ROBOTWIN_ACTION_HORIZON must be 24 or 48." >&2
        exit 2
        ;;
esac
replan_steps="${ROBOTWIN_REPLAN_STEPS:-${action_horizon}}"
observation_interval="${ROBOTWIN_OBSERVATION_INTERVAL:-${default_observation_interval}}"

echo "[SERVER] policy=${policy_name} task=${task_name}"
echo "[SERVER] worldscape_root=${worldscape_root}"
echo "[SERVER] ckpt_name=${ckpt_name} checkpoint_path=${checkpoint_path:-<resolved by model.py>}"
echo "[SERVER] endpoint=${policy_server_host}:${policy_server_port}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

exec env \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="${BENCH_ROOT}:${worldscape_root}:${worldscape_root}/src:${PYTHONPATH:-}" \
    python -u "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            action_type="${action_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            worldscape_root="${worldscape_root}" \
            checkpoint_path="${checkpoint_path}" \
            device="${WORLDSCAPE_DEVICE:-cuda}" \
            mode="${WSP_MODE:-auto}" \
            action_horizon="${action_horizon}" \
            replan_steps="${replan_steps}" \
            observation_interval="${observation_interval}" \
            memory_reset_chunks="${ROBOTWIN_MEMORY_RESET_CHUNKS:-0}" \
            vlm_history_num_frames="${ROBOTWIN_VLM_HISTORY_NUM_FRAMES:-4}" \
            diffusion_view_layout="${ROBOTWIN_DIFFUSION_VIEW_LAYOUT:-}" \
            vlm_cot_prompt="${ROBOTWIN_VLM_COT_PROMPT:-}" \
            t5_prompt_template="${ROBOTWIN_T5_PROMPT_TEMPLATE:-}" \
            validate_checkpoint_artifacts="${WSP_VALIDATE_CHECKPOINT_ARTIFACTS:-false}" \
            log_inference="${WSP_LOG_INFERENCE:-true}" \
            allow_dummy_policy="${allow_dummy_policy}"
