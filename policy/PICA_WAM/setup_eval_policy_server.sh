#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

if [[ "${bench_name}" != "RoboTwin" ]]; then
    echo "[SERVER] ERROR: ${policy_name} supports bench_name=RoboTwin, got '${bench_name}'." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

# model.py resolves ckpt_name (run directory, .pt path, or the release file);
# these variables only override deploy.yml.
overrides=(
    port="${policy_server_port}"
    host="${policy_server_host}"
    bench_name="${bench_name}"
    task_name="${task_name}"
    ckpt_name="${ckpt_name}"
    env_cfg_type="${env_cfg_type}"
    seed="${seed}"
    policy_name="${policy_name}"
    action_type="${action_type}"
    allow_dummy_policy="${WAM_ALLOW_DUMMY_POLICY:-false}"
)
[[ -n "${WAM_CHECKPOINT_PATH:-}" ]] && overrides+=(checkpoint_path="${WAM_CHECKPOINT_PATH}")
[[ -n "${WAM_TRAIN_CONFIG_PATH:-}" ]] && overrides+=(train_config_path="${WAM_TRAIN_CONFIG_PATH}")
[[ -n "${WAM_DATASET_STATS_PATH:-}" ]] && overrides+=(dataset_stats_path="${WAM_DATASET_STATS_PATH}")
[[ -n "${WAM_WAN22_PATH:-}" ]] && overrides+=(base_model_dir="${WAM_WAN22_PATH}")
[[ -n "${WAM_RYNNBRAIN_PATH:-}" ]] && overrides+=(vlm_model_path="${WAM_RYNNBRAIN_PATH}")

echo "[SERVER] policy=${policy_name} task=${task_name} ckpt=${ckpt_name}"

exec env \
  PYTHONWARNINGS=ignore::UserWarning \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
  PYTHONPATH="${XPL_ROOT}:${PYTHONPATH:-}" \
  DIFFSYNTH_SKIP_DOWNLOAD=true \
  python -u "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides "${overrides[@]}"
