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
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

checkpoint_path="${WAM_CHECKPOINT_PATH:-${SCRIPT_DIR}/checkpoints/robodojo.pt}"
stats_path="${WAM_DATASET_STATS_PATH:-${SCRIPT_DIR}/config/dataset_stats.json}"
config_path="${WAM_EVAL_CONFIG_PATH:-${SCRIPT_DIR}/config/eval_model.yaml}"
base_model_dir="${WAM_WAN22_PATH:-${SCRIPT_DIR}/assets/Wan-AI/Wan2.2-TI2V-5B}"
vlm_model_path="${WAM_RYNNBRAIN_PATH:-${SCRIPT_DIR}/assets/Alibaba-DAMO-Academy/RynnBrain1.1-2B}"
allow_dummy_policy="${WAM_ALLOW_DUMMY_POLICY:-false}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

echo "[SERVER] policy=${policy_name} task=${task_name} replan=10"
echo "[SERVER] checkpoint=${checkpoint_path}"

exec env \
  PYTHONWARNINGS=ignore::UserWarning \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
  PYTHONPATH="${XPL_ROOT}:${SCRIPT_DIR}/wam_runtime/src:${PYTHONPATH:-}" \
  DIFFSYNTH_SKIP_DOWNLOAD=true \
  DIFFSYNTH_MODEL_BASE_PATH="${SCRIPT_DIR}/assets" \
  python -u "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
      port="${policy_server_port}" \
      host="${policy_server_host}" \
      bench_name="${bench_name}" \
      task_name="${task_name}" \
      ckpt_name="${ckpt_name}" \
      env_cfg_type="${env_cfg_type}" \
      seed="${seed}" \
      policy_name="${policy_name}" \
      action_type="${action_type}" \
      checkpoint_path="${checkpoint_path}" \
      dataset_stats_path="${stats_path}" \
      train_config_path="${config_path}" \
      base_model_dir="${base_model_dir}" \
      vlm_model_path="${vlm_model_path}" \
      wam_root="${SCRIPT_DIR}/wam_runtime" \
      allow_dummy_policy="${allow_dummy_policy}"
