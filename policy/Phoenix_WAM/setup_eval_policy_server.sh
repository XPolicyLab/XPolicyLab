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

allow_dummy_policy="${WAM_ALLOW_DUMMY_POLICY:-false}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

extra_overrides=()
[[ -z "${WAM_CKPT_SETTING:-}" ]] || extra_overrides+=("ckpt_name=${WAM_CKPT_SETTING}")
[[ -z "${WAM_RUNTIME_MODE:-}" ]] || extra_overrides+=("runtime_mode=${WAM_RUNTIME_MODE}")
[[ -z "${WAM_CHECKPOINT_PATH:-}" ]] || extra_overrides+=("checkpoint_path=${WAM_CHECKPOINT_PATH}")
[[ -z "${WAM_EXPECTED_STEP:-}" ]] || extra_overrides+=("expected_step=${WAM_EXPECTED_STEP}")
[[ -z "${WAM_AUDIT_DIR:-}" ]] || extra_overrides+=("audit_dir=${WAM_AUDIT_DIR}")
# Resolution belongs to model.py and XPolicyLab's shared checkpoint helper.
# Empty explicit paths leave ckpt_name and the shared run naming in control.

echo "[SERVER] policy=${policy_name} task=${task_name} replan=10"
echo "[SERVER] checkpoint=${WAM_CHECKPOINT_PATH:-shared resolver; see ckpt_name}"

exec env \
  PYTHONWARNINGS=ignore::UserWarning \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
  PYTHONPATH="${XPL_ROOT}/..:${PYTHONPATH:-}" \
  DIFFSYNTH_SKIP_DOWNLOAD=true \
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
      allow_dummy_policy="${allow_dummy_policy}" \
      "${extra_overrides[@]}"
