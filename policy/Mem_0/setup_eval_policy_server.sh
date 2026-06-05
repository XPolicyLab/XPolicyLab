#!/bin/bash
set -euo pipefail

dataset_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}
vllm_url=${11:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
POLICY_DIR="${ROOT_DIR}/XPolicyLab/policy/${policy_name}"
UPSTREAM_DIR="${POLICY_DIR}/Mem_0"
yaml_file="${POLICY_DIR}/deploy.yml"
ADAPTER_DIR="${UPSTREAM_DIR}/xpolicylab_adapter"

source "${ADAPTER_DIR}/_artifact_paths.sh"

expert_data_num="${MEM0_EXPERT_DATA_NUM:-}"
ckpt_dir="$(mem0_resolve_ckpt_dir "${POLICY_DIR}" "${dataset_name}" "${ckpt_name}" \
    "${env_cfg_type}" "${action_type}" "${seed}" "${expert_data_num}")"

execution_ckpt="${MEM0_EXECUTION_CKPT:-}"
if [[ -z "${execution_ckpt}" && -d "${ckpt_dir}" ]]; then
    execution_ckpt=$(find "${ckpt_dir}" -maxdepth 1 -type f -name 'final_step*.pt' | sort -V | tail -n 1)
    if [[ -z "${execution_ckpt}" ]]; then
        execution_ckpt=$(find "${ckpt_dir}" -maxdepth 1 -type f -name 'step*.pt' | sort -V | tail -n 1)
    fi
fi

state_stats_path="${MEM0_STATE_STATS_PATH:-$(mem0_resolve_norm_stats_path "${POLICY_DIR}" "${ckpt_name}")}"
planning_cfg="${MEM0_PLANNING_MODULE_CONFIG:-${UPSTREAM_DIR}/source/config/planning_module_inference.yaml}"
global_task="${GLOBAL_TASK:-}"
action_horizon="${MEM0_ACTION_HORIZON:-30}"
threshold="${MEM0_THRESHOLD:-2}"

task_config="${UPSTREAM_DIR}/xpolicylab_adapter/task_config.json"
task_type=$(python3 - "${task_name}" "${task_config}" <<'PY'
import json, sys
task_name, path = sys.argv[1], sys.argv[2]
with open(path) as f:
    cfg = json.load(f)
if task_name in (cfg.get("Mn") or []):
    print("Mn")
elif task_name in (cfg.get("M1") or []):
    print("M1")
else:
    print("M1")
PY
)

echo -e "\033[33m[SERVER] policy=${policy_name} task=${task_name} ckpt=${ckpt_name} type=${task_type}\033[0m"
echo -e "\033[33m[SERVER] ckpt_dir=${ckpt_dir}\033[0m"
echo -e "\033[33m[SERVER] execution_ckpt=${execution_ckpt:-<unset>}\033[0m"
echo -e "\033[33m[SERVER] state_stats_path=${state_stats_path}\033[0m"
echo -e "\033[33m[SERVER] vllm_url=${vllm_url:-<unset>}\033[0m"
echo -e "\033[33m[SERVER] host=${policy_server_host} port=${policy_server_port}\033[0m"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
echo -e "\033[33m[SERVER] action_dim=${action_dim}\033[0m"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${ROOT_DIR}:${UPSTREAM_DIR}:${PYTHONPATH:-}" \
    python -u "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            dataset_name="${dataset_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            execution_ckpt="${execution_ckpt}" \
            state_stats_path="${state_stats_path}" \
            task_type="${task_type}" \
            planning_module_config_path="${planning_cfg}" \
            vllm_url="${vllm_url}" \
            global_task="${global_task}" \
            action_horizon="${action_horizon}" \
            threshold="${threshold}"
