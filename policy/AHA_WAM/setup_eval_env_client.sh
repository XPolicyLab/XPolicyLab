#!/bin/bash
set -euo pipefail

dataset_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env_conda_env=$8
additional_info=$9
policy_server_port=${10}
policy_server_ip=${11:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
yaml_file="${SCRIPT_DIR}/deploy.yml"
eval_episode_num="${AHA_WAM_DEBUG_EVAL_EPISODE_NUM:-${DEBUG_EVAL_EPISODE_NUM:-100}}"
sim_root_dir="${XPOLICYLAB_SIM_ROOT:-${ROOT_DIR}}"
env_cfg_root="${AHA_WAM_ENV_CFG_ROOT:-/mnt/petrelfs/caijisong/env_cfg}"

read eval_batch eval_env < <(python3 - "${yaml_file}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
except Exception:
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            value = value.strip().split("#", 1)[0].strip().strip("\"'")
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.lower() in {"none", "null", ""}:
                value = None
            data[key.strip()] = value

print(str(data.get("eval_batch", False)).lower(), data.get("eval_env"))
PY
)

echo -e "\033[34m[CLIENT] policy=aha-wam task=${task_name} server=${policy_server_ip}:${policy_server_port}\033[0m"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate "${eval_env_conda_env}"

export CUDA_VISIBLE_DEVICES="${env_gpu_id}"
export PYTHONWARNINGS=ignore::UserWarning
export PYTHONPATH="${ROOT_DIR}/XPolicyLab:${ROOT_DIR}:${PYTHONPATH:-}"

if [[ "${eval_env}" == "debug" ]]; then
    echo -e "\033[34m[CLIENT] Activating Conda environment: ${eval_env_conda_env}\033[0m"
    echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${policy_server_port}...\033[0m"
    exec python "${SCRIPT_DIR}/debug_env_client.py" \
        --dataset_name "${dataset_name}" \
        --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" \
        --env_cfg_root "${env_cfg_root}" \
        --host "${policy_server_ip}" \
        --port "${policy_server_port}" \
        --eval_batch "${eval_batch}" \
        --eval_episode_num "${eval_episode_num}"
elif [[ "${eval_env}" == "sim" ]]; then
    echo -e "\033[34m[CLIENT] Activating Conda environment: ${eval_env_conda_env}\033[0m"
    echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${policy_server_port}...\033[0m"
    exec bash "${sim_root_dir}/scripts/eval_policy.sh" \
        --dataset_name "${dataset_name}" \
        --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" \
        --policy_name "AHA_WAM" \
        --host "${policy_server_ip}" \
        --port "${policy_server_port}" \
        --eval_batch "${eval_batch}" \
        --root_dir "${sim_root_dir}" \
        --device_id "${env_gpu_id}" \
        --additional_info "${additional_info}" \
        --seed "${seed}"
elif [[ "${eval_env}" == "real" ]]; then
    exec bash "${UTILS_DIR}/run_real_policy_client.sh" \
        "${eval_batch}" "${eval_env_conda_env}" "${policy_server_port}" \
        "${dataset_name}" "${task_name}" "${env_cfg_type}" "AHA_WAM" \
        "${additional_info}" "${ROOT_DIR}" "${seed}" "${env_gpu_id}" "${policy_server_ip}"
else
    echo "[ERROR] Unknown eval_env: ${eval_env}" >&2
    exit 1
fi
