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
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
SPIRIT_SITECUSTOMIZE_DIR="${SCRIPT_DIR}/deploy"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

echo -e "\033[34m[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}\033[0m"

read eval_batch eval_env < <(python - <<PY
import yaml
with open("${yaml_file}", "r") as f:
    data = yaml.safe_load(f)
print(str(data.get("eval_batch", False)).lower(), data.get("eval_env", "debug"))
PY
)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate "${eval_env_conda_env}"

SPIRIT_CLIENT_ENV=(
    "PYTHONPATH=${SPIRIT_SITECUSTOMIZE_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"
    "SPIRIT_MODEL_CLIENT_TIMEOUT=${SPIRIT_MODEL_CLIENT_TIMEOUT:-600}"
    "PYTHONWARNINGS=ignore::UserWarning"
)

if [[ "${eval_env}" == "debug" ]]; then
    echo -e "\033[34m[CLIENT] Activating Conda environment: ${eval_env_conda_env}\033[0m"
    echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${policy_server_port}...\033[0m"
    env "${SPIRIT_CLIENT_ENV[@]}" \
        python "${ROOT_DIR}/XPolicyLab/debug_env_client.py" \
            --dataset_name "${dataset_name}" \
            --task_name "${task_name}" \
            --env_cfg_type "${env_cfg_type}" \
            --policy_name "${policy_name}" \
            --host "${policy_server_ip}" \
            --port "${policy_server_port}" \
            --eval_batch "${eval_batch}"
    exit 0
fi

bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${yaml_file}" \
    "${eval_env_conda_env}" \
    "${policy_server_port}" \
    "${dataset_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${policy_name}" \
    "${additional_info}" \
    "${ROOT_DIR}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_ip}"
