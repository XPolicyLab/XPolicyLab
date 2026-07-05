#!/bin/bash
set -e

bench_name=$1
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
yaml_file="${SCRIPT_DIR}/deploy.yml"

echo "[CLIENT] policy=A1, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

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

echo -e "\033[34m[CLIENT] Activating Conda environment: ${eval_env_conda_env}\033[0m"
echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${policy_server_port} (eval_env=${eval_env})\033[0m"

if [[ "${eval_env}" == "sim" ]]; then
    PYTHONWARNINGS=ignore::UserWarning \
    bash "${ROOT_DIR}/scripts/eval_policy.sh" \
        --bench_name "${bench_name}" \
        --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" \
        --policy_name "A1" \
        --port "${policy_server_port}" \
        --host "${policy_server_ip}" \
        --eval_batch "${eval_batch}" \
        --root_dir "${ROOT_DIR}" \
        --device_id "${env_gpu_id}" \
        --additional_info "${additional_info}" \
        --seed "${seed}"
    exit 0
fi

bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${yaml_file}" \
    "${eval_env_conda_env}" \
    "${policy_server_port}" \
    "${bench_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "A1" \
    "${additional_info}" \
    "${ROOT_DIR}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_ip}"
