#!/bin/bash
set -e

UTILS_DIR="${1}"
yaml_file="${2}"
eval_env_conda_env="${3}"
policy_server_port="${4}"
dataset_name="${5}"
task_name="${6}"
env_cfg_type="${7}"
policy_name="${8}"
additional_info="${9}"
ROOT_DIR="${10}"
seed="${11}"
env_gpu_id="${12}"
policy_server_ip="${13:-localhost}"

read eval_batch eval_env protocol < <(python - <<PY
import yaml
with open("${yaml_file}", "r") as f:
    data = yaml.safe_load(f)
print(
    str(data.get("eval_batch", False)).lower(),
    data.get("eval_env"),
    data.get("protocol", "legacy_tcp"),
)
PY
)

if [[ "${eval_env}" == "debug" ]]; then
    bash "${UTILS_DIR}/run_debug_env_client.sh" "${eval_batch}" "${eval_env_conda_env}" "${policy_server_port}" "${dataset_name}" "${task_name}" "${env_cfg_type}" "${policy_name}" "${additional_info}" "${ROOT_DIR}" "${seed}" "${env_gpu_id}" "${policy_server_ip}" "${protocol}"
elif [[ "${eval_env}" == "sim" ]]; then
    bash "${UTILS_DIR}/run_sim_env_client.sh" "${eval_batch}" "${eval_env_conda_env}" "${policy_server_port}" "${dataset_name}" "${task_name}" "${env_cfg_type}" "${policy_name}" "${additional_info}" "${ROOT_DIR}" "${seed}" "${env_gpu_id}" "${policy_server_ip}"
elif [[ "${eval_env}" == "real" ]]; then
    bash "${UTILS_DIR}/run_real_policy_client.sh" "${eval_batch}" "${eval_env_conda_env}" "${policy_server_port}" "${dataset_name}" "${task_name}" "${env_cfg_type}" "${policy_name}" "${additional_info}" "${ROOT_DIR}" "${seed}" "${env_gpu_id}" "${policy_server_ip}"
else
    echo "[ERROR] Unknown eval_env: ${eval_env}"
    exit 1
fi
