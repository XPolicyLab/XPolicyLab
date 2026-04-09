#!/bin/bash
set -e

UTILS_DIR="$1"
yaml_file="$2"
eval_env_conda_env="$3"
FREE_PORT="$4"
task_name="$5"
env_cfg_type="$6"
policy_name="$7"
ROOT_DIR="$8"

read eval_batch eval_env < <(python - <<PY
import yaml
with open("${yaml_file}", "r") as f:
    data = yaml.safe_load(f)
print(str(data.get("eval_batch", False)).lower(), data.get("eval_env", "debug"))
PY
)

if [[ "${eval_env}" == "debug" ]]; then
    bash "${UTILS_DIR}/run_debug_policy_client.sh" "${eval_batch}" "${eval_env_conda_env}" "${FREE_PORT}" "${task_name}" "${env_cfg_type}" "${policy_name}" "${ROOT_DIR}"
elif [[ "${eval_env}" == "sim" ]]; then
    bash "${UTILS_DIR}/run_sim_policy_client.sh" "${eval_batch}" "${eval_env_conda_env}" "${FREE_PORT}" "${task_name}" "${env_cfg_type}" "${policy_name}" "${ROOT_DIR}"
elif [[ "${eval_env}" == "real" ]]; then
    bash "${UTILS_DIR}/run_real_policy_client.sh" "${eval_batch}" "${eval_env_conda_env}" "${FREE_PORT}" "${task_name}" "${env_cfg_type}" "${policy_name}" "${ROOT_DIR}"
else
    echo "[ERROR] Unknown eval_env: ${eval_env}"
    exit 1
fi