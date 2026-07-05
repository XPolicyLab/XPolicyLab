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

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

read eval_batch eval_env < <(python - <<PY
import yaml
with open("${yaml_file}", "r") as f:
    data = yaml.safe_load(f)
print(str(data.get("eval_batch", False)).lower(), data.get("eval_env", "debug"))
PY
)

conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
if [ -z "${conda_exe}" ] && [ -x "/opt/conda/bin/conda" ]; then
    conda_exe="/opt/conda/bin/conda"
fi
if [ -z "${conda_exe}" ]; then
    echo "[CLIENT][ERROR] conda executable not found. Set CONDA_EXE or use /opt/conda/bin/conda." >&2
    exit 1
fi
source "$("${conda_exe}" info --base)/etc/profile.d/conda.sh"
conda deactivate || true
client_conda_env="${eval_env_conda_env}"
conda activate "${client_conda_env}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

echo -e "\033[34m[CLIENT] Activating Conda environment: ${client_conda_env}\033[0m"
echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${policy_server_port} (eval_env=${eval_env})\033[0m"

if [[ "${eval_env}" == "sim" ]]; then
    eval_policy_script="${ROOT_DIR}/scripts/eval_policy.sh"
    if [ ! -f "${eval_policy_script}" ]; then
        echo "[CLIENT][ERROR] Simulation entry not found: ${eval_policy_script}" >&2
        echo "[CLIENT][ERROR] This workspace does not include the external sim runner. Set eval_env: debug in deploy.yml for a local smoke test, or place scripts/eval_policy.sh under ${ROOT_DIR}." >&2
        exit 1
    fi
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    DREAMZERO_MODEL_CLIENT_TIMEOUT="${DREAMZERO_MODEL_CLIENT_TIMEOUT:-1800}" \
    PYTHONWARNINGS=ignore::UserWarning \
    bash "${eval_policy_script}" \
        --bench_name "${bench_name}" \
        --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" \
        --policy_name "${policy_name}" \
        --port "${policy_server_port}" \
        --host "${policy_server_ip}" \
        --eval_batch "${eval_batch}" \
        --root_dir "${ROOT_DIR}" \
        --device_id "${env_gpu_id}" \
        --additional_info "${additional_info}" \
        --seed "${seed}"
    exit 0
fi

if [[ "${eval_env}" == "debug" ]]; then
    PYTHONPATH="${ROOT_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}" \
    DREAMZERO_MODEL_CLIENT_TIMEOUT="${DREAMZERO_MODEL_CLIENT_TIMEOUT:-1800}" \
    PYTHONWARNINGS=ignore::UserWarning \
    python "${ROOT_DIR}/XPolicyLab/debug_env_client.py" \
        --bench_name "${bench_name}" \
        --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" \
        --policy_name "${policy_name}" \
        --port "${policy_server_port}" \
        --eval_batch "${eval_batch}"
    exit 0
fi

PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
DREAMZERO_MODEL_CLIENT_TIMEOUT="${DREAMZERO_MODEL_CLIENT_TIMEOUT:-1800}" \
bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${yaml_file}" \
    "${eval_env_conda_env}" \
    "${policy_server_port}" \
    "${bench_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${policy_name}" \
    "${additional_info}" \
    "${ROOT_DIR}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_ip}"
