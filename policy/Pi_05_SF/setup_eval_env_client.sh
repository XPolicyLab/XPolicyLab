#!/bin/bash
set -e

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
find_xpolicylab_root() {
    local dir
    dir="$(cd "${1}" && pwd)"
    while [[ "${dir}" != "/" ]]; do
        if [[ -f "${dir}/setup_policy_server.py" ]]; then
            echo "${dir}"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done
    echo "[CLIENT][ERROR] XPolicyLab root (setup_policy_server.py) not found above ${1}" >&2
    return 1
}
XPL_ROOT="$(find_xpolicylab_root "${SCRIPT_DIR}")"
ROOT_DIR="$(dirname "${XPL_ROOT}")"
UTILS_DIR="${XPL_ROOT}/utils"
yaml_file="${SCRIPT_DIR}/deploy.yml"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
YAML_PYTHON="${CONDA_BASE}/bin/python"

policy_name="$("${YAML_PYTHON}" - <<PYENV
import yaml
cfg = yaml.safe_load(open("${yaml_file}", encoding="utf-8"))
print(cfg["policy_name"])
PYENV
)"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

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
