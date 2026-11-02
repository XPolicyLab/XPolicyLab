#!/bin/bash
# Policy-side server for Auriga. Same arguments as policy/demo_policy/setup_eval_policy_server.sh.
# Differences: sources config.env (model endpoint provided privately by the Auriga team, perception
# paths), starts the local VGGT / SAM 3.1 perception services when nothing answers on their ports,
# and refuses to start a real evaluation without the endpoint variables (EVAL_ENV_TYPE=debug needs none).
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_conda_env=${8}
policy_server_port=${9}
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config_env.sh"
load_config_env "${SCRIPT_DIR}/config.env"

abs_path() {
    local value=$1
    if [[ "${value}" = /* ]]; then echo "${value}"; else echo "${SCRIPT_DIR}/${value#./}"; fi
}

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}"

if [[ "${EVAL_ENV_TYPE:-sim}" != "debug" ]]; then
    if [[ -z "${AURIGA_API_BASE_URL:-}" ]] || [[ -z "${AURIGA_API_KEY:-}" && -z "${AURIGA_API_KEY_FILE:-}" ]]; then
        echo "[SERVER][ERROR] AURIGA_API_BASE_URL and AURIGA_API_KEY (or AURIGA_API_KEY_FILE) are not set." >&2
        echo "[SERVER][ERROR] The Auriga team provides them privately; put them in ${SCRIPT_DIR}/config.env." >&2
        exit 1
    fi
    # The policy server starts, health-checks and stops the perception services itself
    # (auriga_xpl/perception.py).
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
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
            action_type="${action_type}"
