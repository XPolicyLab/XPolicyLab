#!/bin/bash
set -euo pipefail

if (($# < 10 || $# > 11)); then
    echo "Usage: $0 bench task ckpt env_cfg action seed env_gpu eval_env additional_info port [host]" >&2
    exit 2
fi

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
env_gpu_id=${7}
eval_env_conda_env=${8}
additional_info=${9}
policy_server_port=${10}
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

# Pass documented RoboTwin options through its existing additional_info parser.
# No task, success checker, scene sampler, or official template is modified.
if [[ "${bench_name}" != RoboTwin ]]; then
    echo "CR-WM is validated for bench_name=RoboTwin only." >&2; exit 2
fi
additional_info+=",num_workers=1"
if [[ -n "${CRWM_TASK_CONFIG:-}" ]]; then
    [[ "$CRWM_TASK_CONFIG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid CRWM_TASK_CONFIG" >&2; exit 2; }
    additional_info+=",task_config=${CRWM_TASK_CONFIG}"
fi
if [[ -n "${CRWM_INSTRUCTION_TYPE:-}" ]]; then
    [[ "$CRWM_INSTRUCTION_TYPE" == seen || "$CRWM_INSTRUCTION_TYPE" == unseen ]] || { echo "Instruction type must be seen or unseen" >&2; exit 2; }
    additional_info+=",instruction_type=${CRWM_INSTRUCTION_TYPE}"
fi
if [[ -n "${CRWM_TEST_NUM:-}" ]]; then
    [[ "$CRWM_TEST_NUM" =~ ^[1-9][0-9]*$ ]] || { echo "CRWM_TEST_NUM must be positive" >&2; exit 2; }
    additional_info+=",test_num=${CRWM_TEST_NUM}"
fi

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

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
    "${BENCH_ROOT}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_ip}"
