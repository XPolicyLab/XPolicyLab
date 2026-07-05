#!/bin/bash
set -euo pipefail

# Mem_0 eval orchestrator (XPolicyLab 10-arg contract + optional Mn vLLM auto-start).
#
# Usage:
#   bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> \
#                <action_type> <seed> \
#                <policy_gpu_id> <env_gpu_id> \
#                <policy_conda_env> <eval_env_conda_env> [planning_gpu_ids]
#
# `task_name` is the simulator task; `ckpt_name` resolves checkpoint paths.
# For Mn tasks, pass `planning_gpu_ids` (comma-separated) as optional 11th arg to auto-start vLLM.
# Disambiguate legacy artifacts with MEM0_EXPERT_DATA_NUM when multiple datasets match.
#
# Switch debug/sim via deploy.yml `eval_env` (not this script).
#
# Examples:
#   # M1 debug wiring check
#   bash eval.sh RoboDojo swap_blocks swap_blocks arx_x5 joint 0 0 0 mem0 XPolicyLab
#
#   # Mn with auto vLLM on GPUs 4,5,6,7 and execution on GPU 0
#   GLOBAL_TASK="..." bash eval.sh RoboDojo cover_blocks cover_blocks arx_x5 joint 0 \
#       0 0 mem0 XPolicyLab 4,5,6,7

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_conda_env=${9}
eval_env_conda_env=${10}
planning_gpu_ids=${11:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
UPSTREAM_DIR="${SCRIPT_DIR}/Mem_0"
TASK_CONFIG="${UPSTREAM_DIR}/xpolicylab_adapter/task_config.json"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"
PLANNING_SCRIPT="${SCRIPT_DIR}/setup_eval_planning_server.sh"

task_type=$(python3 - "${task_name}" "${TASK_CONFIG}" <<'PY'
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

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"
vllm_url="${VLLM_URL:-}"

additional_info="ckpt_name=${ckpt_name},action_type=${action_type},task_type=${task_type}"

wait_for_port() {
    local host=$1
    local port=$2
    local pid=$3
    local label=$4
    for _ in $(seq 1 180); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo -e "\033[31m[ERROR] ${label} exited before opening port ${port}.\033[0m" >&2
            exit 1
        fi
        if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('${host}', int('${port}'))); s.close()" >/dev/null 2>&1; then
            echo -e "\033[32m[MAIN] ${label} ready on ${host}:${port}\033[0m"
            return 0
        fi
        sleep 2
    done
    echo -e "\033[31m[ERROR] ${label} timed out waiting for port ${port}.\033[0m" >&2
    exit 1
}

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo -e "\033[31m[CLEANUP] kill policy server PID=${SERVER_PID}\033[0m"
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
    if [[ -n "${PLANNING_PID:-}" ]]; then
        echo -e "\033[31m[CLEANUP] kill planning server PID=${PLANNING_PID}\033[0m"
        kill "${PLANNING_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "${task_type}" == "Mn" && -z "${vllm_url}" && -n "${planning_gpu_ids}" ]]; then
    planning_port=$(bash "${UTILS_DIR}/get_free_port.sh")
    echo -e "\033[32m[MAIN] Mn task: start vLLM planning server on port ${planning_port}\033[0m"
    bash "${PLANNING_SCRIPT}" \
        "${bench_name}" \
        "${ckpt_name}" \
        "${env_cfg_type}" \
        "${action_type}" \
        "${seed}" \
        "${planning_gpu_ids}" \
        "${planning_port}" \
        "${SCRIPT_DIR}" &
    PLANNING_PID=$!
    wait_for_port "127.0.0.1" "${planning_port}" "${PLANNING_PID}" "Planning server"
    vllm_url="http://127.0.0.1:${planning_port}/v1"
elif [[ "${task_type}" == "Mn" && -z "${vllm_url}" ]]; then
    echo -e "\033[33m[WARN] Mn task without planning_gpu_ids or VLLM_URL; planner calls will fail.\033[0m" >&2
fi

echo -e "\033[32m[MAIN] start execution policy server, port=${policy_server_port}\033[0m"
bash "${SERVER_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_conda_env}" \
    "${policy_server_port}" \
    "${policy_server_ip}" \
    "${vllm_url}" &
SERVER_PID=$!
wait_for_port "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server"

echo -e "\033[32m[MAIN] start env client (eval_env from deploy.yml), server=${policy_server_ip}:${policy_server_port}\033[0m"
bash "${CLIENT_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${eval_env_conda_env}" \
    "${additional_info}" \
    "${policy_server_port}" \
    "${policy_server_ip}"

echo -e "\033[33m[MAIN] eval finished\033[0m"
