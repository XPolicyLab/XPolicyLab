#!/usr/bin/env bash
set -euo pipefail

# SparkArena is the benchmark identity for both the policy and environment.
# Usage: bash eval.sh SparkArena <task> <checkpoint> tianji_marvin_wuji joint <seed> <policy_gpu> <env_gpu> <policy_env> <eval_env>
bench_name=${1:?bench_name is required}
task_name=${2:?task_name is required}
ckpt_name=${3:?finalized joint checkpoint path is required}
env_cfg_type=${4:?env_cfg_type is required}
action_type=${5:?action_type is required}
seed=${6:?seed is required}
policy_gpu_id=${7:?policy gpu id is required}
env_gpu_id=${8:?env gpu id is required}
policy_env=${9:-${EGOVLA_CONDA_ENV:-}}
eval_env=${10:-${EGOVLA_EVAL_CONDA_ENV:-}}
[[ -n "${eval_env}" ]] || { echo "eval env is required; pass argument 10 or set EGOVLA_EVAL_CONDA_ENV" >&2; exit 2; }

[[ "${bench_name}" == "SparkArena" ]] || {
  echo "SparkArena EgoVLA simulation requires bench_name=SparkArena" >&2; exit 2;
}
[[ "${env_cfg_type}" == "tianji_marvin_wuji" && "${action_type}" == "joint" ]] || {
  echo "SparkArena EgoVLA evaluation requires tianji_marvin_wuji/joint" >&2; exit 2;
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

# Eval Web starts the policy server and the simulator on different machines.
# Each side runs this script with EGOVLA_COMPONENT set.
component="${EGOVLA_COMPONENT:-}"
if [[ -n "${component}" ]]; then
    policy_server_host="${EGOVLA_POLICY_SERVER_HOST:?policy server host is required}"
    policy_server_port="${EGOVLA_POLICY_SERVER_PORT:?policy server port is required}"
    if [[ ! "${policy_server_port}" =~ ^[0-9]+$ ]] || (( policy_server_port < 1 || policy_server_port > 65535 )); then
        echo "[MAIN][ERROR] invalid policy server port" >&2
        exit 2
    fi
    case "${component}" in
        policy)
            echo "[MAIN] split EgoVLA server GPU=${policy_gpu_id}, bind=${policy_server_host}:${policy_server_port}"
            exec bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
                "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
                "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_env}" \
                "${policy_server_port}" "${policy_server_host}"
            ;;
        environment)
            echo "[MAIN] split EgoVLA client GPU=${env_gpu_id}, server=${policy_server_host}:${policy_server_port}"
            ready=0
            for (( attempt=0; attempt<600; attempt++ )); do
                if timeout 1 bash -c 'exec 3<>"/dev/tcp/$1/$2"' _ "${policy_server_host}" "${policy_server_port}" 2>/dev/null; then
                    ready=1
                    break
                fi
                sleep 2
            done
            if [[ "${ready}" != "1" ]]; then
                echo "[MAIN][ERROR] remote EgoVLA policy server did not become ready" >&2
                exit 1
            fi
            exec bash "${SCRIPT_DIR}/setup_eval_env_client.sh" \
                "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
                "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env}" \
                "ckpt_name=${ckpt_name},action_type=joint" \
                "${policy_server_port}" "${policy_server_host}"
            ;;
        *)
            echo "[MAIN][ERROR] unsupported EgoVLA component: ${component}" >&2
            exit 2
            ;;
    esac
fi

port=$(bash "${UTILS_DIR}/get_free_port.sh")
server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then kill "${server_pid}" 2>/dev/null || true; fi
}
trap cleanup EXIT

bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
  "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
  "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_env}" "${port}" &
server_pid=$!
bash "${UTILS_DIR}/wait_for_policy_server.sh" localhost "${port}" "${server_pid}" \
  "EgoVLA SparkArena raw-joint policy server" 1200

bash "${SCRIPT_DIR}/setup_eval_env_client.sh" \
  "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
  "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env}" \
  "ckpt_name=${ckpt_name},action_type=joint" "${port}" localhost
