#!/bin/bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
    echo "Usage: bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>"
    echo "Example: bash eval.sh SparkArena <task_name> sft_spark_0 tianji_marvin_wuji ee 0 0 1 spark_0 <eval_env_conda_env>"
    exit 1
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_conda_env=$9
eval_env_conda_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

# Components are launched independently by Eval Web's remote runner. Keep the
# native wrappers so checkpoint identity, training stats and fast-obs config
# resolution are identical to same-node evaluation.
component="${EGOVLA_COMPONENT:-}"
if [[ -n "${component}" ]]; then
    if [[ "${bench_name}" != "SparkArena" ]]; then
        echo "[MAIN][ERROR] Spark split components require SparkArena" >&2
        exit 2
    fi
    policy_server_host="${EGOVLA_POLICY_SERVER_HOST:?policy server host is required}"
    policy_server_port="${EGOVLA_POLICY_SERVER_PORT:?policy server port is required}"
    if [[ ! "${policy_server_port}" =~ ^[0-9]+$ ]] || (( policy_server_port < 1 || policy_server_port > 65535 )); then
        echo "[MAIN][ERROR] invalid policy server port" >&2
        exit 2
    fi
    case "${component}" in
        policy)
            echo "[MAIN] split Spark server GPU=${policy_gpu_id}, bind=${policy_server_host}:${policy_server_port}"
            exec bash "${SERVER_SCRIPT}" \
                "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
                "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_conda_env}" \
                "${policy_server_port}" "${policy_server_host}"
            ;;
        environment)
            echo "[MAIN] split Spark client GPU=${env_gpu_id}, server=${policy_server_host}:${policy_server_port}"
            ready=0
            # This is a remote endpoint: the local /proc/net/tcp LISTEN check
            # in wait_for_policy_server.sh must not be used here.
            for (( attempt=0; attempt<600; attempt++ )); do
                if timeout 1 bash -c 'exec 3<>"/dev/tcp/$1/$2"' _ "${policy_server_host}" "${policy_server_port}" 2>/dev/null; then
                    ready=1
                    break
                fi
                sleep 2
            done
            if [[ "${ready}" != "1" ]]; then
                echo "[MAIN][ERROR] remote Spark policy server did not become ready" >&2
                exit 1
            fi
            additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"
            exec bash "${CLIENT_SCRIPT}" \
                "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
                "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" \
                "${additional_info}" "${policy_server_port}" "${policy_server_host}"
            ;;
        *)
            echo "[MAIN][ERROR] unsupported Spark component: ${component}" >&2
            exit 2
            ;;
    esac
fi

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_host="localhost"
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[MAIN] kill server ${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[MAIN] start Spark_0 server, policy_server_port=${policy_server_port}"

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
    "${policy_server_host}" &

SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_host}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200

echo "[MAIN] start client, server=${policy_server_host}:${policy_server_port}"

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
    "${policy_server_host}"

echo "[MAIN] eval finished"
