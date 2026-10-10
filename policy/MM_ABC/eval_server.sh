#!/bin/bash
set -euo pipefail

ckpt_name=${1:?usage: eval_server.sh <ckpt_name> [gpu_ids] [base_port] [policy_conda_env] [key=value ...]}
gpu_ids=${2:-0}
base_port=${3:-19000}
policy_conda_env=${4:-none}
shift $(( $# < 4 ? $# : 4 ))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

LOGDIR="${SCRIPT_DIR}/logs/eval_server/$(date +%Y%m%d-%H%M%S)"
mkdir -p "${LOGDIR}"
lan_ip=$(ip -4 -o addr show eth0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
advertise=${PUBLIC_IP:-${lan_ip:-$(hostname -I | awk '{print $1}')}}

PIDS=()
cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

IFS=',' read -r -a gpus <<<"${gpu_ids}"
for i in "${!gpus[@]}"; do
    port=$(( base_port + i ))
    bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
        "${BENCH_NAME:?Set BENCH_NAME to the environment identifier}" "${TASK_NAME:-default}" "${ckpt_name}" "${ENV_CFG_TYPE:-m92uw}" joint "${SEED:-0}" "${gpus[$i]}" \
        "${policy_conda_env}" "${port}" 0.0.0.0 "$@" > "${LOGDIR}/gpu${gpus[$i]}.log" 2>&1 &
    PIDS+=($!)
done

for i in "${!gpus[@]}"; do
    port=$(( base_port + i ))
    bash "${UTILS_DIR}/wait_for_policy_server.sh" localhost "${port}" "${PIDS[$i]}" "MM_ABC server gpu${gpus[$i]}" 1800
    echo "[SERVER] ready: ws://${advertise}:${port}   (gpu ${gpus[$i]}, log ${LOGDIR}/gpu${gpus[$i]}.log)"
done
echo "[SERVER] all ${#gpus[@]} servers up; point each Mobile client at one address above. Ctrl-C to stop."
wait
