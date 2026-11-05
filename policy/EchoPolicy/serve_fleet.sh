#!/usr/bin/env bash
set -euo pipefail
# Usage: bash serve_fleet.sh <checkpoint> [policy_env=uv] [--dry-run]
checkpoint=${1:?Checkpoint path or name required}
policy_env=${2:-uv}
dry_run=${3:-}
[[ -z "$dry_run" || "$dry_run" == --dry-run ]] || { echo "Unknown option" >&2; exit 2; }
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IFS=, read -r -a gpus <<< "${GPU_IDS:-0,1,2,3,4,5,6,7}"
base_port=${BASE_PORT:-17001}
backend_base=${BACKEND_BASE_PORT:-27001}
[[ "$base_port" =~ ^[0-9]+$ && "$backend_base" =~ ^[0-9]+$ ]] || exit 2
(( base_port > 0 && backend_base > 0 && base_port + ${#gpus[@]} <= 65536 && backend_base + ${#gpus[@]} <= 65536 )) || exit 2
declare -A seen=()
for gpu in "${gpus[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ && -z "${seen[$gpu]:-}" ]] || { echo "GPU_IDS must contain distinct numeric GPU indices" >&2; exit 2; }
    seen[$gpu]=1
done
if [[ "$dry_run" != --dry-run ]]; then
    : "${VLM_API_KEY:?Set VLM_API_KEY in the environment}"
    python3 - "$base_port" "$backend_base" "${#gpus[@]}" <<'PORTS'
import socket, sys
start, backend, count = map(int, sys.argv[1:])
sockets = []
try:
    for port in [*range(start, start+count), *range(backend, backend+count)]:
        sock = socket.socket()
        sockets.append(sock)
        sock.bind(('0.0.0.0', port))
finally:
    for sock in sockets:
        sock.close()
PORTS
fi
pids=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for i in "${!gpus[@]}"; do
    port=$((base_port+i))
    backend=$((backend_base+i))
    echo "EchoPolicy GPU=${gpus[$i]} public_port=$port backend_port=$backend"
    [[ "$dry_run" != --dry-run ]] || continue
    BACKEND_PORT=$backend bash "$SCRIPT_DIR/setup_eval_policy_server.sh" \
        "${BENCH_NAME:-RoboDojo}" "${TASK_NAME:-cover_blocks}" "$checkpoint" \
        "${ENV_CFG_TYPE:-arx_x5}" "${ACTION_TYPE:-joint}" "${POLICY_SEED:-0}" \
        "${gpus[$i]}" "$policy_env" "$port" "${ORCHESTRATOR_HOST:-0.0.0.0}" &
    pids+=("$!")
done
[[ "$dry_run" == --dry-run ]] || wait -n "${pids[@]}"
