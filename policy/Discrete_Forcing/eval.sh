#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 10 ]]; then
    echo "Usage: $0 bench task checkpoint env_cfg_type action_type seed policy_gpu env_gpu policy_env eval_env" >&2
    exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
policy_server_port=$(bash "$XPL_ROOT/utils/get_free_port.sh")
cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
bash "$SCRIPT_DIR/setup_eval_policy_server.sh" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$9" "$policy_server_port" &
SERVER_PID=$!
bash "$XPL_ROOT/utils/wait_for_policy_server.sh" localhost "$policy_server_port" "$SERVER_PID" "Policy server" 1200
bash "$SCRIPT_DIR/setup_eval_env_client.sh" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$8" "${10}" \
    "ckpt_name=$3,action_type=$5" "$policy_server_port" localhost
echo "[MAIN] eval finished"
