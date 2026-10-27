#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
if (($# != 10)); then
    echo 'Usage: eval.sh BENCH TASK CHECKPOINT ROBOT joint SEED POLICY_GPU ENV_GPU POLICY_ENV EVAL_CONDA_ENV' >&2
    exit 2
fi
require_xpl
if is_dry_run; then
    port="${PRISMWAM_SERVER_PORT:-12345}"
else
    port="${PRISMWAM_SERVER_PORT:-$(bash "$XPL_ROOT/utils/get_free_port.sh")}"
fi
server=(bash "$POLICY_DIR/setup_eval_policy_server.sh" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$9" "$port")
client=(bash "$POLICY_DIR/setup_eval_env_client.sh" "$1" "$2" "$3" "$4" "$5" "$6" "$8" "${10}"
    "ckpt_name=$3,action_type=$5" "$port" localhost)
if is_dry_run; then
    "${server[@]}"
    show_command bash "$XPL_ROOT/utils/wait_for_policy_server.sh" localhost "$port" '<server_pid>' 'PrismWAM server' 1200
    "${client[@]}"
    exit 0
fi
cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
"${server[@]}" &
SERVER_PID=$!
bash "$XPL_ROOT/utils/wait_for_policy_server.sh" localhost "$port" "$SERVER_PID" 'PrismWAM server' 1200
"${client[@]}"
echo '[MAIN] eval finished'
