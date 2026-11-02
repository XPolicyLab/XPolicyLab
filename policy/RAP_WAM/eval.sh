#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 10 ]]; then
    echo 'Usage: eval.sh bench task checkpoint robot action seed policy_gpu env_gpu policy_env eval_env' >&2
    exit 2
fi
bench_name=$1; task_name=$2; ckpt_name=$3; env_cfg_type=$4; action_type=$5
seed=$6; policy_gpu_id=$7; env_gpu_id=$8; policy_conda_env=$9; eval_env_conda_env=${10}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/runtime_env.sh"
UTILS_DIR="$XPL_ROOT/utils"
policy_server_port=$(bash "$UTILS_DIR/get_free_port.sh")
policy_server_ip=localhost
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"
SERVER_PID=""; CLIENT_PID=""; WAIT_PID=""
SERVER_CONTROL=""; CLIENT_CONTROL=""; WAIT_CONTROL=""

# A live session supervisor owns each command. Callers request cleanup through
# private workspace files; they never signal a recorded, potentially recycled PID.
start_owned_process() {
    local role="$1"
    shift
    OWNED_CONTROL=$(mktemp -d "$RAP_WAM_WORK_DIR/run/${role}.XXXXXXXX")
    "$RAP_WAM_BOOTSTRAP_PYTHON" -B "$SCRIPT_DIR/owned_process.py" --control-dir "$OWNED_CONTROL" --parent-pid "$$" -- "$@" &
    OWNED_PID=$!
}
stop_owned_process() {
    local pid="$1" control="$2"
    [[ -n "$pid" ]] || return 0
    : > "$control/stop"
    wait "$pid" 2>/dev/null || true
    "$RAP_WAM_BOOTSTRAP_PYTHON" -B - "$control/status.json" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    status = json.load(handle)
if status.get("cleanup") not in ("complete", "group_sigkill_requested"):
    raise SystemExit("Missing supervised cleanup result")
PY
}
wait_owned_result() {
    local pid="$1" control="$2"
    wait "$pid" 2>/dev/null || true
    "$RAP_WAM_BOOTSTRAP_PYTHON" -B - "$control/status.json" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    status = json.load(handle)
code = status["exit_code"]
if status.get("cleanup") not in ("complete", "group_sigkill_requested"):
    raise SystemExit("Missing supervised cleanup result")
if not isinstance(code, int) or not 0 <= code <= 255:
    raise SystemExit("Invalid supervised command exit status")
raise SystemExit(code)
PY
}
cleanup() {
    local result=$?
    local cleanup_failed=0
    trap - EXIT INT TERM
    stop_owned_process "$WAIT_PID" "$WAIT_CONTROL" || cleanup_failed=1
    stop_owned_process "$CLIENT_PID" "$CLIENT_CONTROL" || cleanup_failed=1
    stop_owned_process "$SERVER_PID" "$SERVER_CONTROL" || cleanup_failed=1
    if (( cleanup_failed && result == 0 )); then result=1; fi
    exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[MAIN] start server, policy_server_port=$policy_server_port"
start_owned_process server bash "$SCRIPT_DIR/setup_eval_policy_server.sh" \
    "$bench_name" "$task_name" "$ckpt_name" "$env_cfg_type" "$action_type" "$seed" \
    "$policy_gpu_id" "$policy_conda_env" "$policy_server_port"
SERVER_PID=$OWNED_PID
SERVER_CONTROL=$OWNED_CONTROL
start_owned_process wait bash "$UTILS_DIR/wait_for_policy_server.sh" "$policy_server_ip" "$policy_server_port" "$SERVER_PID" \
    'Policy server' "${RAP_WAM_SERVER_WAIT_SECONDS:-1200}"
WAIT_PID=$OWNED_PID
WAIT_CONTROL=$OWNED_CONTROL
wait_owned_result "$WAIT_PID" "$WAIT_CONTROL"
WAIT_PID=""
if [[ -f "$SERVER_CONTROL/status.json" ]]; then
    echo '[MAIN] server exited during startup' >&2
    exit 1
fi

echo "[MAIN] start client, server=$policy_server_ip:$policy_server_port"
start_owned_process client bash "$SCRIPT_DIR/setup_eval_env_client.sh" \
    "$bench_name" "$task_name" "$ckpt_name" "$env_cfg_type" "$action_type" "$seed" \
    "$env_gpu_id" "$eval_env_conda_env" "$additional_info" "$policy_server_port" "$policy_server_ip"
CLIENT_PID=$OWNED_PID
CLIENT_CONTROL=$OWNED_CONTROL
wait_owned_result "$CLIENT_PID" "$CLIENT_CONTROL"
CLIENT_PID=""
stop_owned_process "$SERVER_PID" "$SERVER_CONTROL"
SERVER_PID=""
echo '[MAIN] eval finished (owned server and client processes stopped)'
