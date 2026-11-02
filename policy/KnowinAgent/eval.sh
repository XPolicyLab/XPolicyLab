#!/usr/bin/env bash
# RoboDojo entry for KnowinAgent: start the (no-op) policy server and the perception services, then the RoboDojo
# client, which runs the KnowinAgent agent in deploy.py for every episode.
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_env=${9}
eval_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${XPOLICYLAB_ROOT:-${SCRIPT_DIR}/../..}" && pwd)"
if [[ ! -f "${XPL_ROOT}/setup_policy_server.py" || ! -d "${XPL_ROOT}/utils" ]]; then
    echo "[KnowinAgent][ERROR] set XPOLICYLAB_ROOT to the XPolicyLab checkout" >&2
    exit 2
fi
case "${EVAL_ENV_TYPE:-sim}" in
    sim|"")
        if [[ "${KNOWIN_AGENT_DRY_RUN:-0}" == "1" ]]; then
            echo "[KnowinAgent][ERROR] KNOWIN_AGENT_DRY_RUN=1 requires EVAL_ENV_TYPE=debug" >&2
            exit 2
        fi
        ;;
    debug) export KNOWIN_AGENT_DRY_RUN=1 ;;
    *) echo "[KnowinAgent][ERROR] supported EVAL_ENV_TYPE values: sim, debug" >&2; exit 2 ;;
esac
UTILS_DIR="${XPL_ROOT}/utils"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip=localhost
additional_info="policy=KnowinAgent,model=${KNOWIN_AGENT_MODEL:-gpt-6-astra},effort=${KNOWIN_AGENT_EFFORT:-xhigh},action_type=${action_type}"

cleanup() {
    local exit_status=$?
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        if ! wait "${SERVER_PID}"; then
            echo "[KnowinAgent][ERROR] policy server or cleanup failed" >&2
            if [[ "${exit_status}" == "0" ]]; then exit_status=1; fi
        fi
    fi
    trap - EXIT
    exit "${exit_status}"
}
trap cleanup EXIT

echo "[KnowinAgent] start policy server, port=${policy_server_port}"
# A small process supervisor works on Linux and macOS and owns only this launch's
# process group. It forwards cleanup to that group (no name-based process killing).
"${KNOWIN_AGENT_LAUNCH_PYTHON:-python3}" - "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" \
    "${policy_gpu_id}" "${policy_env}" "${policy_server_port}" "${policy_server_ip}" <<'PY' &
import os
import signal
import subprocess
import sys
import time

child = subprocess.Popen(["bash", *sys.argv[1:]], start_new_session=True)
stop_requested = False

def stop(signum, frame):
    # Signal handlers must not reenter Popen.wait(): its internal lock can deadlock.
    global stop_requested
    stop_requested = True

def signal_group(signum):
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        # macOS may report EPERM after the group has exited. Reap the child before
        # deciding whether cleanup actually failed; never suppress a live failure.
        if child.poll() is None:
            print(f"[KnowinAgent][ERROR] cannot stop owned policy group {child.pid}: {exc}", file=sys.stderr)
            sys.exit(1)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
deadline = None
while child.poll() is None:
    if stop_requested:
        if deadline is None:
            signal_group(signal.SIGTERM)
            deadline = time.monotonic() + 10
        elif time.monotonic() >= deadline:
            signal_group(signal.SIGKILL)
            deadline = float("inf")
    time.sleep(0.1)
sys.exit(0 if stop_requested else child.returncode)
PY
SERVER_PID=$!

if ! bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" \
        "Policy server" 1800 || ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[KnowinAgent][ERROR] policy server did not start; not starting the client" >&2
    exit 1
fi

echo "[KnowinAgent] start RoboDojo client, server=${policy_server_ip}:${policy_server_port}"
bash "${SCRIPT_DIR}/setup_eval_env_client.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" \
    "${env_gpu_id}" "${eval_env}" "${additional_info}" "${policy_server_port}" "${policy_server_ip}"

echo "[MAIN] eval finished"
