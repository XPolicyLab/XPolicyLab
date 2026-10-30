#!/usr/bin/env bash
# AGP: start the stub policy server, then the RoboDojo eval client.
#
#   bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
#        <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env>
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_env=${9:-default}
eval_env=${10:-default}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AGP_HOME="${AGP_HOME:-${HOME}/.agp-robodojo}"
if [[ "${action_type}" != "joint" ]]; then
    echo "[MAIN][ERROR] AGP sends joint targets; action_type must be joint" >&2
    exit 1
fi

if [[ "${policy_env}" != "default" && "${policy_env}" != "uv" && "${policy_env}" != */* && ! -e "${policy_env}" ]]; then
    conda_python="$(
        set +eu
        command -v conda >/dev/null 2>&1 \
            && source "$(conda info --base)/etc/profile.d/conda.sh" >/dev/null 2>&1 \
            && conda activate "${policy_env}" >/dev/null 2>&1 && command -v python
    )" || conda_python=""
    if [[ -n "${conda_python}" ]]; then
        echo "[MAIN] policy env: conda environment ${policy_env}"
        policy_env="${conda_python}"
    else
        echo "[MAIN] policy env '${policy_env}' is not an activatable conda environment; using the default (\$AGP_HOME/.venv-policy)"
        policy_env="default"
    fi
fi
if [[ "${policy_env}" == "default" || "${policy_env}" == "uv" ]]; then
    helper_python="${AGP_HOME}/.venv-policy/bin/python"
elif [[ -x "${policy_env}/bin/python" ]]; then
    helper_python="${policy_env}/bin/python"
else
    helper_python="${policy_env}"
fi

export ROBODOJO_RUN_ID="${ROBODOJO_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_agp-${task_name}-s${seed}-$$}"
if [[ -n "${AGP_SCRIPTED:-}" && -z "${AGP_RESULTS_DIR:-}" ]]; then
    export AGP_RESULTS_DIR="${AGP_HOME}/runs/official/${ROBODOJO_RUN_ID}"
fi
logs_dir="${AGP_RESULTS_DIR:-${AGP_HOME}/runs/official/${ROBODOJO_RUN_ID}}"
eval_root="${AGP_RESULTS_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}/eval_result"
mkdir -p "${logs_dir}"
mode="codex"
if [[ -n "${AGP_SCRIPTED:-}" ]]; then
    mode="scripted-${AGP_SCRIPTED}"
fi
if [[ "${EVAL_ENV_TYPE:-}" == "debug" ]]; then
    mode="debug"
fi
additional_info="agp_${mode}"
policy_server_ip=127.0.0.1
policy_server_port="$("${helper_python}" -B -c 'import socket
s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM -- -"${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[MAIN] run_id=${ROBODOJO_RUN_ID} logs=${logs_dir} eval_result=${eval_root} port=${policy_server_port} mode=${mode}"
setsid bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" \
    "${policy_gpu_id}" "${policy_env}" "${policy_server_port}" "${policy_server_ip}" \
    > "${logs_dir}/policy_server.log" 2>&1 &
SERVER_PID=$!

# Wait for the LISTEN socket (read from /proc, no connection is opened) or the server's exit.
"${helper_python}" -B - "${policy_server_port}" "${SERVER_PID}" <<'PY'
import os, sys, time
port, pid = int(sys.argv[1]), int(sys.argv[2])
deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        sys.exit(f"[MAIN][ERROR] policy server (pid {pid}) exited before listening on {port}")
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            rows = open(table).read().splitlines()[1:]
        except OSError:
            continue
        if any(r.split()[3] == "0A" and int(r.split()[1].rsplit(":", 1)[1], 16) == port for r in rows):
            print(f"[MAIN] policy server listening on {port}", flush=True)
            sys.exit(0)
    time.sleep(0.5)
sys.exit(f"[MAIN][ERROR] policy server did not listen on {port} within 300 s")
PY

bash "${SCRIPT_DIR}/setup_eval_env_client.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" \
    "${env_gpu_id}" "${eval_env}" "${additional_info}" "${policy_server_port}" "${policy_server_ip}" \
    2>&1 | tee "${logs_dir}/simulator.log"
echo "[MAIN] eval finished (run_id=${ROBODOJO_RUN_ID})"
