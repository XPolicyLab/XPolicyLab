#!/usr/bin/env bash
# One-shot launcher: start both LingBot VA backend server and the XPolicyLab
# forward server, each in its own tmux session.
#
#   backend VA server   -> auto-picked free port (internal, 127.0.0.1)
#   forward  server     -> user-specified external port (default 0.0.0.0:<EXPOSE_PORT>)
#
# Only two positional args matter for server startup:
#   bash start_servers.sh <GPU_ID> <EXPOSE_PORT>
#
# task_name / ckpt_name / expert_data_num / seed / action_type are NOT used by
# the servers in websocket-bridge mode (model.py ignores them; inference is
# driven by the upstream VA server). They are only meaningful on the eval
# client side. Defaults are still passed to setup_eval_policy_server.sh to
# satisfy its positional-arg contract, but changing them here has no effect
# on inference. Override via env vars only if you really need to.
#
# Env overrides (all optional):
#   CHECKPOINT_PATH   finetuned ckpt dir for VA server (launch default otherwise)
#   BASE_MODEL_PATH   base model weights dir
#   CONFIG_NAME       wan_va config name (default robotwin30_train)
#   CONDA_ENV         conda env name (default lingbot_va)
#   EXPOSE_HOST       forward server bind host (default 0.0.0.0)
#   TMUX_PREFIX       tmux session name prefix (default lingbot_va)
#
# Examples:
#   bash start_servers.sh 0 10002
#   bash start_servers.sh 1 10003
#   CHECKPOINT_PATH=/path/to/ckpt/ bash start_servers.sh 0 10002
set -eo pipefail

# ----------------------------------------------------------------------------
# positional args (the only two that matter)
# ----------------------------------------------------------------------------
GPU="${1:-0}"
EXPOSE_PORT="${2:-10002}"

# ----------------------------------------------------------------------------
# defaults passed to setup_eval_policy_server.sh to satisfy its arg contract.
# These do NOT affect inference in ws-bridge mode; tweak only if you know why.
# ----------------------------------------------------------------------------
DATASET_NAME="${DATASET_NAME:-RoboDojo}"
TASK_NAME="${TASK_NAME:-stack_bowls}"
CKPT_NAME="${CKPT_NAME:-RoboDojo-cotrain-arx_x5-3500-joint-0}"
ENV_CFG_TYPE="${ENV_CFG_TYPE:-arx_x5}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-3500}"
ACTION_TYPE="${ACTION_TYPE:-joint}"
SEED="${SEED:-0}"

CONDA_ENV="${CONDA_ENV:-lingbot_va}"
EXPOSE_HOST="${EXPOSE_HOST:-0.0.0.0}"
TMUX_PREFIX="${TMUX_PREFIX:-lingbot_va}"
CONFIG_NAME="${CONFIG_NAME:-robotwin30_train}"

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${POLICY_DIR}/.logs"
mkdir -p "${LOG_DIR}"

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
pick_free_port() {
    python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

port_listening() {
    local port="$1"
    python - <<PY
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.settimeout(1.0)
    s.connect(("127.0.0.1", ${port}))
    sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

wait_port() {
    local port="$1"
    local name="$2"
    local timeout="${3:-600}"
    local elapsed=0
    echo "[start_servers] waiting for ${name} on 127.0.0.1:${port} (timeout ${timeout}s)"
    while ! port_listening "${port}"; do
        sleep 3
        elapsed=$((elapsed + 3))
        if (( elapsed >= timeout )); then
            echo
            echo "[start_servers] ERROR: ${name} did not come up within ${timeout}s"
            echo "[start_servers] check tmux: tmux ls ; logs: ${LOG_DIR}/"
            return 1
        fi
        printf "."
    done
    echo
    echo "[start_servers] ${name} is up (waited ${elapsed}s)"
}

tmux_kill_if() {
    tmux kill-session -t "$1" 2>/dev/null || true
}

# ----------------------------------------------------------------------------
# conda activation prefix for tmux windows
# ----------------------------------------------------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "${CONDA_BASE}" ]]; then
    echo "[start_servers] ERROR: conda not found in PATH"
    exit 1
fi
CONDA_ACT="source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}"

# ----------------------------------------------------------------------------
# pick free port for backend VA server
# ----------------------------------------------------------------------------
VA_PORT="$(pick_free_port)"
VA_MASTER_PORT="$(pick_free_port)"
echo "[start_servers] backend VA port  = ${VA_PORT} (auto)"
echo "[start_servers] backend MASTER   = ${VA_MASTER_PORT} (auto)"
echo "[start_servers] forward expose   = ${EXPOSE_HOST}:${EXPOSE_PORT}"
echo "[start_servers] gpu              = ${GPU}"

# ----------------------------------------------------------------------------
# 1) backend VA server (launch_wan_va_server.sh) in tmux
# ----------------------------------------------------------------------------
VA_SESSION="${TMUX_PREFIX}_va"
VA_LOG="${LOG_DIR}/va_server_${VA_PORT}.log"
tmux_kill_if "${VA_SESSION}"

echo "[start_servers] launching backend VA server in tmux session '${VA_SESSION}'"
echo "[start_servers]   log: ${VA_LOG}"

VA_ENV_EXPORT=""
[[ -n "${CHECKPOINT_PATH:-}" ]] && VA_ENV_EXPORT+="export CHECKPOINT_PATH='${CHECKPOINT_PATH}'; "
[[ -n "${BASE_MODEL_PATH:-}" ]]  && VA_ENV_EXPORT+="export BASE_MODEL_PATH='${BASE_MODEL_PATH}'; "

tmux new-session -d -s "${VA_SESSION}" \
    "${CONDA_ACT}; \
     ${VA_ENV_EXPORT} \
     export CONFIG_NAME='${CONFIG_NAME}'; \
     cd '${POLICY_DIR}'; \
     bash launch_wan_va_server.sh ${GPU} ${VA_PORT} 2>&1 | tee '${VA_LOG}'"

# wait for VA server to listen
wait_port "${VA_PORT}" "backend VA server" 600 || {
    echo "[start_servers] backend VA failed; tail of log:"
    tail -n 40 "${VA_LOG}" 2>/dev/null
    exit 1
}

# ----------------------------------------------------------------------------
# 2) forward server (setup_eval_policy_server.sh) in tmux
# ----------------------------------------------------------------------------
FWD_SESSION="${TMUX_PREFIX}_fwd"
FWD_LOG="${LOG_DIR}/forward_server_${EXPOSE_PORT}.log"
tmux_kill_if "${FWD_SESSION}"

echo "[start_servers] launching forward server in tmux session '${FWD_SESSION}'"
echo "[start_servers]   log: ${FWD_LOG}"

tmux new-session -d -s "${FWD_SESSION}" \
    "${CONDA_ACT}; \
     export VA_SERVER_HOST=127.0.0.1 VA_SERVER_PORT=${VA_PORT}; \
     cd '${POLICY_DIR}'; \
     bash setup_eval_policy_server.sh \
         '${DATASET_NAME}' '${TASK_NAME}' '${CKPT_NAME}' \
         '${ENV_CFG_TYPE}' '${EXPERT_DATA_NUM}' '${ACTION_TYPE}' \
         '${SEED}' '${GPU}' '${CONDA_ENV}' \
         '${EXPOSE_PORT}' '${EXPOSE_HOST}' '${CONFIG_NAME}' 2>&1 | tee '${FWD_LOG}'"

# wait briefly for forward server to bind
wait_port "${EXPOSE_PORT}" "forward server" 120 || {
    echo "[start_servers] forward server did not bind in 120s (may still be loading model); check log:"
    tail -n 40 "${FWD_LOG}" 2>/dev/null
    exit 1
}

# ----------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------
cat <<EOF

[start_servers] ============================================================
[start_servers]  ALL UP
[start_servers]  backend VA server : 127.0.0.1:${VA_PORT}  (internal)
[start_servers]  forward  server   : ${EXPOSE_HOST}:${EXPOSE_PORT}  (external, connect clients here)
[start_servers]  GPU               : ${GPU}
[start_servers]  config            : ${CONFIG_NAME}
[start_servers] ------------------------------------------------------------
[start_servers]  tmux sessions:
[start_servers]    backend : tmux attach -t ${VA_SESSION}
[start_servers]    forward : tmux attach -t ${FWD_SESSION}
[start_servers]  logs:
[start_servers]    backend : ${VA_LOG}
[start_servers]    forward : ${FWD_LOG}
[start_servers] ------------------------------------------------------------
[start_servers]  stop both:
[start_servers]    tmux kill-session -t ${VA_SESSION}; tmux kill-session -t ${FWD_SESSION}
[start_servers] ============================================================
EOF
