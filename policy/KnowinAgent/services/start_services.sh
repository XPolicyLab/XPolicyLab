#!/usr/bin/env bash
# Start KnowinAgent's perception services on one GPU, unless something already answers on their ports:
#   SAM3 text segmentation  : KNOWIN_AGENT_SAM3_DIR (service.py, weights/sam3.pt, .venv), port 8035 (KNOWIN_AGENT_SAM3_URL)
#   MoGe-2 point-map depth  : KNOWIN_AGENT_DEPTH_DIR (depth-service project with .venv),   port 8034 (KNOWIN_AGENT_DEPTH_URL)
# Both read only the RGB frame the agent sends them. See README for installing them.
set -euo pipefail

gpu=${1:-0}
SAM3_PORT=${KNOWIN_AGENT_SAM3_PORT:-8035}
DEPTH_PORT=${KNOWIN_AGENT_DEPTH_PORT:-8034}
LOG_DIR=${KNOWIN_AGENT_SERVICE_LOG_DIR:-${TMPDIR:-/tmp}/knowin_agent-services}
mkdir -p "${LOG_DIR}"

up() { curl -s -o /dev/null -m 5 -w "%{http_code}" "http://127.0.0.1:$1/docs" | grep -q 200; }

wait_up() {
    local port=$1 name=$2 deadline=$((SECONDS + ${KNOWIN_AGENT_SERVICE_TIMEOUT:-900}))
    until up "${port}"; do
        if (( SECONDS > deadline )); then
            echo "[KnowinAgent][SERVICES][ERROR] ${name} did not come up on port ${port}; see ${LOG_DIR}" >&2
            exit 1
        fi
        sleep 5
    done
    echo "[KnowinAgent][SERVICES] ${name} ready on port ${port}"
}

if up "${SAM3_PORT}"; then
    echo "[KnowinAgent][SERVICES] SAM3 already running on port ${SAM3_PORT}"
else
    : "${KNOWIN_AGENT_SAM3_DIR:?set KNOWIN_AGENT_SAM3_DIR to the SAM3 service directory (see README)}"
    (cd "${KNOWIN_AGENT_SAM3_DIR}" && CUDA_VISIBLE_DEVICES="${gpu}" SAM3_PORT="${SAM3_PORT}" \
        setsid nohup .venv/bin/python -I service.py > "${LOG_DIR}/sam3.log" 2>&1 < /dev/null &)
    wait_up "${SAM3_PORT}" SAM3
fi

if up "${DEPTH_PORT}"; then
    echo "[KnowinAgent][SERVICES] depth already running on port ${DEPTH_PORT}"
else
    : "${KNOWIN_AGENT_DEPTH_DIR:?set KNOWIN_AGENT_DEPTH_DIR to the depth-service project directory (see README)}"
    (cd "${KNOWIN_AGENT_DEPTH_DIR}" && GPU="${gpu}" PORT="${DEPTH_PORT}" HF_HUB_OFFLINE=1 \
        setsid nohup bash scripts/start.sh > "${LOG_DIR}/depth.log" 2>&1 < /dev/null &)
    wait_up "${DEPTH_PORT}" depth
fi
