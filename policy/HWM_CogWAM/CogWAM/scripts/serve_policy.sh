#!/usr/bin/env bash
# Launch N CogWAM policy servers, one per GPU, on ports BASE_PORT..BASE_PORT+N-1.
#
# The servers are stateless: any evaluation client may talk to any of them, and
# the RoboDojo rollout spreads its requests across all of them. This script
# blocks until every server has completed a WebSocket handshake, prints the
# address to export on the client side, and then supervises the processes.
#
#   COGWAM_ARTIFACT_DIR=/path/to/artifact \
#   COGWAM_BASE_VLM=/path/to/RynnBrain1.1-2B \
#   COGWAM_DINO_MODEL=/path/to/dinov3-vitb16 \
#   NUM_SERVERS=8 bash scripts/serve_policy.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ARTIFACT_DIR="${COGWAM_ARTIFACT_DIR:-}"
CHECKPOINT_PATH="${COGWAM_CKPT_PATH:-}"
BASE_PORT="${BASE_PORT:-7777}"
NUM_SERVERS="${NUM_SERVERS:-8}"
COGWAM_PYTHON="${COGWAM_PYTHON:-python3}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"
READY_CHECK_INTERVAL="${READY_CHECK_INTERVAL:-3}"
ADVERTISE_HOST="${ADVERTISE_HOST:-}"
LOG_ROOT="${COGWAM_SERVER_LOG_ROOT:-}"

if [[ -n "${ARTIFACT_DIR}" && -n "${CHECKPOINT_PATH}" ]]; then
  echo "[serve][ERROR] set exactly one of COGWAM_ARTIFACT_DIR or COGWAM_CKPT_PATH" >&2
  exit 2
fi
if [[ -n "${ARTIFACT_DIR}" ]]; then
  [[ -d "${ARTIFACT_DIR}" ]] || { echo "[serve][ERROR] artifact directory does not exist: ${ARTIFACT_DIR}" >&2; exit 1; }
  SOURCE_ARGS=(--artifact-dir "${ARTIFACT_DIR}")
  VERIFY_ARGS=(--artifact-dir "${ARTIFACT_DIR}")
  SOURCE_LABEL="${ARTIFACT_DIR}"
elif [[ -n "${CHECKPOINT_PATH}" ]]; then
  [[ -f "${CHECKPOINT_PATH}" ]] || { echo "[serve][ERROR] checkpoint does not exist: ${CHECKPOINT_PATH}" >&2; exit 1; }
  SOURCE_ARGS=(--ckpt_path "${CHECKPOINT_PATH}")
  VERIFY_ARGS=(--checkpoint "${CHECKPOINT_PATH}")
  SOURCE_LABEL="${CHECKPOINT_PATH}"
else
  echo "[serve][ERROR] export COGWAM_ARTIFACT_DIR=/path/to/artifact (or COGWAM_CKPT_PATH for a training checkpoint)" >&2
  exit 2
fi

[[ -n "${COGWAM_BASE_VLM:-}" ]] || {
  echo "[serve][ERROR] export COGWAM_BASE_VLM=/path/to/RynnBrain1.1-2B (backbone weights are not redistributed)" >&2
  exit 2
}
[[ -n "${COGWAM_DINO_MODEL:-}" ]] || {
  echo "[serve][ERROR] export COGWAM_DINO_MODEL=/path/to/dinov3-vitb16 (backbone weights are not redistributed)" >&2
  exit 2
}
[[ "${NUM_SERVERS}" =~ ^[1-9][0-9]*$ ]] || { echo "[serve][ERROR] NUM_SERVERS must be a positive integer" >&2; exit 2; }
[[ "${BASE_PORT}" =~ ^[0-9]+$ ]] || { echo "[serve][ERROR] BASE_PORT must be an integer" >&2; exit 2; }
(( BASE_PORT >= 1 && BASE_PORT + NUM_SERVERS - 1 <= 65535 )) || { echo "[serve][ERROR] invalid port range" >&2; exit 2; }
command -v "${COGWAM_PYTHON}" >/dev/null 2>&1 || [[ -x "${COGWAM_PYTHON}" ]] || {
  echo "[serve][ERROR] Python is not executable: ${COGWAM_PYTHON}" >&2
  exit 1
}

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Avoid ABI/architecture-dependent RynnBrain extension kernels on inference
# hosts. This is inference-only; training still follows the checkpoint YAML and
# keeps the required attention implementation.
export COGWAM_QWEN35_DISABLE_CAUSAL_CONV1D="${COGWAM_QWEN35_DISABLE_CAUSAL_CONV1D:-1}"
export COGWAM_QWEN35_DISABLE_FLA="${COGWAM_QWEN35_DISABLE_FLA:-1}"
export COGWAM_QWEN35_ATTN_IMPLEMENTATION="${COGWAM_QWEN35_ATTN_IMPLEMENTATION:-sdpa}"
if [[ "${COGWAM_CUDA_DEBUG:-0}" == "1" ]]; then
  export CUDA_LAUNCH_BLOCKING=1
fi

"${COGWAM_PYTHON}" -m cogwam.eval.verify_contract "${VERIFY_ARGS[@]}"

GPU_COUNT="$("${COGWAM_PYTHON}" - <<'PY'
import torch

print(torch.cuda.device_count())
PY
)"
[[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] || { echo "[serve][ERROR] failed to determine CUDA device count" >&2; exit 1; }
(( GPU_COUNT >= NUM_SERVERS )) || {
  echo "[serve][ERROR] requested ${NUM_SERVERS} servers, but Python sees ${GPU_COUNT} GPUs" >&2
  exit 1
}

declare -a GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  for ((slot = 0; slot < NUM_SERVERS; ++slot)); do
    GPU_IDS+=("${slot}")
  done
fi
(( ${#GPU_IDS[@]} >= NUM_SERVERS )) || {
  echo "[serve][ERROR] CUDA_VISIBLE_DEVICES exposes only ${#GPU_IDS[@]} IDs" >&2
  exit 1
}
GPU_IDS=("${GPU_IDS[@]:0:NUM_SERVERS}")

if [[ -z "${ADVERTISE_HOST}" ]]; then
  ADVERTISE_HOST="$(hostname -I 2>/dev/null | awk '{for (i=1; i<=NF; ++i) if ($i !~ /^127\./) {print $i; exit}}')"
fi
[[ -n "${ADVERTISE_HOST}" ]] || {
  echo "[serve][ERROR] cannot determine the server IP; set ADVERTISE_HOST" >&2
  exit 1
}

LAST_PORT=$((BASE_PORT + NUM_SERVERS - 1))
if [[ -z "${LOG_ROOT}" ]]; then
  LOG_ROOT="${COGWAM_RUN_ROOT:-${PWD}}/cogwam_server_logs/ports_${BASE_PORT}_${LAST_PORT}"
fi
mkdir -p "${LOG_ROOT}"

declare -a PIDS=()

kill_tree() {
  local pid="$1" signal="${2:-TERM}" child
  while read -r child; do
    [[ -n "${child}" ]] && kill_tree "${child}" "${signal}"
  done < <(ps -o pid= --ppid "${pid}" 2>/dev/null || true)
  kill -"${signal}" "${pid}" 2>/dev/null || true
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  for pid in "${PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && kill_tree "${pid}" TERM
  done
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill_tree "${pid}" KILL
    fi
  done
  for pid in "${PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true
  done
  exit "${rc}"
}
trap cleanup EXIT INT TERM

websocket_is_ready() {
  local host="$1" port="$2"
  env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY \
    NO_PROXY="${host},127.0.0.1,localhost" \
    no_proxy="${host},127.0.0.1,localhost" \
    "${COGWAM_PYTHON}" - "${host}" "${port}" <<'PY'
import asyncio
import sys

import websockets


async def main():
    uri = f"ws://{sys.argv[1]}:{int(sys.argv[2])}"
    try:
        async with websockets.connect(
            uri,
            open_timeout=2,
            close_timeout=1,
            ping_interval=None,
            max_size=None,
        ):
            return 0
    except Exception:
        return 1


raise SystemExit(asyncio.run(main()))
PY
}

show_log_tail() {
  local slot="$1"
  local gpu="${GPU_IDS[$slot]}"
  local port=$((BASE_PORT + slot))
  local log_file="${LOG_ROOT}/server_${slot}_gpu${gpu}_port${port}.log"
  echo "[serve][ERROR] tail of ${log_file}:" >&2
  tail -n 80 "${log_file}" >&2 2>/dev/null || true
}

echo "========== CogWAM policy servers =========="
echo "[serve] source=${SOURCE_LABEL}"
echo "[serve] GPUs=${GPU_IDS[*]}"
echo "[serve] ports=${BASE_PORT}-${LAST_PORT}"
echo "[serve] logs=${LOG_ROOT}"
echo "[serve] advertised-host=${ADVERTISE_HOST}"
echo "[serve] topology=one policy process per GPU"
echo "[serve] qwen35-safe-causal-conv=${COGWAM_QWEN35_DISABLE_CAUSAL_CONV1D}"
echo "[serve] qwen35-safe-fla=${COGWAM_QWEN35_DISABLE_FLA}"
echo "[serve] qwen35-attention=${COGWAM_QWEN35_ATTN_IMPLEMENTATION}"

for ((slot = 0; slot < NUM_SERVERS; ++slot)); do
  gpu="${GPU_IDS[$slot]}"
  port=$((BASE_PORT + slot))
  log_file="${LOG_ROOT}/server_${slot}_gpu${gpu}_port${port}.log"
  echo "[serve] launch slot=${slot} gpu=${gpu} port=${port}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${COGWAM_PYTHON}" -u -m cogwam.serve.policy_server \
      "${SOURCE_ARGS[@]}" \
      --port "${port}" \
      --idle_timeout -1 \
      --use_bf16
  ) >"${log_file}" 2>&1 &
  PIDS+=("$!")
done

deadline=$((SECONDS + SERVER_READY_TIMEOUT))
while :; do
  ready=0
  for ((slot = 0; slot < NUM_SERVERS; ++slot)); do
    port=$((BASE_PORT + slot))
    if websocket_is_ready 127.0.0.1 "${port}"; then
      ready=$((ready + 1))
    fi
  done
  (( ready < NUM_SERVERS )) || break

  for slot in "${!PIDS[@]}"; do
    if ! kill -0 "${PIDS[$slot]}" 2>/dev/null; then
      status=0
      wait "${PIDS[$slot]}" || status=$?
      echo "[serve][ERROR] slot ${slot} exited during startup (rc=${status})" >&2
      show_log_tail "${slot}"
      exit 1
    fi
  done
  (( SECONDS < deadline )) || {
    echo "[serve][ERROR] only ${ready}/${NUM_SERVERS} servers ready after ${SERVER_READY_TIMEOUT}s" >&2
    for ((slot = 0; slot < NUM_SERVERS; ++slot)); do show_log_tail "${slot}"; done
    exit 1
  }
  echo "[serve] waiting: ${ready}/${NUM_SERVERS} ready"
  sleep "${READY_CHECK_INTERVAL}"
done

echo "[serve][READY] all ${NUM_SERVERS} servers passed the WebSocket handshake"
echo "[serve][READY] export COGWAM_POLICY_HOST=${ADVERTISE_HOST}"
echo "[serve][READY] export BASE_PORT=${BASE_PORT}"
echo "[serve][READY] export NUM_SERVERS=${NUM_SERVERS}"
echo "[serve][READY] keep this terminal running"

while :; do
  for slot in "${!PIDS[@]}"; do
    if ! kill -0 "${PIDS[$slot]}" 2>/dev/null; then
      status=0
      wait "${PIDS[$slot]}" || status=$?
      echo "[serve][ERROR] slot ${slot} exited (rc=${status})" >&2
      show_log_tail "${slot}"
      exit 1
    fi
  done
  sleep 5
done
