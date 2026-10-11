#!/usr/bin/env bash
# ME_U0 multinode training launcher. It installs nothing and writes all mutable
# state below the approved shared WORK_DIRS root.
set -euo pipefail

DEFAULT_APPROVED_ROOT="./work_dirs"
APPROVED_ROOT="${ME_U0_APPROVED_WORK_ROOT:-${LEAP_WORK_ROOT:-${DEFAULT_APPROVED_ROOT}}}"
CONFIG="leap/configs/experiments/libero_posttraining.yaml"
NNODES=2
NPROC_PER_NODE=8
RESUME=""
ATTEMPT_ID="${ME_U0_ATTEMPT_ID:-}"
WORK_DIR=""
WORK_ROOT=""
RUN_ID="${ME_U0_RUN_ID:-}"
CLI_RUN_ID=""
MASTER_HOST="${MASTER_ADDR:-}"
MASTER_PORT_VALUE="${MASTER_PORT:-29500}"
NODE_RANK_VALUE="${NODE_RANK:-${RANK:-}}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

need_value() {
  [[ $# -ge 2 && -n "${2:-}" ]] || die "$1 requires a non-empty value"
}

canonical_path() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

resolve_master() {
  python3 - "$1" <<'PY'
import os
import socket
import sys
import time

host = sys.argv[1]
retries = int(os.environ.get("ME_U0_MASTER_RESOLVE_RETRIES", "10"))
addresses = []
last_error = None
for attempt in range(max(1, retries)):
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        if addresses:
            break
    except socket.gaierror as exc:
        last_error = exc
    if attempt + 1 < max(1, retries):
        time.sleep(1)
if not addresses:
    detail = f": {last_error}" if last_error is not None else ""
    raise SystemExit(f"cannot resolve MASTER_ADDR {host!r}{detail}")
unique = []
for _family, _type, _proto, _canonname, sockaddr in addresses:
    address = sockaddr[0]
    if address not in unique:
        unique.append(address)
if not unique:
    raise SystemExit(f"cannot resolve MASTER_ADDR {host!r}")
print(unique[0])
PY
}

infer_node_rank() {
  local current_host="${HOSTNAME:-}"
  [[ -n "${current_host}" ]] || current_host="$(hostname)"
  if [[ "${current_host}" =~ -([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

resolve_local_address() {
  python3 - <<'PY'
import ipaddress
import os
import socket

hosts = []
for host in (os.environ.get("HOSTNAME"), socket.gethostname()):
    if host and host not in hosts:
        hosts.append(host)
for host in hosts:
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        continue
    for family, _type, _proto, _canonname, sockaddr in addresses:
        address = sockaddr[0].split("%", 1)[0]
        parsed = ipaddress.ip_address(address)
        if not parsed.is_loopback and not parsed.is_unspecified:
            print(address)
            raise SystemExit(0)
raise SystemExit("cannot discover this node's non-loopback address")
PY
}

require_non_loopback() {
  python3 - "$1" <<'PY'
import ipaddress
import sys

address = ipaddress.ip_address(sys.argv[1].split("%", 1)[0])
if address.is_loopback or address.is_unspecified:
    raise SystemExit(1)
PY
}

assert_fresh_work_dir() {
  [[ -z "${RESUME}" ]] || return 0
  [[ -d "${WORK_DIR}" ]] || return 0
  local artifact
  for artifact in metadata.json config.yaml train.log train_log.jsonl; do
    [[ ! -e "${WORK_DIR}/${artifact}" ]] || die \
      "work directory already contains a prior run (${artifact}); choose a new --run-id/--work-dir or use --resume"
  done
  if [[ -d "${WORK_DIR}/checkpoints" ]] && \
     find "${WORK_DIR}/checkpoints" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    die "work directory already contains checkpoints; choose a new --run-id/--work-dir or use --resume"
  fi
}

if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  NNODES="$1"
  shift
fi

OVERRIDES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) need_value "$@"; CONFIG="$2"; shift 2 ;;
    --nnodes) need_value "$@"; NNODES="$2"; shift 2 ;;
    --nproc-per-node) need_value "$@"; NPROC_PER_NODE="$2"; shift 2 ;;
    --resume) need_value "$@"; RESUME="$2"; shift 2 ;;
    --attempt-id) need_value "$@"; ATTEMPT_ID="$2"; shift 2 ;;
    --work-dir) need_value "$@"; WORK_DIR="$2"; shift 2 ;;
    --work-root) need_value "$@"; WORK_ROOT="$2"; shift 2 ;;
    --run-id|--exp-name)
      need_value "$@"
      if [[ -n "${CLI_RUN_ID}" && "${CLI_RUN_ID}" != "$2" ]]; then
        die "--run-id and --exp-name must match when both are provided"
      fi
      CLI_RUN_ID="$2"
      shift 2
      ;;
    --master-addr) need_value "$@"; MASTER_HOST="$2"; shift 2 ;;
    --master-port) need_value "$@"; MASTER_PORT_VALUE="$2"; shift 2 ;;
    --node-rank) need_value "$@"; NODE_RANK_VALUE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash scripts/ME_U0/run_multinode.sh [NNODES] (--exp-name ID --work-root PATH | --run-id ID [--work-dir PATH]) [--master-addr HOST --node-rank N] [--nproc-per-node N] [--attempt-id ID] [--resume PATH|latest|auto] [key=value ...]"
      exit 0
      ;;
    work_dir=*|work_root=*|run_id=*)
      die "work_dir/work_root/run_id must use the dedicated launcher flags"
      ;;
    *=*) OVERRIDES+=("$1"); shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

if [[ -n "${CLI_RUN_ID}" ]]; then
  RUN_ID="${CLI_RUN_ID}"
fi

[[ "${NNODES}" =~ ^[1-9][0-9]*$ ]] || die "NNODES must be a positive integer"
[[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "NPROC_PER_NODE must be a positive integer"
[[ "${MASTER_PORT_VALUE}" =~ ^[0-9]+$ ]] || die "MASTER_PORT must be an integer"
NNODES=$((10#${NNODES}))
NPROC_PER_NODE=$((10#${NPROC_PER_NODE}))
MASTER_PORT_VALUE=$((10#${MASTER_PORT_VALUE}))
(( MASTER_PORT_VALUE >= 1 && MASTER_PORT_VALUE <= 65535 )) || die "MASTER_PORT must be in [1,65535]"

if (( NNODES > 1 )); then
  if [[ -z "${NODE_RANK_VALUE}" ]]; then
    NODE_RANK_VALUE="$(infer_node_rank)" || die \
      "unable to infer node rank; set NODE_RANK/RANK or pass --node-rank"
    echo "Auto-detected node rank ${NODE_RANK_VALUE} from hostname ${HOSTNAME:-$(hostname)}"
  fi
else
  MASTER_HOST="${MASTER_HOST:-127.0.0.1}"
  NODE_RANK_VALUE="${NODE_RANK_VALUE:-0}"
fi
[[ "${NODE_RANK_VALUE}" =~ ^[0-9]+$ ]] || die "NODE_RANK must be an integer"
NODE_RANK_VALUE=$((10#${NODE_RANK_VALUE}))
(( NODE_RANK_VALUE < NNODES )) || die "NODE_RANK must be in [0, NNODES), got ${NODE_RANK_VALUE}"
if [[ -z "${ATTEMPT_ID}" ]]; then
  CURRENT_HOST="${HOSTNAME:-$(hostname)}"
  NODE_SUFFIX="-${NODE_RANK_VALUE}"
  ATTEMPT_ID="${CURRENT_HOST%"${NODE_SUFFIX}"}"
fi
[[ "${ATTEMPT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || die \
  "attempt ID must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
[[ -z "${WORK_ROOT}" || -z "${WORK_DIR}" ]] || die \
  "--work-root and --work-dir are mutually exclusive"
if [[ -z "${RUN_ID}" && -n "${WORK_DIR}" ]]; then
  RUN_ID="$(basename "${WORK_DIR%/}")"
fi
[[ -n "${RUN_ID}" ]] || die \
  "an explicit --run-id/--exp-name/ME_U0_RUN_ID or --work-dir is required"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || die \
  "run ID must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
# Explicit output paths define the work root unless a root override was provided.
REQUESTED_ROOT="${WORK_ROOT:-${DEFAULT_APPROVED_ROOT}}"
[[ -z "${WORK_DIR}" ]] || REQUESTED_ROOT="$(dirname "${WORK_DIR}")"
APPROVED_ROOT="${ME_U0_APPROVED_WORK_ROOT:-${LEAP_WORK_ROOT:-${REQUESTED_ROOT}}}"
DEFAULT_RUN_ROOT="${APPROVED_ROOT}"
if ! CANONICAL_ROOT="$(canonical_path "${APPROVED_ROOT}")"; then
  die "unable to canonicalize approved work root"
fi
if [[ -n "${WORK_ROOT}" ]]; then
  if ! WORK_ROOT="$(canonical_path "${WORK_ROOT}")"; then
    die "unable to canonicalize --work-root"
  fi
  [[ "${WORK_ROOT}" == "${CANONICAL_ROOT}" || "${WORK_ROOT}" == "${CANONICAL_ROOT}/"* ]] || die \
    "--work-root must canonically resolve at or below ${APPROVED_ROOT}"
  WORK_DIR="${WORK_ROOT}/${RUN_ID}"
else
  WORK_DIR="${WORK_DIR:-${DEFAULT_RUN_ROOT}/${RUN_ID}}"
fi
if ! WORK_DIR="$(canonical_path "${WORK_DIR}")"; then
  die "unable to canonicalize work directory"
fi
[[ "${WORK_DIR}" != "${CANONICAL_ROOT}" && "${WORK_DIR}" == "${CANONICAL_ROOT}/"* ]] || die \
  "--work-dir must canonically resolve below ${APPROVED_ROOT}"
[[ "$(basename "${WORK_DIR}")" == "${RUN_ID}" ]] || die \
  "--work-dir basename must equal run ID ${RUN_ID}"
WORK_ROOT="${WORK_ROOT:-$(dirname "${WORK_DIR}")}"
if [[ -n "${RESUME}" && "${RESUME}" != "latest" && \
      "${RESUME}" != "auto" ]]; then
  if ! RESUME="$(canonical_path "${RESUME}")"; then
    die "unable to canonicalize resume checkpoint"
  fi
  [[ "${RESUME}" == "${WORK_DIR}/checkpoints/"* ]] || die \
    "--resume must be auto/latest or a checkpoint inside this run's work directory"
fi
assert_fresh_work_dir

# LPAI devspace shells do not consistently preserve MASTER_ADDR/NODE_RANK.
# Rank 0 publishes its pod IP through the shared run directory; nodes without
# an explicit MASTER_ADDR wait for that file instead of falling back to loopback.
MASTER_STATE_DIR="${WORK_DIR}/launcher_state/${ATTEMPT_ID}"
MASTER_STATE_FILE="${MASTER_STATE_DIR}/master_addr"
MASTER_IP=""
if (( NNODES > 1 && NODE_RANK_VALUE == 0 )); then
  mkdir -p "${MASTER_STATE_DIR}"
  if [[ -n "${MASTER_HOST}" ]]; then
    MASTER_IP="$(resolve_master "${MASTER_HOST}")" || die \
      "unable to resolve MASTER_ADDR ${MASTER_HOST}"
  else
    MASTER_IP="$(resolve_local_address)" || die \
      "unable to discover rank-0 address; set MASTER_ADDR or pass --master-addr"
  fi
  require_non_loopback "${MASTER_IP}" || die \
    "multi-node MASTER_ADDR must resolve to a non-loopback, non-unspecified address; got ${MASTER_IP}"
  MASTER_STATE_TMP="${MASTER_STATE_FILE}.tmp.${HOSTNAME:-node0}.$$"
  printf '%s\n' "${MASTER_IP}" > "${MASTER_STATE_TMP}"
  mv -f "${MASTER_STATE_TMP}" "${MASTER_STATE_FILE}"
  echo "Published rank-0 address ${MASTER_IP} to ${MASTER_STATE_FILE}"
elif (( NNODES > 1 )) && [[ -z "${MASTER_HOST}" ]]; then
  MASTER_DISCOVERY_TIMEOUT="${ME_U0_MASTER_DISCOVERY_TIMEOUT:-600}"
  [[ "${MASTER_DISCOVERY_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || die \
    "ME_U0_MASTER_DISCOVERY_TIMEOUT must be a positive integer"
  mkdir -p "${MASTER_STATE_DIR}"
  echo "Waiting for rank 0 to publish ${MASTER_STATE_FILE}"
  for ((waited = 0; waited < MASTER_DISCOVERY_TIMEOUT; waited++)); do
    if [[ -s "${MASTER_STATE_FILE}" ]]; then
      read -r MASTER_HOST < "${MASTER_STATE_FILE}"
      if MASTER_IP="$(resolve_master "${MASTER_HOST}")"; then
        break
      fi
      MASTER_IP=""
    fi
    sleep 1
  done
  [[ -n "${MASTER_IP}" ]] || die \
    "timed out waiting for rank-0 address in ${MASTER_STATE_FILE}"
else
  MASTER_IP="$(resolve_master "${MASTER_HOST}")" || die \
    "unable to resolve MASTER_ADDR ${MASTER_HOST}"
fi
if (( NNODES > 1 )) && ! require_non_loopback "${MASTER_IP}"; then
  die "multi-node MASTER_ADDR must resolve to a non-loopback, non-unspecified address; got ${MASTER_IP}"
fi

# Each training process adds its LOCAL_RANK below this node-local root before
# importing torch. Callers may place it on local storage because some mounted
# filesystems do not safely support TorchInductor/Triton compiler-cache I/O.
COMPILER_CACHE_NODE_ROOT="${ME_U0_COMPILER_CACHE_NODE_ROOT:-${WORK_DIR}/compiler_cache/${ATTEMPT_ID}/node_${NODE_RANK_VALUE}}"
TMP_RUN_KEY="$(python3 - "${WORK_DIR}" "${ATTEMPT_ID}" <<'PY'
import hashlib
import sys

print(hashlib.sha256(":".join(sys.argv[1:]).encode("utf-8")).hexdigest()[:12])
PY
)"
# Python multiprocessing creates AF_UNIX sockets below TMPDIR. Some shared
# mounts do not support those sockets, and long work-directory paths can also
# exceed Linux's 108-byte sockaddr limit. Only ephemeral runtime sockets use
# this short node-local path; compiler caches and all durable outputs remain in
# WORK_DIRS.
TORCHRUN_TMP_ROOT="/tmp/ME_U0/${TMP_RUN_KEY}/n${NODE_RANK_VALUE}"
mkdir -p "${COMPILER_CACHE_NODE_ROOT}" "${TORCHRUN_TMP_ROOT}"
export ME_U0_COMPILER_CACHE_NODE_ROOT="${COMPILER_CACHE_NODE_ROOT}"
export TMPDIR="${TORCHRUN_TMP_ROOT}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
[[ -f "${CONFIG}" ]] || die "config not found: ${CONFIG}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
export ME_U0_RUN_ID="${RUN_ID}"
export ME_U0_ATTEMPT_ID="${ATTEMPT_ID}"
export MASTER_ADDR="${MASTER_IP}"
export MASTER_PORT="${MASTER_PORT_VALUE}"
export NODE_RANK="${NODE_RANK_VALUE}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

OVERRIDES+=("work_root=${WORK_ROOT}" "run_id=${RUN_ID}" "work_dir=${WORK_DIR}")
LOG_DIR="${WORK_DIR}/torchrun_logs/${ATTEMPT_ID}/node_${NODE_RANK_VALUE}"
mkdir -p "${LOG_DIR}"
export TORCHELASTIC_ERROR_FILE="${LOG_DIR}/torchelastic_error.json"

echo "ME_U0 training: run_id=${RUN_ID} attempt_id=${ATTEMPT_ID} nodes=${NNODES} gpus/node=${NPROC_PER_NODE} world=${WORLD_SIZE} node_rank=${NODE_RANK_VALUE}"
echo "config=${CONFIG} master=${MASTER_IP}:${MASTER_PORT_VALUE} work_dir=${WORK_DIR}"
echo "compiler_cache_root=${COMPILER_CACHE_NODE_ROOT}"

RESUME_REQUEST="${RESUME}"
if [[ -n "${RESUME}" ]]; then
  RESUME="$(python3 - "${RESUME}" "${WORK_DIR}" "${WORLD_SIZE}" <<'PY'
import sys

from leap.callbacks.atomic_checkpoint_callback import (
    resolve_resume_checkpoint,
    validate_deepspeed_checkpoint_world_size,
)

request, work_dir, raw_world_size = sys.argv[1:]
resolved = resolve_resume_checkpoint(request, work_dir)
if resolved is not None:
    saved_world_size = validate_deepspeed_checkpoint_world_size(
        resolved, int(raw_world_size)
    )
    if saved_world_size is None:
        raise SystemExit(
            f"resume checkpoint has no DeepSpeed optimizer rank shards: {resolved}"
        )
    print(resolved)
PY
)"
  if [[ -n "${RESUME}" ]]; then
    export ME_U0_RESUME_VALIDATED=1
    echo "resume_request=${RESUME_REQUEST} resolved_resume=${RESUME}"
  else
    echo "resume_request=${RESUME_REQUEST} resolved_resume=step_0"
  fi
fi

TRAIN_ARGS=(--config "${CONFIG}")
if [[ -n "${RESUME}" ]]; then
  TRAIN_ARGS+=(--resume "${RESUME}")
fi
TRAIN_ARGS+=("${OVERRIDES[@]}")

exec torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK_VALUE}" \
  --master_addr="${MASTER_IP}" \
  --master_port="${MASTER_PORT_VALUE}" \
  --tee=3 \
  --redirects=3 \
  --log-dir="${LOG_DIR}" \
  scripts/ME_U0/train.py \
  "${TRAIN_ARGS[@]}"
