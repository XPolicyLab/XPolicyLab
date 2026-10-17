#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_zero1.sh <nproc_per_node> [hydra_overrides...]}"
shift

# Project production default (2026-07-16): PyAV is bit-exact on LIBERO,
# faster end-to-end in the 2397216-vs-2397715 A/B, and bypasses the known
# TorchCodec 0.5 native-worker crash path. Keep an explicit override for
# decoder diagnostics and controlled comparisons.
export FASTWAM_VIDEO_BACKEND="${FASTWAM_VIDEO_BACKEND:-${VIDEO_BACKEND:-pyav}}"

EXTRA_ARGS=("$@")
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

if ! is_integer "${NUM_MACHINES}" || ! is_integer "${MACHINE_RANK}"; then
  echo "Error: NUM_MACHINES (${NUM_MACHINES}) and MACHINE_RANK (${MACHINE_RANK}) must be integers." >&2
  exit 1
fi

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

TASK_BASENAME="train"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      ;;
  esac
done

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    RUN_ID_SYNC_TIMEOUT="${RUN_ID_SYNC_TIMEOUT:-180}"
    RUN_ID_SYNC_PORT="${RUN_ID_SYNC_PORT:-$((MAIN_PROCESS_PORT + 11))}"

    export RUN_ID_SYNC_HOST="${MAIN_PROCESS_IP}"
    export RUN_ID_SYNC_PORT
    export RUN_ID_SYNC_TIMEOUT
    export RUN_ID_SYNC_MACHINE_RANK="${MACHINE_RANK}"
    export RUN_ID_SYNC_NUM_MACHINES="${NUM_MACHINES}"
    export RUN_ID_SYNC_TASK_BASENAME="${TASK_BASENAME}"

    RUN_ID="$(
      python - <<'PY'
import datetime
import os
from datetime import timedelta

import torch.distributed as dist

host = os.environ["RUN_ID_SYNC_HOST"]
port = int(os.environ["RUN_ID_SYNC_PORT"])
timeout_s = int(os.environ["RUN_ID_SYNC_TIMEOUT"])
machine_rank = int(os.environ["RUN_ID_SYNC_MACHINE_RANK"])
num_machines = int(os.environ["RUN_ID_SYNC_NUM_MACHINES"])
task_basename = os.environ.get("RUN_ID_SYNC_TASK_BASENAME", "train")

store = dist.TCPStore(
    host_name=host,
    port=port,
    world_size=num_machines,
    is_master=(machine_rank == 0),
    timeout=timedelta(seconds=timeout_s),
)
key = f"run_id::{task_basename}"
if machine_rank == 0:
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    store.set(key, run_id)
run_id = store.get(key).decode("utf-8")
print(run_id)
PY
    )"

    echo "[run_id_sync] mode=tcpstore host=${RUN_ID_SYNC_HOST} port=${RUN_ID_SYNC_PORT} timeout_s=${RUN_ID_SYNC_TIMEOUT} run_id=${RUN_ID}"
  fi
fi

TOTAL_PROCESSES=$(( NPROC_PER_NODE * NUM_MACHINES ))
RUNS_ROOT="${RUNS_ROOT:-./runs}"
OUTPUT_DIR="${RUNS_ROOT%/}/${TASK_BASENAME}/${RUN_ID}"

echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} total_processes=${TOTAL_PROCESSES} main=${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT} run_id=${RUN_ID} video_backend=${FASTWAM_VIDEO_BACKEND} output_dir=${OUTPUT_DIR}"

LAUNCH_ARGS=(
  --config_file "${ACCEL_CONFIG:-scripts/accelerate_configs/accelerate_zero1_ds.yaml}"
  --num_processes "${TOTAL_PROCESSES}"
  --num_machines "${NUM_MACHINES}"
)
if (( NUM_MACHINES > 1 )); then
  # Multi-node: one accelerate process per node (launched by mpirun). Use the
  # `standard` (torchrun-style) DeepSpeed launcher so each node joins a single
  # rendezvous at MAIN_PROCESS_IP:PORT, instead of accelerate's default pdsh
  # launcher which SSHes to hosts and clashes with the mpirun-spawned processes.
  LAUNCH_ARGS+=(
    --machine_rank "${MACHINE_RANK}"
    --main_process_ip "${MAIN_PROCESS_IP}"
    --main_process_port "${MAIN_PROCESS_PORT}"
    --deepspeed_multinode_launcher standard
  )
fi

# This is the final command in the wrapper.  Replacing the shell also pins the
# already-parsed launch path, so editing this file during a long training run
# cannot make bash resume at a shifted byte offset when Accelerate exits.
exec accelerate launch \
  "${LAUNCH_ARGS[@]}" \
  scripts/train.py \
  "output_dir=${OUTPUT_DIR}" \
  "wandb.name=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"
