#!/usr/bin/env bash
# =============================================================================
# FlexPi — RoboTwin 2.0 training. One checkpoint serves every inference regime;
# per-sample dropout over stream presence and joint flags is what makes that
# work (FLEX_P_* below).
#
#   bash scripts/train_flexpi_robotwin.sh
#   # 2D (no depth) — the model flag and the data config are both required
#   bash scripts/train_flexpi_robotwin.sh model.enable_pointmap=false data=robotwin_nodepth
#
# Details: docs/ROBOTWIN.md · knob reference: docs/TRAINING.md
# =============================================================================
set -euo pipefail

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export NCCL_DEBUG=WARN

# Multi-node InfiniBand pinning; a no-op on a single node.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7}"

# Edit both together for a different allocation.
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# ── Data ──────────────────────────────────────────────────────────────────────
TASK_CONFIG="${TASK_CONFIG:-robotwin_unified_flex_3cam_384_1e-4}"
TASK_NAMES="all"                                     # "all", or a space-separated subset
NUM_EPISODES_PER_TASK="${NUM_EPISODES_PER_TASK:-}"   # e.g. 50/100 for the demo-efficiency sweep; empty = all
NUM_EPISODES=""
NUM_EPOCHS="${NUM_EPOCHS:-6}"
VAL_SET_PROPORTION="${VAL_SET_PROPORTION:-0.0}"   # held-out fraction; 0.0 = eval on train
# Size-agnostic mixing: DATASET_WEIGHTS are per-frame draw probabilities parallel
# to the dataset dirs, and SAMPLES_PER_EPOCH defines one epoch (required with
# weights). Both empty = a single uniform pass over the concatenated set.
DATASET_WEIGHTS="${DATASET_WEIGHTS:-}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-}"

# ── Flex randomization ────────────────────────────────────────────────────────
# Independent Bernoulli per sample; p=1.0 disables that dropout.
FLEX_P_PRESENT_VIDEO="0.5"
FLEX_P_PRESENT_DINO="0.5"
FLEX_P_PRESENT_POINTMAP="0.5"
FLEX_P_JV="0.5"
FLEX_P_JD="0.5"
FLEX_P_JP="0.5"

# ── Run labeling / resume / wandb ─────────────────────────────────────────────
RUN_NAME=""
RESUME="${RESUME:-}"

# Off by default: an unset WANDB_API_KEY makes `wandb.init` block or abort on a
# machine with no cached login, and it does so only after the model is already
# built. To enable, export your own key and flip the switch:
#   WANDB_API_KEY=<key> WANDB_ENABLED=true bash scripts/train_flexpi_robotwin.sh
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-flex-pi}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-}"

# =============================================================================
# Resolve TASK_NAMES sentinels → explicit task list.
# =============================================================================
TASK_NAMES_TAG_LABEL=""
if [[ "${TASK_NAMES}" == "all" ]] || { [[ -z "${TASK_NAMES}" ]] && [[ -n "${NUM_EPISODES_PER_TASK}" ]]; }; then
  _FLEXPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  _TASK_MAP_PATH="${_FLEXPI_ROOT}/src/flexpi/datasets/task_episode_map.json"
  TASK_NAMES="$(python - "${_TASK_MAP_PATH}" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    tasks = json.load(f)["tasks"]
print(" ".join(sorted(tasks.keys())))
PY
)"
  TASK_NAMES_TAG_LABEL="all"
fi

# =============================================================================
# Build override args
# =============================================================================
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

EXTRA_ARGS=(
  "task=${TASK_CONFIG}"
  "model.flex_joint.p_present_video=${FLEX_P_PRESENT_VIDEO}"
  "model.flex_joint.p_present_dino=${FLEX_P_PRESENT_DINO}"
  "model.flex_joint.p_present_pointmap=${FLEX_P_PRESENT_POINTMAP}"
  "model.flex_joint.p_jv=${FLEX_P_JV}"
  "model.flex_joint.p_jd=${FLEX_P_JD}"
  "model.flex_joint.p_jp=${FLEX_P_JP}"
)

[[ -n "${NUM_EPISODES}" ]] && EXTRA_ARGS+=("+data.train.num_episodes=${NUM_EPISODES}")
[[ -n "${NUM_EPOCHS}" ]]   && EXTRA_ARGS+=("num_epochs=${NUM_EPOCHS}")
[[ -n "${VAL_SET_PROPORTION}" ]] && EXTRA_ARGS+=("data.train.val_set_proportion=${VAL_SET_PROPORTION}")
[[ -n "${RESUME}" ]]       && EXTRA_ARGS+=("resume=${RESUME}")
[[ -n "${DATASET_WEIGHTS}" ]]   && EXTRA_ARGS+=("++data.train.dataset_weights=${DATASET_WEIGHTS}")
[[ -n "${SAMPLES_PER_EPOCH}" ]] && EXTRA_ARGS+=("++data.train.samples_per_epoch=${SAMPLES_PER_EPOCH}")
if [[ -n "${TASK_NAMES}" ]]; then
  _TASK_LIST="[${TASK_NAMES// /,}]"
  EXTRA_ARGS+=(
    "+data.train.task_names=${_TASK_LIST}"
    "+data.val.task_names=${_TASK_LIST}"
  )
fi
if [[ -n "${NUM_EPISODES_PER_TASK}" ]]; then
  EXTRA_ARGS+=(
    "+data.train.num_episodes_per_task=${NUM_EPISODES_PER_TASK}"
    "+data.val.num_episodes_per_task=${NUM_EPISODES_PER_TASK}"
  )
fi

# ── Warm start (RESUME wins) ──────────────────────────────────────────────────
# Weights only, into a fresh run. Mutually exclusive with RESUME — a restored
# state already holds these weights, so on resume the warm start is skipped.
if [[ -n "${PRETRAINED_CKPT:-}" ]]; then
  if [[ -n "${RESUME}" ]]; then
    echo "[resume] RESUME set — skipping PRETRAINED_CKPT warm-init (resuming full state ${RESUME})."
  else
    if [[ ! -e "${PRETRAINED_CKPT}" ]]; then
      echo "Error: PRETRAINED_CKPT does not exist: ${PRETRAINED_CKPT}" >&2
      exit 1
    fi
    EXTRA_ARGS+=(
      "pretrained_ckpt=${PRETRAINED_CKPT}"
      "pretrained_ckpt_strict_shape=${PRETRAINED_CKPT_STRICT_SHAPE:-false}"
    )
  fi
fi

EXTRA_ARGS+=("$@")

# ── Multi-node bookkeeping ────────────────────────────────────────────────────
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29501}"

is_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }
if ! is_integer "${NUM_MACHINES}" || ! is_integer "${MACHINE_RANK}"; then
  echo "Error: NUM_MACHINES (${NUM_MACHINES}) and MACHINE_RANK (${MACHINE_RANK}) must be integers." >&2
  exit 1
fi

# ── Output dir / run id ───────────────────────────────────────────────────────
TASK_BASENAME="${TASK_CONFIG}"
TASK_NAMES_TAG=""
if [[ -n "${TASK_NAMES}" ]]; then
  if [[ -n "${TASK_NAMES_TAG_LABEL}" ]]; then
    TASK_NAMES_TAG="${TASK_NAMES_TAG_LABEL}"
  else
    TASK_NAMES_TAG="${TASK_NAMES// /_}"
  fi
  [[ -n "${NUM_EPISODES_PER_TASK}" ]] && TASK_NAMES_TAG="${TASK_NAMES_TAG}_perTask${NUM_EPISODES_PER_TASK}"
  [[ -n "${NUM_EPISODES}" ]]          && TASK_NAMES_TAG="${TASK_NAMES_TAG}_demo${NUM_EPISODES}"
  [[ -n "${NUM_EPOCHS}" ]]            && TASK_NAMES_TAG="${TASK_NAMES_TAG}_epoch${NUM_EPOCHS}"
fi

REGIME_TAG="flex_pv${FLEX_P_PRESENT_VIDEO}_pd${FLEX_P_PRESENT_DINO}_pp${FLEX_P_PRESENT_POINTMAP}_jv${FLEX_P_JV}_jd${FLEX_P_JD}_jp${FLEX_P_JP}"
# A 2D run arrives as a passthrough override, so the FLEX_P_* knobs above cannot
# see it — and under enable_pointmap=false the pp/jp halves of the tag are inert.
[[ " ${*,,} " == *"model.enable_pointmap=false "* ]] && REGIME_TAG="${REGIME_TAG}_2d"
# Scratch is the default, so tag the exception — same `_ft` the libero and yam
# launchers use.
[[ -n "${PRETRAINED_CKPT:-}" ]] && REGIME_TAG="${REGIME_TAG}_ft"

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    RUN_ID_SYNC_TIMEOUT="${RUN_ID_SYNC_TIMEOUT:-180}"
    RUN_ID_SYNC_PORT="${RUN_ID_SYNC_PORT:-$((MAIN_PROCESS_PORT + 11))}"
    export RUN_ID_SYNC_HOST="${MAIN_PROCESS_IP}"
    export RUN_ID_SYNC_PORT RUN_ID_SYNC_TIMEOUT
    export RUN_ID_SYNC_MACHINE_RANK="${MACHINE_RANK}"
    export RUN_ID_SYNC_NUM_MACHINES="${NUM_MACHINES}"
    export RUN_ID_SYNC_TASK_BASENAME="${TASK_BASENAME}"
    RUN_ID="$(python - <<'PY'
import datetime, os
from datetime import timedelta
import torch.distributed as dist
host = os.environ["RUN_ID_SYNC_HOST"]
port = int(os.environ["RUN_ID_SYNC_PORT"])
timeout_s = int(os.environ["RUN_ID_SYNC_TIMEOUT"])
machine_rank = int(os.environ["RUN_ID_SYNC_MACHINE_RANK"])
num_machines = int(os.environ["RUN_ID_SYNC_NUM_MACHINES"])
task_basename = os.environ.get("RUN_ID_SYNC_TASK_BASENAME", "train")
store = dist.TCPStore(host_name=host, port=port, world_size=num_machines,
                     is_master=(machine_rank == 0), timeout=timedelta(seconds=timeout_s))
key = f"run_id::{task_basename}"
if machine_rank == 0:
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    store.set(key, run_id)
print(store.get(key).decode("utf-8"))
PY
)"
  fi
fi

echo "[launch] nproc=${NPROC_PER_NODE} machines=${NUM_MACHINES} rank=${MACHINE_RANK} run_id=${RUN_ID}"
echo "[launch] regime=${REGIME_TAG}"

accelerate launch \
  --config_file "${ACCELERATE_CONFIG:-scripts/accelerate_configs/accelerate_zero1_ds.yaml}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --num_processes "$((NPROC_PER_NODE * NUM_MACHINES))" \
  --rdzv_backend static \
  --deepspeed_multinode_launcher standard \
  scripts/train.py \
  "output_dir=./runs/${TASK_BASENAME}/${TASK_NAMES_TAG:+${TASK_NAMES_TAG}/}${RUN_ID}_${REGIME_TAG}${RUN_NAME:+_${RUN_NAME}}" \
  "wandb.enabled=${WANDB_ENABLED}" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=$(date +%m%d)_${TASK_BASENAME}${TASK_NAMES_TAG:+/${TASK_NAMES_TAG}}_${REGIME_TAG}${RUN_NAME:+_${RUN_NAME}}" \
  "wandb.mode=${WANDB_MODE}" \
  "wandb.group=${WANDB_GROUP:-${TASK_BASENAME}}" \
  "${EXTRA_ARGS[@]}"
