#!/usr/bin/env bash
# =============================================================================
# FlexPi — Libero 3D training.
#   bash scripts/train_flexpi_libero.sh
#   # 2D (no depth) — the model flag and the data config are both required
#   bash scripts/train_flexpi_libero.sh model.enable_pointmap=false data=libero_nodepth
#
# Details: docs/LIBERO.md · knob reference: docs/TRAINING.md
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
TASK_CONFIG="${TASK_CONFIG:-libero_unified_flex_2cam224_32d_rotvec_1e-4}"
COMPOSITE_LAYOUT="${COMPOSITE_LAYOUT:-tshape_384x320}"
# The 4-suite mix — the single place to add or drop a suite.
LIBERO_DATA_ROOT="./data/libero_mujoco3.3.2_depth"
LIBERO_SUITES=(libero_spatial libero_object libero_goal libero_10)
# Carved from train. 0.0 = no val rollout; 0.05 keeps a small held-out split.
VAL_SET_PROPORTION="${VAL_SET_PROPORTION:-0.0}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
# Size-agnostic mixing: DATASET_WEIGHTS are per-frame draw probabilities parallel
# to the dataset dirs, and SAMPLES_PER_EPOCH defines one epoch (required with
# weights). Both empty = a single uniform pass over the concatenated set.
DATASET_WEIGHTS="${DATASET_WEIGHTS:-}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-}"

# ── Flex randomization ────────────────────────────────────────────────────────
# Independent Bernoulli per sample; p=1.0 disables that dropout. All 1.0 trains
# the every-stream-present, every-flag-joint regime only. Drop them to 0.5 for
# one checkpoint that serves every regime.
FLEX_P_PRESENT_VIDEO="1.0"
FLEX_P_PRESENT_DINO="1.0"
FLEX_P_PRESENT_POINTMAP="1.0"
FLEX_P_JV="1.0"
FLEX_P_JD="1.0"
FLEX_P_JP="1.0"

# ── Run labeling / resume / wandb ─────────────────────────────────────────────
RUN_NAME=""
RESUME="${RESUME:-}"

WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-flex-pi}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-}"

# =============================================================================
# Build override args
# =============================================================================
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# Build the Hydra dataset_dirs list from LIBERO_SUITES (single source of truth).
_DATASET_DIR_LIST=()
for _suite in "${LIBERO_SUITES[@]}"; do
  _DATASET_DIR_LIST+=("${LIBERO_DATA_ROOT}/${_suite}_no_noops_lerobot")
done
DATASET_DIRS="[$(IFS=,; echo "${_DATASET_DIR_LIST[*]}")]"

# Resolve COMPOSITE_LAYOUT → the geometry it implies. The task config already
# carries the 448×512 values, so only the 384×320 branch has anything to say.
GEOMETRY_ARGS=()
case "${COMPOSITE_LAYOUT}" in
  tshape_libero_2cam_448x512)
    DINO_TEMPORAL_STRIDE="2"
    ;;
  tshape_384x320 | tshape_robotwin_384x320_uniform)
    DINO_TEMPORAL_STRIDE="1"
    GEOMETRY_ARGS=(
      "data.train.video_size=[384,320]"
      # null → fall back to the layout's own map (exterior on top).
      "data.train.composite_layout_slot_key_map=null"
    )
    ;;
  *)
    echo "Error: COMPOSITE_LAYOUT=${COMPOSITE_LAYOUT} is not one of" >&2
    echo "       tshape_libero_2cam_448x512 | tshape_384x320" >&2
    exit 1
    ;;
esac

EXTRA_ARGS=(
  "task=${TASK_CONFIG}"
  "model.composite_layout=${COMPOSITE_LAYOUT}"
  "${GEOMETRY_ARGS[@]}"
  # The flex model preset comes from the task config's defaults list.
  "model.dino_temporal_stride=${DINO_TEMPORAL_STRIDE}"
  # ── Flex-joint per-sample randomization ──
  "model.flex_joint.enabled=true"
  "model.flex_joint.p_present_video=${FLEX_P_PRESENT_VIDEO}"
  "model.flex_joint.p_present_dino=${FLEX_P_PRESENT_DINO}"
  "model.flex_joint.p_present_pointmap=${FLEX_P_PRESENT_POINTMAP}"
  "model.flex_joint.p_jv=${FLEX_P_JV}"
  "model.flex_joint.p_jd=${FLEX_P_JD}"
  "model.flex_joint.p_jp=${FLEX_P_JP}"
  # ── Data (LIBERO: explicit dataset dirs + internal val split; NO data.val) ──
  "data.train.dataset_dirs=${DATASET_DIRS}"
  "data.train.val_set_proportion=${VAL_SET_PROPORTION}"
)

[[ -n "${NUM_EPOCHS}" ]]   && EXTRA_ARGS+=("num_epochs=${NUM_EPOCHS}")
[[ -n "${RESUME}" ]]       && EXTRA_ARGS+=("resume=${RESUME}")
[[ -n "${DATASET_WEIGHTS}" ]]   && EXTRA_ARGS+=("++data.train.dataset_weights=${DATASET_WEIGHTS}")
[[ -n "${SAMPLES_PER_EPOCH}" ]] && EXTRA_ARGS+=("++data.train.samples_per_epoch=${SAMPLES_PER_EPOCH}")

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

REGIME_TAG="flex_pv${FLEX_P_PRESENT_VIDEO}_pd${FLEX_P_PRESENT_DINO}_pp${FLEX_P_PRESENT_POINTMAP}_jv${FLEX_P_JV}_jd${FLEX_P_JD}_jp${FLEX_P_JP}"
# A 2D run arrives as a passthrough override, so the FLEX_P_* knobs above cannot
# see it — and under enable_pointmap=false the pp/jp halves of the tag are inert.
[[ " ${*,,} " == *"model.enable_pointmap=false "* ]] && REGIME_TAG="${REGIME_TAG}_2d"
[[ "${COMPOSITE_LAYOUT}" != "tshape_384x320" ]] && REGIME_TAG="${REGIME_TAG}_${COMPOSITE_LAYOUT}"
[[ "${DINO_TEMPORAL_STRIDE}" != "1" ]] && REGIME_TAG="${REGIME_TAG}_ds${DINO_TEMPORAL_STRIDE}"
[[ -n "${NUM_EPOCHS}" ]] && REGIME_TAG="${REGIME_TAG}_epoch${NUM_EPOCHS}"
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

OUTPUT_DIR="./runs/${TASK_BASENAME}/${RUN_ID}_${REGIME_TAG}${RUN_NAME:+_${RUN_NAME}}"
# Auto-resume from THIS run's own output dir, so a relaunch continues the run
# matching the current config. Skipped when RESUME or a warm start was asked
# for. `|| true` because a no-match under pipefail must not abort.
if [[ -z "${RESUME}" && -z "${PRETRAINED_CKPT:-}" && -d "${OUTPUT_DIR}/checkpoints/state" ]]; then
  RESUME="$(ls -d "${OUTPUT_DIR}"/checkpoints/state/step_*/ 2>/dev/null | sort -V | tail -1 || true)"
  RESUME="${RESUME%/}"
  [[ -n "${RESUME}" ]] && EXTRA_ARGS+=("resume=${RESUME}") && echo "[resume] auto-resume from own output dir: ${RESUME}"
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
  "output_dir=${OUTPUT_DIR}" \
  "wandb.enabled=${WANDB_ENABLED}" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=$(date +%m%d)_${TASK_BASENAME}_${REGIME_TAG}${RUN_NAME:+_${RUN_NAME}}" \
  "wandb.mode=${WANDB_MODE}" \
  "wandb.group=${WANDB_GROUP:-${TASK_BASENAME}}" \
  "${EXTRA_ARGS[@]}"
