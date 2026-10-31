#!/usr/bin/env bash
# =============================================================================
# FlexPi — LIBERO evaluation, one (suite, task) on one GPU.
# The full 4-suite protocol instead: eval_flexpi_libero_4suite.sh
#
#   CKPT=... DATASET_STATS=... TASK_SUITE_NAME=libero_spatial TASK_ID=0 \
#     bash scripts/eval_flexpi_libero_single.sh
#
# Details: docs/LIBERO.md
# =============================================================================
set -euo pipefail

# Repo root first — every default below is relative to it.
cd "$(dirname "$0")/.."

# ── Environment ───────────────────────────────────────────────────────────────
# LIBERO is never pip-installed; it only resolves via PYTHONPATH.
export PYTHONPATH="$(pwd)/third_party/LIBERO:$(pwd)/src:${PYTHONPATH:-}"
# A ~/.local torch would shadow the env's build.
export PYTHONNOUSERSITE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Let prepare_libero() derive the binding; an inherited value binds stale.
unset LIBERO_CONFIG_PATH

[[ -f third_party/LIBERO/libero/libero/__init__.py ]] || {
  echo "ERROR: third_party/LIBERO is empty. Run:"
  echo "       git submodule update --init third_party/LIBERO"; exit 1; }

# ── Checkpoint (required) ─────────────────────────────────────────────────────
CKPT="${CKPT:-}"
DATASET_STATS="${DATASET_STATS:-}"
[[ -n "$CKPT"          ]] || { echo "ERROR: CKPT required (path to step_NNNNNN.pt)"; exit 1; }
[[ -n "$DATASET_STATS" ]] || { echo "ERROR: DATASET_STATS required (path to dataset_stats.json)"; exit 1; }
[[ -f "$CKPT"          ]] || { echo "ERROR: ckpt not found: $CKPT"; exit 1; }
[[ -f "$DATASET_STATS" ]] || { echo "ERROR: stats not found: $DATASET_STATS"; exit 1; }

# ── Task ──────────────────────────────────────────────────────────────────────
HYDRA_TASK_PRESET="${HYDRA_TASK_PRESET:-libero_unified_flex_2cam224_32d_rotvec_1e-4}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_spatial}"
TASK_ID="${TASK_ID:-0}"

# ── GPU ───────────────────────────────────────────────────────────────────────
GPU_ID="${GPU_ID:-0}"

# ── Output / data ─────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-./eval_results/libero_unified_flex_single_${TASK_SUITE_NAME}_t${TASK_ID}_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-./data/libero_mujoco3.3.2_depth}"
CAMERA_INTRINSICS_PATH="${CAMERA_INTRINSICS_PATH:-${DATA_ROOT}/${TASK_SUITE_NAME}_no_noops_lerobot/meta/camera_intrinsics.json}"
[[ -f "$CAMERA_INTRINSICS_PATH" ]] || { echo "ERROR: camera_intrinsics not found: $CAMERA_INTRINSICS_PATH"; exit 1; }

# ── Evaluation knobs ──────────────────────────────────────────────────────────
NUM_TRIALS="${NUM_TRIALS:-50}"
TSHAPE_EVAL_RESIZE="${TSHAPE_EVAL_RESIZE:-stretch}"


# ── Regime ────────────────────────────────────────────────────────────────────
# joint_* = what is generated, present_* = what is encoded as input.
# All three joint flags false = the action-only fast path.
INFER_JOINT_VIDEO="${INFER_JOINT_VIDEO:-true}"
INFER_JOINT_DINO="${INFER_JOINT_DINO:-true}"
INFER_JOINT_POINTMAP="${INFER_JOINT_POINTMAP:-true}"

INFER_PRESENT_VIDEO="${INFER_PRESENT_VIDEO:-true}"
INFER_PRESENT_DINO="${INFER_PRESENT_DINO:-true}"
INFER_PRESENT_POINTMAP="${INFER_PRESENT_POINTMAP:-true}"

# =============================================================================
# Launch
# =============================================================================
mkdir -p "$OUTPUT_DIR"

CMD=(
    python experiments/libero/eval_libero_single.py
    task="${HYDRA_TASK_PRESET}"
    ckpt="${CKPT}"
    EVALUATION.dataset_stats_path="${DATASET_STATS}"
    "+EVALUATION.camera_intrinsics_path=${CAMERA_INTRINSICS_PATH}"
    EVALUATION.output_dir="${OUTPUT_DIR}"
    EVALUATION.num_trials="${NUM_TRIALS}"
    EVALUATION.task_suite_name="${TASK_SUITE_NAME}"
    EVALUATION.task_id="${TASK_ID}"
    EVALUATION.tshape_eval_resize="${TSHAPE_EVAL_RESIZE}"
    "+EVALUATION.infer_present_video=${INFER_PRESENT_VIDEO}"
    "+EVALUATION.infer_present_dino=${INFER_PRESENT_DINO}"
    "+EVALUATION.infer_present_pointmap=${INFER_PRESENT_POINTMAP}"
    "+EVALUATION.infer_joint_video=${INFER_JOINT_VIDEO}"
    "+EVALUATION.infer_joint_dino=${INFER_JOINT_DINO}"
    "+EVALUATION.infer_joint_pointmap=${INFER_JOINT_POINTMAP}"
)
echo "[eval-flex-libero-single] suite=${TASK_SUITE_NAME} task_id=${TASK_ID} gpu=${GPU_ID}"
echo "[eval-flex-libero-single] joint:    video=${INFER_JOINT_VIDEO} dino=${INFER_JOINT_DINO} pointmap=${INFER_JOINT_POINTMAP}"
echo "[eval-flex-libero-single] presence: video=${INFER_PRESENT_VIDEO} dino=${INFER_PRESENT_DINO} pointmap=${INFER_PRESENT_POINTMAP}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
