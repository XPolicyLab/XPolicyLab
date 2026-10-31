#!/usr/bin/env bash
# =============================================================================
# FlexPi — LIBERO 4-suite evaluation, the standard protocol.
# One (suite, task) instead: eval_flexpi_libero_single.sh
#
#   CKPT=runs/.../step_010860.pt \
#   DATASET_STATS=runs/.../dataset_stats.json \
#   GPUS=0,1,2,3 \
#     bash scripts/eval_flexpi_libero_4suite.sh
#
# Details: docs/LIBERO.md
# =============================================================================
set -euo pipefail

# Repo root first — every default below is relative to it.
cd "$(dirname "$0")/.."

# ── Environment ───────────────────────────────────────────────────────────────
# LIBERO is never pip-installed; it only resolves via PYTHONPATH.
export PYTHONPATH="$(pwd)/third_party/LIBERO:$(pwd)/src:${PYTHONPATH:-}"
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

# ── Protocol ──────────────────────────────────────────────────────────────────
# Per-suite max_steps lives in code (eval_libero_single.py::_get_max_steps).
SUITES=(libero_spatial libero_object libero_goal libero_10)
TASKS_PER_SUITE="${TASKS_PER_SUITE:-10}"
NUM_TRIALS="${NUM_TRIALS:-50}"
HYDRA_TASK_PRESET="${HYDRA_TASK_PRESET:-libero_unified_flex_2cam224_32d_rotvec_1e-4}"

# ── Data / intrinsics ─────────────────────────────────────────────────────────
# All four suites ship the same camera_intrinsics.json, so one path serves all.
DATA_ROOT="${DATA_ROOT:-./data/libero_mujoco3.3.2_depth}"
CAMERA_INTRINSICS_PATH="${CAMERA_INTRINSICS_PATH:-${DATA_ROOT}/libero_object_no_noops_lerobot/meta/camera_intrinsics.json}"
[[ -f "$CAMERA_INTRINSICS_PATH" ]] || {
  echo "ERROR: camera_intrinsics not found: $CAMERA_INTRINSICS_PATH"
  echo "       The published dataset ships one per suite under meta/."; exit 1; }

# ── GPUs ──────────────────────────────────────────────────────────────────────
GPUS="${GPUS:-0}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

OUTPUT_DIR="${OUTPUT_DIR:-./eval_results/libero_4suite_$(date +%Y%m%d_%H%M%S)}"

# ── Regime ────────────────────────────────────────────────────────────────────
# joint_* = what is generated, present_* = what is encoded as input.
INFER_JOINT_VIDEO="${INFER_JOINT_VIDEO:-true}"
INFER_JOINT_DINO="${INFER_JOINT_DINO:-true}"
INFER_JOINT_POINTMAP="${INFER_JOINT_POINTMAP:-true}"
INFER_PRESENT_VIDEO="${INFER_PRESENT_VIDEO:-true}"
INFER_PRESENT_DINO="${INFER_PRESENT_DINO:-true}"
INFER_PRESENT_POINTMAP="${INFER_PRESENT_POINTMAP:-true}"

# =============================================================================
# Build the task list, shard it round-robin across GPUs
# =============================================================================
mkdir -p "$OUTPUT_DIR/shards"
for ((g = 0; g < NGPU; g++)); do : > "$OUTPUT_DIR/shards/gpu${GPU_ARR[$g]}.txt"; done

i=0
for suite in "${SUITES[@]}"; do
  for ((t = 0; t < TASKS_PER_SUITE; t++)); do
    g=$((i % NGPU))
    echo "${suite},${t}" >> "$OUTPUT_DIR/shards/gpu${GPU_ARR[$g]}.txt"
    i=$((i + 1))
  done
done

echo "[4suite] ${#SUITES[@]} suites x ${TASKS_PER_SUITE} tasks = ${i} tasks"
echo "[4suite] trials/task=${NUM_TRIALS}  gpus=${GPUS}  out=${OUTPUT_DIR}"
echo "[4suite] joint: v=${INFER_JOINT_VIDEO} d=${INFER_JOINT_DINO} p=${INFER_JOINT_POINTMAP}"

# =============================================================================
# Launch one batch process per GPU
# =============================================================================
pids=()
for gpu in "${GPU_ARR[@]}"; do
  shard="$OUTPUT_DIR/shards/gpu${gpu}.txt"
  [[ -s "$shard" ]] || continue
  CUDA_VISIBLE_DEVICES="$gpu" python experiments/libero/eval_libero_batch.py \
    task="${HYDRA_TASK_PRESET}" \
    ckpt="${CKPT}" \
    gpu_id="${gpu}" \
    EVALUATION.dataset_stats_path="${DATASET_STATS}" \
    "+EVALUATION.camera_intrinsics_path=${CAMERA_INTRINSICS_PATH}" \
    "+EVALUATION.tasks_file=${shard}" \
    EVALUATION.output_dir="${OUTPUT_DIR}" \
    EVALUATION.num_trials="${NUM_TRIALS}" \
    "+EVALUATION.infer_present_video=${INFER_PRESENT_VIDEO}" \
    "+EVALUATION.infer_present_dino=${INFER_PRESENT_DINO}" \
    "+EVALUATION.infer_present_pointmap=${INFER_PRESENT_POINTMAP}" \
    "+EVALUATION.infer_joint_video=${INFER_JOINT_VIDEO}" \
    "+EVALUATION.infer_joint_dino=${INFER_JOINT_DINO}" \
    "+EVALUATION.infer_joint_pointmap=${INFER_JOINT_POINTMAP}" \
    > "$OUTPUT_DIR/gpu${gpu}.log" 2>&1 &
  pids+=($!)
  echo "[4suite] gpu${gpu}: $(wc -l < "$shard") tasks -> $OUTPUT_DIR/gpu${gpu}.log (pid $!)"
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
[[ $fail -eq 0 ]] || echo "[4suite] WARNING: at least one shard exited non-zero; see the per-gpu logs."

# The summarizer averages whatever JSONs exist, so a partial sweep must say so.
_expected=$i
_actual=$(find "$OUTPUT_DIR" -name "*_results.json" | wc -l)
if [[ "$_actual" -ne "$_expected" ]]; then
  echo "[4suite] INCOMPLETE: ${_actual}/${_expected} tasks produced results."
  echo "[4suite] The summary below covers only those — it is NOT the full protocol."
  fail=1
fi

# =============================================================================
# Summarize
# =============================================================================
echo
echo "[4suite] results:"
python experiments/libero/summarize_results_4suite.py --output_dir "$OUTPUT_DIR"
echo
echo "[4suite] summary written to $OUTPUT_DIR/summary_4suite.{csv,json}"
[[ $fail -eq 0 ]] || exit 1
