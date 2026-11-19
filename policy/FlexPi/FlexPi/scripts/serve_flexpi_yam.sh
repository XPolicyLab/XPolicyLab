#!/usr/bin/env bash
# =============================================================================
# FlexPi — YAM real-robot WebSocket policy server.
#
# Serves action chunks over websocket to a raiden bridge. Set SPLIT_DIR to use
# TensorRT engines, or leave it empty for torch.compile.
#
# Edit the config block below, then: bash scripts/serve_flexpi_yam.sh
# Details: docs/YAM.md · docs/INFERENCE_OPTIMIZATION.md §3 (the server takes a
# narrower knob set than the sim eval launchers).
# =============================================================================
set -euo pipefail
# Keeps the allocator from fragmenting under a long-lived server.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Checkpoint ────────────────────────────────────────────────────────────────
# The run's saved config.yaml is autoloaded; only the knobs below are chosen here.
CKPT="./runs/yam_unified_flex_3cam_32d_rel_1e-4/<run_id>/checkpoints/weights/step_NNNNNN.pt"
# Normalizer stats. Empty = the run's dataset_stats.json next to the checkpoint.
DATASET_STATS=""

# ── Prompt ────────────────────────────────────────────────────────────────────
# REQUIRED. Must match the task's wording in the training manifest — a wrong
# prompt costs success rate silently.
PROMPT=""

# ── Server ────────────────────────────────────────────────────────────────────
HOST=0.0.0.0
PORT=8000

# ── Regime ────────────────────────────────────────────────────────────────────
# joint_* = what is generated, present_* = what is encoded as input.
# All three joint flags false = the action-only fast path.
INFER_JOINT_VIDEO=true
INFER_JOINT_DINO=true
INFER_JOINT_POINTMAP=true
INFER_PRESENT_VIDEO=true
INFER_PRESENT_DINO=true
INFER_PRESENT_POINTMAP=true

# ── Inference ─────────────────────────────────────────────────────────────────
NUM_INFERENCE_STEPS=4               # 4 is ~2x faster and close to 10-step performance
ACTION_HORIZON=""                    # empty = derive from the trained config (num_frames-1)
MIXED_PRECISION=bf16
OFFLOAD_TEXT_ENCODER=true            # keeps T5 on CPU: ~10 GB VRAM for ~200-300 ms/call
ATTN_BACKEND=auto                    # sdpa | flex | auto

# ── Acceleration ──────────────────────────────────────────────────────────────
# Engine-free stack only (ignored when SPLIT_DIR is set).
TORCH_COMPILE=true
TORCH_COMPILE_SCOPE=loop             # step | loop

# TensorRT KV-split engines baked for THIS checkpoint. Empty = engine-free.
# See docs/INFERENCE_OPTIMIZATION.md for how to bake them.
SPLIT_DIR=""
PREFILL_ENGINE_NAME=prefill_core_o3.engine
DECODE_ENGINE_NAME=decode_core_o3.engine

# Applies to both stacks.
ENCODER_CUDA_GRAPH=true

# =============================================================================
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[[ -f "${CKPT}" ]] || { echo "ERROR: set CKPT at the top of this script (not found: ${CKPT})" >&2; exit 1; }
[[ -n "${PROMPT}" ]] || { echo "ERROR: set PROMPT at the top of this script — the task instruction the model was trained on." >&2; exit 1; }

ARGS=(
  --ckpt-path "${CKPT}"
  --host "${HOST}" --port "${PORT}"
  --default-prompt "${PROMPT}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --mixed-precision "${MIXED_PRECISION}"
  --attn-backend "${ATTN_BACKEND}"
)
[[ -n "${ACTION_HORIZON}" ]] && ARGS+=(--action-horizon "${ACTION_HORIZON}")
[[ -n "${DATASET_STATS}" ]] && ARGS+=(--dataset-stats-path "${DATASET_STATS}")
[[ "${OFFLOAD_TEXT_ENCODER}" == true ]] && ARGS+=(--offload-text-encoder)

for knob in JOINT_VIDEO JOINT_DINO JOINT_POINTMAP PRESENT_VIDEO PRESENT_DINO PRESENT_POINTMAP; do
  varname="INFER_${knob}"
  value="${!varname:-}"
  [[ -n "${value}" ]] || continue
  flag="--infer-$(echo "${knob}" | tr 'A-Z_' 'a-z-')"
  ARGS+=("${flag}" "${value}")
done

if [[ -n "${SPLIT_DIR}" ]]; then
  PREFILL="${SPLIT_DIR}/${PREFILL_ENGINE_NAME}"
  DECODE="${SPLIT_DIR}/${DECODE_ENGINE_NAME}"
  for engine in "${PREFILL}" "${DECODE}"; do
    [[ -f "${engine}" ]] || { echo "ERROR: engine not found: ${engine}" >&2; exit 1; }
  done
  # --glue-cache pairs with the engines and hangs the joint boot alongside
  # --torch-compile, so the compile stack is skipped entirely here.
  # --no-torch-compile is explicit because serve_yam_flexpi.py defaults
  # --torch-compile to configs/real_yam.yaml's `torch_compile: true`, and the
  # two are mutually exclusive — without it the engine path exits at boot.
  ARGS+=(
    --trt-joint-prefill-split-engine "${PREFILL}"
    --trt-joint-decode-split-engine "${DECODE}"
    --trt-joint-free-video-blocks
    --glue-cache
    --no-torch-compile
  )
  STACK="TensorRT KV-split engines (${SPLIT_DIR})"
else
  if [[ "${TORCH_COMPILE}" == true ]]; then
    ARGS+=(--torch-compile --torch-compile-scope "${TORCH_COMPILE_SCOPE}")
  else
    ARGS+=(--no-torch-compile)
  fi
  STACK="engine-free (torch_compile=${TORCH_COMPILE} scope=${TORCH_COMPILE_SCOPE})"
fi

# encoder CUDA-graph replay is a per-call fixed-cost win on BOTH stacks
# (measured: encoder block 21.0 -> 14.9 ms), so it is applied outside the
# engine/engine-free branch rather than only on the latter.
[[ "${ENCODER_CUDA_GRAPH}" == true ]] && ARGS+=(--encoder-cuda-graph)

echo "[serve_flexpi_yam] ckpt   : ${CKPT}"
echo "[serve_flexpi_yam] listen : ${HOST}:${PORT}"
echo "[serve_flexpi_yam] stack  : ${STACK}"
echo "[serve_flexpi_yam] regime : joint(v=${INFER_JOINT_VIDEO} d=${INFER_JOINT_DINO} p=${INFER_JOINT_POINTMAP}) present(v=${INFER_PRESENT_VIDEO} d=${INFER_PRESENT_DINO} p=${INFER_PRESENT_POINTMAP}) steps=${NUM_INFERENCE_STEPS}"
echo "[serve_flexpi_yam] prompt : ${PROMPT}"

exec python scripts/serve_yam_flexpi.py "${ARGS[@]}"
