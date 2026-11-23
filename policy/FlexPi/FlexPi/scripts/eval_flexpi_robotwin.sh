#!/usr/bin/env bash
# =============================================================================
# FlexPi — RoboTwin 2.0 eval sweep: every task, both phases, across GPUs.
# One pinned task instead: eval_flexpi_robotwin_single.sh
#
# Edit the config block below, then: bash scripts/eval_flexpi_robotwin.sh
# Details: docs/ROBOTWIN.md · docs/INFERENCE_OPTIMIZATION.md
# =============================================================================
set -euo pipefail
# ── Checkpoint ────────────────────────────────────────────────────────────────
# The config saved next to the checkpoint is autoloaded, so the architecture
# comes from the run — `model.*` overrides here would be ignored.
CKPT="./runs/robotwin_unified_flex_3cam_384_1e-4/<run_id>/checkpoints/weights/step_NNNNNN.pt"
DATASET_STATS="./runs/robotwin_unified_flex_3cam_384_1e-4/<run_id>/dataset_stats.json"

# ── Sweep ─────────────────────────────────────────────────────────────────────
PHASES=clean,random                  # clean | random | clean,random
TASKS=""                             # empty = all 50; else "lift_pot,place_shoe"
TASK_SHARD=""                        # "N/K" = run block N of K, for many machines
RESUME_DIR=""                        # an earlier output dir; finished results are skipped
NUM_GPUS=8
MAX_TASKS_PER_GPU=2

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_NUM_EPISODES=100
REPLAN_STEPS=32                      # the whole 32-action chunk, then re-observe.
                                     # Lower is more reactive; 32 is the reported protocol.
INSTRUCTION_TYPE=unseen              # seen | unseen (unseen is the reported protocol)
OFFLOAD_TEXT_ENCODER=true            # T5 on CPU: ~10 GB VRAM for ~200-300 ms/call

# ── Regime ────────────────────────────────────────────────────────────────────
# joint_* = what is generated, present_* = what is encoded as input.
# All three joint flags false = the action-only fast path.
INFER_JOINT_VIDEO=true
INFER_JOINT_DINO=true
INFER_JOINT_POINTMAP=true
INFER_PRESENT_VIDEO=true
INFER_PRESENT_DINO=true
INFER_PRESENT_POINTMAP=true

# ── Speed ─────────────────────────────────────────────────────────────────────
# Set an engine path for TensorRT, or leave it empty for torch.compile; the
# script switches the rest. Engines are machine- and checkpoint-locked (doc §2).
NUM_INFERENCE_STEPS=4
ATTN_BACKEND=auto
GLUE_CACHE=false
TORCH_COMPILE=true                   # 30-60 s warmup per worker; recompiles per regime
TORCH_COMPILE_SCOPE=loop             # step | loop
COMPILE_ENCODERS=false               # torch stack only; a measured no-op here

TRT_JOINT_ENGINE=""                  # this OR the split pair below, never both
TRT_JOINT_PREFILL_SPLIT_ENGINE=""
TRT_JOINT_DECODE_SPLIT_ENGINE=""
TRT_JOINT_FREE_VIDEO_BLOCKS=true     # mandatory with any engine
ENCODER_CUDA_GRAPH=true              # engine path only

# ── Escape hatch ──────────────────────────────────────────────────────────────
HYDRA_TASK_PRESET=robotwin_unified_flex_3cam_384_1e-4
EXTRA_OVERRIDES=""                   # extra Hydra overrides, space-separated

# =============================================================================
cd "$(dirname "$0")/.."

[[ -f "${CKPT}" ]] || { echo "ERROR: set CKPT at the top of this script (not found: ${CKPT})" >&2; exit 1; }
[[ -f "${DATASET_STATS}" ]] || { echo "ERROR: set DATASET_STATS at the top of this script (not found: ${DATASET_STATS})" >&2; exit 1; }

CMD=(
    python experiments/robotwin/run_robotwin_manager.py
    task="${HYDRA_TASK_PRESET}"
    ckpt="${CKPT}"
    EVALUATION.instruction_type="${INSTRUCTION_TYPE}"
    EVALUATION.dataset_stats_path="${DATASET_STATS}"
    EVALUATION.eval_num_episodes="${EVAL_NUM_EPISODES}"
    EVALUATION.num_inference_steps="${NUM_INFERENCE_STEPS}"
    EVALUATION.replan_steps="${REPLAN_STEPS}"
    EVALUATION.offload_text_encoder="${OFFLOAD_TEXT_ENCODER}"
    EVALUATION.attn_backend="${ATTN_BACKEND}"
    MULTIRUN.num_gpus="${NUM_GPUS}"
    MULTIRUN.max_tasks_per_gpu="${MAX_TASKS_PER_GPU}"
    "MULTIRUN.phases=[${PHASES}]"
)

for knob in JOINT_VIDEO JOINT_DINO JOINT_POINTMAP PRESENT_VIDEO PRESENT_DINO PRESENT_POINTMAP; do
    varname="INFER_${knob}"
    value="${!varname:-}"
    [[ -n "${value}" ]] && CMD+=("+EVALUATION.infer_$(echo "${knob}" | tr 'A-Z' 'a-z')=${value}")
done

# Pinned: deploy_policy otherwise infers it from the trained config's
# `data.train._target_`, and falls back to the composite path when that is absent.
CMD+=("+EVALUATION.use_per_cam=true")

# glue_cache and the engine knobs are absent from sim_robotwin.yaml, so they are
# ADDED with +. The engine replaces the compiled torch core, so the two stacks
# are never sent together.
[[ "${GLUE_CACHE}" == true ]] && CMD+=("+EVALUATION.glue_cache=true")
SPLIT="${TRT_JOINT_PREFILL_SPLIT_ENGINE}${TRT_JOINT_DECODE_SPLIT_ENGINE}"
if [[ -n "${TRT_JOINT_ENGINE}" && -n "${SPLIT}" ]]; then
    echo "ERROR: set TRT_JOINT_ENGINE or the KV-split pair, not both." >&2; exit 1
fi
if [[ -n "${SPLIT}" ]]; then
    for engine in "${TRT_JOINT_PREFILL_SPLIT_ENGINE}" "${TRT_JOINT_DECODE_SPLIT_ENGINE}"; do
        [[ -f "${engine}" ]] || { echo "ERROR: the KV-split pair needs both engines (missing: ${engine:-<unset>})" >&2; exit 1; }
    done
    CMD+=(
        "+EVALUATION.trt_joint_prefill_split_engine_path=${TRT_JOINT_PREFILL_SPLIT_ENGINE}"
        "+EVALUATION.trt_joint_decode_split_engine_path=${TRT_JOINT_DECODE_SPLIT_ENGINE}"
    )
    STACK="TRT KV-split"
fi
if [[ -n "${TRT_JOINT_ENGINE}" ]]; then
    [[ -f "${TRT_JOINT_ENGINE}" ]] || { echo "ERROR: engine not found: ${TRT_JOINT_ENGINE}" >&2; exit 1; }
    CMD+=("+EVALUATION.trt_joint_engine_path=${TRT_JOINT_ENGINE}")
    STACK="TRT single engine"
fi
if [[ -n "${TRT_JOINT_ENGINE}" || -n "${SPLIT}" ]]; then
    [[ "${TRT_JOINT_FREE_VIDEO_BLOCKS}" == true ]] && CMD+=("+EVALUATION.trt_joint_free_video_blocks=true")
    [[ "${ENCODER_CUDA_GRAPH}" == true ]] && CMD+=("+EVALUATION.encoder_cuda_graph=true")
else
    CMD+=(
        EVALUATION.torch_compile="${TORCH_COMPILE}"
        EVALUATION.torch_compile_scope="${TORCH_COMPILE_SCOPE}"
        EVALUATION.compile_encoders="${COMPILE_ENCODERS}"
    )
    STACK="torch (compile=${TORCH_COMPILE})"
fi

[[ -n "${TASKS}" ]] && CMD+=("MULTIRUN.tasks=[${TASKS}]")
[[ -n "${TASK_SHARD}" ]] && CMD+=("MULTIRUN.shard=${TASK_SHARD}")
[[ -n "${RESUME_DIR}" ]] && CMD+=("EVALUATION.output_dir=${RESUME_DIR}")
[[ -n "${EXTRA_OVERRIDES}" ]] && CMD+=(${EXTRA_OVERRIDES})

echo "[eval-flex] instr=${INSTRUCTION_TYPE} phases=[${PHASES}] tasks=[${TASKS:-ALL}] shard=${TASK_SHARD:-<none>} resume=${RESUME_DIR:-<fresh>} num_gpus=${NUM_GPUS} max_per_gpu=${MAX_TASKS_PER_GPU}"
echo "[eval-flex] regime: joint(v=${INFER_JOINT_VIDEO} d=${INFER_JOINT_DINO} p=${INFER_JOINT_POINTMAP}) present(v=${INFER_PRESENT_VIDEO} d=${INFER_PRESENT_DINO} p=${INFER_PRESENT_POINTMAP})"
echo "[eval-flex] speed: steps=${NUM_INFERENCE_STEPS} attn=${ATTN_BACKEND} glue_cache=${GLUE_CACHE} stack=${STACK}"
"${CMD[@]}"
