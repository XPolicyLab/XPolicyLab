#!/usr/bin/env bash
set -euo pipefail

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
bench_name=${1:?bench_name is required}
ckpt_name=${2:?ckpt_name is required}
env_cfg_type=${3:?env_cfg_type is required}
action_type=${4:?action_type is required}
seed=${5:?seed is required}
gpu_id=${6:?gpu_id is required}

[[ "${bench_name}" == "SparkArena" ]] || {
  echo "this entry point requires bench_name=SparkArena" >&2; exit 2;
}
[[ "${ckpt_name}" == "egovla" ]] || {
  echo "this entry point requires ckpt_name=egovla" >&2; exit 2;
}
[[ "${env_cfg_type}" == "tianji_marvin_wuji" ]] || {
  echo "SparkArena EgoVLA requires env_cfg_type=tianji_marvin_wuji" >&2; exit 2;
}
[[ "${action_type}" == "joint" ]] || {
  echo "raw-joint EgoVLA requires action_type=joint" >&2; exit 2;
}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
UPSTREAM_ROOT="${EGOVLA_UPSTREAM_ROOT:-${POLICY_DIR}/EgoVLA_Release}"
TRAIN_ENV="${EGOVLA_CONDA_ENV:-}"
if [[ -n "${EGOVLA_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${EGOVLA_PYTHON_BIN}"
elif [[ -n "${TRAIN_ENV}" && -x "${TRAIN_ENV}/bin/python" ]]; then
  PYTHON_BIN="${TRAIN_ENV}/bin/python"
else
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -n "${EGOVLA_TORCHRUN_BIN:-}" ]]; then
  TORCHRUN_BIN="${EGOVLA_TORCHRUN_BIN}"
elif [[ -n "${TRAIN_ENV}" && -x "${TRAIN_ENV}/bin/torchrun" ]]; then
  TORCHRUN_BIN="${TRAIN_ENV}/bin/torchrun"
else
  TORCHRUN_BIN="$(command -v torchrun || true)"
fi
[[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || {
  echo "missing training Python; set EGOVLA_PYTHON_BIN or EGOVLA_CONDA_ENV" >&2; exit 2;
}
if [[ -n "${TORCHRUN_BIN}" && -x "${TORCHRUN_BIN}" ]]; then
  TORCHRUN_CMD=("${TORCHRUN_BIN}")
else
  TORCHRUN_CMD=("${PYTHON_BIN}" -m torch.distributed.run)
fi

if [[ -n "${TRAIN_ENV}" && -d "${TRAIN_ENV}/bin" ]]; then
  export PATH="${TRAIN_ENV}/bin:${PATH}"
fi
unset LD_LIBRARY_PATH
if [[ -n "${TRAIN_ENV}" && -d "${TRAIN_ENV}/bin" ]]; then
  export PATH="${TRAIN_ENV}/bin:${PATH}"
fi
[[ "${gpu_id}" == "all" ]] || {
  echo "SparkArena EgoVLA training requires all eight GPUs; pass gpu_id=all" >&2; exit 2;
}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC=8
[[ "${NPROC}" -eq 8 ]] || { echo "SparkArena EgoVLA requires 8 GPUs" >&2; exit 2; }

RAW_ROOT="${EGOVLA_RAW_ROOT:-${SPARKARENA_RAW_ROOT:-${POLICY_DIR}/data/raw/spark0_bench_7tasks}}"
[[ "$(basename "${RAW_ROOT}")" == "spark0_bench_7tasks" ]] || { echo "raw root must be the spark0_bench_7tasks dataset" >&2; exit 2; }
DATA_DIR="${EGOVLA_DATA_DIR:-${POLICY_DIR}/data/SparkArena-egovla-tianji_marvin_wuji-joint}"
RUN_NAME="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
OUTPUT_DIR="${EGOVLA_OUTPUT_DIR:-${POLICY_DIR}/checkpoints/${RUN_NAME}}"
PRETRAINED_PATH="${EGOVLA_PRETRAINED_PATH:-${UPSTREAM_ROOT}/checkpoints}"

for artifact in train.jsonl val.jsonl joint_state_stats.npz joint_action_stats.npz metadata.json sparkarena_provenance.json; do
  [[ -f "${DATA_DIR}/${artifact}" ]] || {
    echo "missing joint dataset artifact: ${DATA_DIR}/${artifact}" >&2; exit 2;
  }
done
for artifact in config.json llm/model.safetensors vision_tower/model.safetensors mm_projector/model.safetensors traj_decoder/model.safetensors; do
  [[ -f "${PRETRAINED_PATH}/${artifact}" ]] || {
    echo "pretrained model is incomplete: ${PRETRAINED_PATH}/${artifact}" >&2; exit 2;
  }
done
[[ -d "${UPSTREAM_ROOT}/VILA" ]] || {
  echo "vendored EgoVLA release is missing: ${UPSTREAM_ROOT}" >&2; exit 2;
}

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] && [[ "${EGOVLA_RESUME:-0}" != "1" ]]; then
  echo "output directory is non-empty: ${OUTPUT_DIR}; set EGOVLA_RESUME=1 only for an intentional resume" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"

command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required for SparkArena training" >&2; exit 2; }
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || { echo "SparkArena EgoVLA requires exactly 8 visible GPUs" >&2; exit 2; }
for gpu_name in "${GPU_NAMES[@]}"; do
  [[ "${gpu_name}" == *A800* ]] || { echo "all visible GPUs must be A800: ${gpu_name}" >&2; exit 2; }
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
  echo "visible GPUs have active compute processes; refusing to start training" >&2; exit 2
fi

PER_DEVICE_BATCH="${EGOVLA_PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${EGOVLA_GRAD_ACCUM:-2}"
GLOBAL_BATCH_SIZE=$((8 * PER_DEVICE_BATCH * GRAD_ACCUM))
[[ "${GLOBAL_BATCH_SIZE}" -eq 64 ]] || { echo "global batch must be 64 (8 GPUs x batch x accumulation)" >&2; exit 2; }


export EGOVLA_CONDA_ENV="${TRAIN_ENV}"
export EGOVLA_RAW_ROOT="${RAW_ROOT}"
export SPARKARENA_RAW_ROOT="${RAW_ROOT}"
export EGOVLA_DATA_DIR="${DATA_DIR}"
export EGOVLA_OUTPUT_DIR="${OUTPUT_DIR}"
export EGOVLA_PRETRAINED_PATH="${PRETRAINED_PATH}"
export EGOVLA_UPSTREAM_ROOT="${UPSTREAM_ROOT}"
export EGOVLA_SEED="${seed}"
export EGOVLA_PER_DEVICE_BATCH="${PER_DEVICE_BATCH}"
export EGOVLA_GRAD_ACCUM="${GRAD_ACCUM}"
export EGOVLA_GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}"
export PYTHONPATH="${POLICY_DIR}/.deps:${POLICY_DIR}/EgoVLA_Release/VILA:${POLICY_DIR}/EgoVLA_Release:${POLICY_DIR}:${XPL_ROOT}:${UPSTREAM_ROOT}/VILA:${UPSTREAM_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${EGOVLA_WANDB_MODE:-online}"
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB_API_KEY is required" >&2; exit 2; }
if [[ "${WANDB_MODE}" == "online" ]]; then
  [[ "${EGOVLA_MAX_STEPS:-80000}" == "80000" && "${EGOVLA_SAVE_STEPS:-10000}" == "10000" && "${EGOVLA_EVAL_STEPS:-10000}" == "10000" ]] || { echo "online production training requires 80000/10000/10000 schedule" >&2; exit 2; }
  [[ "${EGOVLA_MAX_SAMPLES:-0}" == "0" && "${EGOVLA_TARGET_STRIDE:-1}" == "1" ]] || { echo "online production training cannot use sample/stride overrides" >&2; exit 2; }
  [[ "${EGOVLA_DEBUG_E2E:-False}" == "False" && -z "${EGOVLA_SMOKE_STOP_AFTER_SAVE_STEP:-}" ]] || { echo "online production training cannot use debug/smoke overrides" >&2; exit 2; }
fi
export WANDB_PROJECT="${WANDB_PROJECT:-egovla_sparkarena}"
export RUN_NAME
export OMP_NUM_THREADS="${EGOVLA_OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" - "${DATA_DIR}" "${RAW_ROOT}" <<'PY'
import pathlib
import sys
from XPolicyLab.policy.EgoVLA.EgoVLA_Release.human_plan.dataset_preprocessing.sparkarena.process_data import verify_sparkarena_artifacts
metadata, _ = verify_sparkarena_artifacts(pathlib.Path(sys.argv[1]))
expected = pathlib.Path(sys.argv[2]).resolve()
actual = pathlib.Path(metadata["source_root"]).resolve()
if actual != expected:
    raise ValueError(f"data source mismatch: {actual} != {expected}")
if metadata.get("episodes") != 700 or metadata.get("frames") != 154251:
    raise ValueError(f"SparkArena full inventory mismatch: {metadata.get('episodes')} episodes, {metadata.get('frames')} frames")
expected_tasks = {"click_mouse": 100, "collect_objects": 100, "dual_bottles_pick": 100, "hammer_beat": 100, "put_food_in_microwave": 100, "retrieve_gap": 100, "stack_bowls": 100}
if metadata.get("task_episode_counts") != expected_tasks:
    raise ValueError(f"SparkArena task inventory mismatch: {metadata.get('task_episode_counts')}")
if metadata.get("state_stats_count") != 147897 or metadata.get("action_stats_count") != 147897:
    raise ValueError("SparkArena normalization statistics count mismatch")
print("[egovla-sparkarena] data verified:", metadata["episodes"], "episodes,", metadata["frames"], "frames")
PY
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; assert torch.cuda.device_count() == 8, torch.cuda.device_count(); assert torch.backends.cudnn.version() and torch.backends.cudnn.version() >= 90000, torch.backends.cudnn.version(); print("[egovla-sparkarena] CUDA/cudnn preflight ok:", torch.__version__, torch.backends.cudnn.version())'

MAX_STEPS="${EGOVLA_MAX_STEPS:-80000}"
SAVE_STEPS="${EGOVLA_SAVE_STEPS:-10000}"
EVAL_STEPS="${EGOVLA_EVAL_STEPS:-10000}"
EVAL_STRATEGY="${EGOVLA_EVALUATION_STRATEGY:-steps}"
NUM_WORKERS="${EGOVLA_NUM_WORKERS:-4}"


common_args=(
  --model_name_or_path "${PRETRAINED_PATH}"
  --version vicuna_v1
  --vision_tower google/siglip-so400m-patch14-384
  --data_mixture sparkarena_egovla_train
  --eval_data_mixture sparkarena_egovla_val
  --mm_vision_select_feature cls_patch
  --mm_projector mlp_downsample
  --mm_vision_select_layer -2
  --mm_use_im_start_end False
  --mm_use_im_patch_token False
  --image_aspect_ratio resize
  --bf16 True
  --tf32 True
  --output_dir "${OUTPUT_DIR}"
  --run_name "${RUN_NAME}"
  --seed "${seed}"
  --data_seed "${seed}"
  --per_device_train_batch_size "${PER_DEVICE_BATCH}"
  --per_device_eval_batch_size 4
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --max_steps "${MAX_STEPS}"
  --save_strategy steps
  --save_steps "${SAVE_STEPS}"
  --save_total_limit 8
  --evaluation_strategy "${EVAL_STRATEGY}"
  --eval_steps "${EVAL_STEPS}"
  --logging_steps "${EGOVLA_LOGGING_STEPS:-10}"
  --learning_rate "${EGOVLA_LR:-2e-5}"
  --hand_decoder_lr "${EGOVLA_DECODER_LR:-1e-4}"
  --weight_decay 0.
  --warmup_ratio 0.03
  --lr_scheduler_type constant
  --model_max_length 4096
  --gradient_checkpointing True
  --dataloader_num_workers "${NUM_WORKERS}"
  --dataloader_pin_memory True
  --ddp_find_unused_parameters True
  --debug_e2e "${EGOVLA_DEBUG_E2E:-False}"
  --lazy_preprocess True
  --report_to wandb
  --future_index 0
  --predict_future_step 30
  --max_action 1
  --min_action 0
  --add_his_obs_step 5
  --add_his_imgs True
  --add_his_img_skip 5
  --prompt_version v0
  --num_action_bins 256
  --action_tokenizer uniform
  --invalid_token_weight 0.1
  --mask_input True
  --mask_ignore False
  --traj_decoder_type transformer_split_action_v2
  --raw_action_label True
  --traj_action_output_dim 54
  --proprio_size 54
  --use_proprio True
  --sep_proprio False
  --sep_query_token True
  --input_placeholder_diff_index True
  --next_token_loss_coeff 0.0
  --ee_loss_coeff 1.0
  --hand_loss_coeff 1.0
  --hand_loss_dim 20
  --ee_2d_loss_coeff 0.0
  --ee_rot_loss_coeff 0.0
  --hand_kp_loss_coeff 0.0
  --traj_action_output_ee_2d_dim 0
  --traj_action_output_ee_dim 14
  --traj_action_output_hand_dim 40
  --traj_action_output_ee_rot_dim 0
  --ee_rot_representation ""
  --include_2d_label False
  --include_rot_label False
  --no_norm_ee_label True
  --merge_hand True
  --use_mano False
  --input_hand_dof False
  --loss_use_l1 True
  --tune_vision_tower "${EGOVLA_TUNE_VISION:-False}"
  --tune_mm_projector "${EGOVLA_TUNE_PROJECTOR:-True}"
  --tune_language_model "${EGOVLA_TUNE_LM:-False}"
)

echo "[egovla-sparkarena] run=${RUN_NAME}"
echo "[egovla-sparkarena] raw=${RAW_ROOT} data=${DATA_DIR} output=${OUTPUT_DIR}"
echo "[egovla-sparkarena] batch=${NPROC}x${PER_DEVICE_BATCH}x${GRAD_ACCUM}=${GLOBAL_BATCH_SIZE} steps=${MAX_STEPS} save=${SAVE_STEPS}"
exec "${TORCHRUN_CMD[@]}" --standalone --nproc_per_node=8 "${POLICY_DIR}/EgoVLA_Release/human_plan/vila_train/train_sparkarena_entry.py" "${common_args[@]}"
