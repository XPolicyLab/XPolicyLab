#!/bin/bash
set -euo pipefail

dataset_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
seed=${6}
gpu_id=${7}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
POLICY_DIR="${ROOT_DIR}/XPolicyLab/policy/LDA_1B"
UPSTREAM_DIR="${POLICY_DIR}/LDA-1B"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

# Defaults are derived from POLICY_DIR (script-relative absolute path computed
# above). Override any of them by exporting the corresponding LDA_* env var
# before running train.sh — no path is hard-coded.
base_vlm="${LDA_BASE_VLM:-${POLICY_DIR}/checkpoints/Qwen3-VL-4B-Instruct}"
vision_encoder_path="${LDA_VISION_ENCODER:-${POLICY_DIR}/checkpoints/dinov3-vit-s}"
data_root_dir="${LDA_DATA_ROOT:-${POLICY_DIR}/data}"
data_mix="${LDA_DATA_MIX:-xpolicylab}"
ckpt_root_dir="${LDA_CKPT_ROOT:-${POLICY_DIR}/checkpoints}"

# Feed the 5 CLI args into the generic `xpolicylab` mixture entry registered in
# upstream lda/dataloader/gr00t_lerobot/mixtures.py. The folder name must match
# what LDA-1B/xpolicylab_adapter/process_data.py wrote out (same 5-tuple, hyphen-joined).
export XPOLICYLAB_DATASET_ID="${XPOLICYLAB_DATASET_ID:-${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}}"
export XPOLICYLAB_ROBOT_TYPE="${XPOLICYLAB_ROBOT_TYPE:-${env_cfg_type}}"
ckpt_setting="${LDA_CKPT_SETTING:-${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}}"
pretrained_checkpoint="${LDA_PRETRAINED_CHECKPOINT:-null}"
max_train_steps="${LDA_MAX_TRAIN_STEPS:-200000}"
per_device_batch_size="${LDA_PER_DEVICE_BATCH_SIZE:-64}"

# gr00t mixture loader pads each action key to a fixed per-key width, so the model's
# action_dim must be that padded sum (arx_x5 = 16), NOT the raw physical dim (14).
action_dim=$(python "${UPSTREAM_DIR}/xpolicylab_adapter/gr00t_action_dim.py" "${env_cfg_type}")
mkdir -p "${ckpt_root_dir}/${ckpt_setting}"

cd "${UPSTREAM_DIR}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

training_cfg="${LDA_TRAINING_CONFIG:-${UPSTREAM_DIR}/lda/config/training/LDA_pretrain.yaml}"

# Generate per-run accelerate + deepspeed configs. We MUST go via a separate
# deepspeed JSON file because accelerate's inline `deepspeed_config:` block
# only honors a hard-coded subset of fields (gradient_accumulation_steps,
# gradient_clipping, zero_stage, offload_*); inline train_micro_batch_size_per_gpu
# is silently dropped → accelerate.prepare() then fails to infer batch size
# because LDA's dataloader uses a custom BatchSampler (dataloader.batch_size=None).
# Putting train_micro_batch_size_per_gpu in the external JSON is the only way it
# reaches deepspeed.
accelerate_cfg="${LDA_ACCELERATE_CONFIG:-${ckpt_root_dir}/${ckpt_setting}/accelerate.yaml}"
ds_config="${LDA_DEEPSPEED_CONFIG:-${ckpt_root_dir}/${ckpt_setting}/ds_config.json}"

if [[ -z "${LDA_ACCELERATE_CONFIG:-}" ]]; then
    mkdir -p "$(dirname "${accelerate_cfg}")"
    cat > "${ds_config}" <<EOF
{
    "fp16": {"enabled": false},
    "bf16": {"enabled": true},
    "train_micro_batch_size_per_gpu": ${per_device_batch_size},
    "train_batch_size": "auto",
    "gradient_accumulation_steps": ${LDA_GRADIENT_ACCUMULATION_STEPS:-1},
    "gradient_clipping": ${LDA_GRADIENT_CLIPPING:-1.0},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 500000000,
        "reduce_scatter": true,
        "reduce_bucket_size": 500000000,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "cpu_offload": false
    },
    "steps_per_print": 100
}
EOF
    # Match upstream lda/config/deepseeds/deepspeed_zero2.yaml exactly:
    # when deepspeed_config_file is set, ANY of {mixed_precision,
    # gradient_accumulation_steps, gradient_clipping, zero_stage, offload_*,
    # zero3_save_16bit_model} at the accelerate-yaml level conflicts and aborts.
    # All those settings belong in ds_config.json instead.
    cat > "${accelerate_cfg}" <<EOF
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_config_file: ${ds_config}
  deepspeed_multinode_launcher: standard
  zero3_init_flag: false
distributed_type: DEEPSPEED
num_machines: 1
num_processes: ${LDA_NUM_PROCESSES:-8}
EOF
    echo "[train.sh] generated accelerate config at ${accelerate_cfg}"
    echo "[train.sh] generated deepspeed config at  ${ds_config} (train_micro_batch_size_per_gpu=${per_device_batch_size})"
fi

accelerate launch \
  --config_file "${accelerate_cfg}" \
  --num_processes "${LDA_NUM_PROCESSES:-8}" \
  "${UPSTREAM_DIR}/lda/training/train_LDA.py" \
  --config_yaml "${training_cfg}" \
  --framework.name QwenMMDiT \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.action_model.vision_encoder_path "${vision_encoder_path}" \
  --framework.action_model.action_model_type "${LDA_DIT_TYPE:-DiT-L}" \
  --framework.action_model.max_num_embodiments "${LDA_MAX_NUM_EMBODIMENTS:-1}" \
  --framework.action_model.state_dim "${LDA_STATE_DIM:-null}" \
  --framework.action_model.action_dim "${action_dim}" \
  --framework.action_model.obs_horizon "${LDA_OBS_HORIZON:-1}" \
  --framework.action_model.future_obs_index "${LDA_FUTURE_OBS_INDEX:-5}" \
  --framework.action_model.only_policy "${LDA_ONLY_POLICY:-false}" \
  --framework.action_model.policy_and_video_gen "${LDA_POLICY_AND_VIDEO_GEN:-false}" \
  --framework.action_model.only_wo_video_gen "${LDA_ONLY_WO_VIDEO_GEN:-false}" \
  --datasets.vla_data.use_delta_action "${LDA_USE_DELTA_ACTION:-false}" \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.training_task_weights "${LDA_TRAINING_TASK_WEIGHTS:-[1,1,1,1]}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --datasets.vla_data.return_vlm_inputs "${LDA_RETURN_VLM_INPUTS:-false}" \
  --trainer.freeze_modules "${LDA_FREEZE_MODULES:-qwen_vl_interface,action_model.vision_encoder}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.save_interval "${LDA_SAVE_INTERVAL:-10000}" \
  --trainer.logging_frequency "${LDA_LOGGING_FREQUENCY:-1000}" \
  --trainer.eval_interval "${LDA_EVAL_INTERVAL:-1000}" \
  --trainer.repeated_diffusion_steps "${LDA_REPEATED_DIFFUSION_STEPS:-1}" \
  --trainer.learning_rate.base "${LDA_LEARNING_RATE:-4e-5}" \
  --trainer.pretrained_checkpoint "${pretrained_checkpoint}" \
  --run_root_dir "${ckpt_root_dir}" \
  --run_id "${ckpt_setting}" \
  --wandb_project "${LDA_WANDB_PROJECT:-lda}" \
  --wandb_entity "${LDA_WANDB_ENTITY:-}" \
  --is_debug "${LDA_DEBUG:-False}"
