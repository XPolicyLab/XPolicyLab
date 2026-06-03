#!/bin/bash
# Deprecated: use XPolicyLab entrypoints under policy/Being_H05/ instead.
#
#   cd policy/Being_H05
#   bash process_data.sh RoboDojo cotrain arx_x5 3500 joint
#   export BEINGH_MLLM_PATH=... BEINGH_EXPERT_PATH=... BEINGH_RESUME_PATH=...
#   bash train.sh RoboDojo cotrain arx_x5 3500 joint 0 0,1,2,3
#
# Legacy direct torchrun (only if you maintain paths yourself):

export PYTHONPATH=.
export NCCL_IB_DISABLE=0
export NO_ALBUMENTATIONS_UPDATE=1

PRETRAIN_MODEL="/path/to/InternVL3_5-2B"
EXPERT_MODEL="/path/to/Qwen3-0.6B"
RESUME_PATH="/path/to/Being-H05-2B"

EMBODIMENT="robodojo"
EMBODIMENT_DATASET="robodojo_joint_arx_x5"
DATASET_CONFIG_FILE="configs/posttrain/${EMBODIMENT}/${EMBODIMENT_DATASET}.yaml"
SAVE_MERGED_META=True

NUM_GPUS=4
MAX_STEPS=150000
SAVE_STEPS=25000
SAVE_STEPS_START=0
SAVE_MODEL_ONLY=False
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
WARMUP_RATIO=0

NUM_WORKERS=12
PREFETCH_FACTOR=4

MAX_NUM_TOKENS=8960
EXPECTED_NUM_TOKENS=8960
PREFER_BUFFER_BEFORE=4096
MAX_BUFFER_SIZE=4
ATTN_MODE="causal"

FORCE_IMAGE_SIZE=224
MAX_VIEW_NUM=-1
USE_FIXED_VIEW=False
DOWN_SAMPLE_RATIO=0.5

ACTION_CHUNK_LENGTH=16

FREEZE_MLLM=False
FREEZE_VIT_MLP=False

USE_MPG=True
MPG_LAMBDA=0.1
MPG_NUM_PROJECTIONS=32
MPG_REFINEMENT_ITERS=1
MPG_GATE_TEMPERATURE=1.0
MPG_USE_STOP_GRADIENT=True

USE_TRAINING_TIME_RTC=False
SIMULATED_DELAY=0
RTC_DELAY_EXP_WEIGHT=True
USE_INFERENCE_PREFIX_OVERWRITE=True

MODEL_NAME="post-robodojo_joint_BH05-2B_chunk-${ACTION_CHUNK_LENGTH}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="/path/to/checkpoints/${MODEL_NAME}"
LOG_FILE="${OUTPUT_DIR}/training.log"

mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=29109 \
  BeingH/train/train.py \
  --mllm_path ${PRETRAIN_MODEL} \
  --expert_path ${EXPERT_MODEL} \
  --resume_from ${RESUME_PATH} \
  --resume_model_only True \
  --layer_module Qwen3MoTDecoderLayer \
  --use_expert True \
  --use_flow_matching True \
  --llm_qk_norm True \
  --freeze_mllm ${FREEZE_MLLM} \
  --freeze_vit_mlp ${FREEZE_VIT_MLP} \
  --action_chunk_length ${ACTION_CHUNK_LENGTH} \
  --max_num_tokens ${MAX_NUM_TOKENS} \
  --max_num_tokens_per_sample ${MAX_NUM_TOKENS} \
  --expected_num_tokens ${EXPECTED_NUM_TOKENS} \
  --prefer_buffer_before ${PREFER_BUFFER_BEFORE} \
  --max_buffer_size ${MAX_BUFFER_SIZE} \
  --attn_mode ${ATTN_MODE} \
  --max_view_num ${MAX_VIEW_NUM} \
  --use_fixed_view ${USE_FIXED_VIEW} \
  --force_image_size ${FORCE_IMAGE_SIZE} \
  --down_sample_ratio ${DOWN_SAMPLE_RATIO} \
  --dataset_config_file ${DATASET_CONFIG_FILE} \
  --save_merged_metadata ${SAVE_MERGED_META} \
  --conv_style "being_h0" \
  --vision_select_layer -1 \
  --prompt_template long \
  --output_dir ${OUTPUT_DIR} \
  --num_workers ${NUM_WORKERS} \
  --prefetch_factor ${PREFETCH_FACTOR} \
  --max_steps ${MAX_STEPS} \
  --save_model_only ${SAVE_MODEL_ONLY} \
  --save_steps ${SAVE_STEPS} \
  --save_steps_start ${SAVE_STEPS_START} \
  --logging_steps 10 \
  --learning_rate ${LEARNING_RATE} \
  --weight_decay ${WEIGHT_DECAY} \
  --warmup_ratio ${WARMUP_RATIO} \
  --lr_scheduler cosine \
  --grad_checkpoint False \
  --gradient_accumulation_steps 2 \
  --use_mpg ${USE_MPG} \
  --mpg_lambda ${MPG_LAMBDA} \
  --mpg_num_projections ${MPG_NUM_PROJECTIONS} \
  --mpg_refinement_iters ${MPG_REFINEMENT_ITERS} \
  --mpg_gate_temperature ${MPG_GATE_TEMPERATURE} \
  --mpg_use_stop_gradient ${MPG_USE_STOP_GRADIENT} \
  --use_training_time_rtc ${USE_TRAINING_TIME_RTC} \
  --simulated_delay ${SIMULATED_DELAY} \
  --rtc_delay_exp_weight ${RTC_DELAY_EXP_WEIGHT} \
  --use_inference_prefix_overwrite ${USE_INFERENCE_PREFIX_OVERWRITE} \
  2>&1 | tee "${LOG_FILE}"

echo "Training complete: ${OUTPUT_DIR}"
