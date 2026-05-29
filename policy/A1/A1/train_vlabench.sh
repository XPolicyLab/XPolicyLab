# ============================================
# Load personal environment configuration
# ============================================
if [ -f "$PWD/.env.personal" ]; then
  echo "[env] Loading .env.personal"
  source "$PWD/.env.personal"
fi
# ============================================
# Activate Conda environment
# ============================================
if [ -n "$CONDA_ROOT" ] && [ -n "$CONDA_ENV" ]; then
  echo "[conda] Activating environment from $CONDA_ROOT: $CONDA_ENV"
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export dataset_name=vla_dataset_simulation
export vla_config_path="vlabench.yaml"
checkpoint=model/pretrain
exp_name="a1_vlabench"
save_folder="./model/checkpoints/$exp_name"
# Automatically set nproc_per_node based on visible GPU count
if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
  IFS=',' read -ra DEV_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  nproc_per_node=${#DEV_ARR[@]}
else
  nproc_per_node=$(nvidia-smi -L | wc -l)
fi
BATCH_PER_GPU=16   # Batch size per GPU, can be overridden with --batch_size
STATE_MASK_PROB="0.0"   # State mask probability, can be overridden with --state_mask_prob
# global_batch_size = GPU count * batch size per GPU
global_batch_size=$((nproc_per_node * BATCH_PER_GPU))


# Launch training
torchrun \
  --nproc-per-node=$nproc_per_node \
  --rdzv-endpoint=localhost:13399 \
  launch_scripts/train_vla.py \
  qwen2_7b \
  --checkpoint $checkpoint \
  save_folder=$save_folder \
  --vision_backbone "openai" \
  --action_head "flow_matching" \
  --seq_len 600 \
  --state_mask_prob "${STATE_MASK_PROB}" \
  --device_train_microbatch_size $BATCH_PER_GPU \
  --global_batch_size $global_batch_size \
  --dataset $dataset_name \
  --ft_llm \
  --llm_learning_rate 5e-6 \
  --action_head_learning_rate 5e-5 \
  --vit_learning_rate 2e-6 \
  --connector_learning_rate 2e-6 \
  --warmup_steps 2000 \
  --freeze_steps 1000 \
  --save_interval_unsharded 1000 \
  --save_interval 1000 \
  --crop_mode "resize" \
  --max_crops 3 \
  --train_steps 50000 \
  --vla_config_path $vla_config_path \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --wandb_run_name $exp_name \
  --save_overwrite \
  --log_interval 50 \
  --num_workers 4
