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


# ===== 2. Training hyperparameters =====
export dataset_name=vla_dataset_pretrain
export vla_config_path="pretrain.yaml"
exp_name=molmo_7b_pretrain
save_folder=./model/checkpoints/molmo_7b_pretrain
model_path=model/Molmo-7B-D-0924
load_path=model/checkpoints/molmo_7b_pretrain_v2/latest


# ===== 3. SLURM environment variables =====
export NODE_RANK=${NODE_RANK:-$SLURM_NODEID} # Get current node rank
export OMP_NUM_THREADS=$NPROC_PER_NODE

# Save code snapshot first
if [ "$NODE_RANK" -eq 0 ]; then
    mkdir -p "${save_folder}" && \
    python save_code.py -n "${exp_name}" -o "${save_folder}"
fi

export WANDB_API_KEY=35ed06feafa826b6d6dd0c186d59eeba150e7442
export WANDB_PROJECT=rc_training
export WANDB_ENTITY=
# export WANDB_MODE=offline

# ===== 4. Launch 8 GPU processes on each node =====
# Use srun to ensure all nodes run simultaneously; --nodes=1 --ntasks-per-node=8 to launch 8 tasks on current node
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
launch_scripts/train_vla.py \
    qwen2_7b \
    --checkpoint $model_path \
    save_folder=$save_folder \
    --vision_backbone "openai" \
    --action_head "flow_matching" \
    --seq_len 600 \
    --device_train_microbatch_size 16 \
    --global_batch_size $((128*$NNODES)) \
    --state_mask_prob 0.5 \
    --dataset $dataset_name \
    --vla_config_path $vla_config_path \
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
    --train_steps 500000 \
    --wandb_entity $wandb_entity \
    --wandb_project "pretrain_training" \
    --wandb_run_name $exp_name \
    --save_overwrite \
    --log_interval 50 \
    --num_workers 2