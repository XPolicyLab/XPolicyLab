export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="/data/lingxuan/cutlass"

export WANDB_PROJECT="hrdt"
export OUTPUT_DIR="./checkpoints/robotwin2"

export VISION_ENCODER_NAME="dino-siglip"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# For run in a single node/machine
# accelerate launch main.py \
#     --deepspeed="./configs/zero2.json" \
#     ...

accelerate launch --main_process_port 29500 main.py \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --deepspeed="./configs/zero1.json" \
    --config_path="configs/hrdt_finetune.yaml" \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=32 \
    --sample_batch_size=32 \
    --max_train_steps=1000000 \
    --checkpointing_period=5000 \
    --sample_period=500 \
    --checkpoints_total_limit=40 \
    --lr_scheduler="constant_with_warmup" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=32 \
    --dataset_type="finetune" \
    --report_to=wandb \
    --upsample_rate=3 \
    --image_aug \
    --gradient_checkpointing \
    --precomp_lang_embed \
    --training_mode="lang" \
    --mode="finetune" \
    --pretrained_backbone_path="./checkpoints/pretrain-0618/checkpoint-500000/pytorch_model.bin"

    # For finetune mode with specific robot embodiment, use these parameters instead:
    # --mode="finetune" \
    # --pretrained_backbone_path="./checkpoints/pretrain-0618/pytorch_model.bin" \
    # --config_path="configs/hrdt_finetune.yaml" \  # Config with different action_dim for target robot
    # --dataset_type="finetune" \

    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-36000" \
    # Use this to load from saved lanuage instruction embeddings,
    # instead of calculating it during training