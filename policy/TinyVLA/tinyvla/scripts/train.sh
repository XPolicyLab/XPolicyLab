#!/bin/bash

ACTION_HEAD=droid_diffusion # specify action policy head type
# define OUTPUT path

OUTPUT=/path/to/save_dir

if [ -d "$OUTPUT" ]; then
   echo 'output exists'
else
   echo '!!output not exists!!'
   mkdir -p $OUTPUT
fi
# backup the train scripts
cp ./scripts/train.sh $OUTPUT

# detailed usage of each parameter can be found in train_tinyvla.py

deepspeed --master_port 29600 --num_gpus=8 --num_nodes=1 ./train_tinyvla.py \
  --deepspeed scripts/zero2.json \
  --lora_enable True \
  --lora_module 'vit llm' \
  --load_pretrain False \
  --pretrain_image_size 320 \
  --lora_r 64 \
  --lora_alpha 256 \
  --non_lora_lr 2e-5 \
  --task_name "example_task_config" \
  --model_name_or_path /path/to/pretrained_vlm \
  --version v0 \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower True \
  --freeze_backbone True \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length False \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 10000 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 50 \
  --learning_rate 2e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.005 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True \
  --action_head_type $ACTION_HEAD \
  --use_state True \
  --concat "token_cat" \
  --window_size 6 \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log

for dir in "$OUTPUT"/*/ ; do
    # 检查文件夹名称是否包含'checkpoint'
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp llava-pythia/preprocessor_config.json $dir
    fi
done

