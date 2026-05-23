#!/bin/bash
LLM_MODEL_SIZE=14M

deepspeed --master_port 29600 --num_gpus=8 --num_nodes=1 llava_pythia/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoint_all/pythia_$LLM_MODEL_SIZE/vanilla_pythia_pt_f_vit/llavaPythia-v0-pretrain \
    --version v0 \
    --data_path /data/team/zhumj/data/finetune/data/llava_v1_5_mix665k.json \
    --image_folder /data/team/zhumj/data/finetune/data \
    --tune_mm_mlp_adapter True \
    --freeze_vision_tower False \
    --freeze_backbone Talse \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoint_all/pythia_$LLM_MODEL_SIZE/vanilla_pythia_pt_f_vit/llavaPythia-v0-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

#/data/private/data/llava_data/franka_kitchen_finetune/left_cap2/left_cap2_50k.json
#/data/team/zhumj/data/finetune/data/llava_v1_5_mix665k.json
cp openai/clip-vit-large-patch14-336/preprocessor_config.json ./checkpoint_all/pythia_$LLM_MODEL_SIZE/vanilla_pythia_pt_f_vit/llavaPythia-v0-finetune