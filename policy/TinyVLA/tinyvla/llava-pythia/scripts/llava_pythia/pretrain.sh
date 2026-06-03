#!/bin/bash
LLM_MODEL_SIZE=14M

deepspeed --master_port 29600 llava_pythia/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoint_all/pythia_$LLM_MODEL_SIZE/base_checkpoints_llava_vanilla_pythia \
    --version plain \
    --data_path /data/team/zhumj/data/llava-pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /data/team/zhumj/data/llava-pretrain/images \
    --tune_mm_mlp_adapter True \
    --freeze_vision_tower True \
    --freeze_backbone True \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoint_all/pythia_$LLM_MODEL_SIZE/vanilla_pythia_pt_f_vit/llavaPythia-v0-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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

cp openai/clip-vit-large-patch14-336/preprocessor_config.json ./checkpoint_all/pythia_$LLM_MODEL_SIZE/vanilla_pythia_pt_f_vit/llavaPythia-v0-pretrain