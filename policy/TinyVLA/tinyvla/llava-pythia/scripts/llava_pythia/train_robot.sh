#!/bin/bash
# LLM_MODEL_SIZE=$1
LLM_MODEL_SIZE=2_8B
# ./checkpoint_all/pythia_$LLM_MODEL_SIZE/vanilla_pythia_pt_f_vit/llavaPythia-v0-finetune
OUTPUT=./checkpoint_all/pythia_$LLM_MODEL_SIZE/vanilla_pythia_pt_f_vit/llavaPythia-v0-robot-action-1view_adapter3

# deepspeed --master_port 29601 --include localhost:4,5,6,7 llava_pythia/train/train.py \
# echo "waiting for 20minutes..."
# sleep 20m

deepspeed --master_port 29601 --num_gpus=8 --num_nodes=1 llava_pythia/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path  /data/team/zhumj/model_Param/llava_pythia_checkpoints/pythia_$LLM_MODEL_SIZE/vanilla_pythia_pt_f_vit/llavaPythia-v0-finetune \
    --version v0 \
    --data_path /data/private/data/llava_data/franka_kitchen_finetune/left_cap2/std_train_left_cap2_50k.json \
    --image_folder /data/team/zhumj/data/finetune/data \
    --tune_mm_mlp_adapter True \
    --freeze_vision_tower True \
    --freeze_backbone True \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir $OUTPUT \
    --num_train_epochs 15 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 15 \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.005 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --action_head "fc" \
    --use_state True \
    --lora_enable False \
    --window_size 6 \
    --logging_dir $OUTPUT/log 
    --report_to wandb 

#/data/private/data/llava_data/franka_kitchen_finetune/left_cap2/left_cap2_50k.json
#/data/team/zhumj/data/finetune/data/llava_v1_5_mix665k.json
# cp openai/clip-vit-large-patch14-336/preprocessor_config.json $OUTPUT
for dir in "$OUTPUT"/*/ ; do
    # 检查文件夹名称是否包含'checkpoint'
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp openai/clip-vit-large-patch14-336/preprocessor_config.json $dir
    fi
done

cp ./scripts/llava_pythia/train_robot.sh $OUTPUT
