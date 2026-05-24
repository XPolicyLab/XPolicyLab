#!/bin/bash
set -e

# ==================== 参数定义 ====================
dataset_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
expert_data_num=$5
action_type=$6
seed=$7
gpu_id=$8


gpu_id="${gpu_id//[[:space:]]/}"
if [[ -z "${gpu_id}" ]]; then
   echo "gpu_id is required, e.g. 0 or 0,1,2,3"
   exit 1
fi
IFS=',' read -r -a GPU_IDS <<< "${gpu_id}"
num_gpus="${#GPU_IDS[@]}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


# define OUTPUT path
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ckpt_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
OUTPUT="${POLICY_DIR}/checkpoints/${ckpt_setting}"

if [ -d "$OUTPUT" ]; then
   echo '---------output exists---------'
else
   echo '---------output not exists, create directory---------'
   mkdir -p "$OUTPUT"
fi

# backup the train scripts
cp "${POLICY_DIR}/train.sh" "$OUTPUT"


# download pretrained VLM
pretrained_vlm_path="${OUTPUT}/pretrained_vlm"
has_pretrained_vlm_ckpt() {
  [ -f "${pretrained_vlm_path}/config.json" ] && {
    compgen -G "${pretrained_vlm_path}/*.safetensors" > /dev/null || \
    compgen -G "${pretrained_vlm_path}/pytorch_model*.bin" > /dev/null || \
    compgen -G "${pretrained_vlm_path}/model*.bin" > /dev/null
  }
}

if has_pretrained_vlm_ckpt; then
  echo "Using existing pretrained VLM: ${pretrained_vlm_path}"
else
  echo "No pretrained VLM checkpoint found in ${pretrained_vlm_path}"
  echo "Select pretrained VLM to download:"
  echo "  1) Llava-Pythia(~400M)  For TinyVLA-S  https://huggingface.co/lesjie/Llava-Pythia-400M"
  echo "  2) Llava-Pythia(~700M)  For TinyVLA-B  https://huggingface.co/lesjie/Llava-Pythia-700M"
  echo "  3) Llava-Pythia(~1.3B)  For TinyVLA-H  https://huggingface.co/lesjie/Llava-Pythia-1.3B"
  read -r -p "Enter choice [1-3]: " vlm_choice

  case "${vlm_choice}" in
    1)
      pretrained_vlm_repo="lesjie/Llava-Pythia-400M"
      ;;
    2)
      pretrained_vlm_repo="lesjie/Llava-Pythia-700M"
      ;;
    3)
      pretrained_vlm_repo="lesjie/Llava-Pythia-1.3B"
      ;;
    *)
      echo "Invalid pretrained VLM choice: ${vlm_choice}"
      exit 1
      ;;
  esac

  echo "Downloading ${pretrained_vlm_repo} to ${pretrained_vlm_path}"
  mkdir -p "${pretrained_vlm_path}"
  python - "${pretrained_vlm_repo}" "${pretrained_vlm_path}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    local_dir=sys.argv[2],
    resume_download=True,
)
PY

  if ! has_pretrained_vlm_ckpt; then
    echo "Failed to find pretrained VLM checkpoint after download: ${pretrained_vlm_path}"
    exit 1
  fi
fi



deepspeed --master_port 29600 --num_gpus="${num_gpus}" --num_nodes=1 "${POLICY_DIR}/train.py" \
  --xpl_dataset_name "${dataset_name}" \
  --xpl_task_name "${task_name}" \
  --xpl_env_cfg_type "${env_cfg_type}" \
  --xpl_expert_data_num "${expert_data_num}" \
  --xpl_action_type "${action_type}" \
  --deepspeed "${POLICY_DIR}/tinyvla/llava-pythia/scripts/zero2.json" \
  --lora_enable True \
  --lora_module 'vit llm' \
  --load_pretrain False \
  --pretrain_image_size 320 \
  --lora_r 64 \
  --lora_alpha 256 \
  --non_lora_lr 2e-5 \
  --task_name "${task_name}" \
  --model_name_or_path "${pretrained_vlm_path}" \
  --version v0 \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower True \
  --freeze_backbone True \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length False \
  --bf16 True \
  --output_dir "$OUTPUT" \
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
  --action_head_type act \
  --use_state True \
  --concat "token_cat" \
  --window_size 6 \
  --report_to tensorboard \
  --logging_dir "$OUTPUT/log"

for dir in "$OUTPUT"/*/ ; do
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp "${POLICY_DIR}/tinyvla/llava-pythia/preprocessor_config.json" "$dir"
    fi
done

