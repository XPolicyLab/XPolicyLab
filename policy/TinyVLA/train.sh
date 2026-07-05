#!/bin/bash
set -e


bench_name=$1
ckpt_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5
seed=$6
gpu_id=$7

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
output_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
pretrained_vlm_dir="${output_dir}/pretrained_vlm"

mkdir -p "${output_dir}"


# trainning recipe snapshot
cp "${POLICY_DIR}/train.sh"   "${output_dir}/train.sh"
cp "${POLICY_DIR}/deploy.yml" "${output_dir}/deploy.yml"


# prepare pretrained VLM
has_pretrained_vlm() {
  [ -f "${pretrained_vlm_dir}/config.json" ] && {
    compgen -G "${pretrained_vlm_dir}/*.safetensors"   > /dev/null || \
    compgen -G "${pretrained_vlm_dir}/pytorch_model*.bin" > /dev/null || \
    compgen -G "${pretrained_vlm_dir}/model*.bin"      > /dev/null
  }
}

if has_pretrained_vlm; then
  echo "[TinyVLA] Using existing pretrained VLM: ${pretrained_vlm_dir}"
else
  echo "[TinyVLA] No pretrained VLM in ${pretrained_vlm_dir}"
  echo "Select pretrained VLM to download:"
  echo "  1) Llava-Pythia(~400M)  TinyVLA-S  https://huggingface.co/lesjie/Llava-Pythia-400M"
  echo "  2) Llava-Pythia(~700M)  TinyVLA-B  https://huggingface.co/lesjie/Llava-Pythia-700M"
  echo "  3) Llava-Pythia(~1.3B)  TinyVLA-H  https://huggingface.co/lesjie/Llava-Pythia-1.3B"
  read -r -p "Enter choice [1-3]: " vlm_choice
  case "${vlm_choice}" in
    1) vlm_repo="lesjie/Llava-Pythia-400M" ;;
    2) vlm_repo="lesjie/Llava-Pythia-700M" ;;
    3) vlm_repo="lesjie/Llava-Pythia-1.3B" ;;
    *) echo "Invalid choice: ${vlm_choice}" >&2; exit 1 ;;
  esac
  mkdir -p "${pretrained_vlm_dir}"
  hf download "${vlm_repo}" --local-dir "${pretrained_vlm_dir}"
fi




# effective_batch_size = per_device_train_batch_size × gradient_accumulation_steps × num_gpus

deepspeed --master_port 29600 --include "localhost:${gpu_id}" "${POLICY_DIR}/train.py" \
  --xpl_bench_name                "${bench_name}" \
  --xpl_ckpt_name                   "${ckpt_name}" \
  --xpl_env_cfg_type                "${env_cfg_type}" \
  --xpl_expert_data_num             "${expert_data_num}" \
  --xpl_action_type                 "${action_type}" \
  --xpl_seed                        "${seed}" \
  --max_steps                       10000 \
  --per_device_train_batch_size     32 \
  --save_steps                      1000 \
  --save_total_limit                50 \
  --logging_steps                   10 \
  --gradient_accumulation_steps     1

