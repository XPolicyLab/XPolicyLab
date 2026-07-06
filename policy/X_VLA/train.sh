#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ckpt_setting is the run directory name; pass it verbatim as ckpt_name to eval.sh.
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
meta_path="${XVLA_META_PATH:-${POLICY_DIR}/xvla/meta.json}"
model_path="${XVLA_MODEL_PATH:?set XVLA_MODEL_PATH to your X-VLA-Pt pretrained weights dir}"

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

echo "[X_VLA] meta_path=${meta_path}"
echo "[X_VLA] checkpoint_dir=${ckpt_dir}"

accelerate launch \
    --mixed_precision bf16 \
    xvla/train.py \
    --models "${model_path}" \
    --train_metas_path "${meta_path}" \
    --learning_rate 1e-4 \
    --learning_coef 0.1 \
    --iters 30000 \
    --freeze_steps 1000 \
    --warmup_steps 2000 \
    --batch_size 32 \
    --output_dir "${ckpt_dir}" \
    --seed "${seed}" \
    --save_interval 1000

# save_pretrained only writes model weights/config into ckpt-<step>/; eval needs the
# processor/tokenizer alongside them. Copy those files from the base model, never
# touching config.json / model.safetensors produced by training.
if [[ -d "${model_path}" ]]; then
  for step_dir in "${ckpt_dir}"/ckpt-*/; do
    [[ -d "${step_dir}" ]] || continue
    for fname in preprocessor_config.json tokenizer_config.json tokenizer.json \
                 vocab.json merges.txt special_tokens_map.json added_tokens.json; do
      if [[ -f "${model_path}/${fname}" && ! -e "${step_dir}${fname}" ]]; then
        cp "${model_path}/${fname}" "${step_dir}${fname}"
      fi
    done
  done
fi
