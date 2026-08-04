#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Train Pi0.5 with data augmentation for improved generalization
# ============================================================================
#
# This script trains Pi0.5 with image + prompt augmentation to improve
# performance on:
#   - _random eval variants (randomized object positions)
#   - _by_language eval variants (different language instructions)
#   - Unseen tasks (via overall visual diversity)
#
# Augmentations applied:
#   - Random crop & resize (simulates camera/object position shifts)
#   - Color jitter (brightness, contrast, saturation)
#   - Random erasing (partial occlusion robustness)
#   - Gaussian noise & blur
#   - Prompt paraphrasing (language generalization)
#
# Usage:
#   bash train_aug.sh <gpu_id> [variant]
#
# Variants:
#   aug        - Standard augmentation (default)
#   aug_heavy  - Aggressive augmentation for maximum generalization
#
# Examples:
#   bash train_aug.sh 0              # Standard aug, GPU 0
#   bash train_aug.sh 0,1 aug_heavy  # Heavy aug, 2 GPUs
# ============================================================================

gpu_id=${1:-0}
variant=${2:-aug}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${variant}" in
    aug)
        train_config="pi05_base_aloha_full_sim_arx-x5_seed_0_aug"
        ckpt_setting="RoboDojo-aug-arx_x5-joint-0"
        ;;
    aug_heavy)
        train_config="pi05_base_aloha_full_sim_arx-x5_seed_0_aug_heavy"
        ckpt_setting="RoboDojo-aug_heavy-arx_x5-joint-0"
        ;;
    *)
        echo "Unknown variant: ${variant}. Use 'aug' or 'aug_heavy'." >&2
        exit 1
        ;;
esac

ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
gpu_count=$(awk -F',' '{print NF}' <<<"${gpu_id}")
fsdp_devices="${OPENPI_FSDP_DEVICES:-$(( gpu_count < 2 ? 1 : 2 ))}"

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

LOCAL_CACHE_ROOT="${OPENPI_LOCAL_CACHE_ROOT:-/tmp/openpi-cache-$(hostname)}"
mkdir -p "${LOCAL_CACHE_ROOT}/hf/datasets" "${LOCAL_CACHE_ROOT}/jax"
export HF_DATASETS_CACHE="${LOCAL_CACHE_ROOT}/hf/datasets"
export JAX_COMPILATION_CACHE_DIR="${LOCAL_CACHE_ROOT}/jax"

echo "============================================================"
echo "  Pi0.5 Training with Data Augmentation"
echo "  Config:   ${train_config}"
echo "  Variant:  ${variant}"
echo "  Checkpoint: ${ckpt_dir}"
echo "  GPUs:     ${gpu_id} (fsdp_devices=${fsdp_devices})"
echo "  Started:  $(date)"
echo "============================================================"

cd "${POLICY_DIR}/openpi/"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
  uv run scripts/train.py "${train_config}" \
    --exp-name="${ckpt_setting}" \
    --data.repo-id="RoboDojo-cotrain-arx_x5-joint" \
    --fsdp-devices="${fsdp_devices}" \
    --checkpoint-dir-override="${ckpt_dir}" \
    --seed=0 \
    --overwrite
