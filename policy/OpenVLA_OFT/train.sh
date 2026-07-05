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
# The TFDS default matches eval model.py's `aloha_<ckpt_name>` derivation; an explicit
# OPENVLA_TFDS_DATASET_NAME here (mirrored by deploy.yml tfds_dataset_name) overrides it.
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
tfds_dataset_name="${OPENVLA_TFDS_DATASET_NAME:-aloha_${ckpt_setting}}"

mkdir -p "${ckpt_dir}"

echo "[OpenVLA_OFT] tfds_dataset_name=${tfds_dataset_name}"
echo "[OpenVLA_OFT] checkpoint_dir=${ckpt_dir}"

bash "${POLICY_DIR}/openvla_oft/scripts/finetune.sh" \
  "${ckpt_dir}" \
  "${tfds_dataset_name}" \
  "${gpu_id}" \
  "${seed}"