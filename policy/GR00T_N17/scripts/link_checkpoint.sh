#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> [source_ckpt_dir]" >&2
  exit 1
fi

dataset_name=$1
ckpt_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5
seed=$6
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ckpt_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
target_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
source_ckpt_dir=${7:-}

if [[ -z "${source_ckpt_dir}" ]]; then
  echo "Usage: $0 <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <source_ckpt_dir>" >&2
  echo "  source_ckpt_dir: experiment root containing checkpoint-* (e.g. .../RoboDojo-cotrain-arx_x5-3500-joint-0)" >&2
  exit 1
fi

if [[ ! -d "${source_ckpt_dir}" ]]; then
  echo "Source checkpoint directory not found: ${source_ckpt_dir}" >&2
  exit 1
fi

mkdir -p "${POLICY_DIR}/checkpoints"
ln -sfn "$(cd "${source_ckpt_dir}" && pwd)" "${target_dir}"

echo "[GR00T_N17] linked checkpoint:"
echo "  ${target_dir} -> ${source_ckpt_dir}"
