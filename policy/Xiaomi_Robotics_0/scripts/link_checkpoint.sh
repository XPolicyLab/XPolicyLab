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
source_ckpt_dir=${7:-/vepfs-cnbje63de6fae220/xspark_shared/xiaomi_checkpoints/project_xr0/robodojo_sim}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ckpt_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
target_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"

mkdir -p "${POLICY_DIR}/checkpoints"
ln -sfn "${source_ckpt_dir}" "${target_dir}"

echo "[Xiaomi_Robotics_0] linked checkpoint:"
echo "  ${target_dir} -> ${source_ckpt_dir}"
