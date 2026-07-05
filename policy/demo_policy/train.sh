#!/bin/bash
set -e

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
# Output convention: checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

if [[ -z "${bench_name}" || -z "${ckpt_name}" || -z "${env_cfg_type}" || -z "${action_type}" || -z "${seed}" || -z "${gpu_id}" ]]; then
  echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"

mkdir -p "${ckpt_dir}"
echo "[demo_policy] train stub: would write to ${ckpt_dir}"
echo "[demo_policy] Implement training and invoke it from this script."
