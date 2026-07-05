#!/usr/bin/env bash
set -euo pipefail

# Standard XPolicyLab contract:
#   bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
# LingBot_VA data is prepared by the upstream LeRobot pipeline (see README §数据处理).
# This wrapper validates the standard args; point LEROBOT_DATASET_REPO_ID or
# LINGBOT_VA_DATASET_PATH at the converted dataset before training.

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}

data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
echo "[LingBot_VA] process_data: training data is prepared by the upstream LingBot pipeline."
echo "[LingBot_VA] expected dataset tag: ${data_setting}"
if [[ -n "${expert_data_num}" ]]; then
  echo "[LingBot_VA] note: expert_data_num=${expert_data_num} is not applied here; cap episodes during upstream conversion."
fi
echo "[LingBot_VA] set LEROBOT_DATASET_REPO_ID or LINGBOT_VA_DATASET_PATH, then run train.sh."
