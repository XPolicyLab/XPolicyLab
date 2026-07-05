#!/bin/bash
set -e

# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
# Output convention: data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}

data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[demo_policy] process_data stub: would write to ${POLICY_DIR}/data/${data_setting}"
echo "[demo_policy] Implement process_data.py and invoke it from this script."
