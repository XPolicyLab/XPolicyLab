#!/usr/bin/env bash
set -euo pipefail

# bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]
#
# Optional environment variables:
#   EWAM_RAW_DATA_ROOT     raw task dirs root (default: <parent>/data/<bench_name>)
#   EWAM_DATA_OUT_DIR      output dir (default: policy/EWAM/data/<bench>-<ckpt>-<env>-<action>)
#   EWAM_WAN_PATH          Wan2.2-TI2V-5B dir with the UMT5-XXL encoder (default: checkpoints/Wan2.2-TI2V-5B)
#   EWAM_PROCESS_WORKERS   parallel episode converters (default: 8)
if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]" >&2
  exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

py_args=("$@")
[[ -n "${EWAM_RAW_DATA_ROOT:-}" ]] && py_args+=(--raw_root "${EWAM_RAW_DATA_ROOT}")
[[ -n "${EWAM_DATA_OUT_DIR:-}" ]] && py_args+=(--out_dir "${EWAM_DATA_OUT_DIR}")
[[ -n "${EWAM_WAN_PATH:-}" ]] && py_args+=(--wan_path "${EWAM_WAN_PATH}")
py_args+=(--workers "${EWAM_PROCESS_WORKERS:-8}")

python "${POLICY_DIR}/process_data.py" "${py_args[@]}"
