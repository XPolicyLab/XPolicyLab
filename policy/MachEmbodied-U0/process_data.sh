#!/usr/bin/env bash
# RoboDojo data entry point. The training route consumes an existing LeRobot v3
# dataset; this command validates it and exposes the path under the policy dir.
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]" >&2
  exit 2
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
setting="${bench_name}-${ckpt_name//,/_}-${env_cfg_type}-${action_type}"
source_root="${ME_U0_ROBODOJO_DATA_ROOT:-${RAW_DATA_ROOT:-}}"
output_dir="${OUTPUT_DIR:-${POLICY_DIR}/data/${setting}}"

[[ "${bench_name}" == "RoboDojo" ]] || { echo "[ME_U0] bench_name must be RoboDojo" >&2; exit 1; }
[[ "${env_cfg_type}" == "arx_x5" ]] || { echo "[ME_U0] env_cfg_type must be arx_x5" >&2; exit 1; }
[[ "${action_type}" == "joint" ]] || { echo "[ME_U0] action_type must be joint" >&2; exit 1; }
[[ -n "${source_root}" ]] || { echo "[ME_U0] set ME_U0_ROBODOJO_DATA_ROOT or RAW_DATA_ROOT" >&2; exit 1; }
[[ -d "${source_root}" ]] || { echo "[ME_U0] dataset directory not found: ${source_root}" >&2; exit 1; }
if [[ ! -f "${source_root}/meta/info.json" && ! -f "${source_root}/meta/tasks.parquet" ]]; then
  echo "[ME_U0] expected LeRobot metadata under ${source_root}/meta" >&2
  exit 1
fi

mkdir -p "$(dirname "${output_dir}")"
if [[ "$(realpath -m "${source_root}")" != "$(realpath -m "${output_dir}")" ]]; then
  ln -sfn "$(realpath "${source_root}")" "${output_dir}"
fi

echo "[ME_U0] LeRobot v3 dataset ready: ${output_dir}"
echo "[ME_U0] expert_data_num=${expert_data_num:-all} (dataset is not converted by this adapter)"
echo "[ME_U0] use ME_U0_ROBODOJO_DATA_ROOT=${output_dir} for training"
