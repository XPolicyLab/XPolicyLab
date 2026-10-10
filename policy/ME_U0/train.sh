#!/usr/bin/env bash
# GPU training entry point following the XPolicyLab policy convention.
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [--dry-run] [key=value ...]" >&2
  exit 2
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6
shift 6

dry_run=0
overrides=()
for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    dry_run=1
  else
    overrides+=("${arg}")
  fi
done

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
setting="${bench_name}-${ckpt_name//,/_}-${env_cfg_type}-${action_type}-${seed}"
run_id="${ME_U0_RUN_ID:-${setting}}"
run_root="${RUN_ROOT:-${POLICY_DIR}/train_runs}"
config_rel="configs/experiments/robodojo_sim_posttraining.yaml"
config_path="${POLICY_DIR}/${config_rel}"
data_root="${ME_U0_ROBODOJO_DATA_ROOT:-${POLICY_DIR}/data/${bench_name}-${ckpt_name//,/_}-${env_cfg_type}-${action_type}}"
pretrained_path="${PRETRAINED_PATH:-${ME_U0_PRETRAINED_PTH:-}}"
model_root="${LEAP_MODEL_ROOT:-${POLICY_DIR}/assets}"

[[ "${bench_name}" == "RoboDojo" ]] || { echo "[ME_U0] bench_name must be RoboDojo" >&2; exit 1; }
[[ "${env_cfg_type}" == "arx_x5" ]] || { echo "[ME_U0] env_cfg_type must be arx_x5" >&2; exit 1; }
[[ "${action_type}" == "joint" ]] || { echo "[ME_U0] action_type must be joint" >&2; exit 1; }
[[ -f "${config_path}" ]] || { echo "[ME_U0] missing config: ${config_path}" >&2; exit 1; }

if (( dry_run == 0 )); then
  [[ -d "${data_root}" ]] || { echo "[ME_U0] dataset directory not found: ${data_root}; run process_data.sh first" >&2; exit 1; }
  [[ -n "${pretrained_path}" && -f "${pretrained_path}" ]] || { echo "[ME_U0] set PRETRAINED_PATH or ME_U0_PRETRAINED_PTH to a pretrained model.pt" >&2; exit 1; }
  [[ -d "${model_root}" ]] || { echo "[ME_U0] model assets not found: ${model_root}; set LEAP_MODEL_ROOT" >&2; exit 1; }
fi

gpu_count="$(awk -F, 'NF {print NF}' <<< "${gpu_id}")"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export LEAP_MODEL_ROOT="${model_root}"
export ME_U0_PRETRAINED_PTH="${pretrained_path:-/path/to/ME-U0-Pretrained/model.pt}"
export ME_U0_ROBODOJO_DATA_ROOT="${data_root}"
export PYTHONPATH="${POLICY_DIR}/me_u0:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cmd=(bash "${POLICY_DIR}/scripts/ME_U0/run_multinode.sh" 1
  --nproc-per-node "${gpu_count}"
  --exp-name "${run_id}"
  --work-root "${run_root}"
  --config "${config_rel}"
  "training.seed=${seed}"
  "${overrides[@]}")
printf '[ME_U0] %q ' "${cmd[@]}"; printf '\n'
if (( dry_run == 1 )); then
  exit 0
fi
exec "${cmd[@]}"
