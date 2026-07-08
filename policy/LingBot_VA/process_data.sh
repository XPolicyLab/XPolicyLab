#!/usr/bin/env bash
set -euo pipefail

# Standard XPolicyLab contract:
#   bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
# Runs the full upstream LingBot-VA data pipeline over a RoboDojo LeRobot v2.1
# dataset: 30-dim action mapping, action_config, Wan2.2 VAE latents, empty_emb.pt.

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-0}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBODOJO_TEST_ROOT="$(cd "${POLICY_DIR}/../../.." && pwd)"

resolve_lerobot_repo_id() {
  if [[ -n "${LEROBOT_DATASET_REPO_ID:-}" ]]; then
    echo "${LEROBOT_DATASET_REPO_ID}"
    return
  fi
  case "${env_cfg_type}" in
    arx_x5) echo "RoboDojo_sim_arx-x5_v21" ;;
    *) echo "RoboDojo_sim_${env_cfg_type}" ;;
  esac
}

export LEROBOT_DATA_ROOT="${LEROBOT_DATA_ROOT:-${ROBODOJO_TEST_ROOT}/data}"
repo_id="$(resolve_lerobot_repo_id)"

source_dataset="${LINGBOT_VA_SOURCE_DATASET:-${LEROBOT_DATA_ROOT}/${repo_id}}"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
output_dataset="${LINGBOT_VA_DATASET_PATH:-${POLICY_DIR}/data/${data_setting}}"
base_model="${LINGBOT_VA_BASE_MODEL_PATH:?set LINGBOT_VA_BASE_MODEL_PATH to the lingbot-va-base weights dir}"

gpu_id="${LINGBOT_VA_PROCESS_GPU:-0}"
target_fps="${LINGBOT_VA_TARGET_FPS:-10}"
image_size="${LINGBOT_VA_IMAGE_SIZE:-256}"

echo "[LingBot_VA] source_dataset=${source_dataset}"
echo "[LingBot_VA] output_dataset=${output_dataset}"
echo "[LingBot_VA] base_model=${base_model}"
echo "[LingBot_VA] expert_data_num=${expert_data_num} (0=all) target_fps=${target_fps} gpu=${gpu_id}"

CUDA_VISIBLE_DEVICES="${gpu_id}" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
python "${POLICY_DIR}/process_data.py" \
  --source-dataset "${source_dataset}" \
  --output-dataset "${output_dataset}" \
  --base-model "${base_model}" \
  --num-episodes "${expert_data_num}" \
  --target-fps "${target_fps}" \
  --image-size "${image_size}"

echo "[LingBot_VA] export LINGBOT_VA_DATASET_PATH=${output_dataset}"
