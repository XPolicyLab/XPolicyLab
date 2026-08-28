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

if [[ "${bench_name}" != "RoboDojo" || "${env_cfg_type}" != "arx_x5" || "${action_type}" != "joint" ]]; then
  echo "[KinRT][ERROR] Supported combination: RoboDojo arx_x5 joint." >&2
  exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OPENPI_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)/KinRT_RoboDojo/policy/pi05"
OPENPI_ROOT="${KINRT_OPENPI_ROOT:-${DEFAULT_OPENPI_ROOT}}"
PYTHON_BIN="${KINRT_PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
train_config_name="${OPENPI_TRAIN_CONFIG_NAME:-kinrt_lora_robodojo}"
repo_id="${KINRT_ROBODOJO_REPO_ID:-RoboDojo-KinRT-arx_x5-joint}"
hf_lerobot_home="${HF_LEROBOT_HOME:-${HF_HOME:-${HOME}/.cache/huggingface}/lerobot}"
router_labels_path="${KINRT_ROBODOJO_ROUTER_LABELS_PATH:-${hf_lerobot_home}/${repo_id}/meta/router_labels_k4/router_labels.npy}"
run_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
checkpoint_dir="${POLICY_DIR}/checkpoints/${run_name}"
norm_stats_path="${OPENPI_ROOT}/assets/${train_config_name}/${repo_id}/norm_stats.json"

requires_router_labels=1
case "${train_config_name}" in
  pi05_lora_robodojo)
    requires_router_labels=0
    ;;
  kinrt_lora_robodojo|kinrt_full_robodojo) ;;
  *)
    echo "[KinRT][ERROR] Unsupported OPENPI_TRAIN_CONFIG_NAME: ${train_config_name}" >&2
    exit 1
    ;;
esac
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[KinRT][ERROR] Policy environment not found. Run: bash ${POLICY_DIR}/install.sh" >&2
  exit 1
fi
if (( requires_router_labels == 1 )) && [[ ! -f "${router_labels_path}" ]]; then
  echo "[KinRT][ERROR] Router labels not found: ${router_labels_path}" >&2
  echo "[KinRT][ERROR] Run generate_router_labels.sh before training." >&2
  exit 1
fi
if [[ ! -f "${norm_stats_path}" ]]; then
  echo "[KinRT][ERROR] Normalization statistics not found: ${norm_stats_path}" >&2
  echo "[KinRT][ERROR] Run: bash ${POLICY_DIR}/compute_norm_stats.sh ${train_config_name}" >&2
  exit 1
fi

gpu_count=$(awk -F',' '{print NF}' <<<"${gpu_id}")
fsdp_devices="${OPENPI_FSDP_DEVICES:-${gpu_count}}"
local_cache_root="${OPENPI_LOCAL_CACHE_ROOT:-/tmp/openpi-cache-$(hostname)}"
mkdir -p "${checkpoint_dir}" "${local_cache_root}/hf/datasets" "${local_cache_root}/jax"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export HF_LEROBOT_HOME="${hf_lerobot_home}"
export HF_DATASETS_CACHE="${local_cache_root}/hf/datasets"
export JAX_COMPILATION_CACHE_DIR="${local_cache_root}/jax"
export KINRT_ROBODOJO_REPO_ID="${repo_id}"
if (( requires_router_labels == 1 )); then
  export KINRT_ROBODOJO_ROUTER_LABELS_PATH="${router_labels_path}"
fi

run_mode=(--overwrite)
if [[ "${KINRT_RESUME:-0}" == "1" ]]; then
  run_mode=(--resume)
fi

train_overrides=()
if [[ -n "${OPENPI_NUM_TRAIN_STEPS:-}" ]]; then
  train_overrides+=(--num-train-steps="${OPENPI_NUM_TRAIN_STEPS}")
fi
if [[ -n "${OPENPI_BATCH_SIZE:-}" ]]; then
  train_overrides+=(--batch-size="${OPENPI_BATCH_SIZE}")
fi
if [[ -n "${OPENPI_NUM_WORKERS:-}" ]]; then
  train_overrides+=(--num-workers="${OPENPI_NUM_WORKERS}")
fi
if [[ -n "${OPENPI_SAVE_INTERVAL:-}" ]]; then
  train_overrides+=(--save-interval="${OPENPI_SAVE_INTERVAL}")
fi
if [[ "${OPENPI_WANDB_ENABLED:-1}" == "0" ]]; then
  train_overrides+=(--no-wandb-enabled)
fi

echo "[KinRT] train_config_name=${train_config_name}"
echo "[KinRT] repo_id=${repo_id}"
echo "[KinRT] router_labels=${router_labels_path}"
echo "[KinRT] checkpoint_dir=${checkpoint_dir}"

cd "${OPENPI_ROOT}"
PYTHONPATH="${OPENPI_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
  "${PYTHON_BIN}" scripts/train.py "${train_config_name}" \
    --exp-name="${run_name}" \
    --data.repo-id="${repo_id}" \
    --fsdp-devices="${fsdp_devices}" \
    --checkpoint-dir-override="${checkpoint_dir}" \
    --seed="${seed}" \
    "${train_overrides[@]}" \
    "${run_mode[@]}"
