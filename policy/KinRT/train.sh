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
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
DEFAULT_OPENPI_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)/KinRT_RoboDojo/policy/pi05"
OPENPI_ROOT="${KINRT_OPENPI_ROOT:-${DEFAULT_OPENPI_ROOT}}"
PYTHON_BIN="${KINRT_PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
train_config_name="${OPENPI_TRAIN_CONFIG_NAME:-kinrt_full_robodojo}"
repo_id="${KINRT_ROBODOJO_REPO_ID:-RoboDojo_lerobot_v30_video}"
hf_lerobot_home="${HF_LEROBOT_HOME:-${HF_HOME:-${HOME}/.cache/huggingface}/lerobot}"
dataset_root="${hf_lerobot_home}/${repo_id}"
router_labels_subdir=router_labels_k4
if [[ "${repo_id}" == "RoboDojo_lerobot_v30_video" ]]; then
  router_labels_subdir=router_labels_k4_full35
fi
router_labels_path="${KINRT_ROBODOJO_ROUTER_LABELS_PATH:-${dataset_root}/meta/${router_labels_subdir}/router_labels.npy}"
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
  echo "[KinRT][ERROR] Install the published Full35 assets for the original dataset, or generate labels for a separately named custom dataset." >&2
  exit 1
fi
if [[ ! -f "${norm_stats_path}" ]]; then
  echo "[KinRT][ERROR] Normalization statistics not found: ${norm_stats_path}" >&2
  echo "[KinRT][ERROR] Install the published Full35 assets, or run compute_norm_stats.sh for a custom dataset." >&2
  exit 1
fi
if (( requires_router_labels == 1 )) && [[ "${repo_id}" == "RoboDojo_lerobot_v30_video" ]]; then
  "${PYTHON_BIN}" "${POLICY_DIR}/full35_assets.py" validate-training \
    --dataset-root "${dataset_root}" \
    --labels "${router_labels_path}" \
    --norm-stats "${norm_stats_path}"
fi

run_name="$(PYTHONPATH="${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" - "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" <<'PY'
import sys

from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name

keys = ("bench_name", "ckpt_name", "env_cfg_type", "action_type", "seed")
run_name = build_run_dir_name(dict(zip(keys, sys.argv[1:], strict=True)))
if run_name is None:
    raise ValueError("Training requires a non-empty benchmark, checkpoint name, embodiment, action type, and seed.")
print(run_name)
PY
)"
checkpoint_dir="${POLICY_DIR}/checkpoints/${run_name}"

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

default_batch_size=32
if [[ "${train_config_name}" == "kinrt_full_robodojo" && "${repo_id}" == "RoboDojo_lerobot_v30_video" ]]; then
  default_batch_size=256
fi
train_overrides=(
  --num-train-steps="${OPENPI_NUM_TRAIN_STEPS:-60000}"
  --batch-size="${OPENPI_BATCH_SIZE:-${default_batch_size}}"
  --num-workers="${OPENPI_NUM_WORKERS:-8}"
  --save-interval="${OPENPI_SAVE_INTERVAL:-5000}"
)
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
