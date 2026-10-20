#!/bin/bash
set -euo pipefail

# Offline precompute of state/action normalization statistics for a recipe.
#
# Training launched through train.sh runs with SPARK_0_REQUIRE_PRECOMPUTED_STATS=1
# by default: every rank READS the content-addressed stats caches under
# <dataset>/meta/stats_xpolicy*__<hash>.json and fails fast when one is
# missing, instead of scanning the dataset on rank 0 at launch. This script
# builds those caches ahead of time.
#
# It reuses train.sh's config resolution, so the resolved YAML and the
# values passed into source_starvla are bit-identical to the training launch — the
# cache hash covers pose formats, camera frame, history frames, action
# indices, exclusions and the data/meta fingerprints, and all of them come
# from the same code path here.
#
# Usage (mirrors scripts/finetune.sh; GPUs are accepted but unused — the
# precompute is a single CPU process that reads metadata + parquet only):
#   bash scripts/precompute_stats.sh <train_config>
#   bash scripts/precompute_stats.sh <train_config> 0
#   SPARK_0_DATA_ROOT=/path/to/data bash scripts/precompute_stats.sh <train_config>
#
# Re-run it whenever the data or any stats-relevant config knob changes
# (pose formats, ee_in_camera_frame, state_history_frames, action_horizon,
# exclude_episodes, embodiment layout); a config change produces a NEW hash,
# so old caches are left untouched and the new one is computed once.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_DIR="$(dirname "${SCRIPT_DIR}")"

usage() {
    echo "Usage: bash scripts/precompute_stats.sh <train_config> [gpus] [omega_overrides...]" >&2
    echo "  train_config  YAML stem under model_configs/ (or path to a .yaml)" >&2
    echo "  gpus          accepted for train.sh compatibility; unused (CPU-only)" >&2
    echo "Available configs:" >&2
    for f in "${ADAPTER_DIR}/model_configs/"*.yaml; do
        echo "  $(basename "${f%.yaml}")" >&2
    done
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

train_config="${1}"
shift

gpus="${GPUS:-0}"
if [[ $# -ge 1 && $1 != *=* && $1 != -* ]]; then
    gpus=$1
    shift
fi

config_stem="$(basename "${train_config}")"
config_stem="${config_stem%.yaml}"
ckpt_name="${CKPT_NAME:-${config_stem}}"
seed="${SEED:-0}"
bench_name="${BENCH_NAME:-SparkArena}"
env_cfg_type="${ENV_CFG_TYPE:-tianji_marvin_wuji}"

if [[ -f "${train_config}" ]]; then
    config_yaml="$(realpath "${train_config}")"
else
    config_yaml="${ADAPTER_DIR}/model_configs/${config_stem}.yaml"
fi
if [[ ! -f "${config_yaml}" ]]; then
    echo "[precompute_stats] ERROR: model config not found: ${config_yaml}" >&2
    usage
    exit 1
fi
train_config="${config_stem}"

extra_args=("$@")
if [[ -n "${PER_DEVICE_BS:-}" ]]; then
    extra_args+=("datasets.vla_data.per_device_batch_size=${PER_DEVICE_BS}")
fi

echo "[precompute_stats] train_config=${train_config}"
echo "[precompute_stats] building stats caches only (SPARK_0_PRECOMPUTE_STATS_ONLY=1)"

exec env SPARK_0_PRECOMPUTE_STATS_ONLY=1 bash "${ADAPTER_DIR}/train.sh" \
    "${bench_name}" "${ckpt_name}" "${env_cfg_type}" ee "${seed}" "${gpus}" \
    "${train_config}" ${extra_args[@]+"${extra_args[@]}"}
