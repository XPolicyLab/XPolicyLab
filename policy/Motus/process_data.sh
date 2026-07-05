#!/usr/bin/env bash
# Standard XPolicyLab data-prep entrypoint for Motus.
#
# Usage:
#   bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
#
# Motus trains directly from a LeRobot-format dataset (repo_id + root), so there is NO local
# HDF5 -> zarr/LeRobot conversion step (mirrors SmolVLA). This script therefore:
#   1) resolves the LeRobot dataset (repo_id + root) that train.sh will consume,
#   2) validates that the dataset actually exists (fail fast with clear guidance), and
#   3) optionally pre-computes the T5 embedding cache (set MOTUS_RUN_T5_CACHE=1).
#
# The repo_id/root default to whatever motus/<MOTUS_TRAIN_CONFIG> declares; override with env
# vars (no personal paths are hardcoded here):
#   MOTUS_CONDA_ENV        conda env to activate (default: motus; set MOTUS_SKIP_CONDA_ACTIVATE=1 to skip)
#   MOTUS_TRAIN_CONFIG     upstream config, relative to motus/ (default: configs/lerobot_RoboDojo_sim.yaml)
#   MOTUS_REPO_ID          LeRobot dataset repo_id override
#   MOTUS_DATASET_ROOT / LEROBOT_DATA_ROOT   LeRobot dataset root override (parent dir is OK)
#   MOTUS_RUN_T5_CACHE     when 1, run add_t5_cache_to_lerobot_dataset.py (needs WAN_PATH)
#   WAN_PATH               model root for the T5 encoder (only for the optional cache step)
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]" >&2
    exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
# expert_data_num is accepted for interface parity with other policies; Motus uses all
# episodes present in the LeRobot dataset (subset selection lives in the training config).
expert_data_num=${5:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTUS_ROOT="${SCRIPT_DIR}/motus"

MOTUS_CONDA_ENV="${MOTUS_CONDA_ENV:-motus}"
if [ "${MOTUS_SKIP_CONDA_ACTIVATE:-0}" != "1" ] && command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${MOTUS_CONDA_ENV}"
fi

BASE_CONFIG="${MOTUS_TRAIN_CONFIG:-configs/lerobot_RoboDojo_sim.yaml}"
_dataset_root="${MOTUS_DATASET_ROOT:-${LEROBOT_DATA_ROOT:-}}"

cd "${MOTUS_ROOT}"

echo -e "\033[33m[Motus process_data] bench=${bench_name} ckpt=${ckpt_name} env=${env_cfg_type} action=${action_type}\033[0m"
if [ -n "${expert_data_num}" ]; then
    echo -e "\033[33m[Motus process_data] Note: expert_data_num=${expert_data_num} is ignored (LeRobot dataset is used as-is).\033[0m"
fi

# Resolve final repo_id + root (env override first, else read from the training config).
_resolved="$(
    MOTUS_BASE_CONFIG="${BASE_CONFIG}" \
    MOTUS_REPO_ID="${MOTUS_REPO_ID:-}" \
    MOTUS_DATASET_ROOT="${_dataset_root}" \
    python - <<'PY'
import os
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.environ["MOTUS_BASE_CONFIG"])
params = cfg.dataset.params
repo_id = os.environ.get("MOTUS_REPO_ID") or params.get("repo_id") or ""
root = os.environ.get("MOTUS_DATASET_ROOT") or ""
if root:
    if repo_id and os.path.isdir(os.path.join(root, repo_id)):
        root = os.path.join(root, repo_id)
else:
    root = params.get("root") or ""
print(repo_id)
print(root)
PY
)"
REPO_ID="$(echo "${_resolved}" | sed -n '1p')"
DATASET_ROOT="$(echo "${_resolved}" | sed -n '2p')"

echo -e "\033[33m[Motus process_data] repo_id=${REPO_ID}\033[0m"
echo -e "\033[33m[Motus process_data] root=${DATASET_ROOT}\033[0m"

if [ -z "${DATASET_ROOT}" ]; then
    echo -e "\033[31m[ERROR] No dataset root resolved. Set MOTUS_DATASET_ROOT / LEROBOT_DATA_ROOT or edit ${BASE_CONFIG}.\033[0m" >&2
    exit 1
fi
if [ ! -d "${DATASET_ROOT}" ] || [ ! -f "${DATASET_ROOT}/meta/info.json" ]; then
    echo -e "\033[31m[ERROR] LeRobot dataset not found at: ${DATASET_ROOT}\033[0m" >&2
    echo -e "\033[31m[ERROR] Expected a LeRobot dataset (with meta/info.json). Fix by either:\033[0m" >&2
    echo -e "\033[31m         - export MOTUS_DATASET_ROOT=/abs/path/to/<dataset> (or LEROBOT_DATA_ROOT parent), and/or\033[0m" >&2
    echo -e "\033[31m         - export MOTUS_REPO_ID=<repo_id>, and/or edit ${BASE_CONFIG} dataset.params.\033[0m" >&2
    exit 1
fi

echo -e "\033[32m[Motus process_data] LeRobot dataset OK (no local conversion required).\033[0m"

if [ "${MOTUS_RUN_T5_CACHE:-0}" = "1" ]; then
    if [ -z "${WAN_PATH:-}" ]; then
        echo -e "\033[31m[ERROR] MOTUS_RUN_T5_CACHE=1 but WAN_PATH is unset (needed for the T5 encoder).\033[0m" >&2
        exit 1
    fi
    echo -e "\033[33m[Motus process_data] Building T5 embedding cache...\033[0m"
    python data/lerobot/add_t5_cache_to_lerobot_dataset.py \
        --repo_id "${REPO_ID}" \
        --root "${DATASET_ROOT}" \
        --wan_path "${WAN_PATH}" \
        --device "${MOTUS_T5_DEVICE:-cuda}" \
        --t5_folder_name "${MOTUS_T5_FOLDER_NAME:-t5_embedding}"
    echo -e "\033[32m[Motus process_data] T5 cache done.\033[0m"
else
    echo -e "\033[33m[Motus process_data] Skipping T5 cache (enable_t5_fallback handles it at train time).\033[0m"
    echo -e "\033[33m[Motus process_data] Set MOTUS_RUN_T5_CACHE=1 (with WAN_PATH) to pre-compute it.\033[0m"
fi

echo -e "\033[32m[Motus process_data] Done.\033[0m"
