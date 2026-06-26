#!/bin/bash
set -e

dataset_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
expert_data_num=${5}
action_type=${6}
seed=${7}
policy_gpu_id=${8}
policy_conda_env=${9}
policy_server_port=${10}
policy_server_host=${11:-"localhost"}

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_DIR="$(cd "${CURRENT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_DIR}/utils"
IMPORT_SHIM_DIR="${XPL_DIR}/.xpl_import_shim"
mkdir -p "${IMPORT_SHIM_DIR}"
ln -sfn "${XPL_DIR}" "${IMPORT_SHIM_DIR}/XPolicyLab"

policy_name="$(basename "${CURRENT_DIR}")"
yaml_file="${CURRENT_DIR}/deploy.yml"
checkpoint_root="${CURRENT_DIR}/checkpoints/${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
qwen25_path="${QWEN25_PATH:-/mnt/xspark-data/xspark_shared/model_weights/Qwen2.5-VL-3B-Instruct}"

checkpoint_path=$(python - <<PY
from pathlib import Path

root = Path("${checkpoint_root}")
if not root.exists():
    raise FileNotFoundError(f"Checkpoint root not found: {root}")

candidates = []
for path in (root / "checkpoints").glob("global_step_*"):
    try:
        step = int(path.name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        continue
    hf_ckpt = path / "hf_ckpt"
    if hf_ckpt.exists():
        candidates.append((step, hf_ckpt))

if not candidates:
    raise FileNotFoundError(f"No checkpoints/global_step_*/hf_ckpt found under {root}")

print(max(candidates, key=lambda item: item[0])[1])
PY
)

echo "[SERVER] policy=${policy_name}, task=${task_name}, checkpoint=${checkpoint_path}, policy_server_port=${policy_server_port}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    QWEN25_PATH="${qwen25_path}" \
    PYTHONPATH="${IMPORT_SHIM_DIR}:${XPL_DIR}:${PYTHONPATH:-}" \
    python "${XPL_DIR}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            dataset_name="${dataset_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            env_cfg="${env_cfg_type}" \
            expert_data_num="${expert_data_num}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            checkpoint_path="${checkpoint_path}"
