#!/usr/bin/env bash
# Standard XPolicyLab training entrypoint for Motus.
#
# Usage:
#   bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
#
# This wraps the upstream trainer (motus/train/train.py) and pins the output so that
#   checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>/.../checkpoint_step_<N>/pytorch_model/mp_rank_00_model_states.pt
# is discoverable by eval (see model.py:resolve_motus_checkpoint) WITHOUT manual symlinks.
# The eval ckpt_name is exactly this 5-tuple ckpt_setting.
#
# Optional environment overrides (all have safe defaults; no personal paths are hardcoded):
#   MOTUS_CONDA_ENV        conda env to activate (default: motus; set MOTUS_SKIP_CONDA_ACTIVATE=1 to skip)
#   MOTUS_TRAIN_CONFIG     upstream --config, relative to motus/ (default: configs/lerobot_RoboDojo_sim.yaml)
#   MOTUS_DEEPSPEED_CONFIG upstream --deepspeed (default: configs/zero2_stage2.json)
#   MOTUS_RUN_NAME         run sub-dir under the ckpt_setting dir (default: motus)
#   MOTUS_REPORT_TO        --report_to backend (default: tensorboard)
#   MOTUS_NPROC_PER_NODE   torchrun --nproc_per_node (default: number of GPUs in gpu_id)
#   MOTUS_MASTER_PORT      torchrun --master_port (default: a free port)
#   MOTUS_REPO_ID          LeRobot dataset repo_id override (rendered into a runtime config)
#   MOTUS_DATASET_ROOT / LEROBOT_DATA_ROOT   LeRobot dataset root override
set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
    exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTUS_ROOT="${SCRIPT_DIR}/motus"

ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
CKPT_DIR="${SCRIPT_DIR}/checkpoints/${ckpt_setting}"

MOTUS_CONDA_ENV="${MOTUS_CONDA_ENV:-motus}"
if [ "${MOTUS_SKIP_CONDA_ACTIVATE:-0}" != "1" ] && command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${MOTUS_CONDA_ENV}"
fi

BASE_CONFIG="${MOTUS_TRAIN_CONFIG:-configs/lerobot_RoboDojo_sim.yaml}"
DEEPSPEED_CONFIG="${MOTUS_DEEPSPEED_CONFIG:-configs/zero2_stage2.json}"
RUN_NAME="${MOTUS_RUN_NAME:-motus}"
REPORT_TO="${MOTUS_REPORT_TO:-tensorboard}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
IFS=',' read -ra _GPU_ARRAY <<< "${gpu_id}"
NPROC="${MOTUS_NPROC_PER_NODE:-${#_GPU_ARRAY[@]}}"
MASTER_PORT="${MOTUS_MASTER_PORT:-$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')}"

mkdir -p "${CKPT_DIR}"
cd "${MOTUS_ROOT}"

# Optional dataset override: render a runtime config with repo_id/root swapped in.
# Keeping the file basename stable keeps the checkpoint sub-path predictable; the eval
# resolver descends recursively regardless, so exact nesting is not load-bearing.
TRAIN_CONFIG="${BASE_CONFIG}"
_dataset_root="${MOTUS_DATASET_ROOT:-${LEROBOT_DATA_ROOT:-}}"
if [ -n "${MOTUS_REPO_ID:-}" ] || [ -n "${_dataset_root}" ]; then
    RESOLVED_CONFIG="${CKPT_DIR}/$(basename "${BASE_CONFIG}")"
    MOTUS_BASE_CONFIG="${BASE_CONFIG}" \
    MOTUS_RESOLVED_CONFIG="${RESOLVED_CONFIG}" \
    MOTUS_REPO_ID="${MOTUS_REPO_ID:-}" \
    MOTUS_DATASET_ROOT="${_dataset_root}" \
    python - <<'PY'
import os
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.environ["MOTUS_BASE_CONFIG"])
repo_id = os.environ.get("MOTUS_REPO_ID") or ""
root = os.environ.get("MOTUS_DATASET_ROOT") or ""
params = cfg.dataset.params
if repo_id:
    params.repo_id = repo_id
if root:
    # Allow LEROBOT_DATA_ROOT to be a parent dir; append repo_id if it looks like one.
    if repo_id and os.path.isdir(os.path.join(root, repo_id)):
        root = os.path.join(root, repo_id)
    params.root = root
OmegaConf.save(cfg, os.environ["MOTUS_RESOLVED_CONFIG"])
print(f"[Motus train] rendered runtime config -> {os.environ['MOTUS_RESOLVED_CONFIG']}")
PY
    TRAIN_CONFIG="${RESOLVED_CONFIG}"
fi

echo -e "\033[33m[Motus train] ckpt_setting=${ckpt_setting}\033[0m"
echo -e "\033[33m[Motus train] checkpoint_dir=${CKPT_DIR}\033[0m"
echo -e "\033[33m[Motus train] config=${TRAIN_CONFIG} run_name=${RUN_NAME} nproc=${NPROC} master_port=${MASTER_PORT}\033[0m"

torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    train/train.py \
    --config "${TRAIN_CONFIG}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --seed "${seed}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --run_name "${RUN_NAME}" \
    --report_to "${REPORT_TO}"

echo -e "\033[33m[Motus train] Done. Weights under: ${CKPT_DIR} (eval ckpt_name=${ckpt_setting})\033[0m"
