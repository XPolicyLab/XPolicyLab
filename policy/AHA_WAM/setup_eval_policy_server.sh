#!/bin/bash
set -euo pipefail

dataset_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
expert_data_num=$5
action_type=$6
seed=$7
policy_gpu_id=$8
policy_conda_env=$9
policy_server_port=${10}
policy_server_host=${11:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
yaml_file="${SCRIPT_DIR}/deploy.yml"

apptainer_image="${AHA_WAM_APPTAINER_IMAGE:-/mnt/petrelfs/caijisong/shared_img/new_app.sif}"
elava_root="${AHA_WAM_ELAVA_ROOT:-${SCRIPT_DIR}/AHAWAM}"
checkpoint_path="${AHA_WAM_CHECKPOINT_PATH:-${ROOT_DIR}/XPolicyLab/checkpoint/step_002500.pt}"
dataset_stats_path="${AHA_WAM_DATASET_STATS_PATH:-${ROOT_DIR}/XPolicyLab/checkpoint/dataset_stats.json}"
diffsynth_model_base_path="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/petrelfs/caijisong/dualWAM/checkpoints}"
task_config="${AHA_WAM_TASK_CONFIG:-robodojo_local_history_updated_kv_prior_only}"
allow_dummy_policy="${AHA_WAM_ALLOW_DUMMY_POLICY:-false}"
chunks_per_video_prefill="${AHA_WAM_CHUNKS_PER_VIDEO_PREFILL:-4}"
prepend_episode_first_frame="${AHA_WAM_PREPEND_EPISODE_FIRST_FRAME:-true}"
env_cfg_root="${AHA_WAM_ENV_CFG_ROOT:-/mnt/petrelfs/caijisong/env_cfg}"

action_dim=$(python3 - "${env_cfg_root}" "${env_cfg_type}" <<'PY'
import json
import sys
from pathlib import Path

env_cfg_root = Path(sys.argv[1])
env_cfg_type = sys.argv[2]

try:
    import yaml
    with (env_cfg_root / f"{env_cfg_type}.yml").open("r", encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f) or {}
except Exception:
    env_cfg = {"config": {}}
    in_config = False
    with (env_cfg_root / f"{env_cfg_type}.yml").open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "config:":
                in_config = True
                continue
            if in_config and stripped.startswith("robot:"):
                env_cfg["config"]["robot"] = stripped.split(":", 1)[1].strip().strip("\"'")
                break

robot_name = env_cfg["config"]["robot"]
with (env_cfg_root / "robot" / "_robot_info.json").open("r", encoding="utf-8") as f:
    robot_info = json.load(f)[robot_name]
print(sum(robot_info["arm_dim"]) + sum(robot_info["ee_dim"]))
PY
)

echo -e "\033[33m[SERVER] policy=aha-wam task=${task_name} ckpt=${checkpoint_path}\033[0m"
echo -e "\033[33m[SERVER] stats=${dataset_stats_path}\033[0m"
echo -e "\033[33m[SERVER] conda=${policy_conda_env} server=${policy_server_host}:${policy_server_port}\033[0m"

read -r -d '' SERVER_BODY <<'BASH' || true
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${POLICY_CONDA_ENV}"
export CUDA_VISIBLE_DEVICES="${POLICY_GPU_ID}"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore::UserWarning
export DIFFSYNTH_MODEL_BASE_PATH="${AHA_WAM_DIFFSYNTH_MODEL_BASE_PATH}"
export PYTHONPATH="${ROOT_DIR}/XPolicyLab:${ROOT_DIR}:${ELAVA_ROOT}:${ELAVA_ROOT}/src:${PYTHONPATH:-}"

python -u "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${YAML_FILE}" \
    --overrides \
        port="${POLICY_SERVER_PORT}" \
        host="${POLICY_SERVER_HOST}" \
        dataset_name="${DATASET_NAME}" \
        task_name="${TASK_NAME}" \
        ckpt_name="${CKPT_NAME}" \
        env_cfg_type="${ENV_CFG_TYPE}" \
        env_cfg_root="${ENV_CFG_ROOT}" \
        expert_data_num="${EXPERT_DATA_NUM}" \
        seed="${SEED}" \
        policy_name="AHA_WAM" \
        action_type="${ACTION_TYPE}" \
        action_dim="${ACTION_DIM}" \
        elava_root="${ELAVA_ROOT}" \
        task_config="${TASK_CONFIG}" \
        checkpoint_path="${CHECKPOINT_PATH}" \
        dataset_stats_path="${DATASET_STATS_PATH}" \
        diffsynth_model_base_path="${AHA_WAM_DIFFSYNTH_MODEL_BASE_PATH}" \
        allow_dummy_policy="${ALLOW_DUMMY_POLICY}" \
        chunks_per_video_prefill="${CHUNKS_PER_VIDEO_PREFILL}" \
        prepend_episode_first_frame="${PREPEND_EPISODE_FIRST_FRAME}"
BASH

export ROOT_DIR
export YAML_FILE="${yaml_file}"
export ELAVA_ROOT="${elava_root}"
export POLICY_CONDA_ENV="${policy_conda_env}"
export POLICY_GPU_ID="${policy_gpu_id}"
export POLICY_SERVER_PORT="${policy_server_port}"
export POLICY_SERVER_HOST="${policy_server_host}"
export DATASET_NAME="${dataset_name}"
export TASK_NAME="${task_name}"
export CKPT_NAME="${ckpt_name}"
export ENV_CFG_TYPE="${env_cfg_type}"
export ENV_CFG_ROOT="${env_cfg_root}"
export EXPERT_DATA_NUM="${expert_data_num}"
export SEED="${seed}"
export ACTION_TYPE="${action_type}"
export ACTION_DIM="${action_dim}"
export TASK_CONFIG="${task_config}"
export CHECKPOINT_PATH="${checkpoint_path}"
export DATASET_STATS_PATH="${dataset_stats_path}"
export AHA_WAM_DIFFSYNTH_MODEL_BASE_PATH="${diffsynth_model_base_path}"
export ALLOW_DUMMY_POLICY="${allow_dummy_policy}"
export CHUNKS_PER_VIDEO_PREFILL="${chunks_per_video_prefill}"
export PREPEND_EPISODE_FIRST_FRAME="${prepend_episode_first_frame}"

if command -v apptainer >/dev/null 2>&1; then
    apptainer exec --cleanenv \
        --bind /mnt:/mnt \
        --bind /nvme/caijisong:/nvme/caijisong \
        --nv "${apptainer_image}" \
        env \
            ROOT_DIR="${ROOT_DIR}" \
            YAML_FILE="${YAML_FILE}" \
            ELAVA_ROOT="${ELAVA_ROOT}" \
            POLICY_CONDA_ENV="${POLICY_CONDA_ENV}" \
            POLICY_GPU_ID="${POLICY_GPU_ID}" \
            POLICY_SERVER_PORT="${POLICY_SERVER_PORT}" \
            POLICY_SERVER_HOST="${POLICY_SERVER_HOST}" \
            DATASET_NAME="${DATASET_NAME}" \
            TASK_NAME="${TASK_NAME}" \
            CKPT_NAME="${CKPT_NAME}" \
            ENV_CFG_TYPE="${ENV_CFG_TYPE}" \
            ENV_CFG_ROOT="${ENV_CFG_ROOT}" \
            EXPERT_DATA_NUM="${EXPERT_DATA_NUM}" \
            SEED="${SEED}" \
            ACTION_TYPE="${ACTION_TYPE}" \
            ACTION_DIM="${ACTION_DIM}" \
            TASK_CONFIG="${TASK_CONFIG}" \
            CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
            DATASET_STATS_PATH="${DATASET_STATS_PATH}" \
            AHA_WAM_DIFFSYNTH_MODEL_BASE_PATH="${AHA_WAM_DIFFSYNTH_MODEL_BASE_PATH}" \
            ALLOW_DUMMY_POLICY="${ALLOW_DUMMY_POLICY}" \
            CHUNKS_PER_VIDEO_PREFILL="${CHUNKS_PER_VIDEO_PREFILL}" \
            PREPEND_EPISODE_FIRST_FRAME="${PREPEND_EPISODE_FIRST_FRAME}" \
            bash -lc "${SERVER_BODY}"
else
    echo -e "\033[33m[SERVER] apptainer not found; falling back to local conda environment.\033[0m"
    bash -lc "${SERVER_BODY}"
fi
