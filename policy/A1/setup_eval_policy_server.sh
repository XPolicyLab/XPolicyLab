#!/bin/bash
set -e

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
A1_DIR="${SCRIPT_DIR}/A1"
XPL_DIR="${ROOT_DIR}/XPolicyLab"
yaml_file="${SCRIPT_DIR}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
echo "[SERVER] policy=A1, task=${task_name}, port=${policy_server_port}, host=${policy_server_host}, action_dim=${action_dim}"
if [ -n "${MODEL_PATH:-}" ]; then
    echo "[SERVER] model_path=${MODEL_PATH}"
else
    echo "[SERVER] model_path=<auto-resolve from checkpoints>"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

export PYTHONPATH="${A1_DIR}:${XPL_DIR}:${PYTHONPATH:-}"
export DATA_DIR="${DATA_DIR:-$(cd "${ROOT_DIR}/.." && pwd)/models}"
export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRIPT_DIR}/.cache}"
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}"

OVERRIDES=(
    port="${policy_server_port}"
    bench_name="${bench_name}"
    task_name="${task_name}"
    ckpt_name="${ckpt_name}"
    env_cfg_type="${env_cfg_type}"
    seed="${seed}"
    policy_name="A1"
    action_type="${action_type}"
    action_dim="${action_dim}"
)

if [ -n "${MODEL_PATH:-}" ]; then
    OVERRIDES+=(model_path="${MODEL_PATH}")
fi

if [ -n "${DATA_STATS_PATH:-}" ]; then
    OVERRIDES+=(data_stats_path="${DATA_STATS_PATH}" norm_stats_json_path="${DATA_STATS_PATH}")
    echo "[SERVER] data_stats_path=${DATA_STATS_PATH}"
fi

PYTHON_ARGS=(--config_path "${yaml_file}" --overrides)
for override in "${OVERRIDES[@]}"; do
    PYTHON_ARGS+=("${override}")
done

SERVER_PY="${XPL_DIR}/setup_policy_server.py"
SERVER_ENV=(
    PYTHONWARNINGS=ignore::UserWarning
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}"
    PYTHONPATH="${PYTHONPATH}"
)

if [[ "${policy_server_host}" == "0.0.0.0" ]]; then
    export A1_SPS_ARGV="$(
        python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' \
            "${SERVER_PY}" "${PYTHON_ARGS[@]}"
    )"
    exec env "${SERVER_ENV[@]}" python -c "
import json
import os
import runpy
import sys

import client_server.tcp.model_server as model_server_module

_original_init = model_server_module.ModelServer.__init__

def _bind_all_init(self, model, host='localhost', port=None):
    _original_init(self, model, '0.0.0.0', port)

model_server_module.ModelServer.__init__ = _bind_all_init
sys.argv = json.loads(os.environ['A1_SPS_ARGV'])
runpy.run_path(sys.argv[0], run_name='__main__')
"
fi

exec env "${SERVER_ENV[@]}" python "${SERVER_PY}" "${PYTHON_ARGS[@]}"
