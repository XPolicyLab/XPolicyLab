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
model_path=${11:-${MODEL_PATH:-""}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
XPL_DIR="${ROOT_DIR}/XPolicyLab"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, host=${policy_server_host}, port=${policy_server_port}, action_dim=${action_dim}"
if [ -n "${model_path}" ]; then
    echo "[SERVER] model_path=${model_path}"
else
    echo "[SERVER] model_path=<auto-resolve from checkpoints>"
fi

conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
if [ -z "${conda_exe}" ] && [ -x "/opt/conda/bin/conda" ]; then
    conda_exe="/opt/conda/bin/conda"
fi
if [ -z "${conda_exe}" ]; then
    echo "[SERVER][ERROR] conda executable not found. Set CONDA_EXE or use /opt/conda/bin/conda." >&2
    exit 1
fi
source "$("${conda_exe}" info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

export PYTHONPATH="${SCRIPT_DIR}:${DREAMZERO_DIR:-${SCRIPT_DIR}/dreamzero}:${XPL_DIR}:${PYTHONPATH:-}"

OVERRIDES=(
    port="${policy_server_port}"
    host="${policy_server_host}"
    policy_server_host="${policy_server_host}"
    bench_name="${bench_name}"
    task_name="${task_name}"
    ckpt_name="${ckpt_name}"
    env_cfg_type="${env_cfg_type}"
    seed="${seed}"
    policy_name="${policy_name}"
    action_type="${action_type}"
    action_dim="${action_dim}"
)

if [ -n "${model_path}" ]; then
    OVERRIDES+=(model_path="${model_path}")
fi

PYTHON_ARGS=(--config_path "${yaml_file}" --overrides)
for override in "${OVERRIDES[@]}"; do
    PYTHON_ARGS+=("${override}")
done

SERVER_PY="${XPL_DIR}/setup_policy_server.py"
SERVER_ENV=(
    PYTHONUNBUFFERED=1
    PYTHONWARNINGS=ignore::UserWarning
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}"
    PYTHONPATH="${PYTHONPATH}"
)

if [[ "${policy_server_host}" == "0.0.0.0" ]]; then
    export DREAMZERO_SPS_ARGV="$(
        python -c 'import json, sys; print(json.dumps(sys.argv[1:]))' \
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
sys.argv = json.loads(os.environ['DREAMZERO_SPS_ARGV'])
runpy.run_path(sys.argv[0], run_name='__main__')
"
fi

exec env "${SERVER_ENV[@]}" python "${SERVER_PY}" "${PYTHON_ARGS[@]}"
