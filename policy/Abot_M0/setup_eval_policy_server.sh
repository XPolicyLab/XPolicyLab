#!/bin/bash
set -euo pipefail
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
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}, action_dim=${action_dim}"

# region agent log
python - <<PY >/dev/null 2>&1 || true
import json, time
from pathlib import Path
log_path = Path("/personal/tianxing/RoboDojo/XPolicyLab/.cursor/debug-0684e4.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
with open(log_path, "a", encoding="utf-8") as f:
    f.write(json.dumps({"sessionId":"0684e4","runId":"pre-fix","hypothesisId":"H1,H3","location":"policy/Abot_M0/setup_eval_policy_server.sh:args","message":"parsed Abot_M0 server args","data":{"argc":${#},"bench_name":"${bench_name}","task_name":"${task_name}","ckpt_name":"${ckpt_name}","env_cfg_type":"${env_cfg_type}","action_type":"${action_type}","seed":"${seed}","policy_gpu_id":"${policy_gpu_id}","policy_conda_env":"${policy_conda_env}","policy_server_port":"${policy_server_port}","policy_server_host":"${policy_server_host}","action_dim":"${action_dim}"},"timestamp":int(time.time()*1000)}) + "\n")
PY
# endregion agent log

source "$(conda info --base)/etc/profile.d/conda.sh"
if type deactivate >/dev/null 2>&1 && [[ -n "${VIRTUAL_ENV:-}" ]]; then
    deactivate || true
fi
unset VIRTUAL_ENV
conda activate "${policy_conda_env}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
echo "[SERVER] Activating Conda environment: ${policy_conda_env} (${PYTHON_BIN})"

export PYTHONPATH="${SCRIPT_DIR}/abot_m0:${PYTHONPATH:-}"

exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${PYTHON_BIN}" "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            dataset_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}"
