#!/usr/bin/env bash
# AGP stub policy server: XPolicyLab's setup_policy_server.py with model.py (it holds no model).
#
#   bash setup_eval_policy_server.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
#        <policy_gpu_id> <policy_env> <policy_server_port> [policy_server_host]
#
# policy_env: "default" = $AGP_HOME/.venv-policy, a virtualenv or a Python, or a conda environment.
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_env=${8:-default}
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
AGP_HOME="${AGP_HOME:-${HOME}/.agp-robodojo}"
policy_name="$(basename "${SCRIPT_DIR}")"

if [[ "${policy_env}" != "default" && "${policy_env}" != "uv" && "${policy_env}" != */* && ! -e "${policy_env}" ]]; then
    conda_python="$(
        set +eu
        command -v conda >/dev/null 2>&1 \
            && source "$(conda info --base)/etc/profile.d/conda.sh" >/dev/null 2>&1 \
            && conda activate "${policy_env}" >/dev/null 2>&1 && command -v python
    )" || conda_python=""
    if [[ -n "${conda_python}" ]]; then
        echo "[SERVER] policy env: conda environment ${policy_env}"
        policy_env="${conda_python}"
    else
        echo "[SERVER] policy env '${policy_env}' is not an activatable conda environment; using the default (\$AGP_HOME/.venv-policy)"
        policy_env="default"
    fi
fi
if [[ "${policy_env}" == "default" || "${policy_env}" == "uv" ]]; then
    policy_python="${AGP_HOME}/.venv-policy/bin/python"
elif [[ -x "${policy_env}/bin/python" ]]; then
    policy_python="${policy_env}/bin/python"
else
    policy_python="${policy_env}"
fi
if [[ ! -x "${policy_python}" ]]; then
    echo "[SERVER][ERROR] policy Python not found: ${policy_python}" >&2
    exit 1
fi

debug_env=()
if [[ "${EVAL_ENV_TYPE:-}" == "debug" ]]; then
    debug_env=(EVAL_ENV_TYPE=debug)
fi

cd "${XPL_ROOT}"
echo "[SERVER] policy=${policy_name} task=${task_name} port=${policy_server_port} python=${policy_python}"
exec env -i \
    HOME="${HOME}" \
    PATH="/usr/local/bin:/usr/bin:/bin" \
    LANG="${LANG:-C.UTF-8}" \
    PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}" \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    AGP_STUB_TRACE="${AGP_STUB_TRACE:-1}" \
    ${debug_env[@]+"${debug_env[@]}"} \
    "${policy_python}" -B "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${SCRIPT_DIR}/deploy.yml" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}"
