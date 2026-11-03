#!/usr/bin/env bash
# Perception services (SAM3 segmentation, MoGe-2 depth; RGB only) on the policy GPU, then the XPolicyLab policy
# server with KnowinAgent's no-op model.
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${XPOLICYLAB_ROOT:-${SCRIPT_DIR}/../..}" && pwd)"
if [[ ! -f "${XPL_ROOT}/setup_policy_server.py" || ! -d "${XPL_ROOT}/utils" ]]; then
    echo "[KnowinAgent][ERROR] set XPOLICYLAB_ROOT to the XPolicyLab checkout" >&2
    exit 2
fi
case "${EVAL_ENV_TYPE:-sim}" in
    sim|"")
        if [[ "${KNOWIN_AGENT_DRY_RUN:-0}" == "1" ]]; then
            echo "[KnowinAgent][ERROR] KNOWIN_AGENT_DRY_RUN=1 requires EVAL_ENV_TYPE=debug" >&2
            exit 2
        fi
        ;;
    debug) export KNOWIN_AGENT_DRY_RUN=1 ;;
    *) echo "[KnowinAgent][ERROR] supported EVAL_ENV_TYPE values: sim, debug" >&2; exit 2 ;;
esac
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

# policy_env: a Python interpreter, a venv directory, or a conda environment name.
if [[ -x "${policy_env}" && ! -d "${policy_env}" ]]; then
    python_bin="${policy_env}"
elif [[ -x "${policy_env}/bin/python" ]]; then
    python_bin="${policy_env}/bin/python"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_env}"
    python_bin="$(command -v python)"
fi

if [[ "${EVAL_ENV_TYPE:-sim}" != "debug" && "${KNOWIN_AGENT_START_SERVICES:-1}" != "0" ]]; then
    bash "${SCRIPT_DIR}/services/start_services.sh" "${policy_gpu_id}"
fi

echo "[KnowinAgent][SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}"
cd "${XPL_ROOT}"
exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
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
