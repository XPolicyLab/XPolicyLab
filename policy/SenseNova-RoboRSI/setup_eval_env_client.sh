#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 10 || $# -gt 11 ]]; then
    echo "usage: bash setup_eval_env_client.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <env_gpu_id> <eval_env_conda_env> <additional_info> <port> [host]" >&2
    exit 2
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env_conda_env=$8
additional_info=$9
policy_server_port=${10}
policy_server_ip=${11:-localhost}

ROBORSI_CLIENT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBORSI_CLIENT_XPL_ROOT="$(cd "${ROBORSI_CLIENT_SCRIPT_DIR}/../.." && pwd)"
ROBORSI_CLIENT_BENCH_ROOT="$(cd "${ROBORSI_CLIENT_XPL_ROOT}/.." && pwd)"
ROBORSI_CLIENT_UTILS_DIR="${ROBORSI_CLIENT_XPL_ROOT}/utils"
ROBORSI_CLIENT_POLICY_NAME="$(basename "${ROBORSI_CLIENT_SCRIPT_DIR}")"
ROBORSI_CLIENT_YAML_FILE="${ROBORSI_CLIENT_SCRIPT_DIR}/deploy.yml"
export PYTHONPATH="${ROBORSI_CLIENT_BENCH_ROOT}:${ROBORSI_CLIENT_XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# shellcheck disable=SC1091
source "${ROBORSI_CLIENT_UTILS_DIR}/resolve_eval_env_type.sh"
eval_env_mode="$(resolve_eval_env_type)"
echo "[CLIENT] policy=${ROBORSI_CLIENT_POLICY_NAME}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

if [[ "${eval_env_mode}" == "debug" ]]; then
    debug_path="${eval_env_conda_env}"
    if [[ "${debug_path}" != /* ]]; then
        debug_path="${ROBORSI_CLIENT_SCRIPT_DIR}/${debug_path}"
    fi
    if [[ -x "${debug_path}/bin/python" ]]; then
        client_python="${debug_path}/bin/python"
    elif [[ -x "${debug_path}/.venv/bin/python" ]]; then
        client_python="${debug_path}/.venv/bin/python"
    elif [[ -f "${debug_path}" && -x "${debug_path}" ]]; then
        client_python="${debug_path}"
    elif [[ "${eval_env_conda_env}" == */* || "${eval_env_conda_env}" == .* ]]; then
        echo "[CLIENT][ERROR] Debug Python or virtualenv unavailable: ${debug_path}" >&2
        exit 1
    elif command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base)"
        # shellcheck disable=SC1091
        source "${conda_base}/etc/profile.d/conda.sh"
        conda activate "${eval_env_conda_env}"
        client_python="$(command -v python)"
    else
        echo "[CLIENT][ERROR] Could not resolve debug environment: ${eval_env_conda_env}" >&2
        exit 1
    fi
    cd "${ROBORSI_CLIENT_XPL_ROOT}"
    exec "${client_python}" "${ROBORSI_CLIENT_UTILS_DIR}/debug_env_client.py" \
        --bench_name "${bench_name}" --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" --policy_name "${ROBORSI_CLIENT_POLICY_NAME}" \
        --protocol ws --host "${policy_server_ip}" --port "${policy_server_port}" \
        --eval_batch false --eval_episode_num 1
fi

if [[ "${eval_env_mode}" == "sim" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "[CLIENT][ERROR] Simulator execution requires the benchmark's Conda environment." >&2
        exit 1
    fi
    conda_base="$(conda info --base)"
    # SDK activate.d hooks may read unset shell variables. Preserve their
    # exported SDK paths in this client process, then restore our strict mode.
    set +u
    # shellcheck disable=SC1091
    source "${conda_base}/etc/profile.d/conda.sh" >&2 || { set -u; exit 1; }
    conda activate "${eval_env_conda_env}" >&2 || { set -u; exit 1; }
    set -u
    sim_python="$(command -v python)"
    sim_prefix="$("${sim_python}" -S -c 'import os, sys; print(os.path.realpath(sys.prefix))')"
    export ROBORSI_SIM_BOOTSTRAP=1
    export ROBORSI_SIM_BENCH_ROOT="${ROBORSI_CLIENT_BENCH_ROOT}"
    export ROBORSI_SIM_PYTHON_PREFIX="${sim_prefix}"
    export ROBORSI_SIM_BOOTSTRAP_DIR="${ROBORSI_CLIENT_SCRIPT_DIR}/sim_bootstrap"
    export PYTHONPATH="${ROBORSI_SIM_BOOTSTRAP_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    # Missing native files or incompatible startup hooks fail before Kit/App.
    "${sim_python}" "${ROBORSI_CLIENT_SCRIPT_DIR}/sim_bootstrap/probe.py" --workspace "${ROBORSI_CLIENT_BENCH_ROOT}"

    # The official shared launcher activates Conda again; SDK hooks can replace
    # PYTHONPATH there. Restore our process-local hook only for the official
    # native client invocation, after every activation and on every retry.
    # All other Python commands retain the environment chosen by that launcher.
    python() {
        if [[ $# -ge 2 && "$1" == "-u" && "$2" == "src/eval_client/main.py" ]]; then
            PYTHONPATH="${ROBORSI_SIM_BOOTSTRAP_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
                command python "$@"
        else
            command python "$@"
        fi
    }
    export -f python
fi

# The benchmark's own conda environment owns simulator execution and scoring.
exec bash "${ROBORSI_CLIENT_UTILS_DIR}/setup_env_client.sh" \
    "${ROBORSI_CLIENT_UTILS_DIR}" "${ROBORSI_CLIENT_YAML_FILE}" "${eval_env_conda_env}" \
    "${policy_server_port}" "${bench_name}" "${task_name}" "${env_cfg_type}" \
    "${ROBORSI_CLIENT_POLICY_NAME}" "${additional_info}" "${ROBORSI_CLIENT_BENCH_ROOT}" "${seed}" \
    "${env_gpu_id}" "${policy_server_ip}"
