#!/usr/bin/env bash
# The RoboDojo simulator client. KnowinAgent's agent runs inside it (deploy.py), so this process also needs the KnowinAgent
# runtime environment (see README: KNOWIN_AGENT_CODEX_BIN, KNOWIN_AGENT_CODEX_HOME, model access, perception URLs).
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env=$8
additional_info=$9
policy_server_port=${10}
policy_server_ip=${11:-localhost}

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
UTILS_DIR="${XPL_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"
ROBODOJO_EVAL_ROOT="${ROBODOJO_ROOT:-${BENCH_ROOT}}"

echo "[KnowinAgent][CLIENT] task=${task_name}, server=${policy_server_ip}:${policy_server_port}"
cd "${XPL_ROOT}"

# eval_env: a venv directory (or "uv" for RoboDojo's own .venv) runs RoboDojo's eval_policy.sh directly;
# anything else is a conda environment name handled by XPolicyLab's generic client launcher.
eval_env_path="${eval_env}"
if [[ "${eval_env_path}" == "uv" ]]; then
    eval_env_path="${ROBODOJO_EVAL_ROOT}/.venv"
fi
if [[ "${EVAL_ENV_TYPE:-sim}" == "debug" ]]; then
    if [[ -x "${eval_env_path}" && ! -d "${eval_env_path}" ]]; then
        python_bin="${eval_env_path}"
    elif [[ -x "${eval_env_path}/bin/python" ]]; then
        python_bin="${eval_env_path}/bin/python"
    else
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${eval_env}"
        python_bin="$(command -v python)"
    fi
    exec env PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${python_bin}" "${UTILS_DIR}/debug_env_client.py" \
        --bench_name "${bench_name}" --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" --policy_name "${policy_name}" \
        --protocol ws --host "${policy_server_ip}" --port "${policy_server_port}" \
        --eval_batch false --eval_episode_num "${KNOWIN_AGENT_DEBUG_EPISODES:-1}"
fi

if [[ -x "${eval_env_path}/bin/python" ]]; then
    source "${eval_env_path}/bin/activate"
    export OMNI_KIT_ACCEPT_EULA=YES
    export KNOWIN_AGENT_PYTHON="${KNOWIN_AGENT_PYTHON:-${eval_env_path}/bin/python}"
    export PYTHONPATH="${BENCH_ROOT}:${ROBODOJO_EVAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    exec bash "${ROBODOJO_EVAL_ROOT}/scripts/eval_policy.sh" \
        --root_dir "${ROBODOJO_EVAL_ROOT}" \
        --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" \
        --device_id "${env_gpu_id}" \
        --policy_name "${policy_name}" \
        --host "${policy_server_ip}" \
        --port "${policy_server_port}" \
        --protocol ws \
        --eval_batch false \
        --additional_info "${additional_info}" \
        --seed "${seed}"
fi

bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" "${yaml_file}" "${eval_env}" "${policy_server_port}" "${bench_name}" "${task_name}" \
    "${env_cfg_type}" "${policy_name}" "${additional_info}" "${ROBODOJO_EVAL_ROOT}" "${seed}" "${env_gpu_id}" \
    "${policy_server_ip}"
