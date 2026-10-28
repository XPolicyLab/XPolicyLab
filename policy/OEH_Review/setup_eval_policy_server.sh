#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_conda_env=${8}
policy_server_port=${9}
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"
# The official OpenWAM adapter's vendored source: the OEH runtime wraps XPolicyLab.policy.OpenWAM.model.Model.
OPENWAM_ROOT="${OPENWAM_ROOT:-${XPL_ROOT}/policy/OpenWAM/OpenWAM}"

echo "[SERVER] policy=${policy_name}, task=${task_name}, ckpt=${ckpt_name}, policy_server_port=${policy_server_port}"

if [[ "${bench_name}" != "RoboDojo" || "${action_type}" != "ee" ]]; then
    echo "[SERVER][ERROR] OEH_Review runs RoboDojo with action_type ee; got ${bench_name}/${action_type}" >&2
    exit 1
fi

# The policy Python is the OpenWAM policy environment (policy/OpenWAM/README.md): a conda env name, or an environment
# directory with bin/python (conda prefix or venv).
if [[ -x "${policy_conda_env}/bin/python" ]]; then
    PYTHON_BIN="${policy_conda_env}/bin/python"
else
    set +u  # conda's activation scripts may read unset variables
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
    set -u
    PYTHON_BIN="$(command -v python)"
fi

# The vendored runtime must be the generated one (runtime/MANIFEST.json), and the method configuration is frozen for
# this task on first use (checkpoint, source and kernel identities; a later launch reuses the file).
"${PYTHON_BIN}" "${SCRIPT_DIR}/tools/verify_runtime.py"
freeze_args=(--ckpt "${ckpt_name}" --task "${task_name}" --xpolicylab-root "${XPL_ROOT}")
if [[ -n "${OEH_REVIEW_OPENWAM_CKPT_DIR:-}" ]]; then
    freeze_args+=(--openwam-ckpt-dir "${OEH_REVIEW_OPENWAM_CKPT_DIR}")
fi
PYTHONPATH="${SCRIPT_DIR}/runtime:${BENCH_ROOT}:${OPENWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/tools/freeze_config.py" "${freeze_args[@]}"

# Scope env vars to this server process only; never export them in the orchestrator.
exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${SCRIPT_DIR}/runtime:${BENCH_ROOT}:${OPENWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -u "${XPL_ROOT}/setup_policy_server.py" \
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
