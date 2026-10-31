#!/bin/bash
# Policy server of HexaShell-1.5. Starts the local SAM3 service first when nothing answers on its port
# (it stays up for later runs; scripts/sam3_service.sh stop PORT stops it). Settings come from config.env.
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

if [[ -f "${SCRIPT_DIR}/config.env" ]]; then
    set -a
    source "${SCRIPT_DIR}/config.env"
    set +a
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}"

# The system prompt (prompts/robodojo/) and the task notes (skills/robodojo/) are not public yet: the
# contributors send them to the evaluators, to be unpacked into this directory. A missing note would
# not stop an episode, so the server does not start unless every file is there and matches
# private.sha256 (EVAL_ENV_TYPE=debug runs no agent and needs none of them).
if [[ "${EVAL_ENV_TYPE:-}" != "debug" ]] \
        && ! (cd "${SCRIPT_DIR}" && sha256sum --check --quiet --strict private.sha256); then
    echo "[SERVER] prompts/robodojo/ and skills/robodojo/ must match private.sha256: unpack the archive" \
        "the contributors sent into ${SCRIPT_DIR}" >&2
    exit 1
fi

# <policy_env>: a venv (a directory holding bin/python) or a conda environment
if [[ -x "${policy_conda_env}/bin/python" ]]; then
    export PATH="${policy_conda_env}/bin:${PATH}"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
fi

# the SAM3 service and the simulator's websocket are local: keep them off any proxy
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

sam3_port=${SAM3_PORT:-18300}
export HEXASHELL_SAM3_URL="http://127.0.0.1:${sam3_port}"
if ! curl -sf --noproxy '*' --max-time 10 "${HEXASHELL_SAM3_URL}/health" > /dev/null; then
    bash "${SCRIPT_DIR}/scripts/sam3_service.sh" start "${SAM3_GPU:-${policy_gpu_id}}" "${sam3_port}"
fi

exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
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
            action_type="${action_type}" \
            gpu_id="${policy_gpu_id}"
