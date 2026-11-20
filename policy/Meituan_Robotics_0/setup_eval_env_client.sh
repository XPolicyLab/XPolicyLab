#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 10 ]]; then echo "Expected the standard 10 client arguments, plus optional host." >&2; exit 2; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
resolve_python() {
    local requested="$1"
    if [[ -x "${requested}/bin/python" ]]; then
        POLICY_PYTHON="$(cd "${requested}" && pwd)/bin/python"
    elif [[ -f "${requested}" && -x "${requested}" ]]; then
        POLICY_PYTHON="${requested}"
    else
        if ! command -v conda >/dev/null 2>&1; then
            echo "Provide an environment prefix or Python executable; conda is unavailable." >&2
            return 1
        fi
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${requested}"
        POLICY_PYTHON="$(command -v python)"
    fi
    local prefix
    prefix="$(dirname "$(dirname "${POLICY_PYTHON}")")"
    export PATH="${prefix}/bin:${PATH}"
    export LD_LIBRARY_PATH="${prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}
resolve_python "$8"
export PYTHONPATH="${XPL_ROOT}:${BENCH_ROOT}"
export CUDA_VISIBLE_DEVICES="$7"
mode="${EVAL_ENV_TYPE:-sim}"
if [[ "${mode}" == debug ]]; then
    debug_client="${XPL_ROOT}/utils/debug_env_client.py"
    if [[ ! -f "${debug_client}" ]]; then debug_client="${XPL_ROOT}/debug_env_client.py"; fi
    if [[ ! -f "${debug_client}" ]]; then
        echo "Official XPolicyLab debug client is unavailable in this checkout." >&2
        exit 2
    fi
    exec "${POLICY_PYTHON}" "${debug_client}" \
        --bench_name "$1" --task_name "$2" --env_cfg_type "$4" \
        --policy_name Meituan_Robotics_0 --protocol ws --host "${11:-localhost}" \
        --port "${10}" --eval_batch true --eval_episode_num "${EVAL_EPISODES:-10}"
elif [[ "${mode}" == sim ]]; then
    export EVAL_NUM="${EVAL_NUM:-${EVAL_EPISODES:-native}}"
    exec bash "${BENCH_ROOT}/scripts/eval_policy.sh" \
        --task_name "$2" --env_cfg_type "$4" \
        --policy_name Meituan_Robotics_0 --protocol ws --host "${11:-localhost}" \
        --port "${10}" --eval_batch true --root_dir "${BENCH_ROOT}" \
        --device_id "$7" --additional_info "$9" --seed "$6"
else
    echo "Supported EVAL_ENV_TYPE values: debug, sim." >&2
    exit 2
fi
