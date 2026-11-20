#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 9 ]]; then echo "Expected the standard 9 server arguments, plus optional host." >&2; exit 2; fi
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
export PYTHONUNBUFFERED=1
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK TORCHELASTIC_RUN_ID
exec "${POLICY_PYTHON}" "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${SCRIPT_DIR}/deploy.yml" --overrides \
    port="$9" host="${10:-localhost}" bench_name="$1" task_name="$2" \
    ckpt_name="$3" env_cfg_type="$4" action_type="$5" seed="$6" policy_name=Meituan_Robotics_0
