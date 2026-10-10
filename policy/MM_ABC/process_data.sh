#!/bin/bash
set -euo pipefail

bench_name=${1:?usage: process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]}
ckpt_name=${2:-}
env_cfg_type=${3:-m92uw}
action_type=${4:-joint}
expert_data_num=${5:-}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMABC_ROOT=${MMABC_ROOT:-"${POLICY_DIR}/MM_ABC"}
# shellcheck source=resolve_python.sh
source "${POLICY_DIR}/resolve_python.sh"
PY=${MMABC_PYTHON:-$(resolve_python "${MMABC_ENV:-none}")}

if [[ "${env_cfg_type}" != "m92uw" || "${action_type}" != "joint" ]]; then
    echo "[MM_ABC] supports env_cfg_type=m92uw action_type=joint" >&2
    exit 1
fi

args=(--workers "${WORKERS:-64}")
[[ -n "${MOBILE_SRC:-}" ]] && args+=(--src "${MOBILE_SRC}")
[[ -n "${EPISODE_SUBDIR:-}" ]] && args+=(--episode-subdir "${EPISODE_SUBDIR}")
[[ -n "${expert_data_num}" ]] && args+=(--max-episodes "${expert_data_num}")

echo "[MM_ABC] converting ${bench_name} (${env_cfg_type}/${action_type}) with ${PY}"
cd "${MMABC_ROOT}"
OMP_NUM_THREADS=1 "${PY}" scripts/convert_mobile.py "${args[@]}"
OMP_NUM_THREADS=1 exec "${PY}" scripts/build_mobile_c80.py
