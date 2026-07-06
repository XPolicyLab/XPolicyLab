#!/bin/bash
set -euo pipefail

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
BENCH_ROOT="${BENCH_ROOT:-$(cd "${XPL_ROOT}/.." && pwd)}"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"
eval_episode_num="${AHA_WAM_DEBUG_EVAL_EPISODE_NUM:-${DEBUG_EVAL_EPISODE_NUM:-100}}"
bench_root="${XPOLICYLAB_BENCH_ROOT:-${BENCH_ROOT}}"
env_cfg_root="${AHA_WAM_ENV_CFG_ROOT:-${BENCH_ROOT}/env_cfg}"

read eval_batch < <(python3 - "${yaml_file}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
except Exception:
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            value = value.strip().split("#", 1)[0].strip().strip("\"'")
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.lower() in {"none", "null", ""}:
                value = None
            data[key.strip()] = value

print(str(data.get("eval_batch", False)).lower())
PY
)

# shellcheck source=../../utils/resolve_eval_env_type.sh
source "${UTILS_DIR}/resolve_eval_env_type.sh"
eval_env_mode="$(resolve_eval_env_type)" || exit 1

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"
if [[ -z "${EVAL_ENV_TYPE:-}" ]]; then
    echo "[CLIENT] EVAL_ENV_TYPE=(default sim) -> ${eval_env_mode}"
else
    echo "[CLIENT] EVAL_ENV_TYPE=${EVAL_ENV_TYPE} -> ${eval_env_mode}"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate "${eval_env_conda_env}"

export CUDA_VISIBLE_DEVICES="${env_gpu_id}"
export PYTHONWARNINGS=ignore::UserWarning
export PYTHONPATH="${XPL_ROOT}:${BENCH_ROOT}:${PYTHONPATH:-}"

if [[ "${eval_env_mode}" == "debug" ]]; then
    exec python "${SCRIPT_DIR}/debug_env_client.py" \
        --bench_name "${bench_name}" \
        --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" \
        --env_cfg_root "${env_cfg_root}" \
        --host "${policy_server_ip}" \
        --port "${policy_server_port}" \
        --eval_batch "${eval_batch}" \
        --eval_episode_num "${eval_episode_num}"
fi

if [[ "${eval_env_mode}" == "sim" ]]; then
    exec bash "${BENCH_ROOT}/scripts/eval_policy.sh" \
        --bench_name "${bench_name}" \
        --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" \
        --policy_name "AHA_WAM" \
        --host "${policy_server_ip}" \
        --port "${policy_server_port}" \
        --eval_batch "${eval_batch}" \
        --root_dir "${bench_root}" \
        --device_id "${env_gpu_id}" \
        --additional_info "${additional_info}" \
        --seed "${seed}"
fi

if [[ "${eval_env_mode}" == "real_world" ]]; then
    echo "[ERROR] EVAL_ENV_TYPE=real is not available in open-source release" >&2
    exit 1
fi

echo "[ERROR] Unknown eval env mode: ${eval_env_mode}" >&2
exit 1
