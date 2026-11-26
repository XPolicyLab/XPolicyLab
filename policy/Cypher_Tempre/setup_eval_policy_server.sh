#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_uv_env=${8:-uv}
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
policy_name="${XPL_POLICY_NAME:-$(basename "${SCRIPT_DIR}")}"
yaml_file="${XPL_DEPLOY_YML:-${SCRIPT_DIR}/deploy.yml}"

# RoboDojo invokes this script from the adapter directory. Leave it before
# any helper that may start Python.
cd "${XPL_ROOT}"
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
YAML_PYTHON="${PYTHON:-python3}"
if [[ "${policy_uv_env}" == "uv" ]]; then
    policy_uv_env_path="$("${YAML_PYTHON}" - <<PY
import yaml
from pathlib import Path
script_dir = Path("${SCRIPT_DIR}")
cfg = yaml.safe_load(open("${yaml_file}", encoding="utf-8"))
path = Path(cfg["policy_uv_env_path"]).expanduser()
print((script_dir / path).resolve() if not path.is_absolute() else path)
PY
)"
else
    policy_uv_env_path="$("${YAML_PYTHON}" - <<PY
from pathlib import Path
path = Path("${policy_uv_env}").expanduser()
print((Path("${SCRIPT_DIR}") / path).resolve() if not path.is_absolute() else path)
PY
)"
fi

if [[ ! -x "${policy_uv_env_path}/.venv/bin/python" ]]; then
    echo "[SERVER][ERROR] Harness uv environment not found: ${policy_uv_env_path}/.venv" >&2
    exit 1
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}"
exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONPATH="${BENCH_ROOT}" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${policy_uv_env_path}/.venv/bin/python" "${XPL_ROOT}/setup_policy_server.py" \
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
            action_dim="${action_dim}"
