#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name required}
task_name=${2:?task_name required}
ckpt_name=${3:?ckpt_name required}
env_cfg_type=${4:?env_cfg_type required}
action_type=${5:?action_type required}
seed=${6:?seed required}
policy_gpu_id=${7:?policy_gpu_id required}
policy_env=${8:-base}
policy_server_port=${9:?policy_server_port required}
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
G05_REAL_ROOT="${G05_REAL_ROOT:-${SCRIPT_DIR}/runtime/G05}"
TOKENIZER_ROOT="${G05_REAL_TOKENIZER_ROOT:-${SCRIPT_DIR}/runtime/galaxea_tokenizer}"
DATASET_ROOT="${G05_REAL_DATASET_ROOT:-${SCRIPT_DIR}/runtime/galaxea_dataset}"
PYTHON_BIN="${G05_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${SCRIPT_DIR}/runtime/venv/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/runtime/venv/bin/python"
  elif [[ "${policy_env}" == /* && -x "${policy_env}/bin/python" ]]; then
    PYTHON_BIN="${policy_env}/bin/python"
  elif [[ -x "${policy_env}" ]]; then
    PYTHON_BIN="${policy_env}"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

[[ "${action_type}" == joint ]] || { echo "G05_real supports joint actions only" >&2; exit 2; }
[[ "${env_cfg_type}" == arx_x5 ]] || { echo "This checkpoint supports arx_x5 only" >&2; exit 2; }
[[ -f "${G05_CKPT_PATH:-}" ]] || { echo "Set G05_CKPT_PATH to arx_x5/checkpoint.pt" >&2; exit 3; }
[[ -d "${G05_REAL_ROOT}" ]] || { echo "G0.5 real runtime not found: ${G05_REAL_ROOT}" >&2; exit 3; }

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
exec env \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
  G05_REAL_ROOT="${G05_REAL_ROOT}" \
  G05_ROOT="${G05_REAL_ROOT}" \
  ROBODOJO_G05_ACTION_SOURCE=fm \
  PYTHONPATH="${G05_REAL_ROOT}/src:${TOKENIZER_ROOT}/src:${DATASET_ROOT}/src:${G05_REAL_ROOT}:${BENCH_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -u "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${SCRIPT_DIR}/deploy.yml" \
    --overrides \
      host="${policy_server_host}" port="${policy_server_port}" \
      bench_name="${bench_name}" task_name="${task_name}" ckpt_name="${ckpt_name}" \
      env_cfg_type="${env_cfg_type}" seed="${seed}" policy_name=G05_real \
      action_type=joint action_dim="${action_dim}" ckpt_path="${G05_CKPT_PATH}" \
      g05_root="${G05_REAL_ROOT}"
