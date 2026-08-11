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
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

G05_ROOT="${G05_ROOT:-${SCRIPT_DIR}/G05}"
PYTHON_BIN="${G05_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ "${policy_env}" == /* && -x "${policy_env}/bin/python" ]]; then
    PYTHON_BIN="${policy_env}/bin/python"
  elif [[ -x "${policy_env}" ]]; then
    PYTHON_BIN="${policy_env}"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

resolve_ckpt_path() {
  local raw="$1"
  local run_dir_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
  local candidates=()

  if [[ -n "${G05_CKPT_PATH:-}" ]]; then
    candidates+=("${G05_CKPT_PATH}")
  fi
  if [[ "${raw}" == /* ]]; then
    candidates+=("${raw}")
  elif [[ "${raw}" == */* ]]; then
    candidates+=("${SCRIPT_DIR}/${raw}")
  else
    candidates+=("${SCRIPT_DIR}/checkpoints/${run_dir_name}")
    candidates+=("${SCRIPT_DIR}/checkpoints/${raw}" "${G05_ROOT}/outputs/${run_dir_name}")
  fi

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" || -d "${candidate}" ]]; then
      realpath "${candidate}"
      return 0
    fi
  done

  echo "No checkpoint found. Set G05_CKPT_PATH to a G0.5 .pt file or run directory." >&2
  echo "Tried:" >&2
  printf '  %s\n' "${candidates[@]}" >&2
  return 1
}

if [[ "${action_type}" != "joint" ]]; then
  echo "G05 adapter currently supports action_type=joint, got ${action_type}" >&2
  exit 2
fi

if [[ ! -d "${G05_ROOT}" ]]; then
  echo "G0.5 repo not found: ${G05_ROOT}" >&2
  exit 3
fi

ckpt_path="$(resolve_ckpt_path "${ckpt_name}")"

if ! action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}" 2>/dev/null); then
  echo "Could not resolve action_dim for ${env_cfg_type}; register it in XPolicyLab/utils/robot/_robot_info.json." >&2
  exit 4
fi

echo "[SERVER] policy=${policy_name} task=${task_name} host=${policy_server_host} port=${policy_server_port}"
echo "[SERVER] python=${PYTHON_BIN}"
echo "[SERVER] g05_root=${G05_ROOT}"
echo "[SERVER] ckpt_path=${ckpt_path}"
echo "[SERVER] action_dim=${action_dim}"

cd "${G05_ROOT}"

exec env \
  PYTHONUNBUFFERED=1 \
  PYTHONWARNINGS=ignore::UserWarning \
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
  PYTHONPATH="${G05_ROOT}/src:${G05_ROOT}:${BENCH_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -u "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
      host="${policy_server_host}" \
      port="${policy_server_port}" \
      bench_name="${bench_name}" \
      task_name="${task_name}" \
      ckpt_name="${ckpt_name}" \
      env_cfg_type="${env_cfg_type}" \
      seed="${seed}" \
      policy_name="${policy_name}" \
      action_type="${action_type}" \
      action_dim="${action_dim}" \
      ckpt_path="${ckpt_path}" \
      g05_root="${G05_ROOT}"
