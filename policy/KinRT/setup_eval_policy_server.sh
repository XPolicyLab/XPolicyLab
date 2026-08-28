#!/usr/bin/env bash
set -euo pipefail
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}"

if ! command -v conda >/dev/null 2>&1; then
  for conda_candidate in \
    "${CONDA_EXE:-}" \
    /opt/conda/bin/conda \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/anaconda3/bin/conda"; do
    if [[ -n "${conda_candidate}" && -x "${conda_candidate}" ]]; then
      export PATH="$(dirname "${conda_candidate}"):${PATH}"
      break
    fi
  done
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_uv_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
YAML_PYTHON="${CONDA_BASE}/bin/python"

resolve_uv_env() {
  local raw_path=$1
  if [[ "${raw_path}" == "uv" ]]; then
    "${YAML_PYTHON}" - "${yaml_file}" "${SCRIPT_DIR}" <<'PYENV'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
script_dir = Path(sys.argv[2])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
path = Path(config["policy_uv_env_path"]).expanduser()
print((script_dir / path).resolve() if not path.is_absolute() else path)
PYENV
  else
    "${YAML_PYTHON}" - "${raw_path}" "${SCRIPT_DIR}" <<'PYENV'
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser()
script_dir = Path(sys.argv[2])
print((script_dir / path).resolve() if not path.is_absolute() else path)
PYENV
  fi
}

policy_uv_env_path="$(resolve_uv_env "${policy_uv_env}")"
if [[ ! -x "${policy_uv_env_path}/.venv/bin/python" ]]; then
  echo "[SERVER][ERROR] uv environment not found: ${policy_uv_env_path}/.venv" >&2
  echo "[SERVER][ERROR] Run: bash ${SCRIPT_DIR}/install.sh" >&2
  exit 1
fi

source "${policy_uv_env_path}/.venv/bin/activate"
PYTHON_BIN="${KINRT_PYTHON_BIN:-$(command -v python)}"
OPENPI_ROOT="${KINRT_OPENPI_ROOT:-${policy_uv_env_path}}"
OPENPI_SRC="${OPENPI_ROOT}/src"
EXTRA_PYTHONPATH="${KINRT_EXTRA_PYTHONPATH:-}"
overrides=(
  "port=${policy_server_port}"
  "host=${policy_server_host}"
  "bench_name=${bench_name}"
  "task_name=${task_name}"
  "ckpt_name=${ckpt_name}"
  "env_cfg_type=${env_cfg_type}"
  "seed=${seed}"
  "policy_name=${policy_name}"
  "action_type=${action_type}"
  "action_dim=${action_dim}"
)
if [[ -n "${KINRT_TRAIN_CONFIG_NAME:-}" ]]; then
  overrides+=("train_config_name=${KINRT_TRAIN_CONFIG_NAME}")
fi
if [[ -n "${KINRT_REPO_ID:-}" ]]; then
  overrides+=("repo_id=${KINRT_REPO_ID}")
fi
if [[ -n "${KINRT_CHECKPOINT_PATH:-}" ]]; then
  overrides+=("checkpoint_path=${KINRT_CHECKPOINT_PATH}")
fi
if [[ -n "${KINRT_CHECKPOINT_NUM:-}" ]]; then
  overrides+=("checkpoint_num=${KINRT_CHECKPOINT_NUM}")
fi
if [[ -n "${KINRT_ACTION_CHUNK_SIZE:-}" ]]; then
  overrides+=("action_chunk_size=${KINRT_ACTION_CHUNK_SIZE}")
fi
if [[ -n "${KINRT_PALIGEMMA_LORA_RANK:-}" ]]; then
  overrides+=("paligemma_lora_rank=${KINRT_PALIGEMMA_LORA_RANK}")
fi
if [[ -n "${KINRT_ACTION_EXPERT_LORA_RANK:-}" ]]; then
  overrides+=("action_expert_lora_rank=${KINRT_ACTION_EXPERT_LORA_RANK}")
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}, action_dim=${action_dim}"
echo "[SERVER] OpenPI root=${OPENPI_ROOT}"

exec env \
  PYTHONUNBUFFERED=1 \
  PYTHONWARNINGS=ignore::UserWarning \
  PYTHONPATH="${BENCH_ROOT}:${OPENPI_SRC}${EXTRA_PYTHONPATH:+:${EXTRA_PYTHONPATH}}" \
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
  "${PYTHON_BIN}" "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides "${overrides[@]}"
