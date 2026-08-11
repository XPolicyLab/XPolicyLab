#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
action_type=${4:?action_type required}
seed=${5:?seed required}
gpu_id=${6:?gpu_id required}
shift 6 || true

if [[ "${action_type}" != "joint" ]]; then
  echo "G05 train.sh currently supports action_type=joint, got ${action_type}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
G05_ROOT="${G05_ROOT:-${SCRIPT_DIR}/G05}"
PYTHON_BIN="${G05_PYTHON:-$(command -v python3)}"
TASK_CONFIG="${G05_TASK_CONFIG:-robodojo_arx_x5_joint}"
RUN_ID="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
OUTPUT_ROOT="${G05_OUTPUT_ROOT:-${SCRIPT_DIR}/checkpoints}"
export G05_OUTPUT_DIR="${G05_OUTPUT_DIR:-${OUTPUT_ROOT}}"
export EXP_NAME="${EXP_NAME:-${RUN_ID}}"

if [[ ! -d "${G05_ROOT}" ]]; then
  echo "G05 source checkout not found: ${G05_ROOT}" >&2
  exit 3
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Set G05_PYTHON to a valid Python executable." >&2
  exit 3
fi
if [[ -z "${ROBODOJO_LEROBOT_V30_ROOT:-}" ]]; then
  echo "Set ROBODOJO_LEROBOT_V30_ROOT to the RoboDojo LeRobot v3.0 dataset path." >&2
  exit 3
fi
for required in meta data videos; do
  if [[ ! -e "${ROBODOJO_LEROBOT_V30_ROOT}/${required}" ]]; then
    echo "Invalid ROBODOJO_LEROBOT_V30_ROOT: missing ${required}/" >&2
    exit 3
  fi
done

if [[ "${gpu_id}" == *","* ]]; then
  IFS=',' read -r -a _gpus <<< "${gpu_id}"
  num_gpus="${#_gpus[@]}"
else
  num_gpus=1
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${G05_ROOT}:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-csu-ai-05}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_DIRECT="${WANDB_DIRECT:-1}"
if [[ "${WANDB_DIRECT}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

cd "${G05_ROOT}"
export PYTHON_BIN
logger_overrides=(
  "logger.mode=${G05_LOGGER_MODE:-online}"
  "logger.project=${WANDB_PROJECT}"
)
if [[ -n "${WANDB_ENTITY}" ]]; then
  logger_overrides+=("logger.workspace=${WANDB_ENTITY}")
fi
exec bash scripts/run/finetune.sh \
  "${num_gpus}" \
  "${TASK_CONFIG}" \
  "seed=${seed}" \
  "${logger_overrides[@]}" \
  "model.batch_size=${G05_BATCH_SIZE:-2}" \
  "model.grad_accumulation_steps=${G05_GRAD_ACCUM:-1}" \
  "$@"
