#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
action_type=${4:?action_type required}
seed=${5:?seed required}
gpu_id=${6:?gpu_id required}
shift 6 || true

# WUJI 58-D hybrid (tianji_wuji, action_type=ee): hands as absolute joints, wrists as
# xyz+rot6d poses in the TE2 camera frame. Layout: XPolicyLab/utils/wuji58.py.
wuji58=0
case "${env_cfg_type}:${action_type}" in
  tianji_wuji:ee) wuji58=1 ;;
  *:joint) ;;
  *)
    echo "G05 train.sh supports action_type=joint, or action_type=ee for env_cfg_type=tianji_wuji; got ${env_cfg_type} ${action_type}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G05_ROOT="${G05_ROOT:-${SCRIPT_DIR}/G05}"
PYTHON_BIN="${G05_PYTHON:-$(command -v python3)}"
OUTPUT_ROOT="${G05_OUTPUT_ROOT:-${SCRIPT_DIR}/checkpoints}"
export G05_OUTPUT_DIR="${G05_OUTPUT_DIR:-${OUTPUT_ROOT}}"
export EXP_NAME="${EXP_NAME:-${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}}"
export PYTHON_BIN

if [[ ! -d "${G05_ROOT}" ]]; then
  echo "G05_ROOT does not exist: ${G05_ROOT}" >&2
  exit 3
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Set G05_PYTHON to a valid Python executable." >&2
  exit 3
fi

case "${G05_TRAIN_BENCHMARK:-}" in
  "")
    case "${bench_name}" in
      RoboDojo|robodojo) train_benchmark="robodojo" ;;
      *) train_benchmark="${bench_name}" ;;
    esac
    ;;
  *) train_benchmark="${G05_TRAIN_BENCHMARK}" ;;
esac

case "${G05_TRAIN_MODE:-${ckpt_name}}" in
  fm|fm_only) train_mode="fm_only" ;;
  ar|ar_only) train_mode="ar_only" ;;
  ar_fm|ar-fm|ar+fm|cotrain|joint) train_mode="ar_fm" ;;
  *) train_mode="${G05_TRAIN_MODE:-ar_fm}" ;;
esac

if [[ "${train_benchmark}" != "robodojo" ]]; then
  echo "This released G05 adapter documents RoboDojo simulator training only." >&2
  echo "Set G05_TRAIN_BENCHMARK=robodojo, or use a compatible external G05 checkout via G05_ROOT." >&2
  exit 2
fi

if [[ -z "${ROBODOJO_LEROBOT_V30_ROOT:-}" ]]; then
  echo "Set ROBODOJO_LEROBOT_V30_ROOT to the RoboDojo LeRobot v3.0 dataset path." >&2
  exit 3
fi
for required in meta data videos; do
  if [[ ! -e "${ROBODOJO_LEROBOT_V30_ROOT}/${required}" ]]; then
    echo "Invalid ROBODOJO_LEROBOT_V30_ROOT: missing ${required}/ under ${ROBODOJO_LEROBOT_V30_ROOT}" >&2
    exit 3
  fi
done

if [[ "${wuji58}" == "1" ]]; then
  # The export must be the 58-D hybrid layout, stamped by the official converter.
  "${PYTHON_BIN}" - "${ROBODOJO_LEROBOT_V30_ROOT}/meta/info.json" <<'PY' || exit 3
import json, sys
path = sys.argv[1]
try:
    info = json.load(open(path, encoding="utf-8"))
except Exception as exc:
    sys.exit(f"Cannot read {path}: {exc}")
mode = info.get("repr_mode")
width = (info.get("features", {}).get("action", {}) or {}).get("shape")
if mode != "wrist_rel_te2" or list(width or []) != [58]:
    sys.exit(
        f"{path}: expected repr_mode=wrist_rel_te2 with a 58-D action column, got repr_mode={mode!r} shape={width!r}.\n"
        "Build the dataset with: python XPolicyLab/scripts/transform_lerobot_v30_format.py --representation wuji58 ..."
    )
PY
fi

if [[ "${gpu_id}" == *","* ]]; then
  IFS=',' read -r -a _gpus <<< "${gpu_id}"
  num_gpus="${#_gpus[@]}"
else
  num_gpus=1
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${G05_ROOT}:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-g05_robodojo}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"

if [[ "${WANDB_DIRECT:-1}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

cd "${G05_ROOT}"

schedule_overrides=()
has_schedule_override=0
for arg in "$@"; do
  case "${arg}" in
    model.max_steps=*|model.max_epochs=*) has_schedule_override=1 ;;
  esac
done
if [[ -n "${G05_MAX_STEPS:-}" ]]; then
  schedule_overrides+=("model.max_steps=${G05_MAX_STEPS}" "model.max_epochs=null")
  has_schedule_override=1
fi
if [[ -n "${G05_MAX_EPOCHS:-}" ]]; then
  schedule_overrides+=("model.max_epochs=${G05_MAX_EPOCHS}" "model.max_steps=null")
  has_schedule_override=1
fi
if [[ "${has_schedule_override}" != "1" ]]; then
  echo "Set G05_MAX_STEPS/G05_MAX_EPOCHS or pass model.max_steps=... / model.max_epochs=... ." >&2
  exit 3
fi

case "${train_mode}" in
  fm_only)
    action_overrides=(
      "model.model_arch.discrete_action=false"
      "model.model_arch.continuous_action=true"
      "model.model_arch.fm.joint_training=false"
      "model.model_arch.ar.ce_weight=0.0"
    )
    ;;
  ar_only)
    action_overrides=(
      "model.model_arch.discrete_action=true"
      "model.model_arch.continuous_action=false"
      "model.model_arch.fm.joint_training=false"
      "model.model_arch.fm.fm_weight=0.0"
      "model.model_arch.ar.ce_weight=1.0"
    )
    ;;
  ar_fm)
    action_overrides=(
      "model.model_arch.discrete_action=true"
      "model.model_arch.continuous_action=true"
      "model.model_arch.fm.joint_training=true"
      "model.model_arch.fm.fm_weight=1.0"
      "model.model_arch.ar.ce_weight=1.0"
    )
    ;;
  *)
    echo "Unsupported G05_TRAIN_MODE=${train_mode}" >&2
    exit 2
    ;;
esac

if [[ "${wuji58}" == "1" ]]; then
  default_task_config="robodojo_tianji_wuji_ee"
else
  default_task_config="robodojo_arx_x5_joint"
fi

exec bash scripts/run/finetune.sh \
  "${num_gpus}" \
  "${G05_TASK_CONFIG:-${default_task_config}}" \
  "seed=${seed}" \
  "exp_name=${EXP_NAME}" \
  "logger.mode=${G05_LOGGER_MODE:-online}" \
  "logger.project=${WANDB_PROJECT}" \
  "model.batch_size=${G05_BATCH_SIZE:-8}" \
  "model.grad_accumulation_steps=${G05_GRAD_ACCUM:-1}" \
  "${schedule_overrides[@]}" \
  "${action_overrides[@]}" \
  "$@"
