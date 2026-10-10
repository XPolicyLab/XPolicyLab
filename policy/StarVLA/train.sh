#!/bin/bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra_args...]"
    echo "Example: bash train.sh RoboDojo stack_bowls arx_x5 joint 0 0,1,2,3"
    exit 1
fi

bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
seed=${5}
gpu_id=${6}
shift 6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${SCRIPT_DIR}/source_starvla"

base_config_yaml="${SCRIPT_DIR}/xpolicy_oft_vla.yaml"
data_dir_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
run_id="${data_dir_name}-${seed}"
num_processes=$(awk -F',' '{print NF}' <<< "${gpu_id}")
data_root_dir="${STARVLA_DATA_ROOT:-${SCRIPT_DIR}/data}"
data_mix="${STARVLA_DATA_MIX:-xpolicylab_runtime}"
dataset_name="${STARVLA_XPOLICY_DATASET_NAME:-${data_dir_name}}"
config_yaml="${SCRIPT_DIR}/.generated/xpolicy_oft_vla_${run_id}.yaml"
dataset_path="${data_root_dir}/${dataset_name}"

if [[ ! -f "${dataset_path}/meta/modality.json" && -f "${dataset_path}/${env_cfg_type}/meta/modality.json" ]]; then
    dataset_name="${dataset_name}/${env_cfg_type}"
    dataset_path="${data_root_dir}/${dataset_name}"
fi

if [[ ! -f "${dataset_path}/meta/modality.json" ]]; then
    echo "[starVLA][ERROR] LeRobot dataset not found or incomplete: ${dataset_path}" >&2
    echo "[starVLA][ERROR] expected ${dataset_path}/meta/modality.json" >&2
    echo "[starVLA][ERROR] Run process_data.sh first, or set STARVLA_DATA_ROOT/STARVLA_XPOLICY_DATASET_NAME." >&2
    exit 1
fi

mkdir -p "$(dirname "${config_yaml}")"
python - "${base_config_yaml}" "${config_yaml}" "${data_root_dir}" "${data_mix}" "${run_id}" "${seed}" <<'PY'
import sys
import yaml

src, dst, data_root_dir, data_mix, run_id, seed = sys.argv[1:7]
with open(src, "r", encoding="utf-8") as fp:
    cfg = yaml.safe_load(fp)

cfg["run_id"] = run_id
cfg["seed"] = int(seed)
cfg.setdefault("datasets", {}).setdefault("vla_data", {})["data_root_dir"] = data_root_dir
cfg["datasets"]["vla_data"]["data_mix"] = data_mix

with open(dst, "w", encoding="utf-8") as fp:
    yaml.safe_dump(cfg, fp, sort_keys=False)
PY

echo "[starVLA] config_yaml=${config_yaml}"
echo "[starVLA] run_id=${run_id}"
echo "[starVLA] seed=${seed}"
echo "[starVLA] data_root_dir=${data_root_dir}"
echo "[starVLA] data_mix=${data_mix}, dataset=${dataset_name}, dataset_path=${dataset_path}"
echo "[starVLA] train_entry=starVLA/training/train_starvla.py"
echo "[starVLA] num_processes=${num_processes}, mixed_precision=bf16"

cd "${STARVLA_ROOT}"
PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}" \
STARVLA_XPOLICY_DATASET_NAME="${dataset_name}" \
STARVLA_XPOLICY_DATA_MIX="${data_mix}" \
WANDB_MODE="${WANDB_MODE:-online}" \
NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}" \
NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}" \
CUDA_VISIBLE_DEVICES="${gpu_id}" accelerate launch \
    --num_processes "${num_processes}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --run_id "${run_id}" \
    --seed "${seed}" \
    "$@"
