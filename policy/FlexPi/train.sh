#!/bin/bash
set -euo pipefail

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]
#
# Thin wrapper over flex-pi's own hydra entry (scripts/train.py via accelerate),
# pinned to the RoboDojo task preset. It does not reimplement the training
# recipe — everything not listed below comes from
# configs/task/robodojo_unified_flex_3cam_384_1e-4.yaml.
#
# Checkpoints land in checkpoints/<bench>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/,
# which is what eval.sh resolves from ckpt_name.

bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
seed=${5}
gpu_id=${6}
if [[ $# -ge 7 ]]; then
    num_gpus=${7}
elif [[ "${gpu_id}" == *,* ]]; then
    IFS=',' read -r -a gpu_ids <<< "${gpu_id}"
    num_gpus=${#gpu_ids[@]}
else
    num_gpus=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
FLEXPI_DIR="${SCRIPT_DIR}/FlexPi"
UTILS_DIR="${XPL_ROOT}/utils"

data_key="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
dataset_id="${FLEXPI_DATASET_ID:-${data_key}}"
dataset_dir="${SCRIPT_DIR}/data/${dataset_id}/lerobot"
run_dir="${SCRIPT_DIR}/checkpoints/${data_key}-${seed}"
task_config="${FLEXPI_TASK_CONFIG:-robodojo_unified_flex_3cam_384_1e-4}"
text_cache_dir="${FLEXPI_TEXT_CACHE_DIR:-${SCRIPT_DIR}/data/${dataset_id}/text_embeds_cache}"

if [[ ! -d "${dataset_dir}/meta" ]]; then
    echo "[ERROR] LeRobot dataset not found: ${dataset_dir}/meta" >&2
    echo "Run first: bash process_data.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${action_type}" >&2
    exit 1
fi

# action_dim comes from the env_cfg_type, not from a hardcoded 14, so a future
# robot with different arm/ee dims needs no edit here.
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
echo "[train] task=${task_config} action_dim=${action_dim} num_gpus=${num_gpus}"
echo "[train] dataset=${dataset_dir}"
echo "[train] output=${run_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${FLEXPI_DIR}:${FLEXPI_DIR}/src:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FLEXPI_DIR}/checkpoints}"

cd "${FLEXPI_DIR}"

overrides=(
    "task=${task_config}"
    "seed=${seed}"
    "data.train.dataset_dirs=[${dataset_dir}]"
    "data.val.dataset_dirs=[${dataset_dir}]"
    "data.train.text_embedding_cache_dir=${text_cache_dir}"
    "data.val.text_embedding_cache_dir=${text_cache_dir}"
    # null -> RobotVideoDataset computes stats and writes dataset_stats.json
    # into the run dir itself, which is where eval.sh then looks for it.
    "data.train.pretrained_norm_stats=null"
    "data.val.pretrained_norm_stats=null"
    "data.train.shape_meta.action.0.raw_shape=${action_dim}"
    "data.train.shape_meta.action.0.shape=${action_dim}"
    "data.train.shape_meta.state.0.raw_shape=${action_dim}"
    "data.train.shape_meta.state.0.shape=${action_dim}"
    "data.val.shape_meta.action.0.raw_shape=${action_dim}"
    "data.val.shape_meta.action.0.shape=${action_dim}"
    "data.val.shape_meta.state.0.raw_shape=${action_dim}"
    "data.val.shape_meta.state.0.shape=${action_dim}"
    "data.train.processor.action_output_dim=${action_dim}"
    "data.train.processor.proprio_output_dim=${action_dim}"
    "data.val.processor.action_output_dim=${action_dim}"
    "data.val.processor.proprio_output_dim=${action_dim}"
    "output_dir=${run_dir}"
)

accelerate launch \
    --config_file "${ACCELERATE_CONFIG:-scripts/accelerate_configs/accelerate_zero1_ds.yaml}" \
    --num_processes "${num_gpus}" \
    scripts/train.py "${overrides[@]}" "${@:8}"
