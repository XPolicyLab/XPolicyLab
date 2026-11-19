#!/bin/bash
set -euo pipefail

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra training args]
# Output: checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/
#   SANA_ROBOT_POLICY           ar (SANA-Robot-AR, default) or base (SANA-Robot-base)
#   SANA_ROBOT_ROOT             the checkout of the SANA repository (default: Sana/ in this directory)
#   SANA_ROBOT_CONFIG           the training recipe (default: configs/sana_robot/robodojo_<policy>.yaml of the repository)
#   SANA_ROBOT_PRETRAINED       the weights a new run starts from      (default: the recipe's)
#   SANA_ROBOT_VAE              the LTX-2 video VAE directory          (default: the recipe's)
#   SANA_ROBOT_TEXT_ENCODER     the Gemma 2 2B directory               (default: the recipe's)
if [[ $# -lt 6 ]]; then
    echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra training args]" >&2
    exit 1
fi
bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
seed=${5}
gpu_id=${6}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANA_DIR="${SANA_ROBOT_ROOT:-${SCRIPT_DIR}/Sana}"
policy="${SANA_ROBOT_POLICY:-ar}"
config="${SANA_ROBOT_CONFIG:-${SANA_DIR}/configs/sana_robot/robodojo_${policy}.yaml}"
data_dir="${SCRIPT_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
ckpt_dir="${SCRIPT_DIR}/checkpoints/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"

if [[ "${policy}" != "ar" && "${policy}" != "base" ]]; then
    echo "[SANA_Robot] SANA_ROBOT_POLICY must be ar or base, got ${policy}" >&2
    exit 1
fi
if [[ ! -f "${data_dir}/sana_robot_zip/meta.json" || ! -f "${data_dir}/normalization.json" ]]; then
    echo "[SANA_Robot] no dataset at ${data_dir}; build it with" >&2
    echo "  bash process_data.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${action_type}" >&2
    exit 1
fi

arguments=(
    --work_dir "${ckpt_dir}"
    --data.path "${data_dir}/sana_robot_zip"
    --data.normalization "${data_dir}/normalization.json"
    --seed "${seed}"
)
if [[ -d "${data_dir}/latents" ]]; then
    arguments+=(--data.latent_store "${data_dir}/latents")
fi
if [[ -n "${SANA_ROBOT_PRETRAINED:-}" ]]; then
    arguments+=(--checkpoint "${SANA_ROBOT_PRETRAINED}")
fi
if [[ -n "${SANA_ROBOT_VAE:-}" ]]; then
    arguments+=(--vae "${SANA_ROBOT_VAE}")
fi
if [[ -n "${SANA_ROBOT_TEXT_ENCODER:-}" ]]; then
    arguments+=(--text_encoder "${SANA_ROBOT_TEXT_ENCODER}")
fi

# Evaluation reads the dataset's description and the normalization from the run directory.
mkdir -p "${ckpt_dir}"
cp "${data_dir}/sana_robot_zip/meta.json" "${ckpt_dir}/meta.json"
cp "${data_dir}/normalization.json" "${ckpt_dir}/normalization.json"

IFS=',' read -r -a gpu_ids <<< "${gpu_id}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export GPUS="${#gpu_ids[@]}"

cd "${SANA_DIR}"
bash "scripts/train/train_${policy}.sh" "${config}" "${arguments[@]}" "${@:7}"
