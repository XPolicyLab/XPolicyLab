#!/bin/bash
set -euo pipefail

# XPolicyLab training wrapper for FlexWAM.
# Writes the run to the directory eval resolves by default:
#   <POLICY_DIR>/checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>/
#
# Usage:
#   bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]
#
# Required:
#   FLEXWAM_DATASET_DIR   LeRobot root with action.ee_pose / observation.ee_pose
# Optional:
#   FLEXWAM_MODEL_BASE_PATH   frozen VAE / UMT5 / tokenizer (checkpoints/FlexWAM-base)
#   FLEXWAM_ACTION_DIT_PATH   ActionDiT init; not inside FlexWAM-base
#   FLEXWAM_TRAIN_OVERRIDES   extra Hydra overrides (word-split)

bench_name=${1:?Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]}
ckpt_name=${2:?}
env_cfg_type=${3:?}
action_type=${4:?}
seed=${5:?}
gpu_id=${6:?}

if [[ $# -ge 7 ]]; then
    num_gpus=${7}
elif [[ "${gpu_id}" == *,* ]]; then
    IFS=',' read -r -a gpu_ids <<< "${gpu_id}"
    num_gpus=${#gpu_ids[@]}
else
    num_gpus=1
fi

if [[ "${action_type}" != "ee" ]]; then
    echo "[FlexWAM] ERROR: FlexWAM RoboDojo is an EE-space policy; action_type must be 'ee', got '${action_type}'." >&2
    exit 1
fi

case "${env_cfg_type}" in
    arx_x5) task="wan/robodojo_arx5" ;;
    piper) task="wan/robodojo_piper" ;;
    piper_x) task="wan/robodojo_piperx" ;;
    *)
        echo "[FlexWAM] ERROR: unsupported env_cfg_type '${env_cfg_type}'. Expected arx_x5, piper, or piper_x." >&2
        exit 1
        ;;
esac

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXWAM_ROOT="${FLEXWAM_ROOT:-${POLICY_DIR}/flexwam}"
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
dataset_dir="${FLEXWAM_DATASET_DIR:-${POLICY_DIR}/data/${data_setting}}"

if [[ ! -d "${FLEXWAM_ROOT}/src/flexwam" ]]; then
    echo "[FlexWAM] ERROR: missing vendored FlexWAM at ${FLEXWAM_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${dataset_dir}" ]]; then
    echo "[FlexWAM] ERROR: LeRobot dataset not found at ${dataset_dir}." >&2
    echo "[FlexWAM] Set FLEXWAM_DATASET_DIR or run process_data.sh first." >&2
    exit 1
fi

mkdir -p "${ckpt_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${FLEXWAM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${FLEXWAM_MODEL_BASE_PATH:-}" ]]; then
    export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FLEXWAM_MODEL_BASE_PATH}}"
fi
if [[ -z "${FLEXWAM_ACTION_DIT_PATH:-}" && -n "${FLEXWAM_MODEL_BASE_PATH:-}" ]]; then
    action_dit_candidate="${FLEXWAM_MODEL_BASE_PATH}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
    if [[ -f "${action_dit_candidate}" ]]; then
        export FLEXWAM_ACTION_DIT_PATH="${action_dit_candidate}"
    fi
fi

echo "[FlexWAM train] bench=${bench_name} ckpt=${ckpt_name} env=${env_cfg_type} action=${action_type} seed=${seed}"
echo "[FlexWAM train] task=${task}"
echo "[FlexWAM train] dataset_dir=${dataset_dir}"
echo "[FlexWAM train] output_path=${ckpt_dir}"
echo "[FlexWAM train] gpus=${gpu_id} nproc_per_node=${num_gpus}"

train_args=(
    "task=${task}"
    "data.roots=[${dataset_dir}]"
    "data.val_roots=[${dataset_dir}]"
    "output_dir=${ckpt_dir}"
    "runtime.seed=${seed}"
)
if [[ -n "${FLEXWAM_TRAIN_OVERRIDES:-}" ]]; then
    # shellcheck disable=SC2206
    train_args+=(${FLEXWAM_TRAIN_OVERRIDES})
fi

cd "${FLEXWAM_ROOT}"
python scripts/preprocess/precompute_text_embeds.py "${train_args[@]}"
exec bash scripts/train/train_zero1.sh "${num_gpus}" "${train_args[@]}"
