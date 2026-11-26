#!/usr/bin/env bash
# Post-train HumanWAM-Goal on RoboTwin 2.0 `clean_50` demonstrations with `goalwam_train`.
#   Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra goalwam_train args...]
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

bench_name=${1:?Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>}
ckpt_name=${2:?ckpt_name}
env_cfg_type=${3:?env_cfg_type}
action_type=${4:?action_type}
seed=${5:?seed}
gpu_id=${6:?gpu_id}
shift 6

if [[ "${bench_name}" != RoboTwin || ( "${env_cfg_type}" != aloha_agilex && "${env_cfg_type}" != arx_x5 ) || "${action_type}" != joint ]]; then
    echo "[TRAIN][ERROR] HumanWAM-Goal RoboTwin post-training requires RoboTwin / aloha_agilex / joint" >&2
    exit 2
fi
action_dim=$(bash "${HWGOAL_XPL_ROOT}/utils/get_action_dim.sh" "${env_cfg_type}" "${action_type}")
[[ "${action_dim}" == 14 ]] || { echo "[TRAIN][ERROR] Expected 14-D Aloha-AgileX joint action_dim, got ${action_dim}" >&2; exit 2; }

goalwam_select_python "${HWGOAL_ENV:-${HWGOAL_ADAPTER_DIR}/env}"
torchrun_bin="$(dirname -- "${HWGOAL_EXEC_PYTHON}")/torchrun"
nproc=$(tr ',' '\n' <<< "${gpu_id}" | grep -c .)

setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
data_dir="${HWGOAL_ADAPTER_DIR}/data/${setting}"
clean_dir="${data_dir}/clean_50"
pretrained=${HWGOAL_PRETRAINED:-${HWGOAL_ADAPTER_DIR}/checkpoints/HumanWAM-Goal-Base}
[[ -d "${clean_dir}" ]] || { echo "[TRAIN][ERROR] Missing ${clean_dir}; run process_data.sh first" >&2; exit 2; }
[[ -f "${pretrained}/transformer/config.json" ]] || { echo "[TRAIN][ERROR] Missing pretrained checkpoint ${pretrained}; run: python download_checkpoint.py --ckpt-name HumanWAM-Goal-Base" >&2; exit 2; }

goal_args=()
if [[ -d "${data_dir}/train_goal_cache" ]]; then
    goal_args=(--goal-cache "${data_dir}/train_goal_cache")
elif [[ -n "${HWGOAL_TRAIN_GOAL_CACHE:-}" ]]; then
    goal_args=(--goal-cache "${HWGOAL_TRAIN_GOAL_CACHE}")
fi

ckpt_target="${HWGOAL_ADAPTER_DIR}/checkpoints/${setting}-${seed}"
runs_dir="${ckpt_target}.runs"
mkdir -p "${runs_dir}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${torchrun_bin}" \
    --standalone \
    --nproc-per-node "${nproc}" \
    -m goalwam_train.train \
    --pretrained "${pretrained}" \
    --data "${clean_dir}" \
    "${goal_args[@]}" \
    --output "${runs_dir}" \
    --seed "${seed}" \
    "$@"

latest=$(find "${runs_dir}" -maxdepth 1 -type d -name 'checkpoint-*' | sort | tail -n 1)
[[ -n "${latest}" ]] || { echo "[TRAIN][ERROR] No checkpoint-* written under ${runs_dir}" >&2; exit 2; }
ln -sfn "${latest}" "${ckpt_target}"
echo "[TRAIN] linked ${ckpt_target} -> ${latest}"
