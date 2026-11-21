#!/usr/bin/env bash
# Prepare RoboTwin 2.0 `clean_50` LeRobot v3.0 post-training data, normalization
# (`robot.json`), and pre-generated training goal cache.
#   Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

bench_name=${1:?Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>}
ckpt_name=${2:?ckpt_name}
env_cfg_type=${3:?env_cfg_type}
action_type=${4:?action_type}

if [[ "${bench_name}" != RoboTwin || ( "${env_cfg_type}" != aloha_agilex && "${env_cfg_type}" != arx_x5 ) || "${action_type}" != joint ]]; then
    echo "[DATA][ERROR] HumanWAM-Goal RoboTwin post-training requires RoboTwin / aloha_agilex / joint" >&2
    exit 2
fi

goalwam_select_python "${HWGOAL_ENV:-${HWGOAL_ADAPTER_DIR}/env}"
data_dir="${HWGOAL_ADAPTER_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
clean_dir="${data_dir}/clean_50"
mkdir -p "${data_dir}"

if [[ -n "${HWGOAL_CLEAN_50:-}" ]]; then
    ln -sfn "$(cd -- "${HWGOAL_CLEAN_50}" && pwd)" "${clean_dir}"
elif [[ -n "${HWGOAL_RAW_ROBOTWIN:-}" ]]; then
    raw_root="${HWGOAL_RAW_ROBOTWIN}"
    if [[ -d "${raw_root}/dataset" ]] && ! compgen -G "${raw_root}/*/aloha-agilex_clean_*.zip" >/dev/null; then
        raw_root="${raw_root}/dataset"
    fi
    "${HWGOAL_EXEC_PYTHON}" -m goalwam_train.convert \
        --raw-root "${raw_root}" \
        --output "${clean_dir}" \
        --workers "${HWGOAL_CONVERT_WORKERS:-8}"
elif [[ ! -d "${clean_dir}" ]]; then
    echo "[DATA][ERROR] Set HWGOAL_RAW_ROBOTWIN to TianxingChen/RoboTwin2.0 clean_50 archives, or HWGOAL_CLEAN_50 to converted LeRobot v3.0 data" >&2
    exit 2
fi

pretrained=${HWGOAL_PRETRAINED:-${HWGOAL_ADAPTER_DIR}/checkpoints/HumanWAM-Goal-Base}
if [[ -f "${pretrained}/robot.json" ]]; then
    "${HWGOAL_EXEC_PYTHON}" -m goalwam_train.normalization \
        --data "${clean_dir}" \
        --pretrained "${pretrained}" \
        --output "${data_dir}/robot.json"
fi

train_goals=${HWGOAL_TRAIN_GOAL_CACHE:-${HWGOAL_ADAPTER_DIR}/assets/train_goal_cache}
if [[ -d "${train_goals}" ]]; then
    manifest_sha=$("${HWGOAL_EXEC_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_goal_cache"]["manifest_sha256"])' "${HWGOAL_ADAPTER_DIR}/release.json")
    "${HWGOAL_EXEC_PYTHON}" -m goalwam_train.goal_cache verify --root "${train_goals}" --manifest-sha256 "${manifest_sha}"
    ln -sfn "$(cd -- "${train_goals}" && pwd)" "${data_dir}/train_goal_cache"
fi
echo "[DATA] ready: ${data_dir}"
