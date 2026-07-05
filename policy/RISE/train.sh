#!/bin/bash
set -euo pipefail

# RISE offline training launcher for the RoboDojo LeRobot dataset.
#
# Full offline RISE flow:
#   1. advantage - compute norm, train value/advantage model, and create *_w_adv
#   2. policy    - train the final advantage-conditioned policy on existing *_w_adv
#   3. all       - run advantage -> policy
#
# Usage:
#   bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id> [advantage|policy|all] [extra args]
#
# Examples:
#   bash train.sh RoboDojo stack_bowls arx_x5 100 joint 42 0 advantage
#   bash train.sh RoboDojo stack_bowls arx_x5 100 joint 42 0 policy
#   bash train.sh RoboDojo stack_bowls arx_x5 100 joint 42 0 all

stages_regex="^(advantage|policy|all)$"
usage="Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id> [advantage|policy|all] [extra args]"

if [[ "${1:-}" =~ ${stages_regex} ]]; then
    legacy_usage="Usage: bash train.sh <advantage|policy|all> <gpu_id> <seed> [extra args]"
    stage=${1:?${legacy_usage}}
    gpu_id=${2:?${legacy_usage}}
    seed=${3:?${legacy_usage}}
    extra_args=("${@:4}")

    bench_name="${RISE_BENCH_NAME:-RoboDojo}"
    ckpt_name="${RISE_CKPT_NAME:-stack_bowls}"
    env_cfg_type="${RISE_ENV_CFG_TYPE:-arx_x5}"
    expert_data_num="${RISE_EXPERT_DATA_NUM:-100}"
    action_type="${RISE_ACTION_TYPE:-joint}"
else
    bench_name=${1:?${usage}}
    ckpt_name=${2:?${usage}}
    env_cfg_type=${3:?${usage}}
    expert_data_num=${4:?${usage}}
    action_type=${5:?${usage}}
    seed=${6:?${usage}}
    gpu_id=${7:?${usage}}
    stage=${8:-${RISE_STAGE:-policy}}
    extra_args=("${@:9}")
fi

if [[ ! "${stage}" =~ ${stages_regex} ]]; then
    echo "${usage}" >&2
    echo "[RISE] Unknown stage: ${stage}" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_DIR="${SCRIPT_DIR}/xpolicylab_adapter"
OFFLINE_DIR="${SCRIPT_DIR}/RISE/policy_and_value/policy_offline_and_value"
DEFAULT_PI05_WEIGHTS="${SCRIPT_DIR}/weights/pi05_base_pytorch"
DEFAULT_RAW_DATASET_LINK="${SCRIPT_DIR}/data/RoboDojo_sim_v21_video_abot-lerobot"

source "${ADAPTER_DIR}/_artifact_paths.sh"

STANDARD_CKPT_DIR="$(xpolicylab_resolve_ckpt_dir "${SCRIPT_DIR}" "${bench_name}" "${ckpt_name}" \
    "${env_cfg_type}" "${action_type}" "${seed}" "${expert_data_num}")"

if [[ -n "${RISE_RAW_DATASET:-}" ]]; then
    RAW_DATASET_LINK="${RISE_RAW_DATASET}"
elif [[ -d "$(xpolicylab_resolve_dataset_dir "${SCRIPT_DIR}" "${bench_name}" "${ckpt_name}" \
    "${env_cfg_type}" "${action_type}" "${expert_data_num}")" || \
      -L "$(xpolicylab_resolve_dataset_dir "${SCRIPT_DIR}" "${bench_name}" "${ckpt_name}" \
    "${env_cfg_type}" "${action_type}" "${expert_data_num}")" ]]; then
    RAW_DATASET_LINK="$(xpolicylab_resolve_dataset_dir "${SCRIPT_DIR}" "${bench_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${expert_data_num}")"
elif [[ -e "${DEFAULT_RAW_DATASET_LINK}" ]]; then
    RAW_DATASET_LINK="${DEFAULT_RAW_DATASET_LINK}"
else
    RAW_DATASET_LINK="$(xpolicylab_resolve_dataset_dir "${SCRIPT_DIR}" "${bench_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${expert_data_num}")"
fi

RAW_DATASET="$(readlink -f "${RAW_DATASET_LINK}")"
ADV_DATASET="${RAW_DATASET}_w_adv"
RAW_ASSET_ID="$(basename "${RAW_DATASET}")"
ADV_ASSET_ID="$(basename "${ADV_DATASET}")"

RAW_NORM_DIR="${OFFLINE_DIR}/data/norms/${RAW_ASSET_ID}"
ADV_NORM_DIR="${OFFLINE_DIR}/data/norms/${ADV_ASSET_ID}"
RAW_NORM_PATH="${RAW_NORM_DIR}/norm_stats.json"
ADV_NORM_PATH="${ADV_NORM_DIR}/norm_stats.json"
VALUE_CKPT_ROOT="${STANDARD_CKPT_DIR}/value_release/value_release"

PRECOMPUTED_RAW_NORM="${SCRIPT_DIR}/RISE/assets/norm_stats.json"

PI05_PYTORCH_WEIGHT_PATH="${RISE_PYTORCH_WEIGHT_PATH:-${DEFAULT_PI05_WEIGHTS}}"

resolve_rise_executables() {
    local env_bin=""
    if [[ -n "${RISE_CONDA_ENV:-}" ]]; then
        env_bin="${RISE_CONDA_ENV}/bin"
    elif [[ -n "${CONDA_PREFIX:-}" ]]; then
        env_bin="${CONDA_PREFIX}/bin"
    fi

    if [[ -z "${RISE_PYTHON:-}" ]]; then
        if [[ -n "${env_bin}" && -x "${env_bin}/python" ]]; then
            RISE_PYTHON="${env_bin}/python"
        else
            RISE_PYTHON="$(command -v python)"
        fi
    fi

    if [[ -z "${RISE_TORCHRUN:-}" ]]; then
        if [[ -n "${env_bin}" && -x "${env_bin}/torchrun" ]]; then
            RISE_TORCHRUN="${env_bin}/torchrun"
        else
            RISE_TORCHRUN="$(command -v torchrun)"
        fi
    fi

    if [[ -n "${env_bin}" ]]; then
        export PATH="${env_bin}:${PATH}"
    fi
}

if [[ -n "${RISE_NGPUS_PER_NODE:-}" ]]; then
    ngpus_per_node="${RISE_NGPUS_PER_NODE}"
else
    gpu_list="${gpu_id//,/ }"
    ngpus_per_node=$(wc -w <<< "${gpu_list}")
fi

resolve_rise_executables
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export WANDB_MODE=offline
export PYTHONPATH="${OFFLINE_DIR}/src:${PYTHONPATH:-}"
export RISE_LEROBOT_LAYOUT=robodojo
export RISE_VIDEO_BACKEND=pyav
export RISE_XPOLICYLAB_SEED="${seed}"
export RISE_DEFAULT_PROMPT="stack the bowls"
export RISE_PYTORCH_WEIGHT_PATH="${PI05_PYTORCH_WEIGHT_PATH}"

require_dataset() {
    local dataset_path="$1"
    if [[ ! -d "${dataset_path}/meta" || ! -d "${dataset_path}/data" ]]; then
        echo "[RISE] Missing or invalid LeRobot dataset: ${dataset_path}" >&2
        exit 1
    fi
}

require_pi05_weights() {
    if [[ ! -f "${RISE_PYTORCH_WEIGHT_PATH}/model.safetensors" && ! -f "${RISE_PYTORCH_WEIGHT_PATH}/model.pt" ]]; then
        echo "[RISE] Missing Pi0.5 PyTorch weights: ${RISE_PYTORCH_WEIGHT_PATH}" >&2
        echo "[RISE] Expected model.safetensors or model.pt. See INSTALLATION.md for download/conversion." >&2
        exit 1
    fi
}

require_rise_python() {
    resolve_rise_executables
    if [[ -z "${RISE_PYTHON:-}" || ! -x "${RISE_PYTHON}" ]]; then
        echo "[RISE] Python not found. Activate the RISE conda env, or set RISE_CONDA_ENV / RISE_PYTHON." >&2
        exit 1
    fi
    if [[ -z "${RISE_TORCHRUN:-}" || ! -x "${RISE_TORCHRUN}" ]]; then
        echo "[RISE] torchrun not found. Activate the RISE conda env, or set RISE_CONDA_ENV / RISE_TORCHRUN." >&2
        exit 1
    fi
    "${RISE_PYTHON}" -c "import jax" >/dev/null
}

install_precomputed_raw_norm_if_available() {
    if [[ ! -f "${RAW_NORM_PATH}" && -f "${PRECOMPUTED_RAW_NORM}" ]]; then
        mkdir -p "${RAW_NORM_DIR}"
        cp "${PRECOMPUTED_RAW_NORM}" "${RAW_NORM_PATH}"
        echo "[RISE] Installed precomputed raw norm stats: ${RAW_NORM_PATH}"
    fi
}

require_raw_norm() {
    install_precomputed_raw_norm_if_available
    if [[ ! -f "${RAW_NORM_PATH}" ]]; then
        echo "[RISE] Missing raw norm stats: ${RAW_NORM_PATH}" >&2
        echo "[RISE] Run stage 'advantage' or 'all' to create it." >&2
        exit 1
    fi
}

require_adv_dataset() {
    if [[ ! -d "${ADV_DATASET}/meta" || ! -d "${ADV_DATASET}/data" ]]; then
        echo "[RISE] Missing labeled advantage dataset: ${ADV_DATASET}" >&2
        echo "[RISE] Run stage 'advantage' first, or set RISE_RAW_DATASET to a dataset whose *_w_adv sibling already exists." >&2
        exit 1
    fi
}

ensure_adv_norm() {
    if [[ ! -f "${ADV_NORM_PATH}" ]]; then
        require_raw_norm
        mkdir -p "${ADV_NORM_DIR}"
        cp "${RAW_NORM_PATH}" "${ADV_NORM_PATH}"
        echo "[RISE] Reused raw state/action norm stats for advantage dataset: ${ADV_NORM_PATH}"
    fi
}

run_upstream_train() {
    local config_name="$1"
    shift
    mkdir -p "${STANDARD_CKPT_DIR}"
    bash train.sh "${config_name}" "${ngpus_per_node}" \
        --checkpoint-base-dir "${STANDARD_CKPT_DIR}" \
        --seed "${seed}" \
        --pytorch-weight-path "${RISE_PYTORCH_WEIGHT_PATH}" \
        "$@"
}

latest_checkpoint_dir() {
    local checkpoint_root="$1"
    if [[ ! -d "${checkpoint_root}" ]]; then
        return 1
    fi

    local latest_step
    latest_step=$(
        "${RISE_PYTHON}" - "${checkpoint_root}" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
steps = sorted(
    int(path.name)
    for path in root.iterdir()
    if path.is_dir() and path.name.isdigit()
)
if not steps:
    raise SystemExit(1)
print(steps[-1])
PY
    )
    echo "${checkpoint_root}/${latest_step}"
}

cd "${OFFLINE_DIR}"
require_rise_python

echo "[RISE] stage=${stage}"
echo "[RISE] seed=${seed}"
echo "[RISE] raw_dataset=${RAW_DATASET}"
echo "[RISE] adv_dataset=${ADV_DATASET}"
echo "[RISE] standard_ckpt_dir=${STANDARD_CKPT_DIR}"

case "${stage}" in
    advantage)
        require_dataset "${RAW_DATASET}"
        export RISE_XPOLICYLAB_DATASET="${RAW_DATASET}"

        if [[ ! -f "${RAW_NORM_PATH}" ]]; then
            install_precomputed_raw_norm_if_available
        fi
        if [[ ! -f "${RAW_NORM_PATH}" ]]; then
            echo "[RISE] Step 1/3: computing norm stats for raw dataset"
            "${RISE_PYTHON}" scripts/compute_norm_stats_fast.py --config-name Compute_norm
        else
            echo "[RISE] Step 1/3: raw norm stats already exist: ${RAW_NORM_PATH}"
        fi

        require_raw_norm
        require_pi05_weights
        echo "[RISE] Step 2/3: training value/advantage model on raw dataset: ${RISE_XPOLICYLAB_DATASET}"
        run_upstream_train value_release "${extra_args[@]}"

        value_ckpt_dir="$(latest_checkpoint_dir "${VALUE_CKPT_ROOT}")"
        echo "[RISE] Step 3/3: labeling data with value checkpoint: ${value_ckpt_dir}"
        export RISE_XPOLICYLAB_DATASET="${RAW_DATASET}"
        "${RISE_PYTHON}" examples/label_frame_value.py \
            --config_name vis_value_release_joint_T \
            --ckpt_dir "${value_ckpt_dir}" \
            --split all \
            --no-with_vis
        ensure_adv_norm
        echo "[RISE] Labeled dataset ready: ${ADV_DATASET}"
        ;;

    policy)
        require_adv_dataset
        ensure_adv_norm
        require_pi05_weights
        export RISE_XPOLICYLAB_DATASET="${ADV_DATASET}"
        echo "[RISE] Training advantage-conditioned policy on: ${RISE_XPOLICYLAB_DATASET}"
        run_upstream_train Policy_offline_release "${extra_args[@]}"
        ;;

    all)
        require_dataset "${RAW_DATASET}"
        require_pi05_weights

        export RISE_XPOLICYLAB_DATASET="${RAW_DATASET}"
        if [[ ! -f "${RAW_NORM_PATH}" ]]; then
            install_precomputed_raw_norm_if_available
        fi
        if [[ ! -f "${RAW_NORM_PATH}" ]]; then
            echo "[RISE] Step 1/4: computing norm stats for raw dataset"
            "${RISE_PYTHON}" scripts/compute_norm_stats_fast.py --config-name Compute_norm
        else
            echo "[RISE] Step 1/4: raw norm stats already exist: ${RAW_NORM_PATH}"
        fi

        echo "[RISE] Step 2/4: training value/advantage model"
        run_upstream_train value_release "${extra_args[@]}"

        value_ckpt_dir="$(latest_checkpoint_dir "${VALUE_CKPT_ROOT}")"
        echo "[RISE] Step 3/4: labeling data with value checkpoint: ${value_ckpt_dir}"
        export RISE_XPOLICYLAB_DATASET="${RAW_DATASET}"
        "${RISE_PYTHON}" examples/label_frame_value.py \
            --config_name vis_value_release_joint_T \
            --ckpt_dir "${value_ckpt_dir}" \
            --split all \
            --no-with_vis
        ensure_adv_norm

        require_adv_dataset
        export RISE_XPOLICYLAB_DATASET="${ADV_DATASET}"
        echo "[RISE] Step 4/4: training advantage-conditioned policy"
        run_upstream_train Policy_offline_release "${extra_args[@]}"
        ;;

    *)
        echo "${usage}" >&2
        exit 1
        ;;
esac
