#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra_args...]

Post-trains WorldScape Policy 2.0 on RoboTwin 2.0 through
worldscape-policy/recipes/posttrain/posttrain_robotwin2.sh. Checkpoints are
written to checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/checkpoint-<step>/.
Pass that checkpoint-<step> directory (relative to this policy dir) as
<ckpt_name> to eval.sh. <gpu_id> may be a comma-separated list
(e.g. 0,1,2,3) and becomes CUDA_VISIBLE_DEVICES; NUM_GPUS is derived from it
unless set explicitly. Extra arguments are forwarded to the recipe as Hydra
overrides.

Required environment (defaults in parentheses):
  DATA_ROOT               RoboTwin 2.0 LeRobot v2 dataset root with WorldScape
                          native metadata (data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>
                          from process_data.sh when present)
  ZSCORE_STATS_PATH       14-dim global z-score statistics (DATA_ROOT/dataset_stats.json when present)
  PRETRAINED_MODEL_PATH   WorldScape Policy 2.0 pretrained checkpoint
                          (checkpoints/wsp_2_pretrain when present)

Optional environment (see worldscape-policy/README.md and docs/posttraining.md):
  WORLDSCAPE_POLICY_ROOT  WorldScape Policy source (default: ./worldscape-policy)
  CONDA_ENV / WSP_PYTHON  Python environment; defaults to the active conda env
  ACTION_HORIZON          24 (default) or 48
  WSP_DIFFUSION_VIEW_LAYOUT   mosaic_2x2 (default) or robotwin_concat
  MAX_STEPS, SAVE_STEPS, PER_DEVICE_TRAIN_BATCH_SIZE, LEARNING_RATE
  WANDB_ENABLED           default false
EOF
}

if [ "$#" -lt 6 ]; then
    usage >&2
    exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6
shift 6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

worldscape_root="${WORLDSCAPE_POLICY_ROOT:-${SCRIPT_DIR}/worldscape-policy}"
recipe="${worldscape_root}/recipes/posttrain/posttrain_robotwin2.sh"
if [ ! -f "${recipe}" ]; then
    echo "[WorldScape_Policy_2-train] recipe not found: ${recipe}" >&2
    echo "[WorldScape_Policy_2-train] set WORLDSCAPE_POLICY_ROOT to a WorldScape Policy checkout" >&2
    exit 1
fi

if [ "${action_type}" != "joint" ]; then
    echo "[WorldScape_Policy_2-train] only action_type=joint is supported (got ${action_type})" >&2
    exit 1
fi

# WorldScape Policy 2.0 RoboTwin checkpoints are dual-arm 14-dim (6+1+6+1).
action_dim="$(python3 - "${XPL_ROOT}/utils/robot/_robot_info.json" "${env_cfg_type}" <<'EOF'
import json, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]
print(sum(info["arm_dim"]) + sum(info["ee_dim"]))
EOF
)"
if [ "${action_dim}" != "14" ]; then
    echo "[WorldScape_Policy_2-train] env_cfg_type=${env_cfg_type} has action_dim=${action_dim}; expected 14" >&2
    exit 1
fi

data_tag="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
if [ -z "${DATA_ROOT:-}" ] && [ -d "${SCRIPT_DIR}/data/${data_tag}" ]; then
    DATA_ROOT="${SCRIPT_DIR}/data/${data_tag}"
fi
if [ -z "${ZSCORE_STATS_PATH:-}" ] && [ -n "${DATA_ROOT:-}" ] && [ -f "${DATA_ROOT}/dataset_stats.json" ]; then
    ZSCORE_STATS_PATH="${DATA_ROOT}/dataset_stats.json"
fi
: "${DATA_ROOT:?[WorldScape_Policy_2-train] set DATA_ROOT (or run process_data.sh) to the RoboTwin 2.0 LeRobot dataset root}"
: "${ZSCORE_STATS_PATH:?[WorldScape_Policy_2-train] set ZSCORE_STATS_PATH to dataset_stats.json}"
if [ -z "${PRETRAINED_MODEL_PATH:-}" ]; then
    if [ -d "${SCRIPT_DIR}/checkpoints/wsp_2_pretrain" ]; then
        PRETRAINED_MODEL_PATH="${SCRIPT_DIR}/checkpoints/wsp_2_pretrain"
    else
        echo "[WorldScape_Policy_2-train] set PRETRAINED_MODEL_PATH (or download wsp_2_pretrain into ${SCRIPT_DIR}/checkpoints/)" >&2
        exit 1
    fi
fi
for p in "${DATA_ROOT}" "${ZSCORE_STATS_PATH}" "${PRETRAINED_MODEL_PATH}"; do
    if [ ! -e "${p}" ]; then
        echo "[WorldScape_Policy_2-train] path not found: ${p}" >&2
        exit 1
    fi
done

run_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
output_dir="${SCRIPT_DIR}/checkpoints/${run_name}"
mkdir -p "${output_dir}"

echo "[WorldScape_Policy_2-train] recipe=${recipe}"
echo "[WorldScape_Policy_2-train] DATA_ROOT=${DATA_ROOT}"
echo "[WorldScape_Policy_2-train] ZSCORE_STATS_PATH=${ZSCORE_STATS_PATH}"
echo "[WorldScape_Policy_2-train] PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH}"
echo "[WorldScape_Policy_2-train] OUTPUT_DIR=${output_dir} CUDA_VISIBLE_DEVICES=${gpu_id} SEED=${seed}"

cd "${worldscape_root}"
exec env \
    DATA_ROOT="${DATA_ROOT}" \
    ZSCORE_STATS_PATH="${ZSCORE_STATS_PATH}" \
    PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH}" \
    RUN_NAME="${run_name}" \
    OUTPUT_DIR="${output_dir}" \
    WANDB_RUN_NAME="${WANDB_RUN_NAME:-${run_name}}" \
    WANDB_ENABLED="${WANDB_ENABLED:-false}" \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    SEED="${seed}" \
    ACTION_HORIZON="${ACTION_HORIZON:-24}" \
    bash "${recipe}" "$@"
