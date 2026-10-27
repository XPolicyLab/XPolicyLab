#!/bin/bash
# Convert RoboDojo official HDF5 episodes into the rex_m1 JSON training format,
# then precompute the head-camera thumbnails read as history frames.
#
# Usage:
#   bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
#
# Following the XPolicyLab convention the four positional arguments are
# required; <ckpt_name> doubles as the task filter and <expert_data_num> caps
# the episode count (empty means every episode).
#
#   bench_name       RoboDojo (simulation)
#   ckpt_name        comma-separated task names to convert, or "all"
#   env_cfg_type     robot subdirectory under each task; RoboDojo has arx_x5 only
#   action_type      recorded for bookkeeping; the JSON always carries both
#                    end-effector and joint proprioception
#   expert_data_num  optional cap on the episodes taken from each task
#
# Every path is anchored on this script's own location, so the policy directory
# can be relocated freely. Override the defaults through the environment:
#   RAW_DATA_ROOT   HDF5 root (default: nearest data/<bench_name> above rex_m1/)
#   OUTPUT_DIR      output root        (default: rex_m1/data/<data_setting>)
#   DATA_WORKERS    worker processes   (default: auto)
#   ACTION_LENGTH   action chunk length for the statistics (default: 30)
#   BATCH_SIZE      batch_size written into the generated config (default: 8)
#   HISTORY_RES     thumbnail size, = history_res in the data config (default: 400)
#   THUMBS_STRIDE   thumbnail frame stride, = thumbs_stride (default: 5)
#   EXTRA_ARGS      extra flags forwarded verbatim to process_data.py

set -euo pipefail

usage() {
    sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^#\{1\} \{0,1\}//'
}

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
    usage >&2
    exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}

if [[ "${bench_name}" != "RoboDojo" ]]; then
    echo "[Rex_M1_preview] unsupported bench_name='${bench_name}'; expected 'RoboDojo'." >&2
    exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REX_DIR="${POLICY_DIR}/rex_m1"
PROCESS_PY="${REX_DIR}/scripts/process_data.py"

if [[ ! -f "${PROCESS_PY}" ]]; then
    echo "[Rex_M1_preview] cannot find ${PROCESS_PY}" >&2
    exit 1
fi

# XPolicyLab output convention: data/<bench>-<ckpt>-<env_cfg>-<action_type>.
# Commas in a multi-task ckpt_name become underscores so the directory name
# stays safe to pass through shell and YAML path lists.
data_setting="${bench_name}-${ckpt_name//,/_}-${env_cfg_type}-${action_type}"
output_dir=${OUTPUT_DIR:-"${REX_DIR}/data/${data_setting}"}
config_name="$(echo "${bench_name}" | tr 'A-Z' 'a-z')_${ckpt_name//,/_}_${env_cfg_type}"

args=(
    --bench-name "${bench_name}"
    --dst "${output_dir}"
    --env-cfg-type "${env_cfg_type}"
    --config-name "${config_name}"
    --action-length "${ACTION_LENGTH:-30}"
    --batch-size "${BATCH_SIZE:-8}"
)

# The checked-in configs/data/robodojo_all_arx_x5.yaml is the training recipe:
# history settings plus statistics computed by this script on the full release.
# Converting every task must not overwrite it.
if [[ -f "${REX_DIR}/configs/data/${config_name}.yaml" ]] \
        && grep -q '^ *history_frames:' "${REX_DIR}/configs/data/${config_name}.yaml"; then
    echo "[Rex_M1_preview] keeping the checked-in configs/data/${config_name}.yaml"
    args+=(--skip-stats)
fi

# "all" (or an empty ckpt_name) converts every task; anything else is a filter.
if [[ -n "${ckpt_name}" && "${ckpt_name}" != "all" ]]; then
    args+=(--tasks "${ckpt_name}")
fi
if [[ -n "${expert_data_num}" ]]; then
    args+=(--episodes-per-task "${expert_data_num}")
fi
if [[ -n "${RAW_DATA_ROOT:-}" ]]; then
    args+=(--src "${RAW_DATA_ROOT}")
fi
if [[ -n "${DATA_WORKERS:-}" ]]; then
    args+=(--workers "${DATA_WORKERS}")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # Intentionally word-split so callers can pass several flags in one string.
    # shellcheck disable=SC2206
    args+=(${EXTRA_ARGS})
fi

echo "[Rex_M1_preview] data setting : ${data_setting}"
echo "[Rex_M1_preview] output dir   : ${output_dir}"
echo "[Rex_M1_preview] data config  : configs/data/${config_name}.yaml"

cd "${REX_DIR}"
export PYTHONPATH="${REX_DIR}:${PYTHONPATH:-}"
python -u "scripts/process_data.py" "${args[@]}"

thumb_args=(--data-root "${output_dir}" --res "${HISTORY_RES:-400}" --stride "${THUMBS_STRIDE:-5}")
if [[ -n "${DATA_WORKERS:-}" ]]; then
    thumb_args+=(--workers "${DATA_WORKERS}")
fi
python -u "scripts/precompute_history_thumbs.py" "${thumb_args[@]}"

echo "[Rex_M1_preview] done. Train with:"
echo "  bash train.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${action_type} <seed> <gpu_id>"
