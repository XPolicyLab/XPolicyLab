#!/bin/bash
# Convert RoboDojo official HDF5 episodes into the xr1 JSON training format.
#
# Usage:
#   bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
#
# Following the XPolicyLab convention the four positional arguments are
# required; <ckpt_name> doubles as the task filter and <expert_data_num> caps
# the episode count (empty means every episode).
#
#   bench_name       RoboDojo (simulation) or RoboDojo_real (real robots)
#   ckpt_name        comma-separated task names to convert, or "all"
#   env_cfg_type     robot subdirectory under each task. Only tasks carrying
#                    this robot are converted. RoboDojo has arx_x5 only;
#                    RoboDojo_real has piper_x, piper, and arx_x5, with each
#                    task belonging to exactly one of them
#   action_type      recorded for bookkeeping; the JSON always carries both
#                    end-effector and joint proprioception
#   expert_data_num  optional cap on the episodes taken from each task
#
# Every path is anchored on this script's own location, so the policy directory
# can be relocated freely. Override the defaults through the environment:
#   RAW_DATA_ROOT   HDF5 root (default: nearest data/<bench_name> above xr1/)
#   OUTPUT_DIR      output root        (default: xiaomi_robotics_1/xr1/data/<data_setting>)
#   DATA_WORKERS    worker processes   (default: auto)
#   ACTION_LENGTH   action chunk length for the statistics (default: 30)
#   BATCH_SIZE      batch_size written into the generated config (default: 16)
#   WRIST_FRAME     wuji58 layout only: wrist frame, te2 or world (default: te2);
#                   must match wrist_frame in deploy.yml
#   EXTRA_ARGS      extra flags forwarded verbatim to process_data.py
#
# The packed layout is auto-detected: a 20-wide *_ee_joint_states (tianji_wuji,
# two Wuji hands) selects the WUJI 58-D hybrid layout, anything else the
# arx_x5 layout. Pass --layout in EXTRA_ARGS to force one.

set -euo pipefail

usage() {
    sed -n '2,34p' "${BASH_SOURCE[0]}" | sed 's/^#\{1\} \{0,1\}//'
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

case "${bench_name}" in
    RoboDojo|RoboDojo_real) ;;
    *)
        echo "[Xiaomi_Robotics_1] unsupported bench_name='${bench_name}';" \
             "expected 'RoboDojo' or 'RoboDojo_real'." >&2
        exit 1
        ;;
esac

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XR1_DIR="${POLICY_DIR}/xiaomi_robotics_1/xr1"
PROCESS_PY="${XR1_DIR}/scripts/process_data.py"

if [[ ! -f "${PROCESS_PY}" ]]; then
    echo "[Xiaomi_Robotics_1] cannot find ${PROCESS_PY}" >&2
    exit 1
fi

# XPolicyLab output convention: data/<bench>-<ckpt>-<env_cfg>-<action_type>.
# Commas in a multi-task ckpt_name become underscores so the directory name
# stays safe to pass through shell and YAML path lists.
data_setting="${bench_name}-${ckpt_name//,/_}-${env_cfg_type}-${action_type}"
output_dir=${OUTPUT_DIR:-"${XR1_DIR}/data/${data_setting}"}
# Lowercased bench prefix keeps a sim and a real run on the same task/robot from
# overwriting each other's config.
config_name="$(echo "${bench_name}" | tr 'A-Z' 'a-z')_${ckpt_name//,/_}_${env_cfg_type}"

args=(
    --bench-name "${bench_name}"
    --dst "${output_dir}"
    --env-cfg-type "${env_cfg_type}"
    --config-name "${config_name}"
    --action-length "${ACTION_LENGTH:-30}"
    --batch-size "${BATCH_SIZE:-16}"
)

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
if [[ -n "${WRIST_FRAME:-}" ]]; then
    args+=(--wrist-frame "${WRIST_FRAME}")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # Intentionally word-split so callers can pass several flags in one string.
    # shellcheck disable=SC2206
    args+=(${EXTRA_ARGS})
fi

echo "[Xiaomi_Robotics_1] data setting : ${data_setting}"
echo "[Xiaomi_Robotics_1] output dir   : ${output_dir}"
echo "[Xiaomi_Robotics_1] data config  : configs/data/${config_name}.yaml"

# Run from XR1_DIR so the video paths recorded in the JSON stay relative to it,
# matching how scripts/train.sh launches tools/train.py.
cd "${XR1_DIR}"
export PYTHONPATH="${XR1_DIR}:${PYTHONPATH:-}"
python -u "scripts/process_data.py" "${args[@]}"

echo "[Xiaomi_Robotics_1] done. Train with:"
echo "  cd ${XR1_DIR} && bash scripts/train.sh data=${config_name}"
