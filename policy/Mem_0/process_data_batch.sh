#!/bin/bash
set -euo pipefail

# Merge every task listed in xpolicylab_adapter/task_config.json into ONE
# cotrain Mem_0 LeRobot dataset, but process M1 / Mn tasks with their own
# semantics (M1: single instruction per episode; Mn: per-segment instructions
# from xpolicylab_adapter/language_annotation/<task>/language_annotation.json).
#
# Usage:
#   bash process_data_batch.sh <dataset_name> <env_cfg_type> <expert_data_num> <action_type> [dataset_id] [task_config_path]
# Examples:
#   bash process_data_batch.sh RoboDojo_first100 arx_x5 100 joint
#   bash process_data_batch.sh RoboDojo_first100 arx_x5 100 joint my_cotrain_run
#
# Defaults:
#   dataset_id        = <dataset_name>-cotrain-<env_cfg_type>-<expert_data_num>-<action_type>
#   task_config_path  = <policy>/Mem_0/xpolicylab_adapter/task_config.json
# Output: policy/Mem_0/data/<dataset_id>-lerobot/  (legacy: Mem_0/lerobot_datasets/ with MEM0_LEGACY_PATHS=1)
#
# Fast path (no HDF5 re-encode): set ADAPT_FROM to an existing LeRobot dataset root,
# e.g. xspark_shared/lerobot/RoboDojo_sim_v21_video_abot. The source is read-only;
# videos are symlinked and only parquet/meta are rewritten for Mem_0.

if [[ $# -lt 4 ]]; then
    echo "usage: bash $(basename "${BASH_SOURCE[0]}") <dataset_name> <env_cfg_type> <expert_data_num> <action_type> [dataset_id] [task_config_path]" >&2
    exit 2
fi

dataset_name=${1}
env_cfg_type=${2}
expert_data_num=${3}
action_type=${4}
dataset_id=${5:-}
task_config_path=${6:-}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Match xpolicylab_batch_to_lerobot.py's ROOT_DIR: 3 levels up from policy/Mem_0
# (== 4 levels up from Mem_0/Mem_0) so source data resolves to <root>/data/...
ROOT_DIR="$(cd "${POLICY_DIR}/../../.." && pwd)"
ADAPTER_DIR="${POLICY_DIR}/Mem_0/xpolicylab_adapter"
ADAPTER="${ADAPTER_DIR}/adapt_shared_lerobot_to_mem0.py"
CONVERTER="${ADAPTER_DIR}/xpolicylab_batch_to_lerobot.py"
ANNOTATION_DIR="${ADAPTER_DIR}/language_annotation"
TASK_CFG="${task_config_path:-${ADAPTER_DIR}/task_config.json}"

if [[ -n "${ADAPT_FROM:-}" ]]; then
    if [[ ! -d "${ADAPT_FROM}" ]]; then
        echo "[batch] ADAPT_FROM not found: ${ADAPT_FROM}" >&2
        exit 1
    fi
    if [[ ! -f "${ADAPTER}" ]]; then
        echo "[batch] adapter not found: ${ADAPTER}" >&2
        exit 1
    fi
    default_dataset_id="${dataset_name}-cotrain-${env_cfg_type}-${expert_data_num}-${action_type}"
    resolved_dataset_id="${dataset_id:-${default_dataset_id}}"
    if [[ "${MEM0_LEGACY_PATHS:-}" == "1" ]]; then
        OUT_DIR="${POLICY_DIR}/Mem_0/lerobot_datasets/${resolved_dataset_id}"
    else
        OUT_DIR="${POLICY_DIR}/data/${resolved_dataset_id}-lerobot"
    fi
    echo "[batch] fast adapt from ${ADAPT_FROM}"
    echo "[batch] output dataset_id=${resolved_dataset_id}"
    python "${ADAPTER}" \
        --source "${ADAPT_FROM}" \
        --dest "${OUT_DIR}" \
        --annotation_root "${ANNOTATION_DIR}" \
        --task_config "${TASK_CFG}" \
        --hdf5_root "${ROOT_DIR}/data/${dataset_name}" \
        --env_cfg_type "${env_cfg_type}" \
        --workers "${ADAPT_WORKERS:-16}"
    exit 0
fi

if [[ ! -f "${TASK_CFG}" ]]; then
    echo "[batch] task_config not found: ${TASK_CFG}" >&2
    exit 1
fi
if [[ ! -f "${CONVERTER}" ]]; then
    echo "[batch] converter not found: ${CONVERTER}" >&2
    exit 1
fi

# Read M1 / Mn task lists out of task_config.json once, in a single python call.
read_lists_py=$(python - "${TASK_CFG}" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
for key in ("M1", "Mn"):
    if key not in cfg or not isinstance(cfg[key], list):
        sys.exit(f"task_config missing list field {key!r}")
print(",".join(cfg.get("M1") or []))
print(",".join(cfg.get("Mn") or []))
PY
)
m1_joined=$(printf '%s\n' "${read_lists_py}" | sed -n '1p')
mn_joined=$(printf '%s\n' "${read_lists_py}" | sed -n '2p')

IFS=',' read -r -a m1_tasks <<< "${m1_joined}"
IFS=',' read -r -a mn_tasks <<< "${mn_joined}"
# read -a always produces at least one element ("") for empty input; normalize that.
[[ ${#m1_tasks[@]} -eq 1 && -z "${m1_tasks[0]:-}" ]] && m1_tasks=()
[[ ${#mn_tasks[@]} -eq 1 && -z "${mn_tasks[0]:-}" ]] && mn_tasks=()

if [[ ${#m1_tasks[@]} -eq 0 && ${#mn_tasks[@]} -eq 0 ]]; then
    echo "[batch] task_config has no tasks under M1 or Mn" >&2
    exit 1
fi

default_dataset_id="${dataset_name}-cotrain-${env_cfg_type}-${expert_data_num}-${action_type}"
resolved_dataset_id="${dataset_id:-${default_dataset_id}}"

echo "[batch] dataset_name=${dataset_name} env_cfg_type=${env_cfg_type} expert_data_num=${expert_data_num} action_type=${action_type}"
echo "[batch] task_config=${TASK_CFG}"
echo "[batch] M1 tasks (${#m1_tasks[@]}): ${m1_tasks[*]:-<none>}"
echo "[batch] Mn tasks (${#mn_tasks[@]}): ${mn_tasks[*]:-<none>}"
echo "[batch] annotation_root=${ANNOTATION_DIR}"
echo "[batch] output dataset_id=${resolved_dataset_id}"

# Default to the fast "use preview mp4" path; override with USE_PREVIEW=0.
use_preview="${USE_PREVIEW:-1}"

# Pre-validate sources and Mn annotations so we fail before launching the heavy converter.
missing=()
for t in "${m1_tasks[@]}" "${mn_tasks[@]}"; do
    src="${ROOT_DIR}/data/${dataset_name}/${t}/${env_cfg_type}/data"
    if [[ ! -d "${src}" ]]; then
        missing+=("data: ${src}")
    fi
    if [[ "${use_preview}" == "1" ]]; then
        prev_dir="${ROOT_DIR}/data/${dataset_name}/${t}/${env_cfg_type}/preview_video"
        if [[ ! -d "${prev_dir}" ]]; then
            missing+=("preview_video: ${prev_dir}")
        fi
    fi
done
for t in "${mn_tasks[@]}"; do
    ann="${ANNOTATION_DIR}/${t}/language_annotation.json"
    if [[ ! -f "${ann}" ]]; then
        missing+=("annotation: ${ann}")
    fi
done
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "[batch] missing inputs (${#missing[@]}):" >&2
    for m in "${missing[@]}"; do echo "  - ${m}" >&2; done
    echo "[batch] tip: set USE_PREVIEW=0 to fall back to per-frame jpeg decode." >&2
    exit 1
fi

py_args=( "${dataset_name}" "${env_cfg_type}" "${expert_data_num}" "${action_type}"
          --m1_tasks "${m1_joined}"
          --mn_tasks "${mn_joined}"
          --annotation_root "${ANNOTATION_DIR}"
          --dataset_id "${resolved_dataset_id}" )
# Optional: override the LeRobot video codec via env var, e.g.
#   VCODEC=libsvtav1 bash process_data_batch.sh ...
# Default (no VCODEC set) is the converter's default 'h264', much faster than
# lerobot 0.3.3's baked-in 'libsvtav1'.
if [[ -n "${VCODEC:-}" ]]; then
    py_args+=( --vcodec "${VCODEC}" )
fi
# Disable preview-mp4 fast path with USE_PREVIEW=0 (falls back to slow per-frame
# jpeg decode + PNG dump). Default (no var or =1) uses the fast path.
if [[ "${use_preview}" != "1" ]]; then
    py_args+=( --no-use-preview )
fi

python "${CONVERTER}" "${py_args[@]}"
