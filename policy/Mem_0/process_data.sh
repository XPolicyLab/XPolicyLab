#!/bin/bash
set -euo pipefail

# Convert XPolicyLab trajectory HDF5 -> Mem_0 LeRobot dataset (direct, one step).
# Run inside the Mem_0 policy conda env (needs lerobot, h5py, opencv, XPolicyLab).
#
# Usage:
#   bash process_data.sh <dataset_name> <task_name> <env_cfg_type> <expert_data_num> <action_type> <task_type>
#     task_type = M1 (single-stage) | Mn (multi-stage, needs language_annotation.json)
#
# Examples:
#   bash process_data.sh RoboDojo test_data arx_x5 3 joint M1
#   bash process_data.sh RoboDojo cover_blocks arx_x5 50 joint Mn
# For Mn, first generate sub-task annotations with the VLM segmenter:
#   bash segment_data.sh RoboDojo cover_blocks arx_x5 50   # -> language_annotation.json
#
# Optional:
#   TASK_INSTRUCTION="..."   M1 instruction / Mn global task (default <task_name>)
#   LANGUAGE_ANNOTATION=/path/to/language_annotation.json   (Mn; else auto-discovered
#       at <Mem_0>/language_annotations/<dataset_name>/<task_name>/<env_cfg_type>/language_annotation.json)
# Output: Mem_0/lerobot_datasets/<dataset_name>-<task_name>-<env_cfg_type>-<expert_data_num>-<action_type>

dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
task_type=${6}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="${POLICY_DIR}/Mem_0/xpolicylab_adapter/xpolicylab_to_lerobot.py"

extra=()
[[ -n "${TASK_INSTRUCTION:-}" ]] && extra+=( --instruction "${TASK_INSTRUCTION}" )
[[ -n "${LANGUAGE_ANNOTATION:-}" ]] && extra+=( --language_annotation "${LANGUAGE_ANNOTATION}" )

python "${CONVERTER}" \
    "${dataset_name}" "${task_name}" "${env_cfg_type}" "${expert_data_num}" "${action_type}" \
    --task_type "${task_type}" \
    "${extra[@]}"
