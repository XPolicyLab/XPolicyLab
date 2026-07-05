#!/bin/bash
set -euo pipefail

# Convert XPolicyLab trajectory HDF5 -> Mem_0 LeRobot dataset (direct, one step).
# Run inside the Mem_0 policy conda env (needs lerobot, h5py, opencv, XPolicyLab).
#
# Usage:
#   bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <task_type>
#     task_type = M1 (single-stage) | Mn (multi-stage, needs language_annotation.json)
#
# Examples:
#   bash process_data.sh RoboDojo test_data arx_x5 3 joint M1
#   bash process_data.sh RoboDojo cover_blocks arx_x5 50 joint Mn
#
# Optional:
#   TASK_INSTRUCTION="..."   M1 instruction / Mn global task (default <ckpt_name>)
#   LANGUAGE_ANNOTATION=/path/to/language_annotation.json   (required for Mn unless
#       an existing annotation is present at xpolicylab_adapter/language_annotation/<task>/)
#   MEM0_LEGACY_PATHS=1    write to Mem_0/lerobot_datasets/ (legacy layout)
# Output: policy/Mem_0/data/<dataset>-<ckpt>-<env>-<action>-lerobot

bench_name=${1}
ckpt_name=${2}
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
    "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${expert_data_num}" "${action_type}" \
    --task_type "${task_type}" \
    "${extra[@]}"
