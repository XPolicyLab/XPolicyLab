#!/bin/bash
set -euo pipefail

# Convert XPolicyLab trajectory HDF5 -> Mem_0 LeRobot dataset (direct, one step).
# Run inside the Mem_0 policy conda env (needs lerobot, h5py, opencv, XPolicyLab).
#
# Usage:
#   bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [task_type]
#     expert_data_num: optional episode count; empty = use all episodes
#     task_type       = M1 (single-stage, default) | Mn (multi-stage, needs language_annotation.json)
#
# Examples:
#   bash process_data.sh RoboDojo test_data arx_x5 joint 3 M1
#   bash process_data.sh RoboDojo cover_blocks arx_x5 joint 50 Mn
#   bash process_data.sh RoboDojo cover_blocks arx_x5 joint "" Mn   # all episodes
#
# Optional:
#   TASK_INSTRUCTION="..."   M1 instruction / Mn global task (default <ckpt_name>)
#   LANGUAGE_ANNOTATION=/path/to/language_annotation.json   (required for Mn unless
#       an existing annotation is present at xpolicylab_adapter/language_annotation/<task>/)
# Output: policy/Mem_0/data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-lerobot

bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
expert_data_num=${5:-}
task_type=${6:-M1}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="${POLICY_DIR}/Mem_0/xpolicylab_adapter/xpolicylab_to_lerobot.py"

extra=()
[[ -n "${expert_data_num}" ]] && extra+=( --expert_data_num "${expert_data_num}" )
[[ -n "${TASK_INSTRUCTION:-}" ]] && extra+=( --instruction "${TASK_INSTRUCTION}" )
[[ -n "${LANGUAGE_ANNOTATION:-}" ]] && extra+=( --language_annotation "${LANGUAGE_ANNOTATION}" )

python "${CONVERTER}" \
    "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" \
    --task_type "${task_type}" \
    "${extra[@]}"
