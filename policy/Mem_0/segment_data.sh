#!/bin/bash
set -euo pipefail

# Generate Mn sub-task annotations from XPolicyLab episodes with a VLM, in-project
# (adapts the Ego-X caption operator). Produces language_annotation.json that
# process_data.sh ... Mn then consumes. Run BEFORE process_data.sh for Mn tasks.
#
# Usage:
#   bash segment_data.sh <dataset_name> <task_name> <env_cfg_type> <expert_data_num>
#
# Two segmentation modes (chosen automatically):
#   * free mode: VLM proposes the sub-task list AND the boundaries.
#       GLOBAL_TASK="cover the blocks left-to-right, then uncover red, green, blue" \
#       bash segment_data.sh RoboDojo cover_blocks arx_x5 50
#   * strict template mode: a fixed ordered list of sub-task strings (with
#     possible repeats) is given by Mem_0/xpolicylab_adapter/instruction/<task_name>.json;
#     the VLM only picks the start_frame of each segment, instructions are
#     forced to the template, and adjacent identical instructions are kept as
#     distinct segments. Triggered automatically when that file exists.
#       bash segment_data.sh RoboDojo swap_T arx_x5 50
#     Override the auto-discovered path with TEMPLATE=/abs/path/to/template.json.
#
# Requires a VLM key + the openai (or httpx) client:
#   export VLM_API_PROVIDER=dashscope          # or volcengine_ark
#   export DASHSCOPE_API_KEY=...               # or ARK_API_KEY / VLM_API_KEY
#   export VLM_MODEL=qwen3.5-flash             # optional
# Optional: NUM_FRAMES (default 24), MAX_WORKERS (default 4), TEMPLATE (override).
# Output: XPolicyLab/policy/Mem_0/Mem_0/language_annotations/<dataset_name>/<task_name>/<env_cfg_type>/language_annotation.json

dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${POLICY_DIR}/Mem_0/xpolicylab_adapter/segment_language_annotation.py"

extra=()
[[ -n "${GLOBAL_TASK:-}" ]] && extra+=( --global_task "${GLOBAL_TASK}" )
[[ -n "${NUM_FRAMES:-}" ]] && extra+=( --num_frames "${NUM_FRAMES}" )
[[ -n "${MAX_WORKERS:-}" ]] && extra+=( --max_workers "${MAX_WORKERS}" )
[[ -n "${TEMPLATE:-}" ]] && extra+=( --template "${TEMPLATE}" )

python "${SCRIPT}" \
    "${dataset_name}" "${task_name}" "${env_cfg_type}" "${expert_data_num}" \
    "${extra[@]}"
