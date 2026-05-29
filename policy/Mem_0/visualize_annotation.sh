#!/bin/bash
set -euo pipefail

# Overlay sub-task instructions from language_annotation.json onto the per-episode
# preview MP4s, for sanity-checking VLM segmentation. Run AFTER segment_data.sh.
#
# Usage:
#   bash visualize_annotation.sh <dataset_name> <task_name> <env_cfg_type>
# Example:
#   bash visualize_annotation.sh RoboDojo swap_T arx_x5
#
# Optional:
#   ANNOTATION=/abs/path.json     override the auto-discovered language_annotation.json
#   EPISODES="0,1,2"              limit to specific episode indices
#   CAMERAS="cam_head"            comma list (default cam_head; "all" => every preview MP4)
#   OUT_DIR=/abs/path             override output dir
#   SCALE=0.5                     uniform downscale factor for the output video (e.g. 0.5 = half)
#   MAX_WIDTH=480                 cap output width in pixels (preserves aspect ratio)
# Output: XPolicyLab/policy/Mem_0/Mem_0/language_annotations/<dataset_name>/<task_name>/<env_cfg_type>/preview_video_annotated/

dataset_name=${1}
task_name=${2}
env_cfg_type=${3}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${POLICY_DIR}/Mem_0/xpolicylab_adapter/visualize_annotation.py"

extra=()
[[ -n "${ANNOTATION:-}" ]] && extra+=( --annotation "${ANNOTATION}" )
[[ -n "${EPISODES:-}" ]] && extra+=( --episodes "${EPISODES}" )
[[ -n "${CAMERAS:-}" ]] && extra+=( --cameras "${CAMERAS}" )
[[ -n "${OUT_DIR:-}" ]] && extra+=( --out_dir "${OUT_DIR}" )
[[ -n "${SCALE:-}" ]] && extra+=( --scale "${SCALE}" )
[[ -n "${MAX_WIDTH:-}" ]] && extra+=( --max_width "${MAX_WIDTH}" )

python "${SCRIPT}" \
    "${dataset_name}" "${task_name}" "${env_cfg_type}" \
    "${extra[@]}"
