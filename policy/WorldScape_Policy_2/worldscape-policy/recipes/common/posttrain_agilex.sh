#!/usr/bin/env bash
set -euo pipefail

# Common AgileX expert post-training launcher. Each task recipe selects one
# dataset/mode/visual-prompt profile from configs/posttrain/agilex.yaml.
# For multi-task generalist post-training, use posttrain_libero.sh or
# posttrain_robotwin2.sh instead.
#   WSP_TASK=<task-id>
#   RUN_NAME=<experiment-id>; defaults to baseline
#   WSP_MODE=interactive|auto
#   VISUAL_PROMPT=none|goal|demo
#   NATIVE_DATASET_NAME=
#     worldscape_hdf5_text | worldscape_hdf5_goal |
#     worldscape_hdf5_demo
# Custom datasets can be supplied through data_loader.dataset_plugin in the YAML.
# OmegaConf dot-list overrides may be appended to this script.
# Common environment/model defaults and the MLP torchrun contract live in
# launcher_common.sh. Task wrappers only select task/data.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Public recipes accept an explicit native WSP checkpoint. With no policy
# checkpoint, the builder falls back to the public base components.
export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-none}"
export WSP_TASK="${WSP_TASK:-agilex}"
export RUN_NAME="${RUN_NAME:-baseline}"
export WSP_MODE="${WSP_MODE:-interactive}"
export VISUAL_PROMPT="${VISUAL_PROMPT:-none}"
export NATIVE_DATASET_NAME="${NATIVE_DATASET_NAME:-worldscape_hdf5_text}"
export WANDB_MODE="${WANDB_MODE:-offline}"

case "$WSP_MODE" in
    interactive|auto) ;;
    *) echo "ERROR: WSP_MODE must be interactive or auto" >&2; exit 2 ;;
esac
case "$VISUAL_PROMPT" in
    none|goal|demo) ;;
    *) echo "ERROR: VISUAL_PROMPT must be none, goal, or demo" >&2; exit 2 ;;
esac

# shellcheck source=recipes/common/launcher_common.sh
source "$ROOT/recipes/common/launcher_common.sh"
wsp_distributed_train_launch "$ROOT" "$ROOT/configs/posttrain/agilex.yaml" "$@"
