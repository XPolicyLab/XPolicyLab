#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export WSP_TASK="shell-game-demo"
export RUN_NAME="${RUN_NAME:-baseline}"
export WSP_MODE="${WSP_MODE:-interactive}"
export VISUAL_PROMPT="demo"

# Manifold deployment defaults. Override AGILEX_TRANSPORT=hdf5 for replay.
export AGILEX_TRANSPORT="${AGILEX_TRANSPORT:-manifold}"
export WSP_SERVER_HOST="${WSP_SERVER_HOST:-0.0.0.0}"
export WSP_SERVER_PORT="${WSP_SERVER_PORT:-11451}"
export WSP_NODE_NAME="${WSP_NODE_NAME:-wsp}"

# shellcheck source=recipes/common/eval_checkpoint.sh
source "$ROOT/recipes/common/eval_checkpoint.sh"
export WORLDSCAPE_CHECKPOINT="$(
    wsp_resolve_eval_checkpoint \
        "$ROOT/checkpoints/${WSP_TASK}/${WSP_MODE}-${VISUAL_PROMPT}/${RUN_NAME}" \
        "${SHELL_GAME_DEMO_EVAL_MODEL_PATH:-}"
)"
export WSP_INSTRUCTION="${WSP_INSTRUCTION:-Complete the demonstrated shell-game task.}"
exec "$ROOT/recipes/common/eval_agilex.sh" "$@"
