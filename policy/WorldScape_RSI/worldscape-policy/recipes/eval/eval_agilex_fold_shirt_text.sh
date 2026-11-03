#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export WSP_TASK="fold-shirt-text"
export RUN_NAME="${RUN_NAME:-baseline}"
export WSP_MODE="${WSP_MODE:-auto}"
export VISUAL_PROMPT="none"

# Manifold deployment defaults. Override AGILEX_TRANSPORT=hdf5 for replay.
export AGILEX_TRANSPORT="${AGILEX_TRANSPORT:-manifold}"
export WSP_SERVER_HOST="${WSP_SERVER_HOST:-0.0.0.0}"
export WSP_SERVER_PORT="${WSP_SERVER_PORT:-8887}"
export WSP_NODE_NAME="${WSP_NODE_NAME:-wsp}"
# shellcheck source=recipes/common/eval_checkpoint.sh
source "$ROOT/recipes/common/eval_checkpoint.sh"

export WORLDSCAPE_CHECKPOINT="$(
    wsp_resolve_eval_checkpoint \
        "$ROOT/checkpoints/${WSP_TASK}/${WSP_MODE}-${VISUAL_PROMPT}/${RUN_NAME}" \
        "${FOLD_SHIRT_TEXT_EVAL_MODEL_PATH:-}"
)"
export WSP_INSTRUCTION="${WSP_INSTRUCTION:-Fold a purple short-sleeve shirt and a cream-colored short-sleeve shirt sequentially on a white table using dual robotic arms, then stack the cream shirt on top of the purple one.}"
exec "$ROOT/recipes/common/eval_agilex.sh" "$@"
