#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${WORLDSCAPE_CHECKPOINT:?The outer AgileX eval recipe must set its task-specific model}"

if [[ -z "${WSP_PYTHON:-}" ]]; then
    if [[ -n "${CONDA_ENV:-}" ]]; then
        WSP_PYTHON="$CONDA_ENV/bin/python"
    elif [[ -n "${CONDA_PREFIX:-}" ]]; then
        CONDA_ENV="$CONDA_PREFIX"
        WSP_PYTHON="$CONDA_PREFIX/bin/python"
    else
        WSP_PYTHON="$(command -v python || true)"
    fi
fi
if [[ -z "$WSP_PYTHON" || ! -x "$WSP_PYTHON" ]]; then
    echo "ERROR: Set CONDA_ENV or WSP_PYTHON to a valid Python environment" >&2
    exit 1
fi
if [[ -n "${CONDA_ENV:-}" ]]; then
    export PATH="$CONDA_ENV/bin:$PATH"
fi
export CONDA_ENV WSP_PYTHON
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

export WSP_TASK="${WSP_TASK:-agilex-validation}"
export WSP_MODE="${WSP_MODE:-interactive}"
export VISUAL_PROMPT="${VISUAL_PROMPT:-none}"
export AGILEX_TRANSPORT="${AGILEX_TRANSPORT:-hdf5}"
export WSP_SERVER_HOST="${WSP_SERVER_HOST:-0.0.0.0}"
export WSP_SERVER_PORT="${WSP_SERVER_PORT:-11451}"
export WSP_NODE_NAME="${WSP_NODE_NAME:-WSP}"

case "$WSP_MODE" in
    interactive|auto) ;;
    *) echo "ERROR: WSP_MODE must be interactive or auto" >&2; exit 2 ;;
esac
case "$VISUAL_PROMPT" in
    none|goal|demo) ;;
    *) echo "ERROR: VISUAL_PROMPT must be none, goal, or demo" >&2; exit 2 ;;
esac

LIVE_ARGS=()
case "$AGILEX_TRANSPORT" in
    hdf5) ;;
    manifold)
        export WSP_MAX_STEPS="${WSP_MAX_STEPS:-50000}"
        LIVE_ARGS+=(--live-hardware)
        ;;
    *) echo "ERROR: AGILEX_TRANSPORT must be hdf5 or manifold" >&2; exit 2 ;;
esac

if [[ -n "${WSP_EVAL:-}" ]]; then
    exec "$WSP_EVAL" \
        --config "$ROOT/configs/eval/agilex.yaml" \
        "${LIVE_ARGS[@]}" \
        "$@"
fi

exec "$WSP_PYTHON" -m worldscape_policy.cli.evaluate \
    --config "$ROOT/configs/eval/agilex.yaml" \
    "${LIVE_ARGS[@]}" \
    "$@"
