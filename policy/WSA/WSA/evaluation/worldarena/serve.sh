#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
POLICY_FILE="${SCRIPT_DIR}/policy.py"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WSA_ROOT}/src:${PYTHONPATH:-}"
if [[ -n "${WORLDARENA_ROOT:-}" ]]; then
  export PYTHONPATH="${WORLDARENA_ROOT}:${PYTHONPATH}"
fi

if ! python -c "import real_world_benchmark" >/dev/null 2>&1; then
  echo "Cannot import real_world_benchmark. Install the official A-side package first:" >&2
  echo "  pip install -e \"\${WORLDARENA_ROOT}/real_world_benchmark\"" >&2
  echo "  pip install -r \"\${WORLDARENA_ROOT}/real_world_benchmark/requirements-a.txt\"" >&2
  exit 2
fi

TRANSPORT="${WORLDARENA_TRANSPORT:-hub}"
case "${TRANSPORT}" in
  hub)
    : "${HUB_POLICY_URL:?Set HUB_POLICY_URL to the organizer-provided .../policy URL}"
    : "${POLICY_ID:?Set POLICY_ID to the exact worker-key assigned by the organizer}"
    export WSA_POLICY_ID="${WSA_POLICY_ID:-${POLICY_ID}}"
    ARGS=(
      python -m real_world_benchmark.serve_policy_worldarena
      "${POLICY_FILE}"
      --hub-url "${HUB_POLICY_URL}"
      --worker-key "${POLICY_ID}"
    )
    if [[ -n "${HUB_TOKEN:-}" ]]; then
      ARGS+=(--hub-token "${HUB_TOKEN}")
    fi
    ;;
  ws)
    export WSA_POLICY_ID="${WSA_POLICY_ID:-${POLICY_ID:-wsa_${WORLDARENA_PLATFORM}}}"
    ARGS=(
      python -m real_world_benchmark.serve_policy_worldarena
      "${POLICY_FILE}"
      --host "${WORLDARENA_HOST:-0.0.0.0}"
      --port "${WORLDARENA_PORT:-8000}"
    )
    ;;
  *)
    echo "WORLDARENA_TRANSPORT must be hub or ws, got: ${TRANSPORT}" >&2
    exit 2
    ;;
esac

echo "Starting original WSA-Base WorldArena worker"
echo "  platform=${WORLDARENA_PLATFORM:-<unset>}"
echo "  transport=${TRANSPORT}"
echo "  checkpoint=${WSA_CHECKPOINT:-<dry-run>}"
echo "  execute_chunk_size=${WSA_EXECUTE_CHUNK_SIZE:-1}"
exec "${ARGS[@]}"
