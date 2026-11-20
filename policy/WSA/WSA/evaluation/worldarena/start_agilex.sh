#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WORLDARENA_PLATFORM=agilex
export WSA_ROBOT_TYPE="${WSA_ROBOT_TYPE:-cobot_magic_max}"
export WSA_ACTION_MODE="${WSA_ACTION_MODE:-}"
export WSA_EXECUTE_CHUNK_SIZE="${WSA_EXECUTE_CHUNK_SIZE:-30}"

echo "WorldArena AgileX: loading original WSA-Base checkpoint"
exec bash "${SCRIPT_DIR}/serve.sh"
