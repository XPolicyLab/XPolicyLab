#!/usr/bin/env bash
# Build the agent image. The Codex binary and the robo client are copied into a throwaway context.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
if [ -f "$ROOT/config.env" ]; then set -a; source "$ROOT/config.env"; set +a; fi
CODEX_BIN=$(readlink -f "${CODEX_BIN:-$HOME/.local/bin/codex}")
TAG=${TAG:-roboshell/agent-codex:0.0.1}
CTX=$(mktemp -d "$ROOT/.build-ctx.XXXXXX")
trap 'rm -rf "$CTX"' EXIT
mkdir -p "$CTX/codex-bin"
cp "$CODEX_BIN" "$(dirname "$CODEX_BIN")/codex-code-mode-host" "$CTX/codex-bin/"
cp "$ROOT/roboshell/client/robo.py" "$ROOT/agent/codex/Dockerfile" "$CTX/"
"$ROOT/agent/build_docs.sh" >/dev/null
docker build --build-arg BASE="${BASE_IMAGE:-debian:bookworm-slim}" --build-arg APT_MIRROR="${APT_MIRROR:-deb.debian.org}" --build-arg WITH_PYTHON_TOOLS="${WITH_PYTHON_TOOLS:-1}" -t "$TAG" "$CTX"
"$CODEX_BIN" --version > "$ROOT/agent/codex/CODEX_VERSION"
docker image inspect "$TAG" --format '{{.Id}} {{.Size}}'
