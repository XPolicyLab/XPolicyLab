#!/usr/bin/env bash
# Public packaging: image naming and deployment paths; see metadata/source_manifest.json.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
if [ -f "$ROOT/config.env" ]; then set -a; source "$ROOT/config.env"; set +a; fi
IMAGE=${IMAGE:-joyai-vl/agent:preview}
docker build --build-arg BASE="${BASE_IMAGE:-python:3.11-slim-bookworm}" \
  -t "$IMAGE" "$ROOT/agent/api"
docker image inspect "$IMAGE" --format '{{.Id}} {{.Size}}'
