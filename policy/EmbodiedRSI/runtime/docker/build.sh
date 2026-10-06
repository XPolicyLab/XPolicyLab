#!/usr/bin/env bash
# Build entirely from public downloads; no host CLI or credentials are copied.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${1:-embodiedrsi-agent:xpolicy}"
if [[ $# -gt 0 ]]; then shift; fi
build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT
cp "$SCRIPT_DIR/Dockerfile" "$build_dir/Dockerfile"
cp "$SCRIPT_DIR/codex.sha256" "$build_dir/codex.sha256"
# Docker builds do not inherit the shell's proxy environment automatically.
# Pass variable names so credentials, if any, stay out of the command line.
build_args=()
for proxy_name in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; do
    if [[ -v "$proxy_name" ]]; then
        build_args+=(--build-arg "$proxy_name")
    fi
done
docker build --pull "${build_args[@]}" "$@" --tag "$image" "$build_dir"
docker run --rm --network none "$image" codex --version
docker image inspect --format '{{.Id}}' "$image"
