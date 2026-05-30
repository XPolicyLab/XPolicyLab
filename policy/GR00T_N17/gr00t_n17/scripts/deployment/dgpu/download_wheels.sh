#!/bin/bash
# download_wheels.sh — Download prebuilt aarch64 wheels for uv sync.
#
# These wheels are required by pyproject.toml but are not committed to git
# (see gr00t_n17/.gitignore). Run this once before `uv sync --python 3.10`.
#
# Usage:
#   proxyup   # optional, if GitHub is unreachable directly
#   bash scripts/deployment/dgpu/download_wheels.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELS_DIR="${SCRIPT_DIR}/wheels"
UPSTREAM_REPO="${UPSTREAM_REPO:-NVIDIA/Isaac-GR00T}"
UPSTREAM_REF="${UPSTREAM_REF:-main}"
BASE_URL="https://media.githubusercontent.com/media/${UPSTREAM_REPO}/${UPSTREAM_REF}/scripts/deployment/dgpu/wheels"

# pyproject.toml path sources for aarch64 Linux + Python 3.10
WHEELS=(
    "torchcodec-0.10.0a0-cp310-cp310-linux_aarch64.whl:102400"
    "flash_attn-2.7.4.post1-cp310-cp310-linux_aarch64.whl:1048576"
)

mkdir -p "$WHEELS_DIR"

download_wheel() {
    local spec="$1"
    local name="${spec%%:*}"
    local min_bytes="${spec##*:}"
    local dest="${WHEELS_DIR}/${name}"
    local url="${BASE_URL}/${name}"

    if [[ -f "$dest" ]]; then
        local size
        size="$(stat -c%s "$dest")"
        if [[ "$size" -ge "$min_bytes" ]]; then
            echo "Already exists, skipping: $dest ($(numfmt --to=iec "$size" 2>/dev/null || echo "${size} bytes"))"
            return 0
        fi
        echo "Removing incomplete or invalid wheel: $dest"
        rm -f "$dest"
    fi

    echo "Downloading $name ..."
    curl -fL --retry 3 --retry-delay 5 -C - -o "$dest" "$url"

    local size
    size="$(stat -c%s "$dest")"
    if [[ "$size" -lt "$min_bytes" ]]; then
        echo "Error: $dest is too small (${size} bytes). Got an LFS pointer or a failed download." >&2
        rm -f "$dest"
        exit 1
    fi
}

echo "Downloading GR00T aarch64 wheels from ${UPSTREAM_REPO} (${UPSTREAM_REF})"
echo "Target directory: ${WHEELS_DIR}"
echo ""

for wheel in "${WHEELS[@]}"; do
    download_wheel "$wheel"
done

echo ""
echo "Done. Wheels are ready at:"
ls -lh "$WHEELS_DIR"
