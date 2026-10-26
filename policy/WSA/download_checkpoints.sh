#!/usr/bin/env bash
set -euo pipefail

variant=${1:-base}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${variant}" in
    base)
        repo_id="zaleni/wsa-b-clean2rand-delta-causal"
        default_dir="${SCRIPT_DIR}/checkpoints/WSA-Base-RoboTwin"
        ;;
    *)
        echo "Usage: $0 [base] [output_dir]" >&2
        exit 1
        ;;
esac

output_dir=${2:-${default_dir}}
mkdir -p "${output_dir}"
hf download "${repo_id}" --local-dir "${output_dir}"
echo "[WSA] downloaded ${repo_id} to ${output_dir}"
