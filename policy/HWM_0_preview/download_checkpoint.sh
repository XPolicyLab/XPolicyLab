#!/bin/bash
set -euo pipefail

usage() {
    echo 'Usage: bash download_checkpoint.sh [repo_id] [revision]'
    echo 'Downloads the complete bundle to HWM-0-Preview-RoboDojo beside this script.'
    echo 'Default repository: HorizonRobotics/HWM-0-Preview-RoboDojo; revision: main.'
    echo 'Override the repository with the first argument or HWM_HF_REPO_ID.'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# > 2 )); then
    usage >&2
    exit 2
fi

repo_id="${1:-${HWM_HF_REPO_ID:-HorizonRobotics/HWM-0-Preview-RoboDojo}}"
revision="${2:-main}"
if [[ -z "${repo_id}" ]]; then
    usage >&2
    exit 2
fi
if ! command -v hf >/dev/null 2>&1; then
    echo 'Hugging Face CLI not found. Install huggingface_hub in the active policy environment.' >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec hf download "${repo_id}" --revision "${revision}" \
    --local-dir "${SCRIPT_DIR}/HWM-0-Preview-RoboDojo"
