#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/checkpoints/pi05_robodojo_59999"
REPO_ID="${ECHO_POLICY_CHECKPOINT_REPO:-}"
if [[ -z "${REPO_ID}" ]]; then
  echo "Set ECHO_POLICY_CHECKPOINT_REPO to the public Hugging Face repository before downloading." >&2
  exit 2
fi
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is required (install it in the policy environment)." >&2
  exit 2
fi
mkdir -p "${TARGET}"
exec huggingface-cli download "${REPO_ID}" --local-dir "${TARGET}"
