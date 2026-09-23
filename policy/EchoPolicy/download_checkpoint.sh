#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/checkpoints/pi05_robodojo_59999"
SOURCE_PATH="${ECHO_POLICY_CHECKPOINT_PATH:-}"
REPO_ID="${ECHO_POLICY_CHECKPOINT_REPO:-}"
if [[ -n "${SOURCE_PATH}" ]]; then
  SOURCE_PATH="$(realpath -e "${SOURCE_PATH}")"
  [[ -d "${SOURCE_PATH}" ]] || { echo "Checkpoint path is not a directory: ${SOURCE_PATH}" >&2; exit 2; }
  mkdir -p "$(dirname "${TARGET}")"
  rm -rf "${TARGET}"
  ln -s "${SOURCE_PATH}" "${TARGET}"
  echo "Linked checkpoint: ${TARGET} -> ${SOURCE_PATH}"
  exit 0
fi
if [[ -z "${REPO_ID}" ]]; then
  echo "XPolicyLab publishes the Pi05 adapter, not its binary weights." >&2
  echo "Set ECHO_POLICY_CHECKPOINT_PATH to a mounted checkpoint directory, or" >&2
  echo "set ECHO_POLICY_CHECKPOINT_REPO to an approved public Hugging Face repo." >&2
  exit 2
fi
mkdir -p "${TARGET}"
if command -v hf >/dev/null 2>&1; then
  exec hf download "${REPO_ID}" --local-dir "${TARGET}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  exec huggingface-cli download "${REPO_ID}" --local-dir "${TARGET}"
else
  echo "Install huggingface_hub (hf) in the policy environment." >&2
  exit 2
fi
