#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/checkpoints/pi05_robodojo_${ECHO_POLICY_CHECKPOINT_STEP:-14077}"
SOURCE_PATH="${ECHO_POLICY_CHECKPOINT_PATH:-}"
REPO_ID="${ECHO_POLICY_CHECKPOINT_REPO:-cjgogo/RoboDojo-pi05-checkpoints}"
REMOTE_PATH="${ECHO_POLICY_CHECKPOINT_REMOTE_PATH:-checkpoints/sim-10task/${ECHO_POLICY_CHECKPOINT_STEP:-14077}}"
if [[ -n "${SOURCE_PATH}" ]]; then
  SOURCE_PATH="$(realpath -e "${SOURCE_PATH}")"
  [[ -d "${SOURCE_PATH}" ]] || { echo "Checkpoint path is not a directory: ${SOURCE_PATH}" >&2; exit 2; }
  mkdir -p "$(dirname "${TARGET}")"
  rm -rf "${TARGET}"
  ln -s "${SOURCE_PATH}" "${TARGET}"
  echo "Linked checkpoint: ${TARGET} -> ${SOURCE_PATH}"
  exit 0
fi
mkdir -p "${TARGET}"
TMP_DIR="$(mktemp -d "${SCRIPT_DIR}/.checkpoint-download.XXXXXX")"
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT
if command -v hf >/dev/null 2>&1; then
  hf download "${REPO_ID}" --include "${REMOTE_PATH}/**" --local-dir "${TMP_DIR}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${REPO_ID}" --include "${REMOTE_PATH}/**" --local-dir "${TMP_DIR}"
else
  echo "Install huggingface_hub (hf) in the policy environment." >&2
  exit 2
fi
rm -rf "${TARGET}"
mkdir -p "$(dirname "${TARGET}")"
mv "${TMP_DIR}/${REMOTE_PATH}" "${TARGET}"
echo "Downloaded ${REPO_ID}:${REMOTE_PATH} -> ${TARGET}"
