#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <ckpt_name> <source_ckpt_dir>" >&2
  echo "  ckpt_name: target link name under checkpoints/ (pass as eval.sh ckpt_name)" >&2
  echo "  source_ckpt_dir: experiment root containing checkpoint-* (e.g. .../RoboDojo-cotrain-arx_x5-joint-0)" >&2
  exit 1
fi

ckpt_name=$1
source_ckpt_dir=$2
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${POLICY_DIR}/checkpoints/${ckpt_name}"

if [[ ! -d "${source_ckpt_dir}" ]]; then
  echo "Source checkpoint directory not found: ${source_ckpt_dir}" >&2
  exit 1
fi

mkdir -p "${POLICY_DIR}/checkpoints"
ln -sfn "$(cd "${source_ckpt_dir}" && pwd)" "${target_dir}"

echo "[GR00T_N17] linked checkpoint:"
echo "  ${target_dir} -> ${source_ckpt_dir}"
