#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: bash train.sh <data_mix> <memory_ablation_mode> <keyframe_memory_policy> [extra_args...]"
    echo "Example: bash train.sh robodojo pure_image_keyframe_memory teacher"
    exit 1
fi

data_mix=${1}
memory_ablation_mode=${2}
keyframe_memory_policy=${3}
shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_TRAIN_SCRIPT="${SCRIPT_DIR}/source_eventvla/examples/RoboTwin-Mem/train_files/run_eventvla_train_batch.sh"
DEFAULT_TRAIN_SCRIPT="${LOCAL_TRAIN_SCRIPT}"
TRAIN_SCRIPT="${EVENTVLA_TRAIN_SCRIPT:-${DEFAULT_TRAIN_SCRIPT}}"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "[EventVLA][error] train script not found: ${TRAIN_SCRIPT}" >&2
    echo "[EventVLA][hint] Check LOCAL_TRAIN_SCRIPT or set EVENTVLA_TRAIN_SCRIPT manually." >&2
    exit 1
fi

TRAIN_SCRIPT_DIR="$(cd "$(dirname "${TRAIN_SCRIPT}")" && pwd)"

echo "[EventVLA] train_script=${TRAIN_SCRIPT}"
echo "[EventVLA] data_mix=${data_mix}"
echo "[EventVLA] memory_ablation_mode=${memory_ablation_mode}"
echo "[EventVLA] keyframe_memory_policy=${keyframe_memory_policy}"
if [[ $# -gt 0 ]]; then
    echo "[EventVLA] extra_args=$*"
fi

cd "${TRAIN_SCRIPT_DIR}"
bash "${TRAIN_SCRIPT}" "${data_mix}" "${memory_ablation_mode}" "${keyframe_memory_policy}" "$@"
