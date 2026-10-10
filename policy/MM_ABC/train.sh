#!/bin/bash
set -euo pipefail

bench_name=${1:?usage: train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [overrides...]}
ckpt_name=${2:?}
env_cfg_type=${3:?}
action_type=${4:?}
seed=${5:?}
gpu_id=${6:?}
shift 6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMABC_ROOT=${MMABC_ROOT:-"${POLICY_DIR}/MM_ABC"}
# shellcheck source=resolve_python.sh
source "${POLICY_DIR}/resolve_python.sh"
export MMABC_PYTHON=${MMABC_PYTHON:-$(resolve_python "${MMABC_ENV:-none}")}

case "${MODE:-scratch}" in
    scratch)  default_config=configs/train/mobile.yaml ;;
    pretrain) default_config=configs/train/mobile_pretrain.yaml ;;
    *) echo "[MM_ABC] MODE must be scratch or pretrain, got ${MODE}" >&2; exit 1 ;;
esac
TRAIN_CONFIG=${TRAIN_CONFIG:-${default_config}}

if [[ "${env_cfg_type}" != "m92uw" || "${action_type}" != "joint" ]]; then
    echo "[MM_ABC] supports env_cfg_type=m92uw action_type=joint" >&2
    exit 1
fi

ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
mkdir -p "${POLICY_DIR}/checkpoints/${ckpt_setting}"
echo "[MM_ABC] mode=${MODE:-scratch} config=${TRAIN_CONFIG} python=${MMABC_PYTHON} -> checkpoints/${ckpt_setting}"

args=("${TRAIN_CONFIG}" "checkpoint.root=checkpoints/${ckpt_setting}" "seed=${seed}" "run_name=mmabc-${ckpt_name}" "$@")
cd "${MMABC_ROOT}"
if [[ "${NODES:-single}" == "multi" ]]; then
    if [[ "${SUPERVISED:-0}" == "1" ]]; then
        exec bash scripts/train_supervised.sh "${args[@]}"
    fi
    exec bash scripts/train_multinode.sh "${args[@]}"
fi
if [[ "${gpu_id}" != "all" ]]; then
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export NPROC=$(( $(tr -cd ',' <<<"${gpu_id}" | wc -c) + 1 ))
fi
exec bash scripts/train_single_node.sh "${args[@]}"
