#!/usr/bin/env bash
# =============================================================================
# Hy-VLA RoboDojo training: 2-node x 8-GPU recipe (total batch = 128, same as RoboTwin).
#
# Usage (run on EACH node; one MUST set INDEX=0 and act as chief):
#
#   # Node 0 (chief):
#   export CHIEF_IP=<chief-ip>  INDEX=0
#   bash scripts/train_robodojo_umi.sh
#
#   # Node N (worker):
#   export CHIEF_IP=<chief-ip>  INDEX=N
#   bash scripts/train_robodojo_umi.sh
#
# Env overrides (with defaults shown):
#   EXP_ID            hy_vla_robodojo_umi
#   EXP_ROOT          /path/to/experiments
#   PRETRAIN          tencent/Hy-VLA-UMI (override with local path)
#   HDF5_DIR          /path/to/robodojo/hdf5
#   NORM_PATH         <EXP_ROOT>/<EXP_ID>/norm_stats.pkl
#   NUM_MACHINES      2
#   NPROC_PER_NODE    8
#   MAIN_PORT         6688
#   HY_VLA_DIR        parent of this script
#   CHIEF_IP          (required)
#   INDEX             (required)  rank in [0, NUM_MACHINES)
# =============================================================================

set -euo pipefail

# --------- 1. CLI / env ---------
HY_VLA_DIR=${HY_VLA_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}

EXP_ID=${EXP_ID:-"hy_vla_robodojo_umi"}
EXP_ROOT=${EXP_ROOT:-"/path/to/experiments"}
PRETRAIN=${PRETRAIN:-"tencent/Hy-VLA-UMI"}
HDF5_DIR=${HDF5_DIR:-"/path/to/robodojo/hdf5"}
NORM_PATH=${NORM_PATH:-"${EXP_ROOT}/${EXP_ID}/norm_stats.pkl"}

NUM_MACHINES=${NUM_MACHINES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MAIN_PORT=${MAIN_PORT:-6688}

: "${CHIEF_IP:?CHIEF_IP must be set (rank-0 node IP)}"
: "${INDEX:?INDEX must be set (this node rank in [0, NUM_MACHINES))}"

NUM_PROCESSES=$(( NUM_MACHINES * NPROC_PER_NODE ))
CKPT_SAVE_DIR="${EXP_ROOT}/${EXP_ID}"

# --------- 2. Banner ---------
echo "========================================================"
echo "Hy-VLA RoboDojo training"
echo "EXP_ID         : ${EXP_ID}"
echo "ckpt_save_dir  : ${CKPT_SAVE_DIR}"
echo "pretrain (vla) : ${PRETRAIN}"
echo "hdf5_dir       : ${HDF5_DIR}"
echo "norm_stats.pkl : ${NORM_PATH}"
echo "topology       : ${NUM_MACHINES} nodes x ${NPROC_PER_NODE} gpus = ${NUM_PROCESSES} procs (total batch = $((NUM_PROCESSES * 8)) via yaml)"
echo "this node      : INDEX=${INDEX}  CHIEF_IP=${CHIEF_IP}  PORT=${MAIN_PORT}"
echo "========================================================"

# --------- 3. Launch ---------
cd "${HY_VLA_DIR}"

accelerate launch \
    --multi_gpu \
    --num_machines "${NUM_MACHINES}" \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_ip "${CHIEF_IP}" \
    --main_process_port "${MAIN_PORT}" \
    --machine_rank "${INDEX}" \
    hy_vla/train.py \
    exp_id="${EXP_ID}" \
    exp_name="${EXP_ID}" \
    ckpt_save_dir="${CKPT_SAVE_DIR}" \
    dataset=robodojo_hdf5 \
    dataset.hdf5_dir="${HDF5_DIR}" \
    dataset.mean_std_path="${NORM_PATH}" \
    model.vla_model_path="${PRETRAIN}"
