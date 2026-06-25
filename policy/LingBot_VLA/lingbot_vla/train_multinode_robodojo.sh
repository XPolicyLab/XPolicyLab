#!/usr/bin/env bash
# One-click multi-node LingBot-VLA training for RoboDojo ARX X5.
#
# Usage:
#   # Auto launch master (this node) + worker (remote) from master:
#   bash train_multinode_robodojo.sh
#
#   # Password login (pick one):
#   read -s SSH_PASS && export SSH_PASS && bash train_multinode_robodojo.sh
#   SSH_PASS='your_password' bash train_multinode_robodojo.sh   # avoid saving in shell history
#
#   # Manual two-terminal mode (no remote SSH needed):
#   bash train_multinode_robodojo.sh --master-only   # on 192.168.156.46
#   bash train_multinode_robodojo.sh --worker-only   # on 192.168.156.49
#
#   # Single-node 8-GPU fallback:
#   NNODES=1 bash train_multinode_robodojo.sh --local-only
#
# Environment overrides (optional):
#   MASTER_ADDR WORKER_HOST WORKER_USER MASTER_PORT NNODES
#   CONDA_SH CONDA_ENV TRAIN_DIR OUTPUT_DIR
#   EXTRA_TRAIN_ARGS="--train.micro_batch_size 4"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODE="auto" # auto | master-only | worker-only | local-only
for arg in "$@"; do
  case "${arg}" in
    --master-only) MODE="master-only" ;;
    --worker-only) MODE="worker-only" ;;
    --local-only) MODE="local-only"; NNODES=1 ;;
    -h|--help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      echo "Use --help for usage." >&2
      exit 1
      ;;
  esac
done

# ---------- cluster defaults (edit if needed) ----------
MASTER_ADDR="${MASTER_ADDR:-192.168.156.46}"
WORKER_HOST="${WORKER_HOST:-192.168.156.49}"
WORKER_USER="${WORKER_USER:-root}"
MASTER_PORT="${MASTER_PORT:-62500}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

CONDA_SH="${CONDA_SH:-/mnt/nfs/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-lingbot_vla}"
TRAIN_DIR="${TRAIN_DIR:-${SCRIPT_DIR}}"

MODEL_PATH="${MODEL_PATH:-/mnt/xspark-data/xspark_shared/model_weights/lingbot-vla-4b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/xspark-data/xspark_shared/model_weights/Qwen2.5-VL-3B-Instruct}"
DATA_PATH="${DATA_PATH:-/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21}"
NORM_STATS="${NORM_STATS:-assets/norm_stats/robodojo_sim_arx_x5.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/xspark-data/zijian/XPolicyLab_main/policy/LingBot_VLA/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-0}"

CONFIG="${CONFIG:-configs/vla/robotwin_load20000h.yaml}"
SEED="${SEED:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
MICRO_BATCH="${MICRO_BATCH:-8}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
TOKENIZER_MAX_LEN="${TOKENIZER_MAX_LEN:-72}"
ACTION_DIM="${ACTION_DIM:-14}"

HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/mnt/xspark-data/xspark_shared/lerobot}"
XPOLICY_CACHE_ROOT="${XPOLICY_CACHE_ROOT:-/mnt/xspark-data/zijian/.cache}"
HF_HOME="${HF_HOME:-${XPOLICY_CACHE_ROOT}/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
TMPDIR="${TMPDIR:-/mnt/xspark-data/zijian/tmp}"
WORKER_START_DELAY="${WORKER_START_DELAY:-5}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=no -o ConnectTimeout=10}"

EXTRA_ARGS=()
if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_TRAIN_ARGS}"
fi

TRAIN_ARGS=(
  tasks/vla/train_lingbotvla.py
  "${CONFIG}"
  --model.model_path "${MODEL_PATH}"
  --model.tokenizer_path "${TOKENIZER_PATH}"
  --data.train_path "${DATA_PATH}"
  --data.norm_stats_file "${NORM_STATS}"
  --data.data_name robotwin_robodojo
  --train.output_dir "${OUTPUT_DIR}"
  --train.seed "${SEED}"
  --train.num_train_epochs "${NUM_EPOCHS}"
  --train.save_steps "${SAVE_STEPS}"
  --train.save_epochs 1
  --train.micro_batch_size "${MICRO_BATCH}"
  --train.global_batch_size "${GLOBAL_BATCH}"
  --train.tokenizer_max_length "${TOKENIZER_MAX_LEN}"
  --train.action_dim "${ACTION_DIM}"
  --train.enable_activation_offload true
  --train.use_wandb false
  --train.use_compile false
  "${EXTRA_ARGS[@]}"
)

log() {
  echo "[train_multinode] $*"
}

shell_quote() {
  printf '%q' "$1"
}

build_train_env() {
  local node_rank=$1
  local log_file=$2
  cat <<EOF
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export NNODES=${NNODES}
export NODE_RANK=${node_rank}
export NPROC_PER_NODE=${NPROC_PER_NODE}
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME}
export XPOLICY_CACHE_ROOT=${XPOLICY_CACHE_ROOT}
export HF_HOME=${HF_HOME}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TMPDIR=${TMPDIR}
mkdir -p "\${HF_HOME}" "\${HF_DATASETS_CACHE}" "\${TRANSFORMERS_CACHE}" "\${TMPDIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[train_multinode] ERROR: conda.sh not found: ${CONDA_SH}" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
echo "[train_multinode] node_rank=${node_rank} python=\$(which python)"
python -c "import torch, lingbotvla, lerobot; print('env ok', torch.__version__, lerobot.__file__)"
cd $(shell_quote "${TRAIN_DIR}")
bash train_origin.sh $(printf '%q ' "${TRAIN_ARGS[@]}") 2>&1 | tee "${log_file}"
EOF
}

prewarm_dataset_cache() {
  log "Pre-building HF datasets arrow cache on shared storage (single process)..."
  if [[ ! -f "${CONDA_SH}" ]]; then
    log "ERROR: conda.sh not found: ${CONDA_SH}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  export HF_LEROBOT_HOME="${HF_LEROBOT_HOME}"
  export XPOLICY_CACHE_ROOT="${XPOLICY_CACHE_ROOT}"
  export HF_HOME="${HF_HOME}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
  export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
  export TMPDIR="${TMPDIR}"
  mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}"
  python - <<PY
import os
from datasets import load_dataset
from pathlib import Path

data_dir = Path("${DATA_PATH}") / "data"
cache_dir = os.environ["HF_DATASETS_CACHE"]
print(f"[prewarm] loading parquet from {data_dir}")
ds = load_dataset("parquet", data_dir=str(data_dir), split="train")
print(f"[prewarm] cache ready, rows={len(ds)}, cache={cache_dir}")
PY
}

preflight_local() {
  log "Preflight (local): GPUs, paths, conda"
  command -v nvidia-smi >/dev/null
  nvidia-smi -L | head -3
  [[ -d "${DATA_PATH}" ]] || { log "Missing data: ${DATA_PATH}"; exit 1; }
  [[ -f "${TRAIN_DIR}/${NORM_STATS}" ]] || { log "Missing norm stats: ${TRAIN_DIR}/${NORM_STATS}"; exit 1; }
  [[ -d "${MODEL_PATH}" ]] || { log "Missing model: ${MODEL_PATH}"; exit 1; }
  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
  fi
  python -c "import lingbotvla, torch, lerobot; print('torch', torch.__version__, 'lerobot', lerobot.__file__)"
}

preflight_remote() {
  log "Preflight (remote ${WORKER_USER}@${WORKER_HOST})"
  remote_bash "set -e
command -v nvidia-smi >/dev/null
nvidia-smi -L | wc -l
test -d $(shell_quote "${DATA_PATH}")
test -f $(shell_quote "${TRAIN_DIR}/${NORM_STATS}")
test -d $(shell_quote "${MODEL_PATH}")
if [[ -f $(shell_quote "${CONDA_SH}") ]]; then source $(shell_quote "${CONDA_SH}"); conda activate $(shell_quote "${CONDA_ENV}"); else echo missing conda.sh; exit 1; fi
echo remote_python=\$(which python)
python -c 'import lingbotvla, torch, lerobot; print(\"remote ok\", torch.__version__, lerobot.__file__)'"
}

remote_bash() {
  local remote_cmd=$1
  if [[ -n "${SSH_PASS:-}" ]]; then
    if ! command -v sshpass >/dev/null; then
      log "sshpass not found. Install: apt-get install -y sshpass"
      exit 1
    fi
    SSHPASS="${SSH_PASS}" sshpass -e ssh ${SSH_OPTS} "${WORKER_USER}@${WORKER_HOST}" "bash -lc $(shell_quote "${remote_cmd}")"
    return
  fi
  if ssh ${SSH_OPTS} -o BatchMode=yes "${WORKER_USER}@${WORKER_HOST}" "true" 2>/dev/null; then
    ssh ${SSH_OPTS} "${WORKER_USER}@${WORKER_HOST}" "bash -lc $(shell_quote "${remote_cmd}")"
    return
  fi
  log "SSH key login failed. Enter password for ${WORKER_USER}@${WORKER_HOST}:"
  read -rs SSH_PASS
  echo
  export SSH_PASS
  remote_bash "${remote_cmd}"
}

launch_node() {
  local node_rank=$1
  local log_file="log_node${node_rank}_$(date +%Y%m%d_%H%M%S).txt"
  log "Launch node_rank=${node_rank}, log=${log_file}"
  bash -lc "$(build_train_env "${node_rank}" "${log_file}")"
}

launch_worker_remote() {
  local remote_script
  remote_script="$(build_train_env 1 "log_node1.txt")"
  log "Starting remote worker on ${WORKER_HOST} in background..."
  if [[ -n "${SSH_PASS:-}" ]]; then
    SSHPASS="${SSH_PASS}" sshpass -e ssh ${SSH_OPTS} -f "${WORKER_USER}@${WORKER_HOST}" \
      "bash -lc $(shell_quote "${remote_script}")"
  elif ssh ${SSH_OPTS} -o BatchMode=yes "${WORKER_USER}@${WORKER_HOST}" "true" 2>/dev/null; then
    ssh ${SSH_OPTS} -f "${WORKER_USER}@${WORKER_HOST}" \
      "bash -lc $(shell_quote "${remote_script}")"
  else
    log "SSH key login failed. Enter password for ${WORKER_USER}@${WORKER_HOST}:"
    read -rs SSH_PASS
    echo
    SSHPASS="${SSH_PASS}" sshpass -e ssh ${SSH_OPTS} -f "${WORKER_USER}@${WORKER_HOST}" \
      "bash -lc $(shell_quote "${remote_script}")"
  fi
}

main() {
  if [[ "${MODE}" == "local-only" ]]; then
    NNODES=1
    export NNODES
    preflight_local
    prewarm_dataset_cache
    launch_node 0
    return
  fi

  if [[ "${MODE}" == "master-only" ]]; then
    export NNODES MASTER_ADDR MASTER_PORT
    preflight_local
    prewarm_dataset_cache
    launch_node 0
    return
  fi

  if [[ "${MODE}" == "worker-only" ]]; then
    export NNODES MASTER_ADDR MASTER_PORT
    preflight_local
    launch_node 1
    return
  fi

  # auto: orchestrate from master node
  if [[ "${NNODES}" != "2" ]]; then
    log "auto mode expects NNODES=2 (got ${NNODES}). Use --local-only for single node."
    exit 1
  fi

  preflight_local
  preflight_remote
  prewarm_dataset_cache

  log "Cluster: NNODES=${NNODES}, MASTER=${MASTER_ADDR}:${MASTER_PORT}"
  log "micro_batch=${MICRO_BATCH}, global_batch=${GLOBAL_BATCH}"
  log "output_dir=${OUTPUT_DIR}"

  launch_worker_remote
  log "Waiting ${WORKER_START_DELAY}s for worker to initialize..."
  sleep "${WORKER_START_DELAY}"
  launch_node 0
}

main "$@"
