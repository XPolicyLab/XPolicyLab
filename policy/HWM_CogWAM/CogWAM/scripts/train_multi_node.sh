#!/usr/bin/env bash
# Multi-node launcher for CogWAM training.
#
# Rendezvous comes from plain environment variables, so this works under any
# scheduler that can set them (Slurm, a manual SSH fan-out, docker compose, a
# single box with COGWAM_NUM_MACHINES=1):
#
#   COGWAM_MAIN_PROCESS_IP    address of machine rank 0            (required)
#   COGWAM_MAIN_PROCESS_PORT  rendezvous port                      (default 29600)
#   COGWAM_MACHINE_RANK       this machine's index, 0-based        (required)
#   COGWAM_NUM_MACHINES       total machines in the job            (required)
#   COGWAM_GPUS_PER_NODE      GPUs visible to this machine         (default 8)
#
# Usage:
#   scripts/train_multi_node.sh [CONFIG_YAML] [extra --key value overrides...]

set -Eeuo pipefail
trap 'rc=$?; echo "[cogwam][ERROR] host=${HOSTNAME:-unknown} line=${BASH_LINENO[0]} rc=${rc}" >&2; exit "${rc}"' ERR

export PYTHONUNBUFFERED=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_YAML="${1:-configs/cogwam_robodojo_h25_eventmem_dino_multilayer_50k.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi
ACCEL_CONFIG="${COGWAM_ACCELERATE_CONFIG:-configs/accelerate/zero2.yaml}"

[[ -f "${CONFIG_YAML}" ]] || { echo "[cogwam][ERROR] missing training config: ${CONFIG_YAML}" >&2; exit 1; }
[[ -f "${ACCEL_CONFIG}" ]] || { echo "[cogwam][ERROR] missing accelerate config: ${ACCEL_CONFIG}" >&2; exit 1; }

# --- required assets --------------------------------------------------------
for required_name in COGWAM_BASE_VLM COGWAM_DINO_MODEL COGWAM_DATA_ROOT COGWAM_RUN_ROOT; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "[cogwam][ERROR] missing env: ${required_name} (see the Installation section of README.md)" >&2
    exit 2
  fi
done

# --- file descriptors -------------------------------------------------------
# The LeRobot video backend keeps one descriptor per open shard per dataloader
# worker, and they are opened lazily and never closed. Each rank runs two
# loaders (physical + semantic) with their own worker pools, so the limit has to
# be raised per process. Fail here rather than an hour into training on whichever
# rank happens to sample the widest.
_nofile_int() { [[ "$1" == "unlimited" ]] && echo 1073741824 || echo "$1"; }
REQUIRED_NOFILE=65536
MINIMUM_NOFILE=8192
_soft_nofile="$(_nofile_int "$(ulimit -Sn)")"
_hard_nofile="$(_nofile_int "$(ulimit -Hn)")"
if (( _soft_nofile < REQUIRED_NOFILE )); then
  _target_nofile="${REQUIRED_NOFILE}"
  if (( _hard_nofile < REQUIRED_NOFILE )); then
    _target_nofile="${_hard_nofile}"
  fi
  ulimit -n "${_target_nofile}" 2>/dev/null \
    || echo "[cogwam][WARN] could not raise RLIMIT_NOFILE to ${_target_nofile}" >&2
fi
_soft_nofile="$(_nofile_int "$(ulimit -Sn)")"
echo "[cogwam] RLIMIT_NOFILE soft=$(ulimit -Sn) hard=$(ulimit -Hn)"
if (( _soft_nofile < MINIMUM_NOFILE )); then
  echo "[cogwam][ERROR] RLIMIT_NOFILE soft limit ${_soft_nofile} is below ${MINIMUM_NOFILE}" >&2
  exit 2
fi

# --- VLM preflight ----------------------------------------------------------
# RynnBrain 1.1 is a Qwen3.5 checkpoint. An older transformers silently lacks
# the class, and the failure would otherwise surface only after every rank has
# paid for dataset construction.
python3 - "${COGWAM_BASE_VLM}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
config_path = checkpoint / "config.json"
if not config_path.is_file():
    raise SystemExit(f"[cogwam][ERROR] VLM checkpoint config is missing: {config_path}")
payload = json.loads(config_path.read_text(encoding="utf-8"))
if payload.get("model_type") != "qwen3_5":
    raise SystemExit(
        f"[cogwam][ERROR] base VLM must have model_type=qwen3_5, got {payload.get('model_type')!r}"
    )
if not (checkpoint / "model.safetensors").is_file() and not (
    checkpoint / "model.safetensors.index.json"
).is_file():
    raise SystemExit(f"[cogwam][ERROR] VLM weights are missing under {checkpoint}")

try:
    import transformers
except Exception as exc:
    raise SystemExit(f"[cogwam][ERROR] cannot import transformers: {exc}") from exc
if getattr(transformers, "Qwen3_5ForConditionalGeneration", None) is None:
    raise SystemExit(
        "[cogwam][ERROR] this checkpoint requires transformers>=5.2.0 with "
        f"Qwen3_5ForConditionalGeneration; active version={transformers.__version__}"
    )
print(
    f"[cogwam] VLM preflight ok: checkpoint={checkpoint} transformers={transformers.__version__} "
    f"hidden={payload.get('text_config', {}).get('hidden_size')}"
)
PY

# --- rendezvous -------------------------------------------------------------
num_machines="${COGWAM_NUM_MACHINES:?COGWAM_NUM_MACHINES is required}"
machine_rank="${COGWAM_MACHINE_RANK:?COGWAM_MACHINE_RANK is required}"
main_process_ip="${COGWAM_MAIN_PROCESS_IP:?COGWAM_MAIN_PROCESS_IP is required}"
main_process_port="${COGWAM_MAIN_PROCESS_PORT:-29600}"
gpus_per_node="${COGWAM_GPUS_PER_NODE:-8}"
total_gpus=$((num_machines * gpus_per_node))

[[ "${num_machines}" =~ ^[1-9][0-9]*$ ]] || {
  echo "[cogwam][ERROR] invalid COGWAM_NUM_MACHINES=${num_machines}" >&2
  exit 1
}
[[ "${machine_rank}" =~ ^[0-9]+$ && "${machine_rank}" -lt "${num_machines}" ]] || {
  echo "[cogwam][ERROR] invalid COGWAM_MACHINE_RANK=${machine_rank} for ${num_machines} machines" >&2
  exit 1
}

# Fragmentation from the alternating 12-frame physical / 6-frame semantic
# batches is what drives the tail-end OOMs on this model; expandable segments
# let the caching allocator give the blocks back.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[cogwam] host=${HOSTNAME:-unknown} rank=${machine_rank}/${num_machines} gpus=${gpus_per_node} total=${total_gpus}"
echo "[cogwam] rendezvous=${main_process_ip}:${main_process_port} config=${CONFIG_YAML}"

exec accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  --deepspeed_multinode_launcher standard \
  --main_process_ip "${main_process_ip}" \
  --main_process_port "${main_process_port}" \
  --machine_rank "${machine_rank}" \
  --num_machines "${num_machines}" \
  --num_processes "${total_gpus}" \
  -m cogwam.training.train \
  --config_yaml "${CONFIG_YAML}" \
  "$@"
