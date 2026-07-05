#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_NAME="${CONFIG_NAME:-pi05sf_jax_robodojo_v21_offcache}"
CACHE_DIR="${SF_CACHE_DIR:-${CACHE_DIR:-${ROOT_DIR}/results/sf_cache}}"
LEROBOT_HOME="${HF_LEROBOT_HOME:-${XPL_DATA_ROOT:-${ROOT_DIR}/../data}}"
PI05_BASE_PATH="${PI05_BASE_PATH:-${ROOT_DIR}/checkpoints/pi05_base}"
VGGT_WEIGHT_PATH="${VGGT_WEIGHT_PATH:-${ROOT_DIR}/checkpoints/VGGT-1B}"

PRECACHE_NPROC="${PRECACHE_NPROC:-${NPROC:-1}}"
PRECACHE_BATCHES="${PRECACHE_NUM_BATCHES:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RESET_SF_CACHE="${RESET_SF_CACHE:-0}"
REUSE_SF_CACHE="${REUSE_SF_CACHE:-0}"
MIN_CACHE_SLOTS="${MIN_CACHE_SLOTS:-1000}"
MIN_EPISODES="${MIN_EPISODES:-2}"
MIN_CHUNKS="${MIN_CHUNKS:-2}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if [[ ! -f scripts/precache_vggt_sf_cache.py || ! -f scripts/run_pi05sf_jax_offcache.sh ]]; then
  echo "Run this script from the Spatial_Forcing/open_sf repository root." >&2
  exit 2
fi

if [[ ! -d "${PI05_BASE_PATH}/params" || ! -d "${PI05_BASE_PATH}/assets" ]]; then
  echo "PI05_BASE_PATH must contain params/ and assets/: ${PI05_BASE_PATH}" >&2
  exit 2
fi

if [[ ! -f "${VGGT_WEIGHT_PATH}/model.pt" ]]; then
  echo "VGGT model.pt not found under VGGT_WEIGHT_PATH: ${VGGT_WEIGHT_PATH}" >&2
  exit 2
fi

if [[ -e "${CACHE_DIR}" ]]; then
  if [[ "${RESET_SF_CACHE}" == "1" ]]; then
    backup="${CACHE_DIR}.backup.$(date '+%Y%m%d_%H%M%S')"
    log "Backing up existing cache: ${CACHE_DIR} -> ${backup}"
    mv "${CACHE_DIR}" "${backup}"
  elif [[ "${REUSE_SF_CACHE}" == "1" ]]; then
    log "Reusing existing cache directory and filling missing entries: ${CACHE_DIR}"
  else
    cat >&2 <<EOF
Cache directory already exists: ${CACHE_DIR}

This script refuses to reuse an existing cache by default. Choose one:

  RESET_SF_CACHE=1 $0    # move old cache aside and rebuild
  REUSE_SF_CACHE=1 $0    # reuse existing cache and fill missing entries

EOF
    exit 2
  fi
fi

LOCAL_CACHE_ROOT="${OPENPI_LOCAL_CACHE_ROOT:-/tmp/openpi-cache-$(hostname)}"
mkdir -p "${LOCAL_CACHE_ROOT}/hf/datasets" "${LOCAL_CACHE_ROOT}/jax"

export HF_LEROBOT_HOME="${LEROBOT_HOME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${LOCAL_CACHE_ROOT}/hf/datasets}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${LOCAL_CACHE_ROOT}/jax}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export PI05_BASE_PATH VGGT_WEIGHT_PATH
export SF_CACHE_DIR="${CACHE_DIR}"
export SF_CACHE_MODE=readwrite
export PRECACHE_NUM_BATCHES="${PRECACHE_BATCHES}"
export NPROC="${PRECACHE_NPROC}" MIN_CACHE_SLOTS MIN_EPISODES MIN_CHUNKS

log "Starting VGGT SF precache"
log "config=${CONFIG_NAME} cache_dir=${CACHE_DIR} nproc=${PRECACHE_NPROC} batch_size=${BATCH_SIZE} workers=${NUM_WORKERS}"

if [[ "${PRECACHE_NPROC}" -gt 1 ]]; then
  uv run --no-sync torchrun --standalone --nproc_per_node="${PRECACHE_NPROC}" \
    scripts/precache_vggt_sf_cache.py "${CONFIG_NAME}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}"
else
  uv run --no-sync python scripts/precache_vggt_sf_cache.py "${CONFIG_NAME}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}"
fi

log "Checking VGGT SF cache summaries and on-disk cache shape"
uv run --no-sync python - <<'PY'
import json
import os
from pathlib import Path

cache_dir = Path(os.environ["SF_CACHE_DIR"])
expected_ranks = int(os.environ.get("NPROC", "1"))
min_slots = int(os.environ.get("MIN_CACHE_SLOTS", "1000"))
min_episodes = int(os.environ.get("MIN_EPISODES", "2"))
min_chunks = int(os.environ.get("MIN_CHUNKS", "2"))

if not cache_dir.exists():
    raise SystemExit(f"cache directory does not exist: {cache_dir}")

if expected_ranks == 1:
    summary_paths = [cache_dir / "_precache_summary.json"]
else:
    summary_paths = sorted(cache_dir.glob("_precache_summary.rank*.json"))

missing = [str(path) for path in summary_paths if not path.exists()]
if missing:
    raise SystemExit(f"missing precache summary files: {missing}")
if len(summary_paths) != expected_ranks:
    raise SystemExit(f"expected {expected_ranks} summary files, found {len(summary_paths)}")

total_existing = 0
total_written = 0
total_failed = 0
total_refs = 0
for path in summary_paths:
    with path.open() as f:
        summary = json.load(f)
    existing = int(summary.get("existing", 0))
    written = int(summary.get("written", 0))
    failed = int(summary.get("failed", 0))
    total_existing += existing
    total_written += written
    total_failed += failed
    total_refs += existing + written
    print(
        f"{path.name}: rank={summary.get('rank')} batches={summary.get('num_batches')} "
        f"existing={existing} written={written} failed={failed}"
    )

if total_failed != 0:
    raise SystemExit(f"precache reported failed writes: {total_failed}")
if total_refs <= 0:
    raise SystemExit("precache summary has no processed cache refs")

mask_paths = sorted(cache_dir.glob("ds_*/ep_*/*.mask"))
if not mask_paths:
    raise SystemExit("no cache mask files found")

written_slots = 0
for path in mask_paths:
    written_slots += sum(1 for byte in path.read_bytes() if byte == 1)

episode_dirs = {path.parent for path in mask_paths}
chunk_bases = {path.with_suffix("") for path in mask_paths}
print(
    "cache_shape: "
    f"episodes={len(episode_dirs)} chunks={len(chunk_bases)} written_slots={written_slots} "
    f"total_summary_refs={total_refs}"
)

if written_slots < min_slots:
    raise SystemExit(f"too few written cache slots: {written_slots} < {min_slots}")
if len(episode_dirs) < min_episodes:
    raise SystemExit(f"too few episode cache dirs: {len(episode_dirs)} < {min_episodes}")
if len(chunk_bases) < min_chunks:
    raise SystemExit(f"too few cache chunks: {len(chunk_bases)} < {min_chunks}")

print(
    "VGGT SF cache check passed: "
    f"existing={total_existing}, written={total_written}, failed={total_failed}"
)
PY

log "Starting strict readonly JAX training"
export SF_CACHE_MODE=readonly
bash scripts/run_pi05sf_jax_offcache.sh "$@"
