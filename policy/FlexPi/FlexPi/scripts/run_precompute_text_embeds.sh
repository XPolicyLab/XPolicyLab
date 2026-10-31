#!/usr/bin/env bash
# Precompute T5 embeddings for any dataset with a Hydra data config.
#
# Usage:
#   bash scripts/run_precompute_text_embeds.sh <data_config_name> [extra hydra args...]
#
# Examples:
#   bash scripts/run_precompute_text_embeds.sh robotwin
#   NUM_GPUS=8 bash scripts/run_precompute_text_embeds.sh robotwin
#   bash scripts/run_precompute_text_embeds.sh robotwin overwrite=true
#
# Reads every dataset dir in configs/data/<name>.yaml, encodes each task string
# with Wan2.2 T5, and writes one .pt per unique prompt into the config's
# text_embedding_cache_dir. Idempotent — pass overwrite=true to force.
#
# Env: NUM_GPUS (default 1; >1 uses torchrun), PY_BIN (default `python`).
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <data_config_name> [extra hydra args...]" >&2
    echo "Example: $0 robotwin" >&2
    exit 1
fi

DATA_CFG="$1"; shift
EXTRA_ARGS=( "$@" )

cd "$(dirname "$0")/.."   # FlexPi repo root

CONFIG_PATH="configs/data/${DATA_CFG}.yaml"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: ${CONFIG_PATH} not found." >&2
    exit 1
fi

NUM_GPUS="${NUM_GPUS:-1}"
PY_BIN="${PY_BIN:-python}"

echo "[precompute] data config : configs/data/${DATA_CFG}.yaml"
echo "[precompute] num_gpus    : ${NUM_GPUS}"
echo "[precompute] python      : ${PY_BIN}"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "[precompute] extra args  : ${EXTRA_ARGS[*]}"

# Needed for cfg.model.{model_id, tokenizer_model_id}; any Wan2.2 config works.
MODEL_CFG="${MODEL_CFG:-flexpi}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
        scripts/precompute_text_embeds.py \
        data="${DATA_CFG}" \
        model="${MODEL_CFG}" \
        "${EXTRA_ARGS[@]}"
else
    "${PY_BIN}" scripts/precompute_text_embeds.py \
        data="${DATA_CFG}" \
        model="${MODEL_CFG}" \
        "${EXTRA_ARGS[@]}"
fi

# Post-run sanity — count cached files vs unique prompts
CACHE_DIR=$("${PY_BIN}" -c "
from omegaconf import OmegaConf
c = OmegaConf.load('configs/data/${DATA_CFG}.yaml')
print(c.train.text_embedding_cache_dir)
")
if [[ -n "${CACHE_DIR}" && -d "${CACHE_DIR}" ]]; then
    N_CACHE=$(find "${CACHE_DIR}" -name "*.pt" | wc -l)
    echo
    echo "[precompute] done. ${N_CACHE} .pt files in ${CACHE_DIR}/"
fi
